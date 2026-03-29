#!/usr/bin/env python3
"""
주간 증시 동향 & 우량주 TOP 5 추천 리포트
- US / KR 시장 데이터 수집 (yfinance, FinanceDataReader)
- Gemini AI 시장 전망 생성
- KOSPI 상위 100 종목 스크리닝 (기술적 + 수급 + 밸류에이션)
- Notion 데이터베이스에 리포트 자동 업로드
"""

import os
import re
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import yfinance as yf
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import google.generativeai as genai

# ──────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  환경변수 (GitHub Actions Secrets → env)
# ──────────────────────────────────────────────────────────────
NOTION_TOKEN       = os.environ["NOTION_TOKEN"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
GEMINI_API_KEY     = os.environ["GEMINI_API_KEY"]


NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
    "Referer": "https://finance.naver.com",
}

RANK_EMOJI = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]


# ══════════════════════════════════════════════════════════════
#  STEP 1 · 시장 데이터 수집
# ══════════════════════════════════════════════════════════════

def _last_friday() -> datetime:
    """오늘 기준 가장 최근 금요일"""
    today = datetime.now()
    days_back = (today.weekday() - 4) % 7
    if days_back == 0:
        days_back = 7
    return today - timedelta(days=days_back)


def fetch_us_market() -> dict:
    """yfinance로 지난 금요일 S&P500 / 나스닥 마감 데이터 취득"""
    friday = _last_friday()
    start  = (friday - timedelta(days=10)).strftime("%Y-%m-%d")
    end    = (friday + timedelta(days=1)).strftime("%Y-%m-%d")

    def _download(ticker: str) -> tuple[float, float]:
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 2:
            raise RuntimeError(f"yfinance 데이터 부족: {ticker}")
        close = df["Close"].squeeze()
        last  = float(close.iloc[-1])
        prev  = float(close.iloc[-2])
        return last, (last - prev) / prev * 100.0

    sp_c,  sp_d  = _download("^GSPC")
    nq_c,  nq_d  = _download("^IXIC")

    return dict(
        date=friday.strftime("%Y-%m-%d"),
        sp500_close=sp_c,  sp500_change=sp_d,
        nasdaq_close=nq_c, nasdaq_change=nq_d,
    )


def fetch_kr_market() -> dict:
    """FinanceDataReader로 코스피 / 코스닥 주간 등락 취득"""
    end   = datetime.now()
    start = (end - timedelta(days=20)).strftime("%Y-%m-%d")
    e_str = end.strftime("%Y-%m-%d")

    ki = fdr.DataReader("KS11", start, e_str)["Close"].astype(float)
    kq = fdr.DataReader("KQ11", start, e_str)["Close"].astype(float)

    def _weekly_chg(s: pd.Series) -> float:
        if len(s) < 6:
            return 0.0
        return float((s.iloc[-1] - s.iloc[-6]) / s.iloc[-6] * 100.0)

    return dict(
        kospi_close=float(ki.iloc[-1]),   kospi_weekly=_weekly_chg(ki),
        kosdaq_close=float(kq.iloc[-1]),  kosdaq_weekly=_weekly_chg(kq),
    )


# ══════════════════════════════════════════════════════════════
#  STEP 2 · Gemini AI 시장 전망
# ══════════════════════════════════════════════════════════════

def gemini_analysis(us: dict, kr: dict) -> str:
    """Gemini 1.5 Flash로 이번 주 시장 전망 3~4줄 생성"""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = (
        "아래 지난주 글로벌 증시 데이터를 바탕으로, "
        "이번 주 코스피/코스닥 시장 전망을 한국 개인 투자자 관점에서 "
        "3~4줄로 핵심만 간결하게 작성하세요. 수치 근거를 반드시 포함하세요.\n\n"
        f"[미국 증시 - 지난 금요일 마감]\n"
        f"• S&P 500 : {us['sp500_close']:>10,.2f}pt  ({us['sp500_change']:+.2f}%)\n"
        f"• 나스닥   : {us['nasdaq_close']:>10,.2f}pt  ({us['nasdaq_change']:+.2f}%)\n\n"
        f"[한국 증시 - 주간 등락]\n"
        f"• 코스피   : {kr['kospi_close']:>10,.2f}pt  ({kr['kospi_weekly']:+.2f}%)\n"
        f"• 코스닥   : {kr['kosdaq_close']:>10,.2f}pt  ({kr['kosdaq_weekly']:+.2f}%)\n"
    )

    resp = model.generate_content(prompt)
    return resp.text.strip()


# ══════════════════════════════════════════════════════════════
#  STEP 3 · 종목 스크리너
# ══════════════════════════════════════════════════════════════

def _kospi_top100() -> pd.DataFrame:
    """FinanceDataReader로 코스피 시총 상위 100 종목 리스트 반환"""
    df = fdr.StockListing("KOSPI")

    # 컬럼명 통일 (버전마다 다를 수 있음)
    col_rename = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("symbol", "code", "ticker", "종목코드"):
            col_rename[col] = "Code"
        elif cl in ("marcap", "marketcap", "market_cap", "시가총액"):
            col_rename[col] = "Marcap"
        elif cl in ("name", "종목명"):
            col_rename[col] = "Name"
    if col_rename:
        df = df.rename(columns=col_rename)

    required = {"Code", "Name", "Marcap"}
    missing  = required - set(df.columns)
    if missing:
        raise RuntimeError(f"StockListing 컬럼 없음: {missing}. 실제 컬럼: {list(df.columns)}")

    df["Marcap"] = pd.to_numeric(df["Marcap"], errors="coerce")
    df = df.dropna(subset=["Marcap"]).sort_values("Marcap", ascending=False).head(100)
    return df[["Code", "Name"]].reset_index(drop=True)


def _get_technicals(code: str) -> Optional[dict]:
    """
    FinanceDataReader로 기술적 지표 계산
    - 5 / 20 / 60 / 120일 이동평균
    - 52주 최고·최저가, 현재 위치 (%)
    - 골든크로스(5일선이 20일선을 당일 돌파) 여부
    """
    try:
        end   = datetime.now()
        start = (end - timedelta(days=420)).strftime("%Y-%m-%d")
        e_str = end.strftime("%Y-%m-%d")

        df = fdr.DataReader(code, start, e_str)
        if df is None or df.empty or len(df) < 125:
            return None

        c   = df["Close"].astype(float)
        cur = float(c.iloc[-1])

        ma  = {n: float(c.rolling(n).mean().iloc[-1]) for n in (5, 20, 60, 120)}
        mp  = {n: float(c.rolling(n).mean().iloc[-2]) for n in (5, 20)}

        hi52 = float(c.tail(252).max())
        lo52 = float(c.tail(252).min())
        pos52 = (cur - lo52) / (hi52 - lo52) * 100.0 if hi52 != lo52 else 50.0

        golden_cross = (ma[5] > ma[20]) and (mp[5] <= mp[20])

        return dict(
            current_price=cur,
            ma5=ma[5], ma20=ma[20], ma60=ma[60], ma120=ma[120],
            hi52=hi52, lo52=lo52, pos52=round(pos52, 1),
            golden_cross=golden_cross,
            ma5_gt_ma20=(ma[5] > ma[20]),
        )
    except Exception as e:
        logger.debug(f"[{code}] 기술적 지표 오류: {e}")
        return None


def _scrape_naver_supply(code: str, price: float) -> dict:
    """
    네이버 금융 frgn.naver 크롤링
    → 최근 5영업일 외국인·기관 순매수 수량 → 억원 환산
    """
    try:
        url  = f"https://finance.naver.com/item/frgn.naver?code={code}"
        resp = requests.get(url, headers=NAVER_HEADERS, timeout=12)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table", class_="type2")
        if not table:
            return {}

        # ── 헤더 파싱으로 컬럼 위치 동적 결정 ───────────────
        th_tags      = table.find_all("th")
        header_texts = [th.get_text(strip=True) for th in th_tags]

        def _find_col(keyword_list: list[str], default: int) -> int:
            for i, h in enumerate(header_texts):
                if any(k in h for k in keyword_list):
                    return i
            return default

        f_col = _find_col(["외국인", "외인"], 3)
        i_col = _find_col(["기관"],           5)

        f_qty = i_qty = 0
        count = 0

        for tr in table.find_all("tr"):
            if count >= 5:
                break
            tds = tr.find_all("td")
            if not tds:
                continue

            # 날짜 형식(YYYY.MM.DD) 행만 처리
            date_str = tds[0].get_text(strip=True)
            if not re.match(r"\d{4}\.\d{2}\.\d{2}", date_str):
                continue
            if len(tds) <= max(f_col, i_col):
                continue

            def _parse_qty(td_elem) -> int:
                raw = td_elem.get_text(strip=True).replace(",", "").replace("+", "").strip()
                return int(raw) if raw and re.match(r"-?\d+$", raw) else 0

            f_qty += _parse_qty(tds[f_col])
            i_qty += _parse_qty(tds[i_col])
            count += 1

        bil = 1e8
        return dict(
            foreign_bil=round(f_qty * price / bil, 1),
            inst_bil   =round(i_qty * price / bil, 1),
        )
    except Exception as e:
        logger.debug(f"[{code}] 수급 크롤링 오류: {e}")
        return {}


def _scrape_naver_val(code: str) -> dict:
    """
    네이버 금융 main.naver 크롤링
    → PER, PBR, 시가배당률 추출
    """
    try:
        url  = f"https://finance.naver.com/item/main.naver?code={code}"
        resp = requests.get(url, headers=NAVER_HEADERS, timeout=12)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        def _em_val(elem_id: str) -> Optional[float]:
            tag = soup.find("em", id=elem_id)
            if not tag:
                return None
            try:
                return float(tag.get_text(strip=True).replace(",", ""))
            except ValueError:
                return None

        per = _em_val("_per")
        pbr = _em_val("_pbr")

        # 시가배당률 – id 후보 순서대로 시도
        div = None
        for dvr_id in ("_dvr", "_dps"):
            tag = soup.find("em", id=dvr_id)
            if tag:
                try:
                    div = float(tag.get_text(strip=True).replace(",", "").replace("%", ""))
                    break
                except ValueError:
                    pass

        # id로 못 찾으면 텍스트 패턴 탐색 (배당수익률 N.N%)
        if div is None:
            for tag in soup.find_all(string=re.compile(r"배당수익률|시가배당률")):
                parent = tag.find_parent()
                if parent:
                    next_el = parent.find_next_sibling()
                    if next_el:
                        m = re.search(r"([\d.]+)%", next_el.get_text())
                        if m:
                            try:
                                div = float(m.group(1))
                                break
                            except ValueError:
                                pass

        return dict(per=per, pbr=pbr, div=div)
    except Exception as e:
        logger.debug(f"[{code}] 밸류에이션 크롤링 오류: {e}")
        return dict(per=None, pbr=None, div=None)


# ──────────────────────────────────────────────────────────────
#  점수 산정
# ──────────────────────────────────────────────────────────────

def _score(tech: dict, supply: dict) -> int:
    s = 0

    # ① 이동평균 정배열 (max 40점) ─ 4쌍 각 10점
    ma_pairs = [
        (tech["current_price"], tech["ma5"]),
        (tech["ma5"],           tech["ma20"]),
        (tech["ma20"],          tech["ma60"]),
        (tech["ma60"],          tech["ma120"]),
    ]
    for a, b in ma_pairs:
        if a > b:
            s += 10

    # ② 외국인+기관 5일 합산 수급 (max 30점)
    total_bil = supply.get("foreign_bil", 0.0) + supply.get("inst_bil", 0.0)
    if   total_bil > 500: s += 30
    elif total_bil > 200: s += 22
    elif total_bil > 50:  s += 14
    elif total_bil > 0:   s += 7

    # ③ 골든크로스 (5일선이 20일선을 당일 돌파) +10점
    if tech.get("golden_cross"):
        s += 10

    # ④ 52주 고점 대비 하락폭 + 5>20일선 돌파 → 저평가 반등 (max 20점)
    drop_pct = (tech["hi52"] - tech["current_price"]) / tech["hi52"] * 100.0
    if   drop_pct > 30 and tech.get("ma5_gt_ma20"): s += 20
    elif drop_pct > 15 and tech.get("ma5_gt_ma20"): s += 12
    elif drop_pct > 5:                               s += 5

    return s


def run_screener() -> list[dict]:
    """코스피 시총 상위 100개 스크리닝 → TOP 5 반환"""
    logger.info("KOSPI 상위 100 종목 로딩 중...")
    top100 = _kospi_top100()
    results: list[dict] = []

    for idx, row in top100.iterrows():
        code = str(row["Code"]).zfill(6)
        name = row["Name"]
        logger.info(f"  [{idx+1:3d}/100] {name} ({code})")

        tech = _get_technicals(code)
        if not tech:
            continue

        price  = tech["current_price"]
        supply = _scrape_naver_supply(code, price)
        val    = _scrape_naver_val(code)
        score  = _score(tech, supply)

        results.append(dict(
            code=code, name=name, score=score,
            tech=tech, supply=supply, val=val,
        ))
        time.sleep(0.7)  # 네이버 크롤링 레이트 리밋 방지

    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]
    logger.info("TOP 5: " + " / ".join(f"{s['name']}({s['score']}점)" for s in top5))
    return top5


# ══════════════════════════════════════════════════════════════
#  STEP 4 · 리포트 포매팅
# ══════════════════════════════════════════════════════════════

def _fmt_stock_block(rank: int, s: dict) -> str:
    """
    요구사항의 출력 포맷을 100% 그대로 재현한 종목 블록 문자열 반환
    """
    t   = s["tech"]
    sup = s["supply"]
    val = s["val"]

    price = int(t["current_price"])
    score = s["score"]

    # 분석 상태
    drop_pct = (t["hi52"] - price) / t["hi52"] * 100.0
    if   drop_pct > 25: analysis_status = "저평가 재평가 대기"
    elif drop_pct > 10: analysis_status = "조정 후 회복세"
    else:               analysis_status = "강세 지속"

    # 사이클
    if   score >= 70: cyc_emo, cyc_txt = "🔥", "매수"
    elif score >= 50: cyc_emo, cyc_txt = "📈", "관망"
    else:             cyc_emo, cyc_txt = "⚠️",  "주의"

    # 이평 정배열 판정
    align_conds = [
        t["current_price"] > t["ma5"],
        t["ma5"]           > t["ma20"],
        t["ma20"]          > t["ma60"],
        t["ma60"]          > t["ma120"],
    ]
    n = sum(align_conds)
    if   n == 4: ma_label = "🟢완벽정배열"
    elif n == 3: ma_label = "🟡부분정배열"
    elif n == 2: ma_label = "🟠혼조세"
    else:        ma_label = "🔴역배열"

    # 밸류에이션
    per_s = f"{val['per']:.1f}"  if val.get("per") else "N/A"
    pbr_s = f"{val['pbr']:.2f}"  if val.get("pbr") else "N/A"
    div_s = f"{val['div']:.1f}"  if val.get("div") else "0.0"

    # 수급
    f_bil = sup.get("foreign_bil", 0.0)
    i_bil = sup.get("inst_bil",    0.0)
    tot   = f_bil + i_bil

    # 목표가 (+10%) / 손절가 (-4%)
    tgt  = f"{int(price * 1.10):,}"
    stop = f"{int(price * 0.96):,}"

    lines = [
        f"{RANK_EMOJI[rank - 1]} {s['name']} {price:,}원",
        f"{analysis_status} | {score}점",
        f"사이클: {cyc_emo} {cyc_txt} 52주{t['pos52']}%",
    ]
    if t.get("ma5_gt_ma20"):
        lines.append("✅5일선>20일선")
    lines += [
        f"밸류: PER{per_s} PBR{pbr_s} 배당{div_s}%",
        f"수급5일: 외{f_bil:+.0f}억 기{i_bil:+.0f}억 합{tot:+.0f}억",
        f"이평: {ma_label}",
        f"🎯{tgt} 🛡️{stop}",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  STEP 5 · Notion 업로드
# ══════════════════════════════════════════════════════════════

def _blk_h2(text: str) -> dict:
    return {"object": "block", "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": text}}]}}

def _blk_txt(text: str, bold: bool = False) -> dict:
    return {"object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text",
                                          "text": {"content": text},
                                          "annotations": {"bold": bold}}]}}

def _blk_div() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}

def _blk_callout(text: str, emoji: str) -> dict:
    return {"object": "block", "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": text}}],
                "icon": {"type": "emoji", "emoji": emoji},
            }}


def upload_to_notion(title: str, analysis: str,
                     top5: list[dict], us: dict, kr: dict) -> str:
    """Notion 데이터베이스에 새 페이지 생성 후 리포트 업로드"""
    api_url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization":  f"Bearer {NOTION_TOKEN}",
        "Content-Type":   "application/json",
        "Notion-Version": "2022-06-28",
    }

    children: list[dict] = [
        # ── 섹션 1: 글로벌 시장 동향 ──────────────────────────
        _blk_h2("🌍 글로벌 시장 동향 (지난 금요일 기준)"),
        _blk_txt(
            f"🇺🇸 S&P 500  {us['sp500_close']:>10,.2f}pt"
            f"  ({us['sp500_change']:+.2f}%)"
        ),
        _blk_txt(
            f"🇺🇸 나스닥   {us['nasdaq_close']:>10,.2f}pt"
            f"  ({us['nasdaq_change']:+.2f}%)"
        ),
        _blk_txt(
            f"🇰🇷 코스피   {kr['kospi_close']:>10,.2f}pt"
            f"  (주간 {kr['kospi_weekly']:+.2f}%)"
        ),
        _blk_txt(
            f"🇰🇷 코스닥   {kr['kosdaq_close']:>10,.2f}pt"
            f"  (주간 {kr['kosdaq_weekly']:+.2f}%)"
        ),
        _blk_div(),
        # ── 섹션 2: AI 시장 전망 ──────────────────────────────
        _blk_h2("🔮 이번 주 시장 전망 (Gemini AI 분석)"),
    ]

    for line in analysis.split("\n"):
        stripped = line.strip()
        if stripped:
            children.append(_blk_txt(stripped))

    children += [_blk_div(), _blk_h2("🏆 이번 주 추천 종목 TOP 5")]

    # ── 섹션 3: TOP 5 종목 (callout 블록) ─────────────────────
    for i, stock in enumerate(top5, 1):
        block_text = _fmt_stock_block(i, stock)
        children.append(_blk_callout(block_text, RANK_EMOJI[i - 1]))
        if i < len(top5):
            children.append(_blk_div())

    payload = {
        "parent":     {"database_id": NOTION_DATABASE_ID},
        "icon":       {"type": "emoji", "emoji": "📊"},
        "properties": {
            "Name": {"title": [{"text": {"content": title}}]},
        },
        "children": children,
    }

    resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Notion API 오류 {resp.status_code}: {resp.text[:400]}"
        )

    page_url = resp.json().get("url", "(URL 없음)")
    logger.info(f"Notion 페이지 생성 완료 → {page_url}")
    return page_url


# ══════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("═" * 50)
    logger.info("  주간 증시 리포트 생성 시작")
    logger.info("═" * 50)

    # ── 1. 시장 데이터 ──────────────────────────────────────
    logger.info("[1/4] 글로벌 시장 데이터 수집 중...")
    us = fetch_us_market()
    kr = fetch_kr_market()
    logger.info(
        f"      S&P500={us['sp500_close']:.1f} ({us['sp500_change']:+.2f}%)  "
        f"KOSPI={kr['kospi_close']:.1f} ({kr['kospi_weekly']:+.2f}%)"
    )

    # ── 2. Gemini 분석 ───────────────────────────────────────
    logger.info("[2/4] Gemini AI 시장 전망 생성 중...")
    analysis = gemini_analysis(us, kr)
    logger.info(f"      {analysis[:80].replace(chr(10),' ')}...")

    # ── 3. 종목 스크리닝 ─────────────────────────────────────
    logger.info("[3/4] KOSPI 상위 100 종목 스크리닝 중...")
    top5 = run_screener()

    if not top5:
        logger.error("스크리닝 결과가 없습니다. 프로세스를 종료합니다.")
        sys.exit(1)

    # ── 4. Notion 업로드 ─────────────────────────────────────
    today = datetime.now()
    title = (
        f"📈 주간 증시 리포트 "
        f"{today.strftime('%Y.%m.%d')} "
        f"W{today.isocalendar()[1]}"
    )
    logger.info(f"[4/4] Notion 업로드 중... 제목: {title}")
    upload_to_notion(title, analysis, top5, us, kr)

    logger.info("═" * 50)
    logger.info("  완료!")
    logger.info("═" * 50)


if __name__ == "__main__":
    main()

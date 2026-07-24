"""PDF 근거 텍스트에 등장하는 KOSPI200 기업을 DB(company_profile_chunks/financial_summaries)와
매칭해 재무제표·프로필 요약을 끌어온다.

배경(사용자 지적, 2026-07-24): ERD의 "기업명 및 티커" 노드는 사용자가 티커를 먼저 고르는 게
아니라(사용자는 상용 LLM 쓰듯 PDF+쿼리만 준다) — 파이프라인이 PDF 근거에서 DB가 이미 아는
기업(KOSPI200)이 언급됐는지 스스로 찾아 연결해야 한다는 뜻. dense(임베딩) 매칭을 먼저 시도했으나
`company_profile_chunks`가 영문 위주라 한글 기업명 질의와 임베딩 공간에서 잘 안 붙어 부정확했다
(4건 중 1건만 부분 히트, `파이프라인_최종정리_핸드오프.md` §4). 이름→티커는 의미 유사도가 아니라
정확 조회(lookup) 문제라고 판단해 도구를 바꿨다: `pykrx.stock.get_market_ticker_name()`(로그인
불필요, 실측 199/199 성공)으로 KOSPI200 종목의 정확한 한글명을 미리 받아
`KOSPI200_output/kospi200_korean_names.json`에 캐시해두고, PDF 근거 텍스트에 그 한글명이
문자열로 등장하는지 정확 매칭한다.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KOREAN_NAMES_PATH = ROOT / "KOSPI200_output" / "kospi200_korean_names.json"

_name_map_cache = None


def get_korean_name_map() -> dict:
    """{ticker: 한글명} — 프로세스당 1회 로드(파일 I/O 반복 방지)."""
    global _name_map_cache
    if _name_map_cache is None:
        _name_map_cache = json.loads(KOREAN_NAMES_PATH.read_text(encoding="utf-8"))
    return _name_map_cache


def find_mentioned_companies(text: str, name_map: dict = None) -> list:
    """text 안에 KOSPI200 한글명이 문자열로 등장하는 티커를 전부 찾는다 — 정확 부분문자열 매칭,
    임베딩/LLM 호출 없음(이름→티커는 조회 문제라 dense보다 이게 더 정확하고 공짜).

    [실측으로 발견한 함정] 3자 이하 그룹/약칭명은 계열사 전체 이름의 접두/내장 문자열로 오탐되기
    쉽다 — 예: "GS"(078930.KS, GS그룹 지주사)가 전혀 다른 회사인 "GS건설" 텍스트 안에서 매칭되고,
    "동서"(026960.KS, 동서식품)가 "아이에스동서"(건설사, 우리 199개 목록엔 없음) 안에서 매칭되는
    식. 두 회사 다 우리 KOSPI200 199개 목록에 정확한 전체이름으로는 없어서 겹침-억제만으론 못
    거른다 — 대신 3자 이하 이름은 앞뒤가 한글로 안 이어질 때만(공백/구두점/문자열 끝) 인정한다.
    "한샘"(2자)처럼 뒤에 공백/숫자가 오는 진짜 단독 언급은 그대로 잡히고, "GS건설"처럼 뒤에
    한글이 바로 이어지는 임베디드 오탐만 걸러진다."""
    name_map = name_map or get_korean_name_map()
    found = []
    for ticker, name in name_map.items():
        if len(name) < 2:
            continue
        if len(name) <= 3:
            pattern = re.compile(rf"(?<![가-힣]){re.escape(name)}(?![가-힣])")
        else:
            pattern = re.compile(re.escape(name))
        if pattern.search(text):
            found.append({"ticker": ticker, "name": name})
    return found


def fetch_company_db_context(db_url: str, matched: list) -> str:
    """매칭된 티커들의 financial_summaries.summary + company_profile_chunks.summary를 DB에서
    직접 조회 — 이미 정확한 티커를 알고 있으므로 검색(유사도)이 아니라 PK 조회라 결과가 틀릴
    여지가 없다. 반환: LLM 프롬프트에 그대로 넣을 수 있는 텍스트 블록(매칭 없으면 빈 문자열)."""
    if not matched:
        return ""
    import psycopg2

    tickers = [m["ticker"] for m in matched]
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select ticker, summary from financial_summaries "
                "where ticker = any(%s) and summary is not null",
                (tickers,),
            )
            fin = dict(cur.fetchall())
            cur.execute(
                "select id, summary from company_profile_chunks "
                "where id = any(%s) and summary is not null",
                (tickers,),
            )
            prof = dict(cur.fetchall())
    finally:
        conn.close()

    lines = []
    for m in matched:
        t, name = m["ticker"], m["name"]
        if fin.get(t):
            lines.append(f"[{name}({t}) 재무제표 요약 — DB]\n{fin[t]}")
        if prof.get(t):
            lines.append(f"[{name}({t}) 기업 프로필 요약 — DB]\n{prof[t]}")

    # [재일] README 아키텍처에서 비어 있던 화살표 연결 —
    # `NEWS(관련 뉴스 Sentiment Analysis) -> META(기업 메타데이터 DB) -> LLM`.
    # 뉴스 감성은 지금까지 헤드라인 수집까지만 있고 생성 단계에 도달하지 못했다. 여기서 붙여야
    # 실제로 프롬프트에 실린다(이 함수가 LLM 컨텍스트를 만드는 유일한 지점이기 때문).
    # 테이블이 없거나 해당 티커 데이터가 없으면 빈 문자열이 와서 기존 동작 그대로 유지된다.
    try:
        from news_sentiment_link import fetch_news_sentiment_context
        news_block = fetch_news_sentiment_context(db_url, tickers)
        if news_block:
            lines.append(news_block)
    except Exception:
        pass  # 뉴스 감성은 보조 신호 — 실패해도 재무/프로필 컨텍스트는 그대로 나가야 한다

    return "\n\n".join(lines)


def resolve_and_fetch(db_url: str, text: str) -> tuple:
    """find_mentioned_companies() + fetch_company_db_context() 조합 편의 함수.
    반환: (matched: list[{"ticker","name"}], db_context: str)."""
    matched = find_mentioned_companies(text)
    db_context = fetch_company_db_context(db_url, matched)
    return matched, db_context

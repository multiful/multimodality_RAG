# -*- coding: utf-8 -*-
"""[재일] 뉴스 감성분석 -> 기업 메타데이터 DB 연결 — README 아키텍처 다이어그램에서 유일하게
비어 있던 화살표(`NEWS["관련 뉴스 Sentiment Analysis"] --> META["기업 메타데이터 DB"]`)를 잇는다.

README(§단계별 핵심 내용)에 이렇게 적혀 있었다:
  > "관련 뉴스 Sentiment Analysis"는 현재 뉴스 헤드라인 수집(data_collection/fetch_news.py)까지만
  > 구현됐고, **감성분석 후 기업 메타데이터 DB에 결합하는 부분**과 "Query 타입"별 라우팅은 아직
  > 설계 단계입니다.

이미 있는 조각(전부 로컬, 유료 API 0회):
  - 뉴스 코퍼스 : `KOSPI200_output/kospi200_layer3/{ticker}_layer3_news.md`
                  (네이버 검색 API로 수집 -> 4요소 가중 랭킹 -> LLM 검증으로 선정된 상위 기사들)
  - 감성 모델   : `src/finance/layer3_news_sentiment.py`의 KR-FinBert-SC(snunlp/KR-FinBert-SC)
  - 소비 지점   : `company_entity_linking.fetch_company_db_context()` — 매칭된 티커의 재무/프로필
                  요약을 LLM 프롬프트에 넣어주는 함수. 즉 여기에 붙어야 생성까지 실제로 도달한다.

없던 것은 그 사이의 배선뿐이라, 이 모듈이 그 배선을 담당한다:
  layer3 뉴스 md 파싱 -> KR-FinBert 감성 채점 -> 티커별 집계 -> Supabase `company_news_sentiment`
  적재 -> `fetch_news_sentiment_context()`로 프롬프트 블록 생성.

주의: 네이버 API 자격증명(.env)이 없으면 실시간 수집은 불가하므로 **캐시된 md만** 사용한다.
자격증명이 생기면 `src/finance/layer3_naver_news.search_news()`로 신규 수집분을 같은 경로에
떨어뜨리기만 하면 이 모듈은 그대로 동작한다.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAYER3_DIR = ROOT / "KOSPI200_output" / "kospi200_layer3"
TABLE_DDL = """
create table if not exists company_news_sentiment (
    ticker         text primary key,
    name_ko        text,
    sentiment      double precision,   -- -1.0(매우 부정) ~ +1.0(매우 긍정)
    label          text,               -- very_positive / positive / neutral / negative / very_negative
    n_articles     integer,
    avg_age_days   double precision,   -- 근거 기사 평균 경과일(신선도)
    headlines      text,               -- 근거 헤드라인(프롬프트에 그대로 인용)
    updated_at     timestamptz default now()
)
"""

# 집계 점수를 다시 라벨로 되돌릴 때의 경계(layer3_news_sentiment.LABEL_SCORES와 같은 축)
_LABEL_BOUNDS = [(0.75, "very_positive"), (0.25, "positive"), (-0.25, "neutral"),
                 (-0.75, "negative"), (-9.9, "very_negative")]
NEWS_CONTEXT_MAX_HEADLINES = 3


@dataclass
class NewsItem:
    title: str
    lead: str
    link: str
    pub_date: datetime


_TITLE_RE = re.compile(r"^###\s+\d+\.\s+(.*)$")
_LINK_RE = re.compile(r"^-\s*링크:\s*(\S+)")
_DATE_RE = re.compile(r"^-\s*게재일:\s*(\S+)")
_LEAD_RE = re.compile(r"^-\s*리드문:\s*(.*)$")


def parse_layer3_markdown(path: Path) -> tuple[str, list[NewsItem]]:
    """layer3 뉴스 md에서 (기업 한글명, 기사 목록)을 뽑는다. 파일 형식은 수집기가 만든 고정
    템플릿(`### N. 제목` / `- 링크:` / `- 게재일:` / `- 리드문:`)이라 정규식으로 충분하다."""
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^#\s+\S+\s+\((.+?)\)", text, re.M)
    name_ko = m.group(1).strip() if m else path.stem.split("_")[0]

    items, cur = [], None
    for line in text.splitlines():
        t = _TITLE_RE.match(line)
        if t:
            if cur and cur.get("title"):
                items.append(cur)
            cur = {"title": t.group(1).strip(), "lead": "", "link": "", "pub_date": None}
            continue
        if cur is None:
            continue
        for rx, key in ((_LINK_RE, "link"), (_DATE_RE, "pub_date"), (_LEAD_RE, "lead")):
            mm = rx.match(line)
            if mm:
                cur[key] = mm.group(1).strip()
    if cur and cur.get("title"):
        items.append(cur)

    out = []
    for it in items:
        try:
            pd = datetime.fromisoformat(it["pub_date"]) if it["pub_date"] else datetime.now(timezone.utc)
        except Exception:
            pd = datetime.now(timezone.utc)
        if pd.tzinfo is None:
            pd = pd.replace(tzinfo=timezone.utc)
        out.append(NewsItem(title=it["title"], lead=it["lead"], link=it["link"], pub_date=pd))
    return name_ko, out


def _to_scored_article(item: NewsItem):
    """감성 모델이 기대하는 ScoredArticle 형태로 변환. 랭킹 관련 필드는 감성 채점에 쓰이지
    않으므로(모델 입력은 제목+리드문) 중립값으로 채운다."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))          # 감성 모듈이 `src.finance...` 절대 임포트를 씀
    from src.finance.layer3_news_selection import ScoredArticle
    return ScoredArticle(title=item.title, description=item.lead, link=item.link,
                         originallink=item.link, pub_date=item.pub_date,
                         match_position="title", rel=0.0, recency_decay=1.0,
                         src=1.0, src_tier_label="", event=0, score=0.0)


def _label_for(score: float) -> str:
    for lo, lab in _LABEL_BOUNDS:
        if score >= lo:
            return lab
    return "neutral"


def score_ticker(ticker: str, md_path: Path) -> dict | None:
    """한 티커의 layer3 뉴스를 KR-FinBert로 채점해 집계 레코드를 만든다."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.finance.layer3_news_sentiment import score_news_sentiment

    name_ko, items = parse_layer3_markdown(md_path)
    if not items:
        return None
    articles = [_to_scored_article(i) for i in items]
    s_news, age_days, _results = score_news_sentiment(articles, name_ko)
    return {
        "ticker": ticker, "name_ko": name_ko,
        "sentiment": round(float(s_news), 4), "label": _label_for(s_news),
        "n_articles": len(items), "avg_age_days": round(float(age_days), 2),
        "headlines": " | ".join(i.title for i in items[:NEWS_CONTEXT_MAX_HEADLINES]),
    }


def build_from_cache(db_url: str, layer3_dir: Path = LAYER3_DIR) -> list[dict]:
    """캐시된 layer3 뉴스 전체를 채점해 `company_news_sentiment`에 upsert하고 결과를 돌려준다."""
    import psycopg2

    rows = []
    for md in sorted(layer3_dir.glob("*_layer3_news.md")):
        ticker = md.name.split("_layer3_news")[0]
        rec = score_ticker(ticker, md)
        if rec:
            rows.append(rec)
    if not rows:
        return []

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(TABLE_DDL)
            for r in rows:
                cur.execute(
                    "insert into company_news_sentiment "
                    "(ticker,name_ko,sentiment,label,n_articles,avg_age_days,headlines,updated_at) "
                    "values (%(ticker)s,%(name_ko)s,%(sentiment)s,%(label)s,%(n_articles)s,"
                    "%(avg_age_days)s,%(headlines)s, now()) "
                    "on conflict (ticker) do update set "
                    "name_ko=excluded.name_ko, sentiment=excluded.sentiment, label=excluded.label, "
                    "n_articles=excluded.n_articles, avg_age_days=excluded.avg_age_days, "
                    "headlines=excluded.headlines, updated_at=now()",
                    r)
        conn.commit()
    finally:
        conn.close()
    return rows


def fetch_news_sentiment_context(db_url: str, tickers: list) -> str:
    """매칭된 티커들의 뉴스 감성을 LLM 프롬프트용 텍스트 블록으로 만든다.
    테이블이 아직 없거나 데이터가 없으면 빈 문자열(생성 단계는 그대로 진행)."""
    if not tickers:
        return ""
    import psycopg2

    try:
        conn = psycopg2.connect(db_url)
    except Exception:
        return ""
    try:
        with conn.cursor() as cur:
            cur.execute("select to_regclass('public.company_news_sentiment')")
            if cur.fetchone()[0] is None:
                return ""
            cur.execute(
                "select ticker,name_ko,sentiment,label,n_articles,avg_age_days,headlines "
                "from company_news_sentiment where ticker = any(%s)", (list(tickers),))
            rows = cur.fetchall()
    except Exception:
        return ""
    finally:
        conn.close()

    if not rows:
        return ""
    lines = []
    for t, name, s, lab, n, age, heads in rows:
        lines.append(
            f"[{name}({t}) 최근 뉴스 감성 — DB]\n"
            f"감성 점수 {s:+.2f} ({lab}), 근거 기사 {n}건, 평균 {age:.1f}일 전\n"
            f"주요 헤드라인: {heads}")
    return "\n\n".join(lines)


def main():
    """CLI: 캐시된 layer3 뉴스를 채점해 DB에 적재한다."""
    import os
    for line in open(ROOT / ".env", encoding="utf-8"):
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    rows = build_from_cache(os.environ["SUPABASE_DIRECT_DB_URL"])
    print(f"[뉴스 감성] {len(rows)}개 티커 적재")
    for r in rows:
        print(f"  {r['ticker']:12} {r['name_ko']:10} {r['sentiment']:+.2f} ({r['label']}) "
              f"기사 {r['n_articles']}건 / 평균 {r['avg_age_days']:.1f}일")


if __name__ == "__main__":
    main()

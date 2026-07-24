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

설계문서 정합성 — 배치가 아니라 **읽기-통과 캐시(read-through)**로 간다:
  - `docs/PRD_pdf_pipeline.md` §4는 "확정된 기업 기준으로 주가 조회 / 재무제표 요약 / 관련 뉴스기사"로
    적어, 티커가 확정된 **그 시점에 조회**하는 흐름이다(뉴스 추출 방식은 §5 Open Question #1로 미정).
  - `README.md` 다이어그램은 `NEWS -> META(기업 메타데이터 DB) -> LLM`이라 **DB를 거쳐** 나간다.
  - `src/finance/layer3_news_selection.py`의 Layer3는 애초에 **실시간 파이프라인**이다
    (네이버 검색 API 실시간 수집 -> 하드 필터 -> 4요소 가중 랭킹 -> Qwen3 reasoning 검증 -> top-N).

두 요구를 다 만족시키는 형태가 읽기-통과 캐시다 — 질의 시점에 티커별로 캐시를 보고, 신선하면
그대로 쓰고(지연 0), 없거나 TTL이 지났으면 **Layer3 정식 진입점 `select_news()`를 그 자리에서
호출**해 채운 뒤 같은 테이블에 upsert한다. 그래서 새 기업이 나와도 자동으로 채워지고, 이미 본
기업은 재호출 비용을 안 낸다.

주의: 네이버 API 자격증명(`NAVER_CLIENT_ID`/`NAVER_CLIENT_SECRET`)이 없으면 실시간 경로는
동작할 수 없으므로 캐시(및 기존 layer3 md 산출물)만 사용하고, 조용히 건너뛴다.
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
    collected_at   timestamptz,        -- **뉴스가 실제로 수집된 시각**(신선도/TTL 판단 기준)
    source         text,               -- 'layer3_live'(실시간 수집) | 'layer3_cache'(기존 md 산출물)
    updated_at     timestamptz default now()   -- 이 행을 적재한 시각(운영 추적용)
)
"""

# 기존 배포본에 컬럼이 없을 수 있어 멱등 추가(있으면 무시)
TABLE_MIGRATE = [
    "alter table company_news_sentiment add column if not exists collected_at timestamptz",
    "alter table company_news_sentiment add column if not exists source text",
]

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
_GENERATED_RE = re.compile(r"^_generated:\s*(\S+)_?\s*$", re.M)


def _parse_generated_at(text: str):
    """layer3 md 머리말의 `_generated: 2026-07-23T12:23:38+00:00_`을 읽어 **실제 수집 시각**을
    돌려준다. 이걸 안 쓰고 적재 시각(now())을 신선도로 삼으면, 몇 달 전 산출물도 방금 수집한
    것처럼 보여 읽기-통과 캐시가 재수집을 영원히 건너뛴다(실제로 처음 적재할 때 그렇게 넣었다)."""
    m = _GENERATED_RE.search(text)
    if not m:
        return None
    try:
        return datetime.fromisoformat(m.group(1).rstrip("_"))
    except Exception:
        return None


def parse_layer3_markdown(path: Path) -> tuple[str, list[NewsItem], object]:
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
    return name_ko, out, _parse_generated_at(text)


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

    name_ko, items, generated_at = parse_layer3_markdown(md_path)
    if not items:
        return None
    articles = [_to_scored_article(i) for i in items]
    s_news, age_days, _results = score_news_sentiment(articles, name_ko)
    return {
        "ticker": ticker, "name_ko": name_ko,
        "sentiment": round(float(s_news), 4), "label": _label_for(s_news),
        "n_articles": len(items), "avg_age_days": round(float(age_days), 2),
        "headlines": " | ".join(i.title for i in items[:NEWS_CONTEXT_MAX_HEADLINES]),
        # 캐시 경로는 md가 만들어진 시각이 곧 수집 시각. 못 읽으면 기사들의 최신 게재일로 대체.
        "collected_at": generated_at or max((i.pub_date for i in items), default=None),
        "source": "layer3_cache",
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
            for stmt in TABLE_MIGRATE:
                cur.execute(stmt)
            for r in rows:
                cur.execute(
                    "insert into company_news_sentiment "
                    "(ticker,name_ko,sentiment,label,n_articles,avg_age_days,headlines,"
                    "collected_at,source,updated_at) "
                    "values (%(ticker)s,%(name_ko)s,%(sentiment)s,%(label)s,%(n_articles)s,"
                    "%(avg_age_days)s,%(headlines)s,%(collected_at)s,%(source)s, now()) "
                    "on conflict (ticker) do update set "
                    "name_ko=excluded.name_ko, sentiment=excluded.sentiment, label=excluded.label, "
                    "n_articles=excluded.n_articles, avg_age_days=excluded.avg_age_days, "
                    "headlines=excluded.headlines, collected_at=excluded.collected_at, "
                    "source=excluded.source, updated_at=now()",
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
                "select ticker,name_ko,sentiment,label,n_articles,avg_age_days,headlines,"
                "coalesce(collected_at, updated_at) "
                "from company_news_sentiment where ticker = any(%s)", (list(tickers),))
            rows = cur.fetchall()
    except Exception:
        return ""
    finally:
        conn.close()

    if not rows:
        return ""
    lines = []
    for t, name, s, lab, n, age, heads, collected in rows:
        # 수집일을 같이 노출한다 — LLM이 "최근 뉴스"의 최근이 언제인지 알아야 오래된 감성을
        # 현재 상황처럼 단정하지 않는다(신선도가 프롬프트에 없으면 판단 근거가 없다).
        when = collected.strftime("%Y-%m-%d") if collected else "수집시각 미상"
        lines.append(
            f"[{name}({t}) 최근 뉴스 감성 — DB]\n"
            f"감성 점수 {s:+.2f} ({lab}), 근거 기사 {n}건, 평균 {age:.1f}일 전, 수집일 {when}\n"
            f"주요 헤드라인: {heads}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------- 실시간(Layer3) 경로

NEWS_TTL_HOURS = 24          # 이 시간이 지난 캐시는 오래된 것으로 보고 다시 수집
NEWS_TOP_N = 5               # Layer3가 최종 선정하는 기사 수(기존 산출물과 동일)


def _naver_credentials_available() -> bool:
    import os
    return bool(os.environ.get("NAVER_CLIENT_ID") and os.environ.get("NAVER_CLIENT_SECRET"))


def score_ticker_live(ticker: str, name_ko: str, aliases: list | None = None,
                      topic: str | None = None) -> dict | None:
    """[설계 정합] Layer3 정식 진입점(`select_news`)을 그 자리에서 호출해 감성까지 채점한다.

    md 파일을 파싱하는 캐시 경로와 달리 이쪽이 PRD §4가 말하는 "확정된 기업 기준으로 관련
    뉴스기사 조회"에 해당한다. 네이버 자격증명이 없으면 None을 돌려주고 호출측이 캐시로 폴백한다."""
    if not _naver_credentials_available():
        return None
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.finance.layer3_news_selection import select_news
    from src.finance.layer3_news_sentiment import score_news_sentiment

    try:
        articles = select_news(name_ko=name_ko, query=name_ko, topic=topic or name_ko,
                               aliases=aliases or [], top_n=NEWS_TOP_N)
    except Exception:
        return None            # 수집/랭킹 실패는 보조 신호 실패일 뿐 — 생성은 계속돼야 한다
    if not articles:
        return None
    s_news, age_days, _ = score_news_sentiment(articles, name_ko)
    return {
        "ticker": ticker, "name_ko": name_ko,
        "sentiment": round(float(s_news), 4), "label": _label_for(s_news),
        "n_articles": len(articles), "avg_age_days": round(float(age_days), 2),
        "headlines": " | ".join(a.title for a in articles[:NEWS_CONTEXT_MAX_HEADLINES]),
        "collected_at": datetime.now(timezone.utc),   # 실시간 경로는 지금이 곧 수집 시각
        "source": "layer3_live",
    }


def _stale_or_missing(db_url: str, tickers: list, ttl_hours: int) -> list:
    """캐시에 없거나 TTL이 지난 티커만 골라낸다(있으면 재수집 안 함 = 지연 0)."""
    import psycopg2
    try:
        conn = psycopg2.connect(db_url)
    except Exception:
        return list(tickers)
    try:
        with conn.cursor() as cur:
            cur.execute("select to_regclass('public.company_news_sentiment')")
            if cur.fetchone()[0] is None:
                return list(tickers)
            cur.execute(
                "select ticker from company_news_sentiment "
                "where ticker = any(%s) and coalesce(collected_at, updated_at) "
                "> now() - (%s || ' hours')::interval",
                (list(tickers), str(ttl_hours)))
            fresh = {r[0] for r in cur.fetchall()}
    except Exception:
        return list(tickers)
    finally:
        conn.close()
    return [t for t in tickers if t not in fresh]


def _upsert(db_url: str, rows: list) -> None:
    import psycopg2
    if not rows:
        return
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(TABLE_DDL)
            for stmt in TABLE_MIGRATE:
                cur.execute(stmt)
            for r in rows:
                cur.execute(
                    "insert into company_news_sentiment "
                    "(ticker,name_ko,sentiment,label,n_articles,avg_age_days,headlines,"
                    "collected_at,source,updated_at) "
                    "values (%(ticker)s,%(name_ko)s,%(sentiment)s,%(label)s,%(n_articles)s,"
                    "%(avg_age_days)s,%(headlines)s,%(collected_at)s,%(source)s, now()) "
                    "on conflict (ticker) do update set "
                    "name_ko=excluded.name_ko, sentiment=excluded.sentiment, label=excluded.label, "
                    "n_articles=excluded.n_articles, avg_age_days=excluded.avg_age_days, "
                    "headlines=excluded.headlines, collected_at=excluded.collected_at, "
                    "source=excluded.source, updated_at=now()", r)
        conn.commit()
    finally:
        conn.close()


def refresh_for_matched(db_url: str, matched: list, ttl_hours: int = NEWS_TTL_HOURS) -> int:
    """[읽기-통과 캐시] 매칭된 기업 중 캐시가 없거나 오래된 것만 Layer3로 실시간 채워 넣는다.
    반환: 새로 채운 기업 수. 자격증명이 없거나 수집이 실패하면 0(기존 캐시로 그대로 진행)."""
    if not matched:
        return 0
    by_ticker = {m["ticker"]: m.get("name") or m["ticker"] for m in matched}
    need = _stale_or_missing(db_url, list(by_ticker), ttl_hours)
    if not need or not _naver_credentials_available():
        return 0
    rows = []
    for t in need:
        rec = score_ticker_live(t, by_ticker[t])
        if rec:
            rows.append(rec)
    _upsert(db_url, rows)
    return len(rows)


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
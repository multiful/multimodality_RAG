-- 002_dividend_score.sql
-- [정정] 이 파일은 초안 시절 companies(ticker) FK와 universe_snapshots라는 낡은 테이블명을
-- 담고 있었으나, 실제 운영 Supabase는 그 FK 없이(별도 companies 테이블 부재) 아래 스키마로
-- 생성돼 있다(교차리뷰 스윕에서 불일치 발견 — 운영 DB 실사 + 실제 부트스트랩 SQL로 교체).
-- 신규 DB 부트스트랩 시 이 파일만 실행하면 된다(001 전제 없음, vector 확장만 필요).

-- 저장소 sql/schema.sql 의 financial_chunks 관례를 그대로 따른다:
--   · 티커 = Yahoo 형식 "005930.KS"
--   · 임베딩 = OpenAI text-embedding-3-small → vector(1536)
--   · 검색 = match_* RPC (코사인, HNSW)
--   · companies FK 없음(저장소에 해당 테이블 없음) — ticker 문자열 그대로 사용
-- Supabase SQL Editor에 붙여넣고 Run. 기존 테이블은 건드리지 않는다.

create extension if not exists vector;

-- ── 원시값: ticker × 사업연도 (재계산 가능한 정본, 임베딩 없음) ───────────
create table if not exists dividend_facts (
    ticker          text not null,             -- "005930.KS"
    fiscal_year     int  not null,             -- 사업연도 (예: 2025)
    fs_div          text default 'CFS',        -- CFS(연결) | OFS(별도)
    dps_cash        numeric,                   -- 주당 현금배당금(원, 연간 합산)
    div_total_cash  numeric,                   -- 현금배당금총액(백만원)
    payout_ratio    numeric,                   -- 현금배당성향(%) — DART alotMatter
    div_yield       numeric,                   -- 현금배당수익률(%) — pykrx DIV 우선
    net_income      numeric,                   -- 당기순이익(백만원, 지배주주 우선)
    equity          numeric,                   -- 자본총계·지배(백만원)
    roe             numeric,                   -- 순이익/평균자기자본(%)
    cfo             numeric,                   -- 영업활동현금흐름(백만원)
    capex           numeric,                   -- 유형자산의 취득(백만원)
    fcf             numeric,                   -- cfo - capex
    paid            boolean,                   -- 해당 연도 현금배당 지급 여부
    profitable      boolean,                   -- 흑자 여부
    source          text,                      -- dart_alotmatter|dart_fnltt|pykrx
    data_quality    jsonb default '{}'::jsonb, -- 결측·교차검증 불일치 기록
    fetched_at      timestamptz not null default now(),
    primary key (ticker, fiscal_year)
);
create index if not exists dividend_facts_ticker_idx on dividend_facts (ticker);

-- ── 점수 + 임베딩: ticker × 사업연도 × 버전 (검색·인용 대상) ───────────────
-- financial_chunks 와 동일 관례(id PK, content=임베딩 원문, embedding vector(1536)) +
-- 스코어 메타 컬럼. company_profile_chunks 가 summary 컬럼을 덧붙인 것과 같은 방식.
create table if not exists dividend_scores (
    id              text primary key,          -- "{ticker}_{fiscal_year}_{score_version}"
    ticker          text not null,             -- "005930.KS"
    fiscal_year     int  not null,
    score_version   text not null,             -- 'v1'(논문 원식) | 'v2'(고도화)
    universe        text default 'KOSPI200',   -- 상대평가 모집단
    -- 항목별 점수 (v1: x11~x42 / v2: 전체. 미적용 항목은 null)
    x11 numeric, x12 numeric,                  -- 범주1 지급 이력
    x21 numeric, x22 numeric,                  -- 범주2 배당성향
    x3  numeric,                               -- 범주3 현금배당률 추세
    x41 numeric, x42 numeric,                  -- 범주4 상대 위치
    x51 numeric, x52 numeric, x53 numeric,     -- 범주5 안정성·여력 (v2 전용)
    gp              numeric not null,          -- 총점 (v1: -4~+4, v2: -5~+5)
    score100        numeric,                   -- 0~100 정규화
    grp             text,                      -- 'A' | 'B' | 'C'
    rank_in_universe int,                      -- 유니버스 내 순위 (1 = 최고)
    pct_in_universe  numeric,                  -- 상위 백분율 (0~100)
    flags           text[] default '{}',       -- deficit_year|short_history|special_dividend|non_payer|…
    details         jsonb default '{}'::jsonb, -- 채점 근거 스냅샷 (감사추적)
    content         text not null,             -- 한국어 요약 서술문 (임베딩 원문 = embed_text)
    embedding       vector(1536),              -- OpenAI text-embedding-3-small
    created_at      timestamptz not null default now()
);
create index if not exists dividend_scores_ticker_idx on dividend_scores (ticker);
create index if not exists dividend_scores_embedding_idx
    on dividend_scores using hnsw (embedding vector_cosine_ops);
create index if not exists dividend_scores_grp_idx
    on dividend_scores (score_version, fiscal_year, grp);
create index if not exists dividend_scores_gp_idx
    on dividend_scores (score_version, fiscal_year, gp desc);

-- ── 벡터 검색 RPC (match_financial_chunks 와 동일 패턴 + 정형 필터) ────────
create or replace function match_dividend_scores(
    query_embedding vector(1536),
    match_count     int  default 5,
    filter_version  text default 'v2',
    filter_year     int  default null,         -- null이면 전 연도
    filter_grp      text default null,         -- 'A' 등, null이면 전체
    min_gp          numeric default null,
    filter_ticker   text default null
)
returns table (
    id text, ticker text, fiscal_year int, score_version text,
    gp numeric, grp text, score100 numeric, pct_in_universe numeric,
    content text, similarity float
)
language sql stable
as $$
    select
        dividend_scores.id,
        dividend_scores.ticker,
        dividend_scores.fiscal_year,
        dividend_scores.score_version,
        dividend_scores.gp,
        dividend_scores.grp,
        dividend_scores.score100,
        dividend_scores.pct_in_universe,
        dividend_scores.content,
        1 - (dividend_scores.embedding <=> query_embedding) as similarity
    from dividend_scores
    where dividend_scores.embedding is not null
      and dividend_scores.score_version = filter_version
      and (filter_year   is null or dividend_scores.fiscal_year = filter_year)
      and (filter_grp    is null or dividend_scores.grp = filter_grp)
      and (min_gp        is null or dividend_scores.gp >= min_gp)
      and (filter_ticker is null or dividend_scores.ticker = filter_ticker)
    order by dividend_scores.embedding <=> query_embedding
    limit match_count;
$$;

-- ── 유니버스 스냅샷 (선견편향 방지: 기준일별 구성종목 보존) ────────────────
create table if not exists dividend_universe_snapshots (
    universe    text not null,                 -- 'KOSPI200'
    as_of       date not null,                 -- 스냅샷 기준일
    ticker      text not null,                 -- "005930.KS"
    primary key (universe, as_of, ticker)
);

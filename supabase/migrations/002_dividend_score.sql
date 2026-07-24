-- 002_dividend_score.sql
-- 배당 스코어링(STGP) 적재 스키마 — docs/배당스코어링_STGP_설계.md 참고
-- 전제: 001_schema.sql 실행 완료 (vector 확장, companies 테이블 존재)

-- ── 원시값: ticker × 사업연도 (재계산 가능한 정본) ─────────────────────────
create table if not exists dividend_facts (
  ticker          text not null references companies(ticker),
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
  source          text,                      -- dart_alotmatter|dart_fnltt|pykrx|fnguide
  data_quality    jsonb default '{}'::jsonb, -- 결측·교차검증 불일치 기록
  fetched_at      timestamptz default now(),
  primary key (ticker, fiscal_year)
);

-- ── 점수: ticker × 사업연도 × 버전 (검색·인용 대상) ────────────────────────
create table if not exists dividend_scores (
  ticker          text not null references companies(ticker),
  fiscal_year     int  not null,
  score_version   text not null,             -- 'v1'(논문 원식) | 'v2'(고도화)
  universe        text default 'KOSPI200',   -- 상대평가 모집단 (KOSPI200 | KOSPI200_SECTOR)
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
  details         jsonb default '{}'::jsonb, -- 채점 근거 스냅샷 (Δ배당성향, 백분위 등 감사추적)
  embed_text      text,                      -- 한국어 요약 서술문 (임베딩 원문)
  embedding       halfvec(1024),             -- BGE-M3
  scored_at       timestamptz default now(),
  primary key (ticker, fiscal_year, score_version)
);

create index if not exists dividend_scores_embedding_idx
  on dividend_scores using hnsw (embedding halfvec_cosine_ops);
create index if not exists dividend_scores_grp_idx
  on dividend_scores (score_version, fiscal_year, grp);
create index if not exists dividend_scores_gp_idx
  on dividend_scores (score_version, fiscal_year, gp desc);

-- ── 벡터 검색 RPC (기존 match_chunks와 동일 패턴 + 정형 필터) ──────────────
create or replace function match_dividend_scores(
  query_embedding halfvec(1024),
  match_count     int  default 10,
  filter_version  text default 'v2',
  filter_year     int  default null,         -- null이면 전 연도
  filter_grp      text default null,         -- 'A' 등, null이면 전체
  min_gp          numeric default null
)
returns table (
  ticker text, fiscal_year int, gp numeric, grp text,
  pct_in_universe numeric, embed_text text, similarity float
)
language sql stable as $$
  select ticker, fiscal_year, gp, grp, pct_in_universe, embed_text,
         1 - (embedding <=> query_embedding) as similarity
  from dividend_scores
  where score_version = filter_version
    and embedding is not null
    and (filter_year is null or fiscal_year = filter_year)
    and (filter_grp  is null or grp = filter_grp)
    and (min_gp      is null or gp >= min_gp)
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- ── 유니버스 스냅샷 (선견편향 방지: 기준일별 구성종목 보존) ────────────────
create table if not exists universe_snapshots (
  universe    text not null,                 -- 'KOSPI200'
  as_of       date not null,                 -- 스냅샷 기준일
  ticker      text not null,
  primary key (universe, as_of, ticker)
);

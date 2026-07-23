-- Supabase SQL Editor에서 실행. output/kospi200_financials/*.md를 인덱싱할 벡터 테이블.
-- 임베딩 모델: text-embedding-3-small (1536차원). 다른 모델로 바꾸면 vector(N) 차원도 함께 바꿀 것.

create extension if not exists vector;

create table if not exists financial_chunks (
    id text primary key,              -- "{ticker}_{statement_type}_{period_type}"
    ticker text not null,             -- "000080.KS"
    statement_type text not null,     -- income_statement | balance_sheet | cash_flow
    period_type text not null,        -- annual | quarterly
    content text not null,            -- 마크다운 테이블 원문(임베딩 대상 텍스트)
    embedding vector(1536) not null,
    created_at timestamptz not null default now()
);

create index if not exists financial_chunks_ticker_idx on financial_chunks (ticker);

-- HNSW: 코사인 유사도 기준 근사 최근접 이웃 검색용 인덱스
create index if not exists financial_chunks_embedding_idx
    on financial_chunks using hnsw (embedding vector_cosine_ops);

-- 검색 시 사용할 RPC. Supabase JS/Python 클라이언트에서 rpc("match_financial_chunks", {...})로 호출.
create or replace function match_financial_chunks(
    query_embedding vector(1536),
    match_count int default 5,
    filter_ticker text default null
)
returns table (
    id text,
    ticker text,
    statement_type text,
    period_type text,
    content text,
    similarity float
)
language sql stable
as $$
    select
        financial_chunks.id,
        financial_chunks.ticker,
        financial_chunks.statement_type,
        financial_chunks.period_type,
        financial_chunks.content,
        1 - (financial_chunks.embedding <=> query_embedding) as similarity
    from financial_chunks
    where filter_ticker is null or financial_chunks.ticker = filter_ticker
    order by financial_chunks.embedding <=> query_embedding
    limit match_count;
$$;

-- KOSPI200_output/kospi200_profiles/*.md (기업 프로필, 티커당 1개 파일)를 인덱싱할 벡터 테이블.
create table if not exists company_profile_chunks (
    id text primary key,              -- ticker, "000080.KS"
    ticker text not null,
    content text not null,            -- 프로필 마크다운 원문(임베딩 대상 텍스트)
    embedding vector(1536) not null,
    created_at timestamptz not null default now()
);

create index if not exists company_profile_chunks_ticker_idx on company_profile_chunks (ticker);

create index if not exists company_profile_chunks_embedding_idx
    on company_profile_chunks using hnsw (embedding vector_cosine_ops);

create or replace function match_company_profile_chunks(
    query_embedding vector(1536),
    match_count int default 5,
    filter_ticker text default null
)
returns table (
    id text,
    ticker text,
    content text,
    similarity float
)
language sql stable
as $$
    select
        company_profile_chunks.id,
        company_profile_chunks.ticker,
        company_profile_chunks.content,
        1 - (company_profile_chunks.embedding <=> query_embedding) as similarity
    from company_profile_chunks
    where filter_ticker is null or company_profile_chunks.ticker = filter_ticker
    order by company_profile_chunks.embedding <=> query_embedding
    limit match_count;
$$;

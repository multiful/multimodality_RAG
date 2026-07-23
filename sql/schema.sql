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

-- [엔티티 합성] 텍스트/테이블/이미지 세 브랜치의 근거를 한 곳에 통합 저장 — ERD의
-- "엔티티 합성(PDF 객체 비율 가중치 정제) -> DB(supabase) 기업명 및 티커" 단계.
-- 임베딩은 pdf_pipeline이 쓰는 dragonkue/BGE-m3-ko(1024차원) 기준 — 위 두 테이블(OpenAI
-- text-embedding-3-small, 1536차원)과 모델이 달라 벡터 차원도 다름, 별도 테이블로 분리.
create table if not exists document_evidence (
    id text primary key,               -- "{pdf_id}_{source_type}_{seq}"
    pdf_id text not null,
    ticker text,                       -- 기업명/티커로 매칭 안 되면 null 허용(문서 단위 임시 인덱싱)
    source_type text not null check (source_type in ('text', 'table', 'image')),
    page int,
    content text not null,             -- 임베딩 대상 텍스트(브랜치별로 근거 요약해서 채움)
    weight float not null default 1.0, -- 소스 타입별 가중치(엔티티 합성의 "가중치 정제")
    metadata jsonb not null default '{}'::jsonb,  -- canonical_field/structured_metadata/section_path 등
    embedding vector(1024) not null,
    created_at timestamptz not null default now()
);

create index if not exists document_evidence_pdf_id_idx on document_evidence (pdf_id);
create index if not exists document_evidence_ticker_idx on document_evidence (ticker);
create index if not exists document_evidence_source_type_idx on document_evidence (source_type);

create index if not exists document_evidence_embedding_idx
    on document_evidence using hnsw (embedding vector_cosine_ops);

create or replace function match_document_evidence(
    query_embedding vector(1024),
    match_count int default 5,
    filter_pdf_id text default null,
    filter_ticker text default null
)
returns table (
    id text,
    pdf_id text,
    ticker text,
    source_type text,
    page int,
    content text,
    weight float,
    metadata jsonb,
    similarity float
)
language sql stable
as $$
    select
        document_evidence.id,
        document_evidence.pdf_id,
        document_evidence.ticker,
        document_evidence.source_type,
        document_evidence.page,
        document_evidence.content,
        document_evidence.weight,
        document_evidence.metadata,
        1 - (document_evidence.embedding <=> query_embedding) as similarity
    from document_evidence
    where (filter_pdf_id is null or document_evidence.pdf_id = filter_pdf_id)
      and (filter_ticker is null or document_evidence.ticker = filter_ticker)
    order by document_evidence.embedding <=> query_embedding
    limit match_count;
$$;

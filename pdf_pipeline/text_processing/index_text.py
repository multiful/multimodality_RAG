"""[35] 텍스트 인덱싱 틀 — BM25 + BGE-m3-ko 하이브리드 검색 스켈레톤.

사용자 요청: "텍스트의 bm25 + bge-m3-ko로 인덱싱 과정 연결해놓게 틀 잡아놓자." Supabase
스키마가 아직 팀원 쪽에서 확정 전이라(실제 Supabase 프로젝트 확인 결과 스키마 미정, 대화 기록
참고), 이번엔 저장소를 **인메모리**로 두고 인터페이스(`build_index`/`hybrid_search`)만 먼저
잡는다 — 나중에 Supabase(pgvector + full-text search)로 교체할 때 이 두 함수의 내부 구현만
바꾸면 되고, 호출부(인덱싱 스크립트/쿼리 핸들러)는 그대로 재사용 가능하도록 설계.

다이어그램 반영: "핵심모델: DENSE(BGE-m3-ko), 보조모델: BM25" — dense_weight를 bm25_weight보다
높게 기본 설정. Rank Fusion은 가장 단순한 min-max 정규화 후 가중합(RRF 등 더 정교한 방식은
데이터 늘어나면 재검토 — 지금은 틀만 잡는 단계라 오버엔지니어링 방지).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list:
    """BM25용 토크나이저 — 형태소 분석기 없이 한글 어절/영숫자 단위로만 쪼갬(정교한 한국어
    토크나이저는 이후 개선 과제로 남김 — 지금은 dense가 메인이라 BM25는 보조 신호 정도로 충분)."""
    return re.findall(r"[가-힣]+|[A-Za-z0-9]+", text.lower())


@dataclass
class TextIndex:
    pdf_id: str
    chunk_ids: list = field(default_factory=list)
    chunks: list = field(default_factory=list)   # process_pdf() chunk dict 그대로(text/raw_chunk/section_path/page/structured_metadata 등)
    embeddings: object = None                     # np.ndarray (N, dim), normalize_embeddings=True
    bm25: object = None                            # rank_bm25.BM25Okapi


def build_index(pdf_id: str, process_pdf_result: dict, embed_model=None) -> TextIndex:
    """`text_extraction.process_pdf()`(또는 `process_pdf_streaming()`을 다 모은 결과)의 pages에서
    모든 chunks를 모아 (1) BGE-m3-ko 임베딩, (2) BM25 인덱스를 만든다. 임베딩 대상은 `c["text"]`
    (컨텍스트 접두어 포함본, [5]/[9]에서 검증된 대로 이게 실제 검색 품질이 더 좋았음) — raw_chunk가
    아님에 주의."""
    if embed_model is None:
        from embedding import get_embedding_model
        embed_model = get_embedding_model()

    chunks = [c for page in process_pdf_result["pages"] for c in page["chunks"]]
    if not chunks:
        return TextIndex(pdf_id=pdf_id)

    chunk_ids = [f"{pdf_id}_p{c['page']}_{i}" for i, c in enumerate(chunks)]
    texts = [c["text"] for c in chunks]

    from embedding import embed_texts
    embeddings = embed_texts(texts)

    bm25 = BM25Okapi([_tokenize(t) for t in texts])

    return TextIndex(pdf_id=pdf_id, chunk_ids=chunk_ids, chunks=chunks, embeddings=embeddings, bm25=bm25)


def hybrid_search(index: TextIndex, query: str, embed_model=None, top_k: int = 5,
                   dense_weight: float = 0.7, bm25_weight: float = 0.3) -> list:
    """Dense(코사인 유사도) + BM25 점수를 각각 min-max 정규화 후 가중합으로 결합(Rank Fusion
    초안). 반환: [{chunk_id, chunk, score, dense_score, bm25_score}, ...] score 내림차순 top_k."""
    if not index.chunks:
        return []
    if embed_model is None:
        from embedding import get_embedding_model
        embed_model = get_embedding_model()

    import numpy as np
    from embedding import embed_texts
    query_emb = embed_texts([query])[0]
    dense_scores = np.asarray(index.embeddings) @ query_emb
    bm25_scores = np.asarray(index.bm25.get_scores(_tokenize(query)))

    def _normalize(arr):
        span = arr.max() - arr.min()
        return (arr - arr.min()) / span if span > 0 else np.zeros_like(arr)

    fused = dense_weight * _normalize(dense_scores) + bm25_weight * _normalize(bm25_scores)
    order = np.argsort(-fused)[:top_k]
    return [
        {"chunk_id": index.chunk_ids[i], "chunk": index.chunks[i], "score": float(fused[i]),
         "dense_score": float(dense_scores[i]), "bm25_score": float(bm25_scores[i])}
        for i in order
    ]


def save_index(index: TextIndex, path: Path) -> None:
    """[35] 임시 저장 — Supabase 스키마 확정 전까지 로컬 pickle로 인덱스를 보존해두는 용도.
    실제 서비스 저장소(Supabase pgvector 등)로 교체할 때 이 함수만 바꾸면 됨."""
    import pickle
    Path(path).write_bytes(pickle.dumps(index))


def load_index(path: Path) -> TextIndex:
    import pickle
    return pickle.loads(Path(path).read_bytes())

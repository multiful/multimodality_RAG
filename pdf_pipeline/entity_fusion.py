"""엔티티 합성(Entity Fusion) — 텍스트/테이블/이미지 세 브랜치의 출력을 하나의 통합 증거
(evidence) 리스트로 합치고, 소스 타입별 가중치를 매겨 document_evidence 테이블에 저장 +
하이브리드 검색 인덱스(BM25+BGE-m3-ko)를 만든다.

ERD의 "엔티티 합성: PDF 객체 비율 가중치 정제" 반영 — "가중치 정제"는 소스 타입별 신뢰도/정밀도
차이를 가중치로 표현한 것: 표(canonical field 매칭)/이미지(구조화 추출)처럼 이미 한 번 구조화를
거친 근거는 자유 텍스트 청크보다 수치가 명확하므로 기본 가중치를 더 준다. 이 가중치는
hybrid_search()의 fused score에 곱해져 검색 순위에 반영된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "text_processing"))
from index_text import TextIndex, _tokenize  # noqa: E402

SOURCE_WEIGHTS = {
    "table": 1.3,   # canonical_field 매칭 등 구조화된 수치 — 가장 신뢰
    "image": 1.15,  # 구조화 추출(entities/key_values) 거쳤지만 OCR/VLM 오차 가능성 있어 표보다 낮게
    "text": 1.0,    # 자유 텍스트 청크(기준)
}


def from_text_chunks(pdf_id: str, chunks: list) -> list:
    """text_processing.text_extraction.process_pdf() 결과의 chunk들(process_pdf_result["pages"][i]
    ["chunks"]를 모은 것)을 evidence 아이템으로 변환."""
    return [
        {
            "id": f"{pdf_id}_text_{i}", "pdf_id": pdf_id, "source_type": "text",
            "page": c.get("page"), "content": c["text"], "weight": SOURCE_WEIGHTS["text"],
            "metadata": {"section_path": c.get("section_path")},
        }
        for i, c in enumerate(chunks)
    ]


def from_table_records(pdf_id: str, row_records: list) -> list:
    """table_processing.run_table_metadata_pipeline.build_records()가 반환한 row_records(순수
    재무항목 필터 통과한 모든 행, canonical 매칭 여부 무관)를 evidence 아이템으로 변환. 한 행 =
    "라벨: 값들" 짧은 문장으로 직렬화해 임베딩 대상 텍스트를 만든다."""
    items = []
    for i, r in enumerate(row_records):
        cells = r.get("cells") or r.get("numeric_values") or []
        if not cells:
            continue
        items.append({
            "id": f"{pdf_id}_table_{i}", "pdf_id": pdf_id, "source_type": "table",
            "page": r.get("page"), "content": f"{r['raw_label']}: {cells}",
            "weight": SOURCE_WEIGHTS["table"],
            "metadata": {"canonical_field": r.get("canonical_field"), "table_idx": r.get("table_idx")},
        })
    return items


def from_image_cards(pdf_id: str, cards: list) -> list:
    """image_processing 카드(status="useful"인 것만)를 evidence 아이템으로 변환. embed_text
    (캡션+각주+OCR+narrative 조합, image_processing/s2_onestop_mineru.py의 build_embed_text())를
    임베딩 대상으로 쓰고, structured_metadata(있으면)는 metadata로 함께 저장."""
    items = []
    for c in cards:
        if c.get("status") != "useful":
            continue
        content = c.get("embed_text") or c.get("caption") or ""
        if not content:
            continue
        items.append({
            "id": f"{pdf_id}_image_{c['image_id']}", "pdf_id": pdf_id, "source_type": "image",
            "page": c.get("page"), "content": content, "weight": SOURCE_WEIGHTS["image"],
            "metadata": {"block_type": c.get("block_type"),
                         "structured_metadata": c.get("structured_metadata")},
        })
    return items


def fuse(pdf_id: str, text_chunks: list = None, table_records: list = None, image_cards: list = None) -> list:
    """세 브랜치 출력을 하나의 evidence 리스트로 합친다."""
    return (from_text_chunks(pdf_id, text_chunks or [])
            + from_table_records(pdf_id, table_records or [])
            + from_image_cards(pdf_id, image_cards or []))


def build_fused_index(pdf_id: str, evidence_items: list, embed_model=None) -> TextIndex:
    """fuse()가 만든 evidence 리스트로 하이브리드(BM25+dense) 검색 인덱스를 만든다.
    text_processing.index_text.build_index()와 같은 인터페이스(TextIndex)를 쓰되, 소스가
    process_pdf() 결과 하나가 아니라 세 브랜치를 합친 evidence이므로 별도 구현 — hybrid_search()는
    그대로 재사용 가능(TextIndex.chunks에 뭐가 들어있는지 상관 안 함)."""
    if not evidence_items:
        return TextIndex(pdf_id=pdf_id)
    if embed_model is None:
        from embedding import get_embedding_model
        embed_model = get_embedding_model()

    from embedding import embed_texts
    texts = [it["content"] for it in evidence_items]
    embeddings = embed_texts(texts)
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([_tokenize(t) for t in texts])

    return TextIndex(
        pdf_id=pdf_id,
        chunk_ids=[it["id"] for it in evidence_items],
        chunks=evidence_items,
        embeddings=embeddings,
        bm25=bm25,
    )


def store_evidence(db_url: str, pdf_id: str, index: TextIndex, ticker: str = None) -> int:
    """build_fused_index()가 만든 인덱스(evidence_items + 이미 계산된 embeddings)를
    document_evidence 테이블에 적재. 이미 계산된 임베딩을 재사용(재임베딩 안 함).
    반환: 적재된 행 수."""
    if not index.chunks:
        return 0
    import psycopg2
    from psycopg2.extras import Json, execute_values

    rows = [
        (it["id"], pdf_id, ticker, it["source_type"], it.get("page"), it["content"],
         it.get("weight", 1.0), Json(it.get("metadata") or {}), emb.tolist())
        for it, emb in zip(index.chunks, index.embeddings)
    ]
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "insert into document_evidence "
                "(id, pdf_id, ticker, source_type, page, content, weight, metadata, embedding) "
                "values %s on conflict (id) do update set "
                "content=excluded.content, weight=excluded.weight, metadata=excluded.metadata, "
                "embedding=excluded.embedding",
                rows,
            )
    finally:
        conn.close()
    return len(rows)


def weighted_hybrid_search(index: TextIndex, query: str, top_k: int = 5,
                            dense_weight: float = 0.7, bm25_weight: float = 0.3) -> list:
    """index_text.hybrid_search()와 동일한 min-max 정규화 + 가중합 융합에, evidence 아이템별
    `weight`(소스 타입 가중치, SOURCE_WEIGHTS)를 최종 곱해서 정렬 — "엔티티 합성의 가중치 정제"."""
    if not index.chunks:
        return []
    from embedding import embed_texts
    query_emb = embed_texts([query])[0]
    dense_scores = np.asarray(index.embeddings) @ query_emb
    bm25_scores = np.asarray(index.bm25.get_scores(_tokenize(query)))

    def _normalize(arr):
        span = arr.max() - arr.min()
        return (arr - arr.min()) / span if span > 0 else np.zeros_like(arr)

    fused = dense_weight * _normalize(dense_scores) + bm25_weight * _normalize(bm25_scores)
    source_weights = np.array([c.get("weight", 1.0) for c in index.chunks])
    fused = fused * source_weights

    order = np.argsort(-fused)[:top_k]
    return [
        {"chunk_id": index.chunk_ids[i], "chunk": index.chunks[i], "score": float(fused[i]),
         "dense_score": float(dense_scores[i]), "bm25_score": float(bm25_scores[i]),
         "source_type": index.chunks[i].get("source_type")}
        for i in order
    ]

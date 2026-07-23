"""엔티티 합성(Entity Fusion) — 텍스트/테이블/이미지 세 브랜치의 출력을 하나의 통합 증거
(evidence) 리스트로 합치고, 소스 타입별 가중치를 매겨 document_evidence 테이블에 저장 +
하이브리드 검색 인덱스(BM25+BGE-m3-ko)를 만든다.

ERD의 "엔티티 합성: PDF 객체 비율 가중치 정제" 반영 — "가중치 정제"는 소스 타입별 신뢰도/정밀도
차이를 가중치로 표현한 것: 표(canonical field 매칭)/이미지(구조화 추출)처럼 이미 한 번 구조화를
거친 근거는 자유 텍스트 청크보다 수치가 명확하므로 기본 가중치를 더 준다. 이 가중치는
hybrid_search()의 fused score에 곱해져 검색 순위에 반영된다.

[48] 사용자 지적("정적으로 고정하면 안 되겠지?") 반영 — 표 evidence를 소스 타입 하나로만
뭉뚱그려 항상 고정 가중치를 주던 걸, canonical_field 매칭 여부(이미 계산돼 있는 품질 신호)로
행마다 동적 결정하도록 변경(`from_table_records()`). 가중치 학습(경사하강법 등)은 채택 안 함 —
학습 데이터가 없고, 이 관계는 이미 실측(C밴드.pdf)으로 확인된 이산 신호라 학습 없이 규칙으로
바로 반영 가능. `실험_pipeline_baseline_비교.md`/text_processing 실험.md [48] 참고.
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

# [48] 사용자 지적("정적으로 고정하면 안 되겠지?") 반영 — 표 evidence를 소스 타입 하나로만
# 뭉뚱그려 항상 1.3을 주면 안 됨을 실측으로 확인(C밴드.pdf: "투자의견 변동 내역"류 표 8개,
# 64행 전부 canonical_field 매칭 0% — 표준 필드로 확인 안 된 행은 회사명 등 식별자가 아예
# 없이 "24.12.18: [BUY,40000,...]"만 남는데, 이게 검색 상위에 뜨면 LLM이 엉뚱한 회사에
# 갖다붙이는 실제 오답을 냄, [47] 참고). canonical_field가 실제로 매칭된 행만 "구조화 확인
# 완료"로 보고 기존 가중치(1.3)를 주고, 매칭 안 된 행은 text 기준(1.0)보다도 낮게(0.6) —
# 학습이 아니라 이미 계산된 품질 신호(canonical_field 매칭 여부)로 조건부 결정하는 규칙
# 기반 접근(학습 데이터도 없고, 이 관계는 이미 실측으로 확인된 이산 신호라 학습이 불필요).
UNMATCHED_TABLE_WEIGHT = 0.6


def from_text_chunks(pdf_id: str, chunks: list) -> list:
    """text_processing.text_extraction.process_pdf() 결과의 chunk들(process_pdf_result["pages"][i]
    ["chunks"]를 모은 것)을 evidence 아이템으로 변환.

    [49] 사용자 지적("구조화 메타데이터가 정말 병렬/비동기로 붙는 거 맞아?") 검증 중 발견 —
    이 함수가 chunk의 structured_metadata(entities/sector_mentioned/sentiment 등)를 metadata에
    안 담고 있었음. process_pdf_streaming()의 설계 의도(구조화 출력은 별도 느린 백그라운드 잡)를
    실제로 쓰려면 나중에 이 필드가 채워진 chunk로 다시 부를 때 그걸 저장 스키마에 실어야 하는데,
    그 통로 자체가 없었던 것 — 있으면 담고, 없으면(아직 안 끝난 경우) 생략."""
    items = []
    for i, c in enumerate(chunks):
        metadata = {"section_path": c.get("section_path")}
        if c.get("structured_metadata"):
            metadata["structured_metadata"] = c["structured_metadata"]
        items.append({
            "id": f"{pdf_id}_text_{i}", "pdf_id": pdf_id, "source_type": "text",
            "page": c.get("page"), "content": c["text"], "weight": SOURCE_WEIGHTS["text"],
            "metadata": metadata,
        })
    return items


def from_table_records(pdf_id: str, row_records: list) -> list:
    """table_processing.run_table_metadata_pipeline.build_records()가 반환한 row_records(순수
    재무항목 필터 통과한 모든 행, canonical 매칭 여부 무관)를 evidence 아이템으로 변환. 한 행 =
    "라벨: 값들" 짧은 문장으로 직렬화해 임베딩 대상 텍스트를 만든다.

    [48] 가중치를 행마다 canonical_field 매칭 여부로 동적 결정 — 매칭된 행은 표준 필드로
    확인된 만큼 기존대로 신뢰(SOURCE_WEIGHTS["table"]=1.3), 매칭 안 된 행은 UNMATCHED_TABLE_
    WEIGHT(0.6, text 기준 1.0보다 낮음)로 강등. 모든 행에 소스 하나로 뭉뚱그린 고정값을 주지
    않는다."""
    items = []
    for i, r in enumerate(row_records):
        cells = r.get("cells") or r.get("numeric_values") or []
        if not cells:
            continue
        weight = SOURCE_WEIGHTS["table"] if r.get("canonical_field") else UNMATCHED_TABLE_WEIGHT
        items.append({
            "id": f"{pdf_id}_table_{i}", "pdf_id": pdf_id, "source_type": "table",
            "page": r.get("page"), "content": f"{r['raw_label']}: {cells}",
            "weight": weight,
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


def embed_items(items: list, embed_model=None):
    """evidence 아이템 리스트를 임베딩한다. 반환: (items, embeddings ndarray) — items가 비어있으면
    (items, None). [40] 브랜치별로 끝나는 즉시 호출해서 store_evidence()로 바로 적재하고, 나중에
    build_index_from_items()로 그 임베딩을 재사용(재임베딩 없이) 통합 인덱스를 만들기 위해 items/
    embeddings를 분리된 반환값으로 둔다."""
    if not items:
        return items, None
    if embed_model is None:
        from embedding import get_embedding_model
        embed_model = get_embedding_model()
    from embedding import embed_texts
    embeddings = embed_texts([it["content"] for it in items])
    return items, embeddings


def build_index_from_items(pdf_id: str, items: list, embeddings) -> TextIndex:
    """이미 계산된 (items, embeddings)로 하이브리드(BM25+dense) 검색 인덱스를 만든다 — 재임베딩
    없음. 여러 브랜치의 embed_items() 결과를 이어붙여서 넘기면 전체 통합 인덱스가 된다."""
    if not items or embeddings is None or len(embeddings) == 0:
        return TextIndex(pdf_id=pdf_id)
    from rank_bm25 import BM25Okapi
    texts = [it["content"] for it in items]
    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    return TextIndex(
        pdf_id=pdf_id,
        chunk_ids=[it["id"] for it in items],
        chunks=items,
        embeddings=embeddings,
        bm25=bm25,
    )


def build_fused_index(pdf_id: str, evidence_items: list, embed_model=None) -> TextIndex:
    """fuse()가 만든 evidence 리스트를 한 번에 임베딩해 인덱스를 만드는 편의 함수(embed_items()+
    build_index_from_items() 조합). 브랜치별로 이미 임베딩을 따로 계산해뒀다면 그쪽을 직접
    build_index_from_items()에 넘기는 게 재임베딩을 피할 수 있어 더 낫다."""
    items, embeddings = embed_items(evidence_items, embed_model)
    return build_index_from_items(pdf_id, items, embeddings)


def store_evidence(db_url: str, pdf_id: str, items: list, embeddings, ticker: str = None) -> int:
    """embed_items()가 반환한 (items, embeddings)를 document_evidence 테이블에 즉시 적재.

    [40] 사용자 지적("Entity Fusion sync barrier" — 세 브랜치를 다 모은 뒤 한 번에 적재하면,
    가장 느린 브랜치(이미지/VLM, 문서당 최대 152초+)가 끝날 때까지 몇 초면 끝나는 텍스트/테이블
    결과까지 DB 적재가 막혀버림) 반영 — 브랜치 하나가 끝나는 즉시 그 브랜치분만 호출해서 저장하도록
    분리. "엔티티 합성"은 이제 Python 메모리 안에서의 사전 병합이 아니라, 같은 pdf_id/ticker로
    태그된 채 같은 document_evidence 테이블에 각자 도착하는 것 자체가 합성 지점이 된다."""
    if not items or embeddings is None or len(embeddings) == 0:
        return 0
    import psycopg2
    from psycopg2.extras import Json, execute_values

    rows = [
        (it["id"], pdf_id, ticker, it["source_type"], it.get("page"), it["content"],
         it.get("weight", 1.0), Json(it.get("metadata") or {}), emb.tolist())
        for it, emb in zip(items, embeddings)
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


def load_evidence_from_db(db_url: str, pdf_id: str = None, ticker: str = None) -> TextIndex:
    """[41] 사용자 지적("질의 시점에 저장소에서 다시 읽어오는 경로가 없음") 반영 — store_evidence()가
    document_evidence에 적재해둔 임베딩을 재계산 없이 그대로 읽어와 TextIndex(BM25+dense)를
    재구성한다. 이 함수가 있으면 검색은 수집(ingest)과 같은 프로세스/실행일 필요가 없다 — 문서를
    한 번 적재해두면, 그 뒤로는 이 함수 하나만 호출해서 매 질의마다 PDF 3브랜치를 재실행하지 않고
    바로 검색할 수 있다.

    pdf_id/ticker 중 최소 하나는 지정해야 함(의도치 않은 전체 테이블 스캔 방지). 반환된 TextIndex는
    build_index_from_items()가 만든 것과 동일한 구조라 weighted_hybrid_search()에 그대로 넘기면 됨.

    psycopg2는 pgvector 어댑터가 없으면 embedding 컬럼을 '[0.1,0.2,...]' 문자열로 반환하므로
    (image_processing/common.py의 vec_to_pg()와 반대 방향 변환) 여기서 직접 파싱한다."""
    if not pdf_id and not ticker:
        raise ValueError("pdf_id 또는 ticker 중 하나는 지정해야 합니다(전체 스캔 방지).")

    import psycopg2

    conditions, params = [], []
    if pdf_id:
        conditions.append("pdf_id = %s")
        params.append(pdf_id)
    if ticker:
        conditions.append("ticker = %s")
        params.append(ticker)
    where = " and ".join(conditions)

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"select id, source_type, page, content, weight, metadata, embedding "
                f"from document_evidence where {where} order by id",
                params,
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    index_id = pdf_id or ticker
    if not rows:
        return TextIndex(pdf_id=index_id)

    items = [
        {"id": row_id, "pdf_id": index_id, "source_type": source_type, "page": page,
         "content": content, "weight": weight, "metadata": metadata}
        for row_id, source_type, page, content, weight, metadata, _ in rows
    ]
    embeddings = np.asarray([
        np.fromstring(embedding_str.strip("[]"), sep=",") for *_, embedding_str in rows
    ])
    return build_index_from_items(index_id, items, embeddings)


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

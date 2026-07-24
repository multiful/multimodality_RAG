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
import threading
import time
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
        meta = {"canonical_field": r.get("canonical_field"), "table_idx": r.get("table_idx")}
        # [재일] 텍스트/이미지 브랜치와 동일하게 표 브랜치도 structured_metadata를 evidence에 부착
        # (기존엔 여기서 안 실려 표 entities_mentioned가 DB에 전무 — §8.4 갭). 호출측(데모)이 표
        # 단위 table_metadata + 행 content에서 매칭한 기업 entities를 row record에 넣어 넘긴다.
        if r.get("structured_metadata"):
            meta["structured_metadata"] = r["structured_metadata"]
        # [재일] 컬럼명이 회수된 표(run_table_metadata_pipeline._detect_column_headers)는 "컬럼명=값"
        # 으로 직렬화 — 기존 "2026-07-20: ['대우건설','상도15구역...','14,367']" 형태는 어떤 숫자가
        # 계약금액인지 알 수 없고 "계약 금액" 같은 컬럼명 토큰이 본문에 없어 BM25/dense 둘 다
        # 못 걸렸다(실측: Construct p5 주간 수주 공시). 헤더가 없으면 기존 형태 그대로.
        headers = r.get("column_headers")
        row_values = [r["raw_label"]] + list(cells)
        if headers and len(headers) >= 2:
            pairs = [f"{h}={v}" for h, v in zip(headers, row_values) if str(v).strip()]
            extra = [str(v) for v in row_values[len(headers):] if str(v).strip()]
            content = " | ".join(pairs + extra)
            meta["column_headers"] = headers
        else:
            content = f"{r['raw_label']}: {cells}"
        # [재일 — C밴드 사례] 표 캡션(있으면)을 행 본문 앞에 붙인다. 캡션에만 회사명이 있는 표
        # ("투자의견 변동 내역 및 목표주가 괴리율 / 케이엠더블유")에서 행이 `26.5.19: ['BUY',
        # '70,000',...]`로만 저장되면 회사명 질의로는 어휘·의미 어느 쪽으로도 매칭될 수 없어
        # **검색 후보가 되지 못한다**(실측: c밴드 table 청크 63개 중 회사명 포함 0건 -> 회사명으로
        # 도달 가능한 유일한 근거가 축 눈금만 담긴 차트 카드였고, 생성 단계가 Y축 9개·X축 9개를
        # 순서대로 짝지어 없는 시계열을 만들어냈다).
        caption = (r.get("table_caption") or "").strip()
        if caption:
            content = f"[{caption[:80]}] {content}"
        items.append({
            "id": f"{pdf_id}_table_{i}", "pdf_id": pdf_id, "source_type": "table",
            "page": r.get("page"), "content": content,
            "weight": weight, "metadata": meta,
        })
    return items + _table_digest_items(pdf_id, row_records)


TABLE_DIGEST_MAX_ROWS = 25      # 요약 청크에 담을 최대 행 수(아주 긴 재무표가 임베딩을 삼키지 않게)
TABLE_DIGEST_MAX_CHARS = 2000


def _table_digest_items(pdf_id: str, row_records: list) -> list:
    """[재일] 표 하나를 통째로 담은 '요약 청크'를 표마다 1개 더 만든다.

    배경(실측): 표를 행 단위 청크로만 적재하면 "수주 공시 중 계약금액이 **가장 큰** 프로젝트가
    뭐야?" 같은 최댓값/집계 질의를 원리적으로 못 푼다 — 정답을 고르려면 6개 행을 한꺼번에 비교해야
    하는데 각 행은 서로 독립된 청크라 top-k 안에 다 들어오지도 않고, 어느 행에도 "가장 크다"는
    신호가 없다(골든셋 H4: 검색 9개 방법 전부 hit=0). 표 단위로 헤더+모든 행을 한 청크에 모아두면
    그 청크 하나만 검색돼도 LLM이 행끼리 비교해 답할 수 있다. 행 단위 청크는 그대로 두므로
    (핀포인트 질의는 여전히 행이 더 정확) 두 입도가 공존한다."""
    by_table, order = {}, []
    for r in row_records:
        key = (r.get("page"), r.get("table_idx"))
        if key not in by_table:
            by_table[key] = []
            order.append(key)
        by_table[key].append(r)

    items = []
    for key in order:
        rows = by_table[key]
        if len(rows) < 2:
            continue                      # 한 줄짜리는 요약해도 행 청크와 같음
        page, table_idx = key
        headers = next((r.get("column_headers") for r in rows if r.get("column_headers")), None)
        sm = next((r.get("structured_metadata") for r in rows if r.get("structured_metadata")), None) or {}
        title = sm.get("table_title") or ""
        lines = []
        for r in rows[:TABLE_DIGEST_MAX_ROWS]:
            vals = [r.get("raw_label") or ""] + [str(c) for c in (r.get("cells") or [])]
            lines.append(" | ".join(v for v in vals if str(v).strip()))
        body = "\n".join(lines)[:TABLE_DIGEST_MAX_CHARS]
        note = next((r.get("table_note") for r in rows if r.get("table_note")), None)
        caption = next((r.get("table_caption") for r in rows if r.get("table_caption")), None)
        head = f"[표 전체] {caption or ''} {title or ''} {note or ''}".strip()
        if headers:
            head += f"\n컬럼: {' | '.join(headers)}"
        n_more = max(0, len(rows) - TABLE_DIGEST_MAX_ROWS)
        tail = f"\n(외 {n_more}행 생략)" if n_more else ""
        items.append({
            "id": f"{pdf_id}_tabledigest_{page}_{table_idx}", "pdf_id": pdf_id,
            "source_type": "table", "page": page,
            "content": f"{head}\n{body}{tail}",
            "weight": SOURCE_WEIGHTS["table"],
            "metadata": {"table_idx": table_idx, "granularity": "table_digest",
                         "n_rows": len(rows), "column_headers": headers,
                         "structured_metadata": sm or None},
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


# [수정] 사용자 지적("load_evidence_from_db가 매 호출마다 BM25 재구축 — 캐싱 고려") 반영 —
# (pdf_id, ticker) 키의 프로세스 메모리 캐시. DB 재조회(네트워크 왕복) + 임베딩 문자열 파싱
# (np.fromstring, evidence 수에 비례) + BM25 전체 재토큰화를 매 호출마다 반복하던 것을 없앤다.
# Redis 등 외부 인프라는 불필요 — 지금 배포는 단일 프로세스(README: "배포는 아직 구현 안 됨")
# 라 여러 프로세스가 캐시를 공유할 필요가 없고, TextIndex 자체가 numpy 배열+BM25Okapi 객체를
# 들고 있어 JSON으로 못 담아 Redis에 넣으려면 pickle 등 추가 직렬화가 필요해지는데 지금 규모에선
# 오버엔지니어링. [49]처럼 백그라운드 잡이 나중에 upsert하면 캐시가 stale해질 수 있어 기본
# TTL(60초)을 두고, 그걸 아는 호출측(예: RQ 워커 완료 콜백)은 invalidate_evidence_cache()로
# 즉시 무효화할 수 있다.
_EVIDENCE_CACHE_TTL_S = 60
_evidence_cache: dict = {}
_evidence_cache_lock = threading.Lock()


def invalidate_evidence_cache(pdf_id: str = None, ticker: str = None) -> None:
    """load_evidence_from_db()의 캐시를 무효화. 인자를 안 주면 캐시 전체를 비운다 — 예:
    store_evidence()로 같은 pdf_id/ticker에 새로 적재한 직후 호출측이 명시적으로 불러 스테일
    캐시를 지울 수 있다."""
    with _evidence_cache_lock:
        if pdf_id is None and ticker is None:
            _evidence_cache.clear()
            return
        _evidence_cache.pop((pdf_id, ticker), None)


def load_evidence_from_db(db_url: str, pdf_id: str = None, ticker: str = None,
                           use_cache: bool = True, cache_ttl_s: float = _EVIDENCE_CACHE_TTL_S) -> TextIndex:
    """[41] 사용자 지적("질의 시점에 저장소에서 다시 읽어오는 경로가 없음") 반영 — store_evidence()가
    document_evidence에 적재해둔 임베딩을 재계산 없이 그대로 읽어와 TextIndex(BM25+dense)를
    재구성한다. 이 함수가 있으면 검색은 수집(ingest)과 같은 프로세스/실행일 필요가 없다 — 문서를
    한 번 적재해두면, 그 뒤로는 이 함수 하나만 호출해서 매 질의마다 PDF 3브랜치를 재실행하지 않고
    바로 검색할 수 있다.

    pdf_id/ticker 중 최소 하나는 지정해야 함(의도치 않은 전체 테이블 스캔 방지). 반환된 TextIndex는
    build_index_from_items()가 만든 것과 동일한 구조라 weighted_hybrid_search()에 그대로 넘기면 됨.

    psycopg2는 pgvector 어댑터가 없으면 embedding 컬럼을 '[0.1,0.2,...]' 문자열로 반환하므로
    (image_processing/common.py의 vec_to_pg()와 반대 방향 변환) 여기서 직접 파싱한다.

    [수정] use_cache=True(기본)면 (pdf_id, ticker) 키로 TTL 캐시를 먼저 확인 — 히트하면 DB 왕복/
    파싱/BM25 재구축 없이 즉시 반환. use_cache=False로 캐시를 완전히 우회할 수 있음(디버깅,
    "방금 적재한 최신 상태를 반드시 봐야 함"이 확실한 호출부)."""
    if not pdf_id and not ticker:
        raise ValueError("pdf_id 또는 ticker 중 하나는 지정해야 합니다(전체 스캔 방지).")

    cache_key = (pdf_id, ticker)
    if use_cache:
        with _evidence_cache_lock:
            cached = _evidence_cache.get(cache_key)
        if cached is not None:
            index, cached_at = cached
            if time.monotonic() - cached_at < cache_ttl_s:
                return index

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
        index = TextIndex(pdf_id=index_id)
    else:
        items = [
            {"id": row_id, "pdf_id": index_id, "source_type": source_type, "page": page,
             "content": content, "weight": weight, "metadata": metadata}
            for row_id, source_type, page, content, weight, metadata, _ in rows
        ]
        embeddings = np.asarray([
            np.fromstring(embedding_str.strip("[]"), sep=",") for *_, embedding_str in rows
        ])
        index = build_index_from_items(index_id, items, embeddings)

    if use_cache:
        with _evidence_cache_lock:
            _evidence_cache[cache_key] = (index, time.monotonic())
    return index


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

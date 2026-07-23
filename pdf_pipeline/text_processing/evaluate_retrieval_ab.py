"""[43] 사용자 요청 — "하이브리드 서치의 성능 테스트와 각각 MQE, HyDE, 쿼리 타입 분류 라우팅
등을 A/B테스트로 도입해보면서 비교". 텍스트 청킹->인덱싱->검색 파이프라인만 대상(이미지/테이블
라우팅의 구조화 출력 엔티티 추출은 팀원 담당 영역이라 그대로 두고 건드리지 않음).

시나리오: 사용자가 PDF를 업로드하고 "이 PDF에 나오는 기업의 이벤트 요약과 투자 인사이트를
도출해줘"(summary형 질의, golden_set_event_insight.json 참고)라고 묻는다고 가정. 1차 테스트는
어려운 문서(KWave, 산업분석 73p, 10개 기업), 2차는 쉬운 문서(LGCNS, 단일기업 6p)로 같은 질의를
비교 — "테스트는 어려운 걸로 해야 적합"하다는 사용자 지적 반영.

비교 대상 6가지 검색 전략(모두 index_text.py의 실제 프로덕션 함수를 그대로 호출, 이 스크립트는
평가 로직만 다룸):
  1) dense-only              : BGE-m3-ko 코사인 유사도만
  2) dense+BM25(weighted_sum): 기존 hybrid_search() 기본값
  3) dense+BM25(RRF)         : [42] hybrid_search(fusion="rrf")
  4) MQE                     : [43] mqe_search() — LLM으로 하위질의 4개 생성 후 RRF로 합침
  5) HyDE                    : [43] hyde_search() — LLM 가상 문단 임베딩으로 dense 검색
  6) 쿼리타입 라우팅          : [43] route_search() — classify_query_type()으로 분류 후 전략 선택
                                (이 질의는 summary로 분류되어 MQE와 동일 경로를 타지만, 분류
                                자체의 정확성은 기존 15개 factoid/list/comparison 질의로도 검증)

지표: NDCG@10, MAP@10, MRR, Recall@10, Precision@10, F1@10, 지연(초). Golden set이 문서당
5~20개(summary형이라 다답형)라 랭크 품질 지표가 recall@1류 단일정답 지표보다 적합.

주의(정직하게 기록): 질의 1개 x 문서 2개짜리 평가라 통계적 유의성은 없음 — 경향 확인용.
LGCNS는 전체 청크가 17개뿐이라 top_k=10이 전체의 절반 이상을 차지, recall@10 수치가 코퍼스
크기 때문에 원래도 높게 나올 수밖에 없다는 점을 감안해서 읽을 것.
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "page_classification"))

from PIL import Image
from ultralytics import YOLO

from index_text import (TextIndex, hybrid_search, mqe_search, hyde_search, route_search,
                         classify_query_type, _tokenize, precompute_entity_count)
from rank_bm25 import BM25Okapi
from text_extraction import process_pdf

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
GOLD_PATH = OUT_DIR / "golden_set_event_insight.json"
CHUNK_CACHE_DIR = OUT_DIR / "_ab_chunk_cache"
TOP_K = 10


# ---------------------------------------------------------------- 코퍼스 준비

def build_corpus_index(pdf_id: str, pdf_path: Path, yolo_model, doc_title: str) -> TextIndex:
    """process_pdf()로 문서 전체를 실제 프로덕션 경로 그대로 청킹 -> BGE-m3-ko 임베딩 + BM25
    인덱스. 73페이지짜리 KWave는 YOLO+청킹에 20초+ 걸리므로 결과를 로컬에 캐싱해 재실행 비용을
    없앤다(원문 PDF가 안 바뀌는 한 캐시 재사용 — 실험 스크립트라 이 정도 캐싱이면 충분, 프로덕션
    코드의 [41] Supabase read-path와는 별개)."""
    CHUNK_CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CHUNK_CACHE_DIR / f"{pdf_id}_chunks.json"
    if cache_path.exists():
        chunks = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"  [{pdf_id}] 캐시된 청크 {len(chunks)}개 재사용")
    else:
        t0 = time.perf_counter()
        result = process_pdf(pdf_path, yolo_model, doc_title=doc_title, chunk_backend="rulebased",
                              remove_boilerplate=True)
        chunks = [c for p in result["pages"] for c in p["chunks"]]
        print(f"  [{pdf_id}] process_pdf() {time.perf_counter()-t0:.1f}s, 청크 {len(chunks)}개, "
              f"hard_pages={result['hard_page_numbers']}")
        cache_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    from embedding import embed_texts
    texts = [c["text"] for c in chunks]
    t0 = time.perf_counter()
    embeddings = embed_texts(texts)
    print(f"  [{pdf_id}] 임베딩 {time.perf_counter()-t0:.1f}s")
    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    chunk_ids = [f"{pdf_id}_p{c['page']}_{i}" for i, c in enumerate(chunks)]
    index = TextIndex(pdf_id=pdf_id, chunk_ids=chunk_ids, chunks=chunks, embeddings=embeddings, bm25=bm25)

    # [46] 사용자 지적("캐시 미스 해결") 반영 — route_search() 첫 호출에서 entity_count를 그때야
    # 계산하면 그 지연이 첫 질의 응답 시간에 그대로 노출된다. 인제스트(인덱스 빌드) 시점에 미리
    # 채워서 이후 모든 route_search() 호출이 캐시만 읽게 함(하나증권 포맷이면 정규식이라 무료).
    t0 = time.perf_counter()
    precompute_entity_count(index, pdf_path=pdf_path)
    print(f"  [{pdf_id}] entity_count={index.entity_count} 사전계산 {time.perf_counter()-t0:.2f}s")
    return index


def resolve_gold_indices(index: TextIndex, anchors: list) -> set:
    idx_set = set()
    unmatched = []
    for anchor in anchors:
        hits = [i for i, c in enumerate(index.chunks) if anchor in c["raw_chunk"]]
        if not hits:
            unmatched.append(anchor)
        else:
            idx_set.add(hits[0])
    if unmatched:
        print(f"  경고: anchor {len(unmatched)}개가 청크에서 안 잡힘: {unmatched}")
    return idx_set


# ---------------------------------------------------------------- IR 지표 (binary relevance)

def _binary_rel(chunk_ids_ranked: list, id_to_idx: dict, gold: set) -> list:
    return [1 if id_to_idx[cid] in gold else 0 for cid in chunk_ids_ranked]


def precision_at_k(rel, k): return sum(rel[:k]) / k if k else 0.0


def recall_at_k(rel, k, n_gold): return sum(rel[:k]) / n_gold if n_gold else 0.0


def f1_at_k(p, r): return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mrr_score(rel):
    for i, r in enumerate(rel):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(rel, n_gold):
    if n_gold == 0:
        return 0.0
    hits, s = 0, 0.0
    for i, r in enumerate(rel):
        if r:
            hits += 1
            s += hits / (i + 1)
    return s / n_gold


def ndcg_at_k(rel, k):
    rel_k = rel[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel_k))
    ideal = sorted(rel, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def score_ranking(chunk_ids_ranked: list, id_to_idx: dict, gold: set, k: int = TOP_K) -> dict:
    rel = _binary_rel(chunk_ids_ranked, id_to_idx, gold)
    p, r = precision_at_k(rel, k), recall_at_k(rel, k, len(gold))
    return {
        "ndcg@k": round(ndcg_at_k(rel, k), 4), "map@k": round(average_precision(rel[:k], len(gold)), 4),
        "mrr": round(mrr_score(rel), 4), "recall@k": round(r, 4), "precision@k": round(p, 4),
        "f1@k": round(f1_at_k(p, r), 4), "n_gold_found_in_top_k": sum(rel[:k]), "n_gold_total": len(gold),
    }


# ---------------------------------------------------------------- 방법별 실행

def run_dense_only(index, query, k):
    import numpy as np
    from embedding import embed_texts
    q_emb = embed_texts([query])[0]
    scores = np.asarray(index.embeddings) @ q_emb
    order = np.argsort(-scores)[:k]
    return [index.chunk_ids[i] for i in order]


def run_method(name, index, query, client, k=TOP_K):
    t0 = time.perf_counter()
    if name == "dense-only":
        ranked = run_dense_only(index, query, k)
    elif name == "dense+BM25(weighted_sum)":
        ranked = [h["chunk_id"] for h in hybrid_search(index, query, top_k=k, fusion="weighted_sum")]
    elif name == "dense+BM25(RRF)":
        ranked = [h["chunk_id"] for h in hybrid_search(index, query, top_k=k, fusion="rrf")]
    elif name == "MQE":
        hits, _ = mqe_search(index, query, client=client, top_k=k)
        ranked = [h["chunk_id"] for h in hits]
    elif name == "MQE(BM25 제외)":
        hits, _ = mqe_search(index, query, client=client, top_k=k, use_bm25=False)
        ranked = [h["chunk_id"] for h in hits]
    elif name == "HyDE":
        hits, _ = hyde_search(index, query, client=client, top_k=k)
        ranked = [h["chunk_id"] for h in hits]
    elif name == "HyDE(BM25 제외)":
        hits, _ = hyde_search(index, query, client=client, top_k=k, use_bm25=False)
        ranked = [h["chunk_id"] for h in hits]
    elif name == "쿼리타입 라우팅(구, 항상 MQE)":
        hits, qtype = route_search(index, query, client=client, top_k=k, entity_aware=False)
        ranked = [h["chunk_id"] for h in hits]
        print(f"    (route_search 분류: {qtype}, entity_aware=False)")
    elif name == "쿼리타입 라우팅(신, 엔티티인식)":
        hits, qtype = route_search(index, query, client=client, top_k=k, entity_aware=True)
        ranked = [h["chunk_id"] for h in hits]
        n_ent = index.entity_count
        strategy = "HyDE" if n_ent is not None and n_ent <= 1 else "MQE"
        print(f"    (route_search 분류: {qtype}, entity_count={n_ent} -> {strategy})")
    else:
        raise ValueError(name)
    elapsed = time.perf_counter() - t0
    return ranked, elapsed


def main():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(ROOT / ".env"))
    from openai import OpenAI
    client = OpenAI()

    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    query = gold["query"]
    print(f"질의: {query}")
    print(f"규칙 기반 분류 결과: {classify_query_type(query)} (기대값: {gold['query_type_expected']})\n")

    yolo_model = YOLO(str(ROOT / "pdf_pipeline/page_classification/models/yolo11n_doc_layout.pt"))
    yolo_model.predict(Image.new("RGB", (595, 842), (255, 255, 255)), conf=0.25, verbose=False)

    methods = ["dense-only", "dense+BM25(weighted_sum)", "dense+BM25(RRF)", "MQE", "MQE(BM25 제외)",
               "HyDE", "HyDE(BM25 제외)", "쿼리타입 라우팅(구, 항상 MQE)", "쿼리타입 라우팅(신, 엔티티인식)"]
    all_results = {}

    for doc_id, doc_info in gold["docs"].items():
        print(f"=== {doc_id} ({doc_info['note'][:40]}...) ===")
        index = build_corpus_index(doc_id, ROOT / doc_info["pdf_path"], yolo_model, doc_title=doc_id)
        gold_idx = resolve_gold_indices(index, doc_info["gold_anchors"])
        print(f"  golden set: {len(gold_idx)}/{len(doc_info['gold_anchors'])}개 anchor 매칭됨, "
              f"코퍼스 {len(index.chunks)}청크\n")
        id_to_idx = {cid: i for i, cid in enumerate(index.chunk_ids)}

        doc_results = {}
        for m in methods:
            ranked, elapsed = run_method(m, index, query, client)
            metrics = score_ranking(ranked, id_to_idx, gold_idx)
            metrics["latency_s"] = round(elapsed, 3)
            doc_results[m] = metrics
            print(f"  {m:26s} ndcg={metrics['ndcg@k']:.3f} map={metrics['map@k']:.3f} "
                  f"mrr={metrics['mrr']:.3f} recall={metrics['recall@k']:.3f} "
                  f"prec={metrics['precision@k']:.3f} f1={metrics['f1@k']:.3f} "
                  f"({metrics['n_gold_found_in_top_k']}/{metrics['n_gold_total']} gold in top{TOP_K}) "
                  f"lat={metrics['latency_s']:.2f}s")
        all_results[doc_id] = doc_results
        print()

    (OUT_DIR / "result_retrieval_ab.json").write_text(
        json.dumps({"query": query, "top_k": TOP_K, "results": all_results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[result] saved to {OUT_DIR / 'result_retrieval_ab.json'}")


if __name__ == "__main__":
    main()

"""[42] 사용자 요청 — "dense임베딩만(bge-m3-ko) vs dense+bm25(RRF 먼저 도입)" 비교.

`evaluate_embeddings.py`가 이미 검증해둔 코퍼스(3문서 실제 청크) + 정답셋(Claude 작성 질의 15개,
anchor 기반 정답 청크 식별)을 그대로 재사용 — 새 코퍼스/GT를 따로 만들지 않는다(오버엔지니어링
방지, 두 스크립트가 같은 기준으로 비교돼야 의미가 있음).

비교 대상 3가지:
  1) dense-only            : BGE-m3-ko 코사인 유사도만
  2) dense+BM25(weighted)  : index_text.hybrid_search(fusion="weighted_sum", 기존 기본값)
  3) dense+BM25(RRF)       : index_text.hybrid_search(fusion="rrf", [42]에서 신규 도입)
지표는 evaluate_embeddings.py와 동일하게 Recall@1/@3, MRR(코퍼스 15개 질의 기준이라 규모가 작음 —
경향 확인용이지 통계적으로 유의미한 정밀 비교는 아님, 데이터 늘어나면 재평가 필요).
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_embeddings import ROOT, YOLO_MODEL_PATH, build_corpus, assign_ground_truth  # noqa: E402
from index_text import TextIndex, hybrid_search, _tokenize  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def _rank_of_correct(order: np.ndarray, correct: set) -> Optional[int]:
    return next((r + 1 for r, idx in enumerate(order) if idx in correct), None)


def _score(name: str, orders: list, queries: list) -> dict:
    """orders[qi] = 그 질의에 대해 코퍼스 인덱스를 점수 내림차순으로 정렬한 배열."""
    recall_1, recall_3, mrr_sum, n_scored = 0, 0, 0.0, 0
    details = []
    for qi, q in enumerate(queries):
        correct = set(q["correct_indices"])
        if not correct:
            continue
        n_scored += 1
        rank = _rank_of_correct(orders[qi], correct)
        if rank == 1:
            recall_1 += 1
        if rank and rank <= 3:
            recall_3 += 1
        if rank:
            mrr_sum += 1.0 / rank
        details.append({"query": q["query"], "rank": rank})
    return {
        "method": name,
        "recall_at_1": round(recall_1 / n_scored, 4) if n_scored else 0.0,
        "recall_at_3": round(recall_3 / n_scored, 4) if n_scored else 0.0,
        "mrr": round(mrr_sum / n_scored, 4) if n_scored else 0.0,
        "n_queries_scored": n_scored,
        "details": details,
    }


def main():
    print("YOLO 로딩 + 코퍼스 구성 중...", flush=True)
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    yolo_model.predict(Image.new("RGB", (595, 842), (255, 255, 255)), conf=0.25, verbose=False)
    corpus = build_corpus(yolo_model)
    print(f"코퍼스: {len(corpus)}개 청크(3문서)", flush=True)

    gt = json.loads((OUT_DIR / "ground_truth_embedding_eval.json").read_text(encoding="utf-8"))
    queries = assign_ground_truth(corpus, gt["queries"])
    unmatched = [q["query"] for q in queries if not q["correct_indices"]]
    if unmatched:
        print(f"경고: anchor가 코퍼스에서 안 잡힌 질의 {len(unmatched)}개: {unmatched}", flush=True)

    corpus_texts = [c["text"] for c in corpus]
    query_texts = [q["query"] for q in queries]

    print("BGE-m3-ko 임베딩 중...", flush=True)
    from embedding import embed_texts
    t0 = time.perf_counter()
    corpus_embs = np.asarray(embed_texts(corpus_texts))
    corpus_embed_s = time.perf_counter() - t0
    query_embs = np.asarray(embed_texts(query_texts))

    bm25 = BM25Okapi([_tokenize(t) for t in corpus_texts])
    chunk_ids = [f"{c['doc']}_{i}" for i, c in enumerate(corpus)]  # doc 키만으로는 중복(청크마다 고유해야 함)
    index = TextIndex(pdf_id="eval_corpus", chunk_ids=chunk_ids,
                       chunks=corpus, embeddings=corpus_embs, bm25=bm25)

    dense_orders, weighted_orders, rrf_orders = [], [], []
    for qi, q in enumerate(queries):
        query_emb = query_embs[qi]
        dense_scores = corpus_embs @ query_emb
        dense_orders.append(np.argsort(-dense_scores))

        for fusion, bucket in (("weighted_sum", weighted_orders), ("rrf", rrf_orders)):
            hits = hybrid_search(index, q["query"], top_k=len(corpus), fusion=fusion)
            # hybrid_search가 내부에서 쿼리 임베딩을 다시 계산하지만(코퍼스 15개 규모라 비용
            # 무시 가능), 반환된 chunk_id 순서를 코퍼스 인덱스 순서로 변환해 공통 채점 함수로 넘김
            id_to_idx = {cid: i for i, cid in enumerate(index.chunk_ids)}
            bucket.append(np.array([id_to_idx[h["chunk_id"]] for h in hits]))

    results = [
        _score("dense-only", dense_orders, queries),
        _score("dense+BM25(weighted_sum)", weighted_orders, queries),
        _score("dense+BM25(RRF)", rrf_orders, queries),
    ]

    print(f"\n=== 결과 (코퍼스 {len(corpus)}청크, 질의 {len(queries)}개, "
          f"코퍼스 임베딩 {corpus_embed_s:.2f}s) ===")
    for r in results:
        print(f"  {r['method']:28s} recall@1={r['recall_at_1']*100:5.1f}%  "
              f"recall@3={r['recall_at_3']*100:5.1f}%  mrr={r['mrr']:.3f}")

    (OUT_DIR / "result_hybrid_search_eval.json").write_text(
        json.dumps({"n_corpus_chunks": len(corpus), "n_queries": len(queries), "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[result] saved to {OUT_DIR / 'result_hybrid_search_eval.json'}")


if __name__ == "__main__":
    main()

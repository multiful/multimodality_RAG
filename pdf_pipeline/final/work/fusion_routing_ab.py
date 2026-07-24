# -*- coding: utf-8 -*-
"""[재일] 퓨전·라우팅 정밀 비교 — "고정 가중 하이브리드가 정말 최선인가"를 골든셋으로 증명한다.

비교군(8):
  dense_only      : BGE-m3-ko 코사인만
  bm25_only       : BM25Okapi만
  linear_0.7/0.3  : min-max 정규화 후 선형가중 결합(= 현재 entity_fusion.weighted_hybrid_search)
  linear_0.5/0.5  : 가중치가 0.7/0.3이어야 하는지 확인용 대조군
  rrf             : Reciprocal Rank Fusion(k=60) — 점수 스케일에 무관한 순위 기반 결합
  hyde            : 가설문서 생성 후 dense 검색
  mqe             : 다중 질의 확장 + RRF
  route(gpt-4o)   : 질의 타입 분류 -> 추상=HyDE/MQE, 키워드=하이브리드
  decompose+route : 현재 프로덕션 기본 경로

채점: gold_any(사람이 PDF 읽고 만든 식별 문자열) 포함 여부로 relevance 이진 판정.
      nDCG@k / MRR / Recall@k / 지연. 쿼리타입(사람 라벨)별로 나눠 집계한다."""
import os, sys, json, math, time
from pathlib import Path
from collections import defaultdict
ROOT = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP = ROOT / "pdf_pipeline"
for d in [str(PP), str(PP/"text_processing")]: sys.path.insert(0, d)
for line in open(ROOT/".env", encoding="utf-8"):
    line = line.strip()
    if line and "=" in line and not line.startswith("#"): k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
DB = os.environ["SUPABASE_DIRECT_DB_URL"]
GOLD = json.loads((PP/"final"/"golden_set_construct_routing.json").read_text(encoding="utf-8"))
OUT = PP/"final"/"results_fusion_routing_ab.json"
TOPK = 8

import numpy as np
import entity_fusion, index_text
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ---------- 채점 ----------
def is_rel(chunk, gold_any):
    c = chunk.get("content") or ""
    return any(g in c for g in gold_any)

def score(hits, gold_any, k=TOPK):
    rels = [1 if is_rel(h["chunk"], gold_any) else 0 for h in hits[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rels, reverse=True)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    mrr = next((1 / (i + 1) for i, r in enumerate(rels) if r), 0.0)
    return {"ndcg": round(ndcg, 3), "mrr": round(mrr, 3), "hit": int(any(rels)), "n_rel": sum(rels)}


# ---------- 검색기(공통 인덱스 위에서 순수 함수로 구현해 공정 비교) ----------
def _dense_scores(index, query):
    from embedding import embed_texts
    return np.asarray(index.embeddings) @ embed_texts([query])[0]

def _bm25_scores(index, query):
    return np.asarray(index.bm25.get_scores(index_text._tokenize(query)))

def _minmax(a):
    a = np.asarray(a, dtype=float)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

def _weights(index):
    return np.asarray([c.get("weight", 1.0) for c in index.chunks], dtype=float)

def _top(index, s, k=TOPK):
    order = np.argsort(-s)[:k]
    return [{"chunk": index.chunks[i], "score": float(s[i])} for i in order]

def run_dense(index, q):   return _top(index, _dense_scores(index, q) * _weights(index))
def run_bm25(index, q):    return _top(index, _bm25_scores(index, q) * _weights(index))

def run_linear(index, q, wd=0.7, wb=0.3):
    s = (wd * _minmax(_dense_scores(index, q)) + wb * _minmax(_bm25_scores(index, q))) * _weights(index)
    return _top(index, s)

def run_rrf(index, q, k=60):
    d, b = _dense_scores(index, q), _bm25_scores(index, q)
    rr = np.zeros(len(index.chunks))
    for s in (d, b):
        ranks = np.empty(len(s), dtype=int)
        ranks[np.argsort(-s)] = np.arange(len(s))
        rr += 1.0 / (k + ranks + 1)
    return _top(index, rr * _weights(index))

def _retry(fn, tries=6):
    """조직 TPM(gpt-4o 30K) 한도로 429가 나면 지수 백오프 — 429로 인한 빈 결과가 그 방법의
    점수로 잘못 기록되는 걸 막는다(초기 측정에서 route/decompose 점수가 이 때문에 오염됐음)."""
    import time as _t
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if "rate_limit" not in str(e).lower() or i == tries - 1:
                raise
            _t.sleep(2 ** i)


def _hits(x):
    """검색 함수마다 반환형이 다르다(hits 단독 / (hits, 부가정보) 튜플) — hits 리스트만 꺼낸다."""
    while isinstance(x, tuple):
        x = x[0]
    return x or []

def run_hyde(index, q):    return _hits(_retry(lambda: index_text.hyde_search(index, q, client=client, top_k=TOPK)))
def run_mqe(index, q):     return _hits(_retry(lambda: index_text.mqe_search(index, q, client=client, top_k=TOPK)))
def run_route(index, q):   return _hits(_retry(lambda: index_text.route_search(index, q, client=client, top_k=TOPK)))
def run_decomp(index, q):  return _hits(_retry(lambda: index_text.decompose_and_route_search(index, q, client=client, top_k=TOPK)))

METHODS = [
    ("dense_only",      run_dense),
    ("bm25_only",       run_bm25),
    ("linear_0.7/0.3",  lambda i, q: run_linear(i, q, 0.7, 0.3)),
    ("linear_0.5/0.5",  lambda i, q: run_linear(i, q, 0.5, 0.5)),
    ("rrf_k60",         run_rrf),
    ("hyde",            run_hyde),
    ("mqe",             run_mqe),
    ("route(gpt-4o)",   run_route),
    ("decompose+route", run_decomp),
]


def main():
    index = entity_fusion.load_evidence_from_db(DB, pdf_id="Construct", use_cache=False)
    index.entity_count = 8
    print(f"[index] {len(index.chunks)}청크")
    run_dense(index, "warm")  # BGE 콜드로드 분리

    qs = GOLD["queries"]
    # gpt-4o 질의 타입 분류(사람 라벨과 대조)
    cls = {}
    for item in qs:
        try:
            cls[item["id"]] = _retry(lambda: index_text.classify_query_type_llm(item["q"], client=client))
        except Exception as e:
            cls[item["id"]] = f"error:{e}"
    agree = sum(1 for it in qs if _norm(cls[it["id"]]) == _norm_human(it["type"]))
    print(f"[분류] gpt-4o vs 사람라벨 일치 {agree}/{len(qs)}")

    results = {"topk": TOPK, "n_queries": len(qs), "classifier": cls,
               "classifier_agreement": f"{agree}/{len(qs)}", "per_method": {}, "per_query": defaultdict(dict)}
    for name, fn in METHODS:
        agg, lat = defaultdict(list), []
        for item in qs:
            t = time.time()
            try:
                hits = fn(index, item["q"])
            except Exception as e:
                print(f"  !! {name} / {item['id']} 실패: {e}"); hits = []
            dt = time.time() - t
            s = score(hits, item["gold_any"])
            s["latency_s"] = round(dt, 3)
            results["per_query"][item["id"]][name] = s
            agg[item["type"]].append(s); agg["ALL"].append(s); lat.append(dt)
        row = {}
        for t_, ss in agg.items():
            row[t_] = {"ndcg": round(sum(x["ndcg"] for x in ss)/len(ss), 3),
                       "mrr": round(sum(x["mrr"] for x in ss)/len(ss), 3),
                       "recall": round(sum(x["hit"] for x in ss)/len(ss), 3)}
        row["latency_s_mean"] = round(sum(lat)/len(lat), 3)
        results["per_method"][name] = row
        a = row["ALL"]
        print(f"  {name:17} ndcg={a['ndcg']:.3f} mrr={a['mrr']:.3f} recall={a['recall']:.3f} "
              f"lat={row['latency_s_mean']:.2f}s")

    results["per_query"] = dict(results["per_query"])
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT.name} 저장")


def _norm(x):
    x = (x or "").strip().lower()
    if "keyword" in x or "키워드" in x: return "keyword"
    if "abstract" in x or "추상" in x: return "abstract"
    return "hybrid"

def _norm_human(x):
    return {"키워드형": "keyword", "추상형": "abstract", "하이브리드형": "hybrid"}[x]


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""4축 교차 평가: 검색정확도(추출커버리지 + retrieval hit@k, 근접성 기반) + 구조 + 페이지분류 + 라우팅.
검색 백본은 4축 공통(BGE-m3-ko dense 0.7 + BM25 0.3). 결과 -> results.json."""
import sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_exp as C

import os
GT = C.load_json(HERE.parent / "ground_truth_smartphone.json")
AXES = ["baseline", "enhanced", "docling", "mineru"]
TOPK = int(os.environ.get("EVAL_TOPK", "5"))
WINDOW = 220
_OUTNAME = os.environ.get("EVAL_OUT", "results.json")

def get_embed_fn():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("dragonkue/BGE-m3-ko")
    def fn(texts):
        return m.encode(list(texts), normalize_embeddings=True)
    return fn

def eval_retrieval(axis_out, embed_fn):
    chunks = axis_out["chunks"]
    full = axis_out.get("full_text", "")
    idx = C.HybridIndex(chunks, embed_fn)
    per_q = []
    for q in GT["qa"]:
        keys = q["answer_keys"]
        # 추출 커버리지: 전체 파싱 텍스트에서 근접 hit 여부
        cov_present, cov_prox, cov_span = C.keys_hit(full, keys, WINDOW)
        # 검색: top-k 청크 중 근접 hit 되는 청크가 있나
        hits = idx.search(q["q"], top_k=TOPK)
        ret_prox = False; ret_present = False; best_rank = None
        for rank, h in enumerate(hits, start=1):
            p, px, _ = C.keys_hit(h["text"], keys, WINDOW)
            if p: ret_present = True
            if px:
                ret_prox = True; best_rank = rank; break
        per_q.append({"id": q["id"], "type": q["type"],
                       "cov_present": cov_present, "cov_prox": cov_prox,
                       "ret_present": ret_present, "ret_prox": ret_prox, "rank": best_rank})
    return per_q

def rate(lst, key):
    n = len(lst); s = sum(1 for x in lst if x[key])
    return {"n": n, "hit": s, "rate": round(s/max(1,n), 3)}

def by_type(per_q, key):
    types = {}
    for x in per_q:
        types.setdefault(x["type"], []).append(x)
    return {t: rate(v, key) for t, v in types.items()}

def eval_page_classification(page_pred):
    """enhanced page_pred vs GT page_types. table / image(chart) 라벨에 대한 P/R/F1."""
    gt = GT["page_types"]
    labels = {"table": "table", "image": "chart"}  # pred label -> gt label
    res = {}
    for pred_lab, gt_lab in labels.items():
        tp=fp=fn=0
        for pg, pred in page_pred.items():
            if pg == "_doc": continue
            gtset = gt.get(pg, [])
            g = gt_lab in gtset
            p = bool(pred.get(pred_lab))
            if p and g: tp+=1
            elif p and not g: fp+=1
            elif (not p) and g: fn+=1
        prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
        f1 = 2*prec*rec/max(1e-9,prec+rec)
        res[pred_lab] = {"tp":tp,"fp":fp,"fn":fn,"precision":round(prec,3),"recall":round(rec,3),"f1":round(f1,3)}
    return res

def main():
    outs = {}
    for a in AXES:
        p = HERE / f"out_{a}.json"
        if p.exists():
            outs[a] = C.load_json(p)
        else:
            print(f"[warn] missing out_{a}.json — skip")
    embed_fn = get_embed_fn()

    results = {"topk": TOPK, "window": WINDOW, "axes": {}}
    for a, o in outs.items():
        t = time.time()
        per_q = eval_retrieval(o, embed_fn)
        results["axes"][a] = {
            "latency": {"parse_time_s": o.get("parse_time_s"), "total_time_s": o.get("total_time_s"),
                         "stage_timing": o.get("stage_timing")},
            "structure": o.get("structure"),
            "n_chunks": o.get("n_chunks"),
            "retrieval": {
                "extraction_coverage_prox": rate(per_q, "cov_prox"),
                "extraction_coverage_present": rate(per_q, "cov_present"),
                "retrieval_hit_prox@k": rate(per_q, "ret_prox"),
                "retrieval_hit_present@k": rate(per_q, "ret_present"),
                "by_type_ret_prox": by_type(per_q, "ret_prox"),
                "by_type_cov_prox": by_type(per_q, "cov_prox"),
            },
            "per_q": per_q,
        }
        if o.get("page_pred"):
            results["axes"][a]["page_classification"] = eval_page_classification(o["page_pred"])
        if o.get("routing"):
            results["axes"][a]["routing"] = o["routing"]
        print(f"[{a}] eval {time.time()-t:.1f}s  "
              f"cov_prox={results['axes'][a]['retrieval']['extraction_coverage_prox']['rate']} "
              f"ret_prox@{TOPK}={results['axes'][a]['retrieval']['retrieval_hit_prox@k']['rate']} "
              f"ret_present@{TOPK}={results['axes'][a]['retrieval']['retrieval_hit_present@k']['rate']}")

    C.dump_json(HERE.parent / _OUTNAME, results)
    print("\n==== SUMMARY ====")
    hdr = f"{'axis':10} {'parse_s':>8} {'chunks':>7} {'cov_prox':>9} {'retP@k':>7} {'retPres':>7}"
    print(hdr)
    for a in outs:
        r = results["axes"][a]
        print(f"{a:10} {str(r['latency']['parse_time_s']):>8} {str(r['n_chunks']):>7} "
              f"{r['retrieval']['extraction_coverage_prox']['rate']:>9} "
              f"{r['retrieval']['retrieval_hit_prox@k']['rate']:>7} "
              f"{r['retrieval']['retrieval_hit_present@k']['rate']:>7}")
    print("\nby-type retrieval_hit_prox@k:")
    for a in outs:
        bt = results["axes"][a]["retrieval"]["by_type_ret_prox"]
        print(f"  {a:10}", {t: bt[t]["rate"] for t in sorted(bt)})

if __name__ == "__main__":
    main()

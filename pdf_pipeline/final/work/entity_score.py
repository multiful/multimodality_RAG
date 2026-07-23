# -*- coding: utf-8 -*-
"""엔티티 추출 채점: 팀 evaluate.py 관례(별칭 정규화 substring 매칭) 재사용.
브랜치별 Recall(text/table/image) + 전체 Recall/Precision(strict·lenient)/F1 + 메타데이터 + 지연."""
import sys, json, re, unicodedata
from pathlib import Path
from collections import Counter
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_exp as C

GT = C.load_json(HERE.parent/"ground_truth_entities_smartphone.json")
AXES = ["baseline", "enhanced", "enhanced_v2", "docling", "mineru"]

def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"[\s\.\,\-_/()]+", "", s)

ALIASES = GT["aliases"]
BRANCHES = GT["branches"]
TARGET = sorted(set(e for b in BRANCHES.values() for e in b))
KNOWN_NON = set(norm(x) for x in GT["known_non_entities"])

def candidates(ent):
    return [ent] + ALIASES.get(ent, [])

def entity_found(ent, extracted_norm_list, combined):
    """GT 엔티티가 추출목록에 있나 — 별칭 정규화 substring 양방향."""
    for c in candidates(ent):
        nc = norm(c)
        if not nc: continue
        if nc in combined:  # 팀 방식: 후보가 결합문자열에 포함
            return True
        for e in extracted_norm_list:
            if nc in e or e in nc:
                return True
    return False

def score_axis(axis):
    o = C.load_json(HERE/f"entity_out_{axis}.json")
    extracted = o["entities"]
    ex_norm = [norm(e) for e in extracted]
    combined = " | ".join(ex_norm)

    # ---- Recall (전체 + 브랜치별) ----
    def recall_of(entset):
        hits = [e for e in entset if entity_found(e, ex_norm, combined)]
        return hits, len(hits)/max(1,len(entset))
    all_hits, rec_all = recall_of(TARGET)
    branch_rec = {}
    for b, ents in BRANCHES.items():
        hits, r = recall_of(sorted(set(ents)))
        miss = [e for e in sorted(set(ents)) if e not in hits]
        branch_rec[b] = {"recall": round(r,3), "hit": len(hits), "n": len(set(ents)), "miss": miss}

    # ---- Precision (strict/lenient) ----
    tp=imprecise=fp=0; fp_items=[]; imp_items=[]
    for e, en in zip(extracted, ex_norm):
        if not en: continue
        matched = any(any(norm(c) in en or en in norm(c) for c in candidates(t)) for t in TARGET)
        if matched: tp += 1
        elif en in KNOWN_NON: fp += 1; fp_items.append(e)
        else: imprecise += 1; imp_items.append(e)
    tot = tp+imprecise+fp
    p_strict = tp/max(1,tot)
    p_lenient = (tp+imprecise)/max(1,tot)
    def f1(p,r): return 0.0 if p+r==0 else round(2*p*r/(p+r),3)

    sc = Counter(o.get("sentiments") or [])
    res = {
        "axis": axis,
        "latency": {"extract_latency_s": o["extract_latency_s"], "n_windows": o["n_windows"], "n_api_calls": o["n_api_calls"]},
        "recall_overall": round(rec_all,3), "recall_hit": f"{len(all_hits)}/{len(TARGET)}",
        "recall_by_branch": branch_rec,
        "precision_strict": round(p_strict,3), "precision_lenient": round(p_lenient,3),
        "f1_strict": f1(p_strict, rec_all), "f1_lenient": f1(p_lenient, rec_all),
        "n_extracted": tot, "tp": tp, "out_of_scope(imprecise)": imprecise, "non_entity_fp": fp,
        "fp_examples": fp_items[:10], "out_of_scope_examples": imp_items[:12],
        "metadata": {
            "sentiment_dist": dict(sc),
            "sector_detected": o.get("sectors"),
            "time_periods": (o.get("time_periods") or [])[:12],
            "n_metric_mentions": len(o.get("metric_mentions") or []),
        },
    }
    return res

def main():
    results = {"target_set_size": len(TARGET), "axes": {}}
    for a in AXES:
        if not (HERE/f"entity_out_{a}.json").exists():
            print(f"[skip] {a} (no entity_out)"); continue
        r = score_axis(a); results["axes"][a] = r
    C.dump_json(HERE.parent/"results_entities.json", results)

    print(f"\n==== ENTITY EXTRACTION (target set {len(TARGET)}) ====")
    print(f"{'axis':10} {'lat_s':>7} {'R_all':>6} {'R_text':>7} {'R_tbl':>6} {'R_img':>6} {'P_len':>6} {'F1_len':>7} {'#ext':>5}")
    for a in results["axes"]:
        r = results["axes"][a]; b=r["recall_by_branch"]
        print(f"{a:10} {r['latency']['extract_latency_s']:>7} {r['recall_overall']:>6} "
              f"{b['text']['recall']:>7} {b['table']['recall']:>6} {b['image']['recall']:>6} "
              f"{r['precision_lenient']:>6} {r['f1_lenient']:>7} {r['n_extracted']:>5}")
    print("\nmisses (table branch):")
    for a in results["axes"]:
        print(f"  {a:10}", results["axes"][a]["recall_by_branch"]["table"]["miss"])
    print("\nsentiment dist:")
    for a in results["axes"]:
        print(f"  {a:10}", results["axes"][a]["metadata"]["sentiment_dist"])

if __name__ == "__main__":
    main()

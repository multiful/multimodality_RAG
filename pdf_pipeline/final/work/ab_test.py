# -*- coding: utf-8 -*-
"""A/B 테스트: 개선판 enhanced_v2(B) vs 4개 모델(A) 문항별 paired 비교.
같은 문항셋을 두 모델에 돌린 대응표본이므로 McNemar(불일치쌍 부호검정, 정확 이항)로 유의성 판정.
지표: 검색 hit@5, hit@10, 추출 커버리지(문항단위) + 엔티티 recall(엔티티단위) + 지연/구조 집계."""
import json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
def load(p): return json.loads((FINAL/p).read_text(encoding="utf-8"))

R5 = load("results.json"); R10 = load("results_k10.json"); RE = load("results_entities.json")
B = "enhanced_v4"; A_LIST = ["enhanced_v3", "baseline", "docling", "mineru"]

def per_q_map(results, axis, field):
    return {q["id"]: q[field] for q in results["axes"][axis]["per_q"]}

def mcnemar_exact(b, c):
    """불일치쌍 b(=B만 성공), c(=A만 성공) 양측 정확 이항검정 p-value."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k+1)) / (2**n)
    return min(1.0, 2*p)

def paired(results, field, A):
    bmap = per_q_map(results, B, field); amap = per_q_map(results, A, field)
    ids = sorted(set(bmap) & set(amap))
    bw = sum(1 for i in ids if bmap[i] and not amap[i])   # B win
    aw = sum(1 for i in ids if amap[i] and not bmap[i])   # A win
    tie = len(ids) - bw - aw
    bw_ids = [i for i in ids if bmap[i] and not amap[i]]
    aw_ids = [i for i in ids if amap[i] and not bmap[i]]
    return {"n": len(ids), "B_win": bw, "A_win": aw, "tie": tie,
            "net": bw-aw, "p": round(mcnemar_exact(bw, aw), 4),
            "B_win_ids": bw_ids, "A_win_ids": aw_ids}

def entity_found_set(axis):
    """exhaustive 채점 기준 axis가 맞춘 target 엔티티 집합."""
    br = RE["axes"][axis]["recall_by_branch"]
    target = set()
    for b in br.values(): target |= set(b.get("miss", [])) | set()  # placeholder
    # target 전체 = 브랜치 합집합; found = target - union(miss)
    all_target = set()
    for b, d in br.items():
        # branch entity set = hit + miss; we only have miss + counts. reconstruct from n via GT
        pass
    return None

import re, unicodedata
_GT = load("ground_truth_entities_smartphone.json")
TARGET = sorted(set(e for b in _GT["branches"].values() for e in b))
ALIASES = _GT["aliases"]
def _norm(s): return re.sub(r"[\s\.\,\-_/()]+","",unicodedata.normalize("NFKC",str(s)).lower())
def _found_set(entities):
    exn=[_norm(e) for e in entities]; combined=" | ".join(exn); out=set()
    for t in TARGET:
        for c in [t]+ALIASES.get(t,[]):
            nc=_norm(c)
            if nc and (nc in combined or any(nc in e or e in nc for e in exn)): out.add(t); break
    return out
def entity_paired(A, field="entities"):
    """엔티티 단위 paired. field='entities'(exhaustive 병합) 또는 'entities_sidefield_only'(파이프라인 기본 추출기)."""
    fB=_found_set(load(f"work/entity_out_{B}.json").get(field) or [])
    fA=_found_set(load(f"work/entity_out_{A}.json").get(field) or [])
    bw=[e for e in TARGET if e in fB and e not in fA]
    aw=[e for e in TARGET if e in fA and e not in fB]
    tie=len(TARGET)-len(bw)-len(aw)
    return {"n":len(TARGET),"B_win":len(bw),"A_win":len(aw),"tie":tie,"net":len(bw)-len(aw),
            "p":round(mcnemar_exact(len(bw),len(aw)),4),"B_win_ents":bw,"A_win_ents":aw}

def lat(axis, res=R5): return res["axes"][axis]["latency"]["parse_time_s"]

print("="*74)
print(f"A/B TEST  —  B = {B}  vs  A = 각 모델   (McNemar 정확 이항검정, 30문항/46엔티티)")
print("="*74)
report={"B":B,"matchups":{}}
for A in A_LIST:
    r5=paired(R5,"ret_prox",A); r10=paired(R10,"ret_prox",A); cov=paired(R5,"cov_prox",A)
    ent=entity_paired(A,"entities"); ent_sf=entity_paired(A,"entities_sidefield_only")
    latB, latA = lat(B), lat(A)
    print(f"\n### {B}  vs  {A}")
    print(f"  {'metric':26} {'B_win':>5} {'A_win':>5} {'tie':>4} {'net':>4} {'p':>7}")
    for name,r in [("검색 hit@5",r5),("검색 hit@10",r10),("추출 커버리지",cov),
                   ("엔티티(기본추출기)",ent_sf),("엔티티(exhaustive)",ent)]:
        sig = "*" if r["p"]<0.05 else ""
        print(f"  {name:26} {r['B_win']:>5} {r['A_win']:>5} {r['tie']:>4} {r['net']:>+4} {r['p']:>7}{sig}")
    spd = (latA/latB) if (latB and latA) else 0
    print(f"  파싱 지연(s): B={latB}  A={latA}  → B가 {spd:.1f}x {'빠름' if spd>=1 else '느림(단 구조/분류 없음)'}")
    wins=sum(1 for r in [r5,r10,cov,ent] if r["net"]>0); losses=sum(1 for r in [r5,r10,cov,ent] if r["net"]<0)
    verdict = "B 우세" if wins>losses else ("A 우세" if losses>wins else "혼전")
    print(f"  → 품질(검색·커버리지·엔티티) {wins}승 {losses}패 {4-wins-losses}무 | 유의(p<.05) 지표: {[n for n,r in [('ret5',r5),('ret10',r10),('cov',cov),('ent',ent),('entSF',ent_sf)] if r['p']<0.05] or '없음'}")
    report["matchups"][A]={"ret@5":r5,"ret@10":r10,"cov":cov,"entity_exhaustive":ent,"entity_sidefield":ent_sf,
                            "lat_B":latB,"lat_A":latA,"speedup":round(spd,2),"verdict":verdict}

# 집계 스코어카드
print("\n"+"="*74); print("집계 스코어카드 (지표별 값)"); print("="*74)
def cov_rate(a,res): return res["axes"][a]["retrieval"]["extraction_coverage_prox"]["rate"]
def ret_rate(a,res): return res["axes"][a]["retrieval"]["retrieval_hit_prox@k"]["rate"]
print(f"{'axis':12} {'parse_s':>8} {'cov':>6} {'ret@5':>6} {'ret@10':>7} {'entR':>6} {'entP':>6}")
for a in [B]+A_LIST:
    er=RE["axes"][a]
    print(f"{a:12} {str(lat(a)):>8} {cov_rate(a,R5):>6} {ret_rate(a,R5):>6} {ret_rate(a,R10):>7} "
          f"{er['recall_overall']:>6} {er['precision_lenient']:>6}")
(FINAL/"results_abtest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print("\n-> results_abtest.json 저장")

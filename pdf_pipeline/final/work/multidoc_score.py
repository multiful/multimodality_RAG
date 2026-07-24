# -*- coding: utf-8 -*-
"""다문서 A/B 채점 — 자동 정답셋(consensus): 문서별 2개 이상 파서가 뽑은 엔티티를 '실재'로 보고
각 파서 recall 측정. 3문서 pooled McNemar(정확 이항)로 enhanced_v3 vs 각 파서 검정.
+ 지연·구조(캡션/표) 집계."""
import sys, re, unicodedata, math, json
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); import common_exp as C
MD=HERE/"multidoc"
DOCS=["doc1_construction","doc2_pharma","doc3_preview","doc4_retail","doc5_steel","doc06_aircraft","doc07_mart","doc08_solar","doc09_bank","doc10_food","doc11_delivery","doc12_tourism","doc13_construction2","doc14_semi","doc15_earnings"]; AXES=["baseline","enhanced_v5","docling","mineru"]
def norm(s): return re.sub(r"[\s\.\,\-_/()·]+","",unicodedata.normalize("NFKC",str(s)).lower())

def load_ents(docid,axis):
    p=MD/f"entity_{docid}_{axis}.json"
    return set(norm(e) for e in C.load_json(p)["entities"] if len(norm(e))>=2) if p.exists() else None

def match(a,b): return a==b or (len(a)>=2 and len(b)>=2 and (a in b or b in a))

def consensus_and_recall(docid):
    per={ax:load_ents(docid,ax) for ax in AXES}
    if any(v is None for v in per.values()): return None
    # 풀: 모든 정규화 엔티티. union-find로 유사(substring) 클러스터링
    pool=[]; owner=[]
    for ax in AXES:
        for e in per[ax]: pool.append(e); owner.append(ax)
    n=len(pool); parent=list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for i in range(n):
        for j in range(i+1,n):
            if match(pool[i],pool[j]): parent[find(i)]=find(j)
    clusters={}
    for i in range(n): clusters.setdefault(find(i),set()).add(owner[i])
    # consensus = >=2 axes; recall per axis
    cons=[axset for axset in clusters.values() if len(axset)>=2]
    rec={ax:sum(1 for axset in cons if ax in axset)/max(1,len(cons)) for ax in AXES}
    # per-cluster found bool per axis (for McNemar)
    found={ax:[ax in axset for axset in cons] for ax in AXES}
    return {"n_consensus":len(cons),"recall":rec,"found":found}

def mcnemar(b,c):
    n=b+c
    if n==0: return 1.0
    k=min(b,c); return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))

def main():
    B="enhanced_v5"
    pooled={ax:[] for ax in AXES}; per_doc={}
    for docid in DOCS:
        r=consensus_and_recall(docid)
        if r is None: print(f"[{docid}] incomplete — skip"); continue
        per_doc[docid]=r
        for ax in AXES: pooled[ax]+=r["found"][ax]
    print("="*70); print("다문서 엔티티 A/B (consensus 자동 정답셋, sidefield 추출기)"); print("="*70)
    print(f"{'doc':20} {'n_cons':>6} " + " ".join(f"{ax[:9]:>9}" for ax in AXES))
    for docid in DOCS:
        if docid not in per_doc: continue
        r=per_doc[docid]; print(f"{docid:20} {r['n_consensus']:>6} " + " ".join(f"{r['recall'][ax]:>9.3f}" for ax in AXES))
    # pooled recall
    tot=len(pooled[B])
    print(f"\n[POOLED recall over {tot} consensus entities]")
    for ax in AXES: print(f"  {ax:12} {sum(pooled[ax])/max(1,tot):.3f}  ({sum(pooled[ax])}/{tot})")
    # McNemar B vs each
    print(f"\n[McNemar paired: B={B} vs A]")
    ab={}
    for ax in AXES:
        if ax==B: continue
        bw=sum(1 for i in range(tot) if pooled[B][i] and not pooled[ax][i])
        aw=sum(1 for i in range(tot) if pooled[ax][i] and not pooled[B][i])
        p=mcnemar(bw,aw); sig="*" if p<0.05 else ""
        print(f"  vs {ax:10}  B_win={bw:>3} A_win={aw:>3} net={bw-aw:>+3} p={p:.4f}{sig}")
        ab[ax]={"B_win":bw,"A_win":aw,"net":bw-aw,"p":round(p,4)}
    # latency + structure
    print("\n[지연/구조 집계 (문서 평균)]")
    print(f"{'axis':12} {'parse_s(avg)':>12} {'chart_cap%':>10} {'table_cap%':>10}")
    agg={}
    for ax in AXES:
        ps=[]; cc=[]; tc=[]
        for docid in DOCS:
            p=MD/f"out_{docid}_{ax}.json"
            if not p.exists(): continue
            o=C.load_json(p); ps.append(o["parse_time_s"])
            # caption preservation vs raw
            raw=C.load_json(MD/f"out_{docid}_baseline.json")["structure"]
            st=o["structure"]
            cc.append(st.get("chart_titles",0)/max(1,raw["chart_titles"]) if raw["chart_titles"] else 1.0)
            tc.append(st.get("table_caps",0)/max(1,raw["table_caps"]) if raw["table_caps"] else 1.0)
        agg[ax]={"parse_avg":round(sum(ps)/max(1,len(ps)),2),"chart_cap":round(sum(cc)/max(1,len(cc)),3),"table_cap":round(sum(tc)/max(1,len(tc)),3)}
        print(f"{ax:12} {agg[ax]['parse_avg']:>12} {agg[ax]['chart_cap']:>10.3f} {agg[ax]['table_cap']:>10.3f}")
    C.dump_json(HERE.parent/"results_multidoc_ab.json",
        {"per_doc":{d:per_doc[d]["recall"] for d in per_doc},"pooled_recall":{ax:round(sum(pooled[ax])/max(1,tot),3) for ax in AXES},
         "mcnemar":ab,"latency_structure":agg,"n_pooled_consensus":tot})
    print("\n-> results_multidoc_ab.json 저장")

if __name__=="__main__": main()

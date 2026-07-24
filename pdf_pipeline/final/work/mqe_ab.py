# -*- coding: utf-8 -*-
"""MQE 하위질의 생성 A/B — 핸드오프 §5 지적: route_search/decompose가 쓰는 결합호출
_classify_and_expand()(분류+확장 1콜, 지연최적화 [44])가 전용 mqe_search() 하위질의 프롬프트보다
품질 낮은 하위질의를 만들 수 있다(납기 ndcg 0.693 vs 0.871). Construct 건자재 질의로 재현·측정."""
import os, sys, math, json
from pathlib import Path
ROOT=Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP=ROOT/"pdf_pipeline"
for d in [str(PP),str(PP/"text_processing")]: sys.path.insert(0,d)
for line in open(ROOT/".env",encoding="utf-8"):
    line=line.strip()
    if line and "=" in line and not line.startswith("#"): k,v=line.split("=",1); os.environ.setdefault(k.strip(),v.strip())
DB=os.environ["SUPABASE_DIRECT_DB_URL"]
QUERY="이 PDF에 나오는 기업의 인사이트 도출해주고, 건자재업종 종목 중에 주간 수익률이 가장 높은 기업이 어딘지 알려줘"
GOLD=["건자재업종 종목 주간 수익률","KCC글라스","도표 4"]   # 건자재 관련 evidence 앵커
OUT=PP/"final"/"results_mqe_ab.json"
import entity_fusion, index_text
from openai import OpenAI
client=OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def relevant(chunk):
    c=chunk.get("content") or ""; return any(g in c for g in GOLD)
def ndcg(hits,k=8):
    rels=[1 if relevant(h["chunk"]) else 0 for h in hits[:k]]
    dcg=sum(r/math.log2(i+2) for i,r in enumerate(rels))
    ideal=sorted(rels,reverse=True); idcg=sum(r/math.log2(i+2) for i,r in enumerate(ideal))
    return round(dcg/idcg,3) if idcg>0 else 0.0, sum(rels)

def main():
    idx=entity_fusion.load_evidence_from_db(DB, pdf_id="Construct", use_cache=False); idx.entity_count=8
    idx2=entity_fusion.load_evidence_from_db(DB, pdf_id="Construct", use_cache=False); idx2.entity_count=8
    entity_fusion.weighted_hybrid_search(idx,"warm",top_k=1)  # BGE 콜드로드 분리
    res={"query":QUERY,"gold_anchors":GOLD}
    # A) 전용 mqe_search
    hitsA, subsA = index_text.mqe_search(idx, QUERY, client=client, top_k=8, use_bm25=False)
    nA,relA=ndcg(hitsA)
    # B) 결합 _classify_and_expand + _fuse_multi_query
    qtype, subsB = index_text._classify_and_expand(QUERY, client=client)
    hitsB = index_text._fuse_multi_query(idx2, [QUERY]+(subsB or []), top_k=8, use_bm25=False)
    nB,relB=ndcg(hitsB)
    print(f"[A/전용 mqe_search]     ndcg@8={nA} rel@8={relA}")
    print(f"    subqueries: {subsA[1:]}")
    print(f"[B/결합 classify+expand] ndcg@8={nB} rel@8={relB}  (qtype={qtype})")
    print(f"    subqueries: {subsB}")
    res["dedicated_mqe"]={"ndcg@8":nA,"rel@8":relA,"subqueries":subsA[1:]}
    res["combined_classify_expand"]={"ndcg@8":nB,"rel@8":relB,"qtype":qtype,"subqueries":subsB}
    res["verdict"]="dedicated 우세" if nA>nB else ("combined 우세" if nB>nA else "동률")
    OUT.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"\n판정: {res['verdict']}  -> results_mqe_ab.json 저장")

if __name__=="__main__": main()

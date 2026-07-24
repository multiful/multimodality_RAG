# -*- coding: utf-8 -*-
"""추상 질의 생성모델 A/B — gpt-4o-mini vs gpt-4.1, 전체 DB 컨텍스트 포함(8.6K자 재무요약).
쉬운 factoid 대신 '인사이트/핵심요약' 추상 질의로 4.1의 강한 추론 이점이 드러나는지 검증.
측정: 지연 / 입력·출력 토큰(토큰 최적화 근거) / LLM-judge(gpt-4o 블라인드 우열)."""
import os, sys, time, json
from pathlib import Path
ROOT=Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP=ROOT/"pdf_pipeline"
for d in [str(PP),str(PP/"text_processing")]: sys.path.insert(0,d)
for line in open(ROOT/".env",encoding="utf-8"):
    line=line.strip()
    if line and "=" in line and not line.startswith("#"): k,v=line.split("=",1); os.environ.setdefault(k.strip(),v.strip())
DB=os.environ["SUPABASE_DIRECT_DB_URL"]
PDF=str(PP/"reference"/"Construct"/"20260721_industry_362851000.pdf")
OUT=PP/"final"/"results_abstract_gen_ab.json"
import entity_fusion, index_text, company_entity_linking
from openai import OpenAI
import fitz
client=OpenAI(api_key=os.environ["OPENAI_API_KEY"])
QUERIES={
  "insight":"이 리포트의 핵심 투자 인사이트를 도출해줘 — 업종 방향성, 종목별 시사점, 리스크를 종합해서.",
  "summary":"이 리포트를 핵심만 요약·분석해줘. 무엇이 중요한 변화이고 투자자가 알아야 할 포인트가 뭔지.",
}
def build_ctx():
    idx=entity_fusion.load_evidence_from_db(DB, pdf_id="Construct", use_cache=False); idx.entity_count=8
    entity_fusion.weighted_hybrid_search(idx,"warm",top_k=1)
    doc=fitz.open(PDF); txt="\n".join(p.get_text() for p in doc); doc.close()
    matched=company_entity_linking.find_mentioned_companies(txt)
    dbctx=company_entity_linking.fetch_company_db_context(DB, matched)
    return idx, dbctx
def gen(model, prompt):
    t=time.time()
    r=client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}])
    dt=time.time()-t; u=r.usage
    return r.choices[0].message.content, dt, u.prompt_tokens, u.completion_tokens
def judge(q, a1, a2):
    """gpt-4o 블라인드: 두 답 중 어느 게 더 나은 인사이트인가(순서 무작위화는 생략, A/B 라벨만)."""
    p=(f"질문: {q}\n\n[답변A]\n{a1[:2500]}\n\n[답변B]\n{a2[:2500]}\n\n"
       "두 답변 중 근거 활용·통찰 깊이·정확성이 더 나은 쪽을 고르라. "
       "'A' 또는 'B' 또는 'TIE' 한 단어로만 답하라.")
    r=client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":p}], temperature=0)
    return r.choices[0].message.content.strip()[:4]
def main():
    idx, dbctx=build_ctx()
    print(f"[ctx] db_context={len(dbctx)}자")
    res={"db_context_len":len(dbctx),"queries":{}}
    for qname,q in QUERIES.items():
        hits=index_text.decompose_and_route_search(idx, q, client=client, top_k=8)[0]
        ev="\n\n".join(f"[{h['chunk'].get('source_type')} / p{h['chunk'].get('page')}] {h['chunk']['content']}" for h in hits)
        ctx=ev+"\n\n=== 기업 DB 재무요약 ===\n\n"+dbctx
        prompt=f"[통합 근거]\n{ctx}\n\n[요청]\n{q}\n\n근거의 구체 수치를 인용해 작성하라."
        outs={}
        for model in ["gpt-4o-mini","gpt-4.1"]:
            a,dt,pin,pout=gen(model,prompt)
            outs[model]={"answer":a,"latency_s":round(dt,2),"in_tok":pin,"out_tok":pout}
            print(f"[{qname}/{model:11}] {dt:6.2f}s in={pin} out={pout}")
        # LLM-judge (A=4o-mini, B=4.1) 양방향으로 2번(위치편향 완화)
        j1=judge(q, outs["gpt-4o-mini"]["answer"], outs["gpt-4.1"]["answer"])  # A=mini B=4.1
        j2=judge(q, outs["gpt-4.1"]["answer"], outs["gpt-4o-mini"]["answer"])  # A=4.1 B=mini
        # 승자 집계
        win={"gpt-4o-mini":0,"gpt-4.1":0,"tie":0}
        win["gpt-4o-mini" if j1=="A" else ("gpt-4.1" if j1=="B" else "tie")]+=1
        win["gpt-4.1" if j2=="A" else ("gpt-4o-mini" if j2=="B" else "tie")]+=1
        print(f"[{qname}/judge] j1={j1} j2={j2} -> {win}")
        for m in outs: outs[m]["answer"]=outs[m]["answer"][:600]
        res["queries"][qname]={"models":outs,"judge":{"j1":j1,"j2":j2,"win":win}}
    OUT.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
    print("-> results_abstract_gen_ab.json 저장")
if __name__=="__main__": main()

# -*- coding: utf-8 -*-
"""Construct(건설업 Weekly) 최종 A/B — 재파싱 없이 Supabase evidence 로드 후:
  (A) 검색 전략: weighted_hybrid vs route_search vs decompose_and_route_search
      지표: 건설업종(도표3) evidence가 건자재(도표4)보다 위로 랭크되는가 + 지연
  (B) 생성 모델: gpt-4o-mini vs gpt-4.1 — 지연 / 재시도 / 정답성(1위=IPARK현대산업개발)
  + 단계별 지연(병목).
쿼리: "여기 나온 기업의 인사이트 도출 + 건설업종 종목 주간 수익률 최고 기업 추출"
"""
import os, sys, time, json
from pathlib import Path
ROOT=Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP=ROOT/"pdf_pipeline"
for d in [str(PP),str(PP/"text_processing"),str(PP/"table_processing"),str(PP/"page_classification")]: sys.path.insert(0,d)
for line in open(ROOT/".env",encoding="utf-8"):
    line=line.strip()
    if line and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k.strip(),v.strip())
DB=os.environ["SUPABASE_DIRECT_DB_URL"]
QUERY="여기 나온 기업의 인사이트 도출해주고, 건설업종 종목 주간 수익률이 가장 높은 기업은 어딘지 추출해"
PDF=str(PP/"reference"/"Construct"/"20260721_industry_362851000.pdf")
OUT=PP/"final"/"results_construct_ab.json"

import entity_fusion, index_text, citation_check, company_entity_linking
from openai import OpenAI
client=OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def rank_of(hits, needle):
    for i,h in enumerate(hits,1):
        if needle in (h["chunk"].get("content") or ""): return i
    return None
def img_rank(hits, needle):
    """image-소스 hit 중 needle 포함 첫 순위(도표3 image vs 도표4 image를 정확히 분리)."""
    for i,h in enumerate(hits,1):
        if h["chunk"].get("source_type")=="image" and needle in (h["chunk"].get("content") or ""): return i
    return None

def main():
    T={}
    t=time.time(); idx=entity_fusion.load_evidence_from_db(DB, pdf_id="Construct"); T["load_evidence"]=round(time.time()-t,2)
    # entity_count 사전계산(route_search가 사용)
    t=time.time(); ec=index_text.precompute_entity_count(idx, pdf_path=PDF, client=client); T["precompute_entity_count"]=round(time.time()-t,2)
    # BGE 워밍업(첫 검색 콜드로드 분리)
    t=time.time(); entity_fusion.weighted_hybrid_search(idx, "워밍업", top_k=1); T["bge_warmup"]=round(time.time()-t,2)
    print(f"[load] evidence={len(idx.chunks)} entity_count={ec} warmup={T['bge_warmup']}s")

    # ===== (A) 검색 전략 A/B =====
    C3="건설업종 종목 주간 수익률"; C4="건자재업종 종목 주간 수익률"
    strategies={
        "weighted_hybrid": lambda: (entity_fusion.weighted_hybrid_search(idx, QUERY, top_k=8), None),
        "route_search":    lambda: index_text.route_search(idx, QUERY, client=client, top_k=8),
        "decompose_route": lambda: index_text.decompose_and_route_search(idx, QUERY, client=client, top_k=8),
    }
    retr={}
    for name,fn in strategies.items():
        t=time.time()
        res=fn(); hits=res[0] if isinstance(res,tuple) else res
        dt=round(time.time()-t,2)
        i3,i4=img_rank(hits,C3), img_rank(hits,C4)
        top=hits[0]["chunk"]; topprev=(top.get("content") or "")[:60].replace("\n"," ")
        # 정답성: 도표3(건설업종) image가 도표4(건자재) image보다 위 (질의가 '건설업종'이므로)
        ok = (i3 is not None) and (i4 is None or i3 < i4)
        retr[name]={"latency_s":dt,"건설업종_도표3_img_rank":i3,"건자재_도표4_img_rank":i4,
                    "correct(건설>건자재)":ok,"top1":f"[{top.get('source_type')}]p{top.get('page')} {topprev}"}
        print(f"[A/검색] {name:16} {dt:5.2f}s  도표3(img)={i3} 도표4(img)={i4} correct={ok}")

    # ===== (B) 생성 모델 A/B (동일 컨텍스트) =====
    # 최선 검색(decompose)으로 컨텍스트 고정
    hits=index_text.decompose_and_route_search(idx, QUERY, client=client, top_k=8)[0]
    import fitz; doc=fitz.open(PDF); full_text="\n".join(doc[i].get_text() for i in range(doc.page_count)); doc.close()
    matched=company_entity_linking.find_mentioned_companies(full_text)
    db_context=company_entity_linking.fetch_company_db_context(DB, matched)
    evidence_context="\n\n".join(f"[{h['chunk'].get('source_type')} / p{h['chunk'].get('page')}] {h['chunk']['content']}" for h in hits)
    full_context=evidence_context + (("\n\n=== 기업 DB 참고 정보 ===\n\n"+db_context) if db_context else "")
    prompt=("다음 근거로 투자 인사이트를 작성하고 질문에 답하라. [image]는 차트 OCR이라 괄호숫자=음수(하락)로 해석.\n\n"
            f"[통합 근거]\n{full_context}\n\n[사용자 요청]\n{QUERY}\n")
    print(f"[B/생성] 기업매칭 {len(matched)}건, 컨텍스트 {len(full_context)}자")
    gen={}
    for model in ["gpt-4o-mini","gpt-4.1"]:
        t=time.time()
        r=citation_check.generate_with_citation_check(client, prompt, context=full_context, model=model, max_retries=2)
        dt=round(time.time()-t,2)
        ans=r["answer"]
        # 정답성 키워드 체크
        names_top=lambda s: s in ans and ("가장 높" in ans or "최고" in ans or "1위" in ans)
        correct = ("IPARK" in ans or "아이파크" in ans or "현대산업개발" in ans)
        wrong = any(w in ans and ("가장 높" in ans) for w in ["GS건설이 -14","금호건","한샘 11.7"])  # 오답 신호(대략)
        gen[model]={"latency_s":dt,"attempts":r["attempts"],"unsupported_numbers":r["unsupported_numbers"],
                    "correct(IPARK언급)":correct,"answer_head":ans[:220].replace("\n"," ")}
        print(f"[B/생성] {model:12} {dt:6.2f}s attempts={r['attempts']} unsupported={r['unsupported_numbers']} IPARK={correct}")

    # ===== 엔드투엔드 지연(검색+생성, 병목 관점) =====
    e2e={
        "current(decompose_route + gpt-4.1)": round(retr["decompose_route"]["latency_s"] + gen["gpt-4.1"]["latency_s"],2),
        "opt_A(decompose_route + gpt-4o-mini)": round(retr["decompose_route"]["latency_s"] + gen["gpt-4o-mini"]["latency_s"],2),
        "opt_B(weighted_hybrid + gpt-4o-mini)": round(retr["weighted_hybrid"]["latency_s"] + gen["gpt-4o-mini"]["latency_s"],2),
    }
    print("\n[E2E 검색+생성 지연]")
    for k,v in e2e.items(): print(f"   {k:42} {v}s")
    res={"query":QUERY,"stage_timings":T,"retrieval_ab":retr,"generation_ab":gen,
         "e2e_search_plus_gen":e2e,"n_matched_companies":len(matched)}
    OUT.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\n-> results_construct_ab.json 저장")

if __name__=="__main__": main()

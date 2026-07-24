# -*- coding: utf-8 -*-
"""[재일] RAGAS 평가용 생성 산출 — 골든셋 질의를 프로덕션 경로(검색+DB컨텍스트)로 돌려
gpt-4o-mini / gpt-4.1 두 모델의 답변과 그때 실제로 준 컨텍스트를 함께 저장한다.
토큰이 아니라 '성능'(환각/근거활용/논리)을 재기 위한 입력 데이터 생성 단계."""
import os, sys, json, time
from pathlib import Path
ROOT = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP = ROOT / "pdf_pipeline"
for d in [str(PP), str(PP/"text_processing")]: sys.path.insert(0, d)
for line in open(ROOT/".env", encoding="utf-8"):
    line = line.strip()
    if line and "=" in line and not line.startswith("#"): k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
DB = os.environ["SUPABASE_DIRECT_DB_URL"]
PDF = PP/"reference"/"Construct"/"20260721_industry_362851000.pdf"
GOLD = json.loads((PP/"final"/"golden_set_construct_routing.json").read_text(encoding="utf-8"))
OUT = PP/"final"/"ragas_input_construct.json"
# 타입별 3개씩 — 생성 품질 비교는 추상형이 핵심이지만 키워드/하이브리드도 환각 검사를 위해 포함
PICK = ["K2", "K4", "K6", "H1", "H3", "H4", "A1", "A2", "A6"]
MODELS = ["gpt-4o-mini", "gpt-4.1"]

import entity_fusion, index_text, company_entity_linking, fitz
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def main():
    index = entity_fusion.load_evidence_from_db(DB, pdf_id="Construct", use_cache=False)
    index.entity_count = 8
    entity_fusion.weighted_hybrid_search(index, "warm", top_k=1)
    doc = fitz.open(PDF); txt = "\n".join(p.get_text() for p in doc); doc.close()
    dbctx = company_entity_linking.fetch_company_db_context(
        DB, company_entity_linking.find_mentioned_companies(txt))
    print(f"[ctx] db_context={len(dbctx)}자")

    items = {q["id"]: q for q in GOLD["queries"]}
    out = {"db_context_len": len(dbctx), "samples": []}
    for qid in PICK:
        it = items[qid]; q = it["q"]
        hits = index_text.decompose_and_route_search(index, q, client=client, top_k=8)[0]
        contexts = [f"[{h['chunk'].get('source_type')}/p{h['chunk'].get('page')}] {h['chunk']['content']}"
                    for h in hits]
        ctx = "\n\n".join(contexts) + "\n\n=== 기업 DB 재무요약 ===\n\n" + dbctx
        prompt = (f"[통합 근거]\n{ctx}\n\n[요청]\n{q}\n\n"
                  "근거의 구체 수치를 인용해 답하라. 근거에 없는 내용은 절대 추측해 쓰지 말 것.")
        rec = {"id": qid, "type": it["type"], "question": q, "reference": it["answer"],
               "contexts": contexts, "db_context": dbctx, "answers": {}}
        for m in MODELS:
            t = time.time()
            r = client.chat.completions.create(model=m, messages=[{"role": "user", "content": prompt}],
                                               temperature=0)
            rec["answers"][m] = {"text": r.choices[0].message.content,
                                 "latency_s": round(time.time()-t, 2),
                                 "in_tok": r.usage.prompt_tokens, "out_tok": r.usage.completion_tokens}
            print(f"  {qid}/{m:12} {rec['answers'][m]['latency_s']:5.1f}s out={r.usage.completion_tokens}")
        out["samples"].append(rec)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT.name} 저장 ({len(out['samples'])}질의 x {len(MODELS)}모델)")

if __name__ == "__main__":
    main()

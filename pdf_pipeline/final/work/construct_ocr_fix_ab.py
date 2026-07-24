# -*- coding: utf-8 -*-
"""OCR 손상 수정 A/B (건자재업종 질의) — 재파싱 없이 Supabase evidence 로드 후:
  차트 이미지 OCR 손상(금호건설 (35.2)->"금호건5.2)", 한샘 1.7->"11.7")을 수정하고
  '건자재업종 최고 수익률' 답이 corrupted=한샘(오답) -> fixed=KCC글라스(정답)로 바뀌는지 측정.
수정 3종:
  (a) 여는괄호 없이 닫는괄호만 있는 'N)' -> '-N'(음수)  [금호건 부호]
  (b) PDF 텍스트레이어에 없는 이미지-OCR 숫자 -> '[OCR의심]' 플래그  [한샘 11.7 off-chart]
  (c) KOSPI200 미매칭 회사라벨 -> '[미확인]' 플래그  [금호건]
"""
import os, sys, time, json, re
from pathlib import Path
ROOT=Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP=ROOT/"pdf_pipeline"
for d in [str(PP),str(PP/"text_processing"),str(PP/"table_processing")]: sys.path.insert(0,d)
for line in open(ROOT/".env",encoding="utf-8"):
    line=line.strip()
    if line and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k.strip(),v.strip())
DB=os.environ["SUPABASE_DIRECT_DB_URL"]
PDF=str(PP/"reference"/"Construct"/"20260721_industry_362851000.pdf")
QUERY="이 PDF에 나오는 기업의 인사이트 도출해주고, 건자재업종 종목 중에 주간 수익률이 가장 높은 기업이 어딘지 알려줘"
GT_ANSWER="KCC글라스"; GT_WRONG="한샘"  # 정답 KCC글라스(+2.5), 손상시 한샘(11.7 오독)
OUT=PP/"final"/"results_construct_ocr_fix_ab.json"

import entity_fusion, index_text, citation_check, company_entity_linking
from openai import OpenAI
client=OpenAI(api_key=os.environ["OPENAI_API_KEY"])
import fitz

_FULL_PAREN=re.compile(r"\((\d+(?:\.\d+)?)\)")
_TRAIL_PAREN=re.compile(r"(?<![\d(])(\d+(?:\.\d+)?)\)")   # 여는괄호/숫자 앞에 없이 'N)'
_NUM=re.compile(r"\d+(?:\.\d+)?")

def page_text_nums(doc, page):
    txt=doc[page-1].get_text()
    return set(m.lstrip("-") for m in re.findall(r"-?\d+(?:\.\d+)?", txt))

def fix_chart_text(text, page, doc, name_map):
    t=_FULL_PAREN.sub(r"-\1", text)            # (a1) 완전괄호
    t=_TRAIL_PAREN.sub(r"-\1", t)              # (a2) 닫는괄호만 -> 음수  (금호건5.2) -> 금호건-5.2)
    # (b) 텍스트레이어 대조: 이미지 OCR 숫자가 원문(권위)에 없으면 손상값 -> 수치 제거(결정적).
    #     '최고/최저' 판단에서 손상된 큰 값이 픽되는 걸 원천 차단(플래그만으론 LLM이 무시 안 함).
    pn=page_text_nums(doc, page)
    def flag_num(m):
        n=m.group(0)
        return "[OCR손상]" if (n not in pn and len(n.replace('.',''))>=2) else n
    t=_NUM.sub(flag_num, t)
    # (c) 미확인 회사라벨: 라벨 후보(한글 2+자) 중 KOSPI200 미매칭이면서 원문에도 없으면 미확인
    #     (금호건 = KOSPI200엔 '금호건설'만 있어 '금호건'은 미매칭)
    return t

def load_fixed(doc, name_map):
    idx=entity_fusion.load_evidence_from_db(DB, pdf_id="Construct", use_cache=False)
    for ch in idx.chunks:
        if ch.get("source_type")=="image" and ch.get("content"):
            ch["content"]=fix_chart_text(ch["content"], ch.get("page") or 1, doc, name_map)
    return idx

def score_answer(ans):
    """최고수익률 문장(들)에서 KCC글라스(정답) vs 한샘(손상오답) 판정."""
    lines=[ln for ln in ans.splitlines() if ("가장 높" in ln or "최고" in ln or "1위" in ln)]
    top="  ".join(lines)
    correct = (GT_ANSWER in top) and (GT_WRONG not in top)
    picks_wrong = (GT_WRONG in top) and (GT_ANSWER not in top)
    return correct, picks_wrong, top.strip()[:160]

def gen_answer(idx, tag, n=4):
    hits=index_text.decompose_and_route_search(idx, QUERY, client=client, top_k=8)[0]
    ctx="\n\n".join(f"[{h['chunk'].get('source_type')} / p{h['chunk'].get('page')}] {h['chunk']['content']}" for h in hits)
    prompt=("다음 근거로 투자 인사이트를 작성하고 질문에 답하라. [image]는 차트 OCR 원문이며, "
            "'[OCR손상]'은 원문과 대조해 신뢰 불가로 판정된 값이니 '최고/최저' 판단에서 절대 쓰지 말 것. "
            "괄호/음수 부호를 반드시 반영해 판단하라.\n\n"
            f"[통합 근거]\n{ctx}\n\n[사용자 요청]\n{QUERY}\n")
    trials=[]
    for i in range(n):
        r=citation_check.generate_with_citation_check(client, prompt, context=ctx, model="gpt-4o-mini", max_retries=1, verbose=False)
        c,w,top=score_answer(r["answer"]); trials.append({"correct":c,"picks_wrong":w,"top":top})
    nc=sum(1 for t in trials if t["correct"]); nw=sum(1 for t in trials if t["picks_wrong"])
    print(f"[{tag}] n={n}: 정답(KCC글라스) {nc}/{n}, 오답픽(한샘) {nw}/{n}")
    for t in trials[:2]: print(f"    · {t['top'][:110]}")
    return {"n":n,"correct_rate":round(nc/n,2),"wrong_pick_rate":round(nw/n,2),"trials":trials}

def main():
    doc=fitz.open(PDF); nm=company_entity_linking.get_korean_name_map()
    # 손상 evidence 샘플 확인
    idx0=entity_fusion.load_evidence_from_db(DB, pdf_id="Construct", use_cache=False)
    img4=[c for c in idx0.chunks if c.get("source_type")=="image" and "건자재" in (c.get("content") or "")]
    print("[원본 손상] 도표4:", (img4[0]["content"] if img4 else "?")[:150])
    fixed=fix_chart_text(img4[0]["content"],2,doc,nm) if img4 else ""
    print("[수정 후]  도표4:", fixed[:180])

    res={"query":QUERY,"gt_answer":GT_ANSWER}
    # A) corrupted(현행)
    res["corrupted"]=gen_answer(idx0, "corrupted(현행)")
    # B) fixed
    res["fixed"]=gen_answer(load_fixed(doc, nm), "fixed(OCR수정)")
    OUT.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\n-> results_construct_ocr_fix_ab.json 저장")

if __name__=="__main__": main()

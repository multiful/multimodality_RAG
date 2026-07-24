# -*- coding: utf-8 -*-
"""[재일] OCR 손상 필터 수정 A/B — 오탐(정상값 삭제)은 없애면서 진성 손상은 계속 잡는가.
  오탐 케이스 : c밴드 차트들(축 라벨 "24.07"을 OCR이 "24,07"로 읽어 조각나던 것)
  진성 케이스 : Construct 도표4 한샘 — 겹친 텍스트로 1.7이 11.7로 깨진 값(원문에 11.7 없음)
OpenAI 호출 0회."""
import os, sys, json
from pathlib import Path
ROOT = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP = ROOT / "pdf_pipeline"
sys.path.insert(0, str(PP))
for line in open(ROOT/".env", encoding="utf-8"):
    line = line.strip()
    if line and "=" in line and not line.startswith("#"): k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
OUT = PP/"final"/"results_ocr_filter_fix_ab.json"

import psycopg2
from run_investment_opinion_demo import _normalize_chart_card_signs

DOCS = {"c밴드(오탐 검사)": ("upload_44b76ed9", PP/"reference"/"C밴드"/"c밴드.pdf"),
        "Construct(진성 검사)": ("Construct", PP/"reference"/"Construct"/"20260721_industry_362851000.pdf")}

def main():
    conn = psycopg2.connect(os.environ["SUPABASE_DIRECT_DB_URL"]); cur = conn.cursor()
    res = {}
    for label, (pdf_id, pdf) in DOCS.items():
        cur.execute("""select page, content from document_evidence
                       where pdf_id=%s and source_type='image' order by page""", (pdf_id,))
        rows = cur.fetchall()
        if not rows:
            print(f"[{label}] evidence 없음 — 스킵"); continue
        # DB에 저장된 content는 이미 예전 로직으로 정규화된 결과라, 손상표시를 원복할 수 없다.
        # 대신 "지금 로직을 다시 적용했을 때 새로 손상 판정되는 양"을 재서 과잉 여부를 본다.
        cards = [{"block_type": "chart", "page": p, "embed_text": c} for p, c in rows]
        before = sum(c["embed_text"].count("[OCR손상]") for c in cards)
        out = _normalize_chart_card_signs(cards, pdf_path=pdf)
        after = sum(c["embed_text"].count("[OCR손상]") for c in out)
        print(f"\n[{label}] 카드 {len(cards)}개 | 기존 저장본의 손상표시 {before}개 -> 새 로직 재적용 후 {after}개")
        for c in out:
            if "[OCR손상]" in c["embed_text"]:
                import re
                print(f"    p{c['page']}: {re.findall(r'\\S*\\[OCR손상\\]\\S*', c['embed_text'])[:5]}")
        res[label] = {"cards": len(cards), "damaged_before": before, "damaged_after": after}
    conn.close()

    # 진성 손상 단위테스트: 원문에 없는 값(11.7)은 여전히 잡히는가 / 정상 축라벨(24,07)은 살아남는가
    print("\n=== 단위테스트 ===")
    import fitz
    con = PP/"reference"/"Construct"/"20260721_industry_362851000.pdf"
    fake = [{"block_type": "chart", "page": 2,
             "embed_text": "도표 4. 건자재업종 종목 주간 수익률 2.5 KCC글라스 11.7 한샘 (8.1) 동화기업"}]
    got = _normalize_chart_card_signs(fake, pdf_path=con)[0]["embed_text"]
    print(f"  진성손상(한샘 11.7) 잡히나: {'11.7' not in got}  -> {got[:110]}")
    cb = PP/"reference"/"C밴드"/"c밴드.pdf"
    fake2 = [{"block_type": "chart", "page": 3,
              "embed_text": "쏠리드 수정TP 35,000 30,000 25,000 20,000 15,000 10,000 5,000 0 24,07 24,10 25,01"}]
    got2 = _normalize_chart_card_signs(fake2, pdf_path=cb)[0]["embed_text"]
    print(f"  오탐(24,07 등) 안 지워지나: {'[OCR손상]' not in got2}  -> {got2[:110]}")
    res["unit_test"] = {"true_positive_kept": "11.7" not in got, "false_positive_gone": "[OCR손상]" not in got2}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT.name} 저장")

if __name__ == "__main__":
    main()

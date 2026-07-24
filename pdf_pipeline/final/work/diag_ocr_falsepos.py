# -*- coding: utf-8 -*-
"""[재일] `[OCR손상]` 과잉 치환 진단 — 남은과제 13(텍스트레이어 대조 오탐률)이 실제로 터진 건지 확인.

현재 로직(run_investment_opinion_demo._normalize_chart_card_signs 단계3)은 이미지 OCR의 숫자가
PDF 텍스트레이어에 없으면 [OCR손상]으로 지운다. 두 가지 실패 가설을 검사한다:
  H1 토큰화 불일치 — OCR은 "35,000"/"24,07"(쉼표), 텍스트레이어는 "35,000"/"24.07".
     정규식 `\\d+(?:\\.\\d+)?`은 쉼표를 모르니 OCR쪽은 "35","000"으로 쪼개지는데
     텍스트레이어쪽은 "24.07"을 한 토큰으로 잡아 서로 매칭이 안 된다.
  H2 텍스트레이어 부재 — 그 페이지 차트가 래스터 이미지라 축 라벨이 텍스트레이어에 아예 없다.
"""
import os, sys, re, json
from pathlib import Path
ROOT = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP = ROOT / "pdf_pipeline"
sys.path.insert(0, str(PP))
for line in open(ROOT/".env", encoding="utf-8"):
    line = line.strip()
    if line and "=" in line and not line.startswith("#"): k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
PDF = PP/"reference"/"C밴드"/"c밴드.pdf"
OUT = PP/"final"/"results_ocr_falsepos_diag.json"

import fitz, psycopg2
from run_investment_opinion_demo import _CHART_NUM_RE, _page_number_set

def main():
    doc = fitz.open(str(PDF))
    conn = psycopg2.connect(os.environ["SUPABASE_DIRECT_DB_URL"]); cur = conn.cursor()
    cur.execute("""select page, content from document_evidence
                   where pdf_id='upload_44b76ed9' and source_type='image' order by page""")
    cards = cur.fetchall(); conn.close()

    res = {"pages": {}}
    for pg in sorted({p for p, _ in cards}):
        pn = _page_number_set(doc, pg)
        raw = doc[pg-1].get_text()
        res["pages"][pg] = {"textlayer_number_tokens": len(pn),
                            "sample": sorted(list(pn))[:12],
                            "has_axis_like": any(t in pn for t in ("10,000", "10000", "24.07"))}
        print(f"\n[p{pg}] 텍스트레이어 숫자 토큰 {len(pn)}개  샘플={sorted(list(pn))[:10]}")

    print("\n=== 카드별로 어떤 토큰이 왜 지워지는가 ===")
    detail = []
    for pg, content in cards:
        if "[OCR손상]" not in content:
            continue
        pn = _page_number_set(doc, pg)
        # 원본 OCR을 알 수 없으니 손상표시 주변 문맥만 본다
        ctx = [m for m in re.findall(r"\S*\[OCR손상\]\S*", content)][:6]
        print(f"  p{pg}: {ctx}")
        detail.append({"page": pg, "damaged_tokens": ctx})
        # 같은 카드의 살아남은 숫자들이 텍스트레이어에 있는지 교차확인
        alive = _CHART_NUM_RE.findall(content)[:10]
        miss = [a for a in alive if a not in pn]
        print(f"      살아있는 숫자 중 텍스트레이어에 없는 것: {miss[:8]}")
    res["damaged"] = detail

    print("\n=== H1 검증: 쉼표 토큰화 불일치 ===")
    for probe in ["35,000", "24.07", "24,07"]:
        toks = _CHART_NUM_RE.findall(probe)
        print(f"  '{probe}' -> _CHART_NUM_RE 토큰 {toks}")
    p3 = _page_number_set(doc, 3)
    print(f"  p3 텍스트레이어에 '000' 있나? {'000' in p3} / '35' {'35' in p3} / '07' {'07' in p3}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    doc.close(); print(f"\n-> {OUT.name} 저장")

if __name__ == "__main__":
    main()

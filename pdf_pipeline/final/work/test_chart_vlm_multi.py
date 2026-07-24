# -*- coding: utf-8 -*-
"""[재일] 차트 VLM(4a) 다중 유형 정확도 — "VLM이 차트의 추세/흐름을 텍스트로 파악할 정도가 되나?"

차트 1장(스텝차트)만으로는 답할 수 없어 유형을 나눠 실측한다. 4b(narrative, OpenAI)는 빼고
4a(chart_table_extract: 차트 이미지 -> 표 복원)만 돌린다 = OpenAI 토큰 0.

유형:
  A. 스텝차트(값 라벨 없음, 눈금선에 정렬)  — c밴드 목표주가 추이 4종
  B. 값 라벨 막대차트                       — Construct 도표3(건설업종 주간수익률) / 도표4(건자재)
     * 도표3에는 겹친 텍스트로 OCR이 깨졌던 금호건설 -35.2가 들어있다 -> VLM이 이걸 바로 읽으면
       기존 OCR 손상 문제까지 함께 해결되는지 확인 가능.
정답은 PDF 원문을 직접 읽어 아래 GOLD에 하드코딩했다."""
import sys, time, json, re
from pathlib import Path
ROOT = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP = ROOT / "pdf_pipeline"
for d in [str(PP), str(PP/"image_processing")]: sys.path.insert(0, d)
SCRATCH = Path("C:/Users/wodlf/AppData/Local/Temp/claude/c--Users-wodlf-OneDrive-Desktop----------/16e3e5ba-1dad-48c5-939c-215a4d9f6a4e/scratchpad")
OUT = PP/"final"/"results_chart_vlm_multi.json"
CBAND = PP/"reference"/"C밴드"/"c밴드.pdf"
CONSTRUCT = PP/"reference"/"Construct"/"20260721_industry_362851000.pdf"

import fitz

# (이름, pdf, 페이지, 앵커텍스트(이 텍스트 아래를 자름), 폭(left/right/full), 유형, 정답값들)
CASES = [
    ("c밴드 우리넷 스텝차트", CBAND, 4, "우리넷", "left", "A",
     ["10,000", "15,000", "25,000"]),
    ("c밴드 아이씨티케이 스텝차트", CBAND, 4, "아이씨티케이", "left", "A",
     ["28,000", "40,000", "60,000"]),
    ("Construct 도표3 건설업종 주간수익률", CONSTRUCT, 2, "도표 3", "left", "B",
     ["7.4", "7.6", "14.3", "13.2", "4.9", "1.1", "1.3", "11.6", "10.1", "5.9", "35.2", "2.7"]),
    ("Construct 도표4 건자재 주간수익률", CONSTRUCT, 2, "도표 4", "right", "B",
     ["2.5", "8.1", "1.7", "0.6", "5.4", "3.0", "8.9"]),
]


def crop(pdf, page_no, anchor, side, out_png, dpi=200, height=200):
    doc = fitz.open(str(pdf)); page = doc[page_no - 1]
    hits = page.search_for(anchor)
    if not hits:
        doc.close(); return None
    r = hits[0]
    W, H = page.rect.width, page.rect.height
    x0, x1 = (0, W/2) if side == "left" else ((W/2, W) if side == "right" else (0, W))
    clip = fitz.Rect(max(0, x0), r.y0, min(W, x1), min(H, r.y0 + height))
    page.get_pixmap(clip=clip, matrix=fitz.Matrix(dpi/72, dpi/72)).save(str(out_png))
    doc.close(); return out_png


def main():
    import s2_onestop_mineru as s2
    res = []
    for name, pdf, pg, anchor, side, kind, gold in CASES:
        png = crop(pdf, pg, anchor, side, SCRATCH / (re.sub(r"\W+", "_", name) + ".png"))
        if png is None:
            print(f"\n[{name}] 앵커 '{anchor}' 못 찾음 — 스킵"); continue
        t = time.time()
        table, _ = s2.chart_table_extract(png, "chart", s2.CHART_MAX_NEW_TOKENS)
        dt = time.time() - t
        got = set(re.findall(r"\d[\d,]*\.?\d*", table or ""))
        hit = [g for g in gold if g in got or g.replace(",", "") in {x.replace(",", "") for x in got}]
        print(f"\n{'='*72}\n[{name}]  유형{kind}  {dt:.1f}s   정답값 회수 {len(hit)}/{len(gold)}")
        print(f"미회수: {[g for g in gold if g not in hit]}")
        print((table or "!! VLM이 표를 반환하지 않음")[:900])
        res.append({"name": name, "kind": kind, "latency_s": round(dt, 1),
                    "recovered": f"{len(hit)}/{len(gold)}", "missed": [g for g in gold if g not in hit],
                    "table": table})
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT.name} 저장")


if __name__ == "__main__":
    main()

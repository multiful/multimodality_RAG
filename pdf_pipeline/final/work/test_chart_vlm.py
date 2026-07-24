# -*- coding: utf-8 -*-
"""[재일] 차트 VLM(4a) 실효성 테스트 — "차트 OCR이 축 눈금만 담아 시계열이 조작된다"는 사례를
MinerU2.5-Pro VLM이 실제로 고치는지 확인한다. 4b(narrative)는 OpenAI를 쓰므로 호출하지 않고
4a(chart_table_extract: 차트 이미지 -> 표 복원)만 단독 실행한다 = OpenAI 토큰 0.

대상: c밴드 p4 '케이엠더블유 수정TP' 스텝차트. 정답(같은 페이지 표):
  24.3.20=25,000 / 24.10.16=20,000 / 24.12.18=15,000 / 25.7.3=25,000 / 26.2.3=35,000 /
  26.3.17=50,000 / 26.5.19=70,000  (= 상승 추세)
현재 OCR이 준 것: Y축 눈금 9개(0~80,000) + X축 날짜 9개(24.07~26.07) 뿐."""
import sys, time, json
from pathlib import Path
ROOT = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP = ROOT / "pdf_pipeline"
for d in [str(PP), str(PP/"image_processing")]: sys.path.insert(0, d)
PDF = PP/"reference"/"C밴드"/"c밴드.pdf"
SCRATCH = Path("C:/Users/wodlf/AppData/Local/Temp/claude/c--Users-wodlf-OneDrive-Desktop----------/16e3e5ba-1dad-48c5-939c-215a4d9f6a4e/scratchpad")
OUT = PP/"final"/"results_chart_vlm_test.json"

import fitz

def crop_chart(page_no: int, rect_pt, out_png: Path, dpi=200):
    doc = fitz.open(str(PDF))
    page = doc[page_no - 1]
    pix = page.get_pixmap(clip=fitz.Rect(*rect_pt), matrix=fitz.Matrix(dpi/72, dpi/72))
    pix.save(str(out_png)); doc.close()
    return out_png

def main():
    # p4 케이엠더블유 차트: 표(x 305~546, y 121~216) 왼쪽 같은 높이 영역이 차트
    png = crop_chart(4, (30, 100, 300, 230), SCRATCH/"kmw_chart.png")
    print(f"[crop] {png} ({png.stat().st_size} bytes)")
    import s2_onestop_mineru as s2
    t = time.time()
    table, secs = s2.chart_table_extract(png, "chart", s2.CHART_MAX_NEW_TOKENS)
    dt = time.time() - t
    print(f"\n[VLM] {dt:.1f}s (내부보고 {secs}s)")
    print("=" * 70)
    print(table if table else "!! VLM이 표를 반환하지 않음(None)")
    print("=" * 70)
    OUT.write_text(json.dumps({"latency_s": round(dt, 1), "chart_table": table,
                                "gold": "24.3.20=25,000 / 24.10.16=20,000 / 24.12.18=15,000 / "
                                        "25.7.3=25,000 / 26.2.3=35,000 / 26.3.17=50,000 / 26.5.19=70,000"},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT.name} 저장")

if __name__ == "__main__":
    main()

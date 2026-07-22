"""테이블 고도화의 베이스라인: 표 위치는 이미 안다고 가정(YOLO 분류 결과 재사용)하고,
그 표 영역에 pdfplumber `extract_table()`만 적용했을 때 얼마나 잘 뽑아내는지 측정.

- YOLOv11n으로 표 bbox 재검출(픽셀 좌표) → 150dpi 픽셀좌표를 PDF pt 좌표로 환산
  → pdfplumber page.crop(bbox_pt).extract_table()
- Docling(TableFormer) 결과(result_docling_parallel.json, 240행)와 "동일 표 영역" 기준으로 직접 비교
- 페이지/객체 분류(어디가 표인지) 자체는 이미 잘 된다고 가정하고 다루지 않음 — 순수하게
  "표 내용을 얼마나 잘 추출하는가"만 봄
"""

import json
from pathlib import Path

import fitz
import pdfplumber
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "20260721_company_279243000.pdf"
OUT_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"
RESULT_PATH = OUT_DIR / "result_table_baseline_pdfplumber.json"

CONF_THRESHOLD = 0.25
RENDER_DPI = 150
SCALE = RENDER_DPI / 72  # 150dpi 픽셀 -> pt 환산


def main():
    model = YOLO(str(YOLO_MODEL_PATH))
    doc_fitz = fitz.open(str(PDF_PATH))

    per_table = []
    with pdfplumber.open(str(PDF_PATH)) as pdf:
        for i, (page_pp, page_fz) in enumerate(zip(pdf.pages, doc_fitz), start=1):
            pix = page_fz.get_pixmap(dpi=RENDER_DPI)
            tmp_path = OUT_DIR / f"_tmp_base_p{i}.png"
            pix.save(str(tmp_path))
            img = Image.open(tmp_path).convert("RGB")

            results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
            names = model.names
            boxes = results.boxes
            tmp_path.unlink(missing_ok=True)
            if boxes is None:
                continue

            t_idx = 0
            for cls_idx, xyxy in zip(boxes.cls, boxes.xyxy):
                if names[int(cls_idx)] != "Table":
                    continue
                t_idx += 1
                x1, y1, x2, y2 = [float(v) for v in xyxy.tolist()]
                bbox_pt = (x1 / SCALE, y1 / SCALE, x2 / SCALE, y2 / SCALE)
                try:
                    cropped_page = page_pp.crop(bbox_pt)
                    table = cropped_page.extract_table()
                    n_rows = len(table) if table else 0
                except Exception as e:
                    n_rows = 0
                per_table.append({"page": i, "table_idx": t_idx, "n_rows": n_rows,
                                   "height_px": round(y2 - y1, 1)})
                print(f"page{i} table{t_idx}: pdfplumber {n_rows}행 (bbox 높이 {y2-y1:.0f}px)", flush=True)
    doc_fitz.close()

    total_rows = sum(t["n_rows"] for t in per_table)
    zero_row = sum(1 for t in per_table if t["n_rows"] == 0)

    docling_path = OUT_DIR / "result_docling_parallel.json"
    docling_rows = None
    if docling_path.exists():
        docling_rows = json.loads(docling_path.read_text(encoding="utf-8"))["total_rows_extracted"]

    result = {
        "method": "pdfplumber extract_table() on YOLO-confirmed Table bbox (표 위치는 주어진 것으로 가정)",
        "n_tables": len(per_table),
        "total_rows_extracted": total_rows,
        "zero_row_count": zero_row,
        "per_table": per_table,
        "comparison": {"pdfplumber_baseline_rows": total_rows, "docling_tableformer_rows": docling_rows},
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n총 {len(per_table)}개 표, pdfplumber 총 {total_rows}행 추출, 0행 표 {zero_row}개")
    if docling_rows:
        print(f"(참고) Docling/TableFormer 동일 표 기준: {docling_rows}행")
    print(f"[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

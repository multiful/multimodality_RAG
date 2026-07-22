"""Docling Framework 단독(1-Stage, PDF 전체를 그대로 입력) 벤치마크 — 페이지별로 측정.

YOLO crop 없이 원본 PDF를 페이지 단위로 Docling에 바로 넣어 레이아웃+표 구조를 한 번에 추론.
PDF 자체의 텍스트 레이어를 활용해서(래스터 크롭과 달리 OCR 불필요) 상대적으로 빠를 것으로 예상.
"""

import json
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "20260721_company_279243000.pdf"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / "result_docling_standalone.json"
GROUND_TRUTH_PATH = OUT_DIR / "ground_truth_pages.json"


def main():
    gt = {p["page"]: p for p in json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))["pages"]}
    converter = DocumentConverter()

    per_page = []
    for i in range(1, 7):
        t0 = time.perf_counter()
        res = converter.convert(str(PDF_PATH), page_range=(i, i))
        elapsed = round(time.perf_counter() - t0, 3)
        doc = res.document
        n_tables = len(doc.tables)
        n_rows = sum(t.export_to_dataframe(doc).shape[0] for t in doc.tables)
        n_pictures = len(doc.pictures)
        has_table = n_tables > 0
        has_image = n_pictures > 0
        correct = (has_table == gt[i]["has_table"]) and (has_image == gt[i]["has_image"])
        per_page.append({
            "page": i, "elapsed_s": elapsed, "n_tables": n_tables, "n_rows": n_rows,
            "n_pictures": n_pictures, "has_table": has_table, "has_image": has_image,
            "matches_ground_truth": correct,
        })
        print(f"page {i}: {elapsed}s, tables={n_tables}(rows={n_rows}), pictures={n_pictures}, "
              f"gt_match={correct}", flush=True)

    total_s = round(sum(p["elapsed_s"] for p in per_page), 3)
    total_rows = sum(p["n_rows"] for p in per_page)
    matches = sum(1 for p in per_page if p["matches_ground_truth"])

    result = {
        "method": "Docling Framework standalone (1-Stage, whole PDF page fed directly, no YOLO crop)",
        "per_page": per_page,
        "total_time_s_6pages": total_s,
        "avg_time_s_per_page": round(total_s / 6, 3),
        "total_rows_extracted": total_rows,
        "page_classification_match": f"{matches}/6",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n총 {total_s}s (평균 {total_s/6:.3f}s/page), 총 행수 {total_rows}, 분류 일치 {matches}/6")
    print(f"[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

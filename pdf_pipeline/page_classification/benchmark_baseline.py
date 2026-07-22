"""페이지 분류 baseline("PyMuPDF Fast Scan") 성능/지연 벤치마크.

- 분류 로직: pdfplumber find_tables() + PyMuPDF 래스터 이미지 수 / 벡터 드로잉 수
  (run_baseline.py의 classify_page()와 동일 — LLM 미사용, 메타데이터만 봄)
- ground_truth_pages.json과 비교해 라벨별(has_text/has_table/has_image) Accuracy/Precision/Recall/F1 계산
- 페이지당 분류 소요시간(ms) 측정
"""

import json
import time
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "20260721_company_279243000.pdf"
OUT_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = OUT_DIR / "ground_truth_pages.json"
RESULT_PATH = OUT_DIR / "result_baseline_pymupdf_fastscan.json"

VECTOR_DRAWING_THRESHOLD = 40


def classify_page(page_pdfplumber, page_fitz) -> dict:
    tables = page_pdfplumber.find_tables()
    has_table = len(tables) > 0
    raster_images = page_fitz.get_images()
    drawings = page_fitz.get_drawings()
    has_image = len(raster_images) > 0 or len(drawings) > VECTOR_DRAWING_THRESHOLD
    text = page_pdfplumber.extract_text() or ""
    has_text = len(text.strip()) > 20
    return {
        "has_text": has_text, "has_table": has_table, "has_image": has_image,
        "n_tables": len(tables), "n_raster_images": len(raster_images), "n_drawings": len(drawings),
    }


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main():
    gt = {p["page"]: p for p in json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))["pages"]}

    doc_fitz = fitz.open(str(PDF_PATH))
    predictions, timings = {}, {}

    with pdfplumber.open(str(PDF_PATH)) as pdf:
        for i, (page_pp, page_fz) in enumerate(zip(pdf.pages, doc_fitz), start=1):
            t0 = time.perf_counter()
            pred = classify_page(page_pp, page_fz)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            predictions[i] = pred
            timings[i] = elapsed_ms
            print(f"page {i}: pred={pred}  ({elapsed_ms}ms)")
    doc_fitz.close()

    labels = ["has_text", "has_table", "has_image"]
    per_label = {}
    correct_total, n_total = 0, 0
    exact_match_pages = 0

    for label in labels:
        tp = fp = fn = tn = 0
        for pg, g in gt.items():
            p = predictions[pg][label]
            g_val = g[label]
            n_total += 1
            if p == g_val:
                correct_total += 1
            if p and g_val:
                tp += 1
            elif p and not g_val:
                fp += 1
            elif not p and g_val:
                fn += 1
            else:
                tn += 1
        precision, recall, f1 = prf(tp, fp, fn)
        per_label[label] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round((tp + tn) / len(gt), 4),
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        }

    for pg, g in gt.items():
        if all(predictions[pg][l] == g[l] for l in labels):
            exact_match_pages += 1

    overall_accuracy = correct_total / n_total
    avg_latency_ms = round(sum(timings.values()) / len(timings), 3)
    total_latency_ms = round(sum(timings.values()), 3)

    result = {
        "method": "PyMuPDF Fast Scan (pdfplumber find_tables + PyMuPDF image/drawing count) — baseline",
        "per_label": per_label,
        "overall_label_accuracy": round(overall_accuracy, 4),
        "exact_match_page_accuracy": round(exact_match_pages / len(gt), 4),
        "avg_latency_ms_per_page": avg_latency_ms,
        "total_latency_ms_6pages": total_latency_ms,
        "predictions": predictions,
        "per_page_latency_ms": timings,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 라벨별 성능 ===")
    for label, r in per_label.items():
        print(f"{label}: acc={r['accuracy']:.1%} precision={r['precision']:.1%} recall={r['recall']:.1%} f1={r['f1']:.1%}")
    print(f"\n전체 라벨 정확도(3라벨x6페이지=18개 중): {overall_accuracy:.1%}")
    print(f"페이지 완전일치율(3라벨 다 맞은 페이지): {exact_match_pages}/{len(gt)}")
    print(f"평균 지연: {avg_latency_ms}ms/page, 총 {total_latency_ms}ms(6p)")
    print(f"[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

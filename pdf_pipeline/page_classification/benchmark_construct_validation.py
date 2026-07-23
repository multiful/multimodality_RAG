"""[5] 두 번째 다른 섹터 PDF(Construct, 건설 Weekly, 하나증권, 10페이지)로 일반화 검증 — baseline
vs YOLOv11n 페이지 분류. K-Wave([4])는 엔터/음식료/미디어 섹터였고, 이번엔 건설업 섹터 — 10페이지라
샘플링 없이 전 페이지 육안 전수 검수로 정답 작성(ground_truth_construct_pages.json).
"""

import json
import time
from pathlib import Path

import fitz
import pdfplumber
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "Construct" / "20260721_industry_362851000.pdf"
OUT_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = OUT_DIR / "models" / "yolo11n_doc_layout.pt"
RESULT_PATH = OUT_DIR / "result_construct_page_classification.json"
CROP_DIR = OUT_DIR / "table_crops_construct"

CONF_THRESHOLD = 0.25
RENDER_DPI = 150


def run_baseline(doc_fitz, pdf_pp):
    predictions = {}
    for i, (page_pp, page_fz) in enumerate(zip(pdf_pp.pages, doc_fitz), start=1):
        text = page_pp.extract_text() or ""
        has_text = len(text.strip()) > 20
        try:
            tables = page_pp.find_tables()
            has_table = len(tables) > 0
        except Exception:
            has_table = False
        pix = page_fz.get_pixmap(dpi=RENDER_DPI)
        n_images = len(page_fz.get_images())
        drawings = page_fz.get_drawings()
        has_image = n_images > 0 or len(drawings) > 30
        predictions[i] = {"has_text": has_text, "has_table": has_table, "has_image": has_image}
    return predictions


def run_yolo(doc_fitz):
    CROP_DIR.mkdir(exist_ok=True)
    model = YOLO(str(YOLO_MODEL_PATH))
    warmup_img = Image.new("RGB", (595, 842), (255, 255, 255))
    model.predict(warmup_img, conf=CONF_THRESHOLD, verbose=False)

    predictions, detections_log, timings = {}, {}, {}
    for i, page_fz in enumerate(doc_fitz, start=1):
        t0 = time.perf_counter()
        pix = page_fz.get_pixmap(dpi=RENDER_DPI)
        tmp_path = OUT_DIR / f"_tmp_kwave_p{i}.png"
        pix.save(str(tmp_path))
        img = Image.open(tmp_path).convert("RGB")
        results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        names = model.names
        boxes = results.boxes
        classes_found = [names[int(c)] for c in boxes.cls] if boxes is not None else []
        has_table = "Table" in classes_found
        has_image = "Picture" in classes_found
        text_classes = {"Text", "Title", "Section-header", "Caption", "List-item",
                         "Page-header", "Page-footer", "Footnote", "Formula"}
        has_text = any(c in text_classes for c in classes_found)

        table_crops = []
        if boxes is not None:
            t_idx = 0
            for cls_idx, xyxy in zip(boxes.cls, boxes.xyxy):
                if names[int(cls_idx)] == "Table":
                    t_idx += 1
                    x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
                    crop = img.crop((x1, y1, x2, y2))
                    crop_path = CROP_DIR / f"page_{i}_table_{t_idx}.png"
                    crop.save(crop_path)
                    table_crops.append(str(crop_path.relative_to(ROOT)))
        tmp_path.unlink(missing_ok=True)

        predictions[i] = {"has_text": has_text, "has_table": has_table, "has_image": has_image}
        detections_log[i] = {"classes_found": classes_found, "table_crops": table_crops}
        timings[i] = elapsed_ms
        print(f"page {i}: {predictions[i]} classes={set(classes_found)} ({elapsed_ms}ms)", flush=True)
    return predictions, detections_log, timings


def score_against_gt(preds: dict, gt_pages: dict) -> dict:
    """페이지별 has_text/has_table/has_image 3개 필드를 정답과 비교해 전체 정확도(accuracy) 계산.
    10페이지 x 3필드 = 30개 판정 중 몇 개가 맞았는지."""
    n_correct, n_total = 0, 0
    mismatches = []
    for page_str, gt in gt_pages.items():
        page = int(page_str)
        pred = preds.get(page, {})
        for field in ("has_text", "has_table", "has_image"):
            n_total += 1
            if pred.get(field) == gt[field]:
                n_correct += 1
            else:
                mismatches.append({"page": page, "field": field, "pred": pred.get(field), "gt": gt[field]})
    return {"accuracy": round(n_correct / n_total, 4) if n_total else 0.0,
            "n_correct": n_correct, "n_total": n_total, "mismatches": mismatches}


def main():
    gt = json.loads((OUT_DIR / "ground_truth_construct_pages.json").read_text(encoding="utf-8"))
    doc_fitz = fitz.open(str(PDF_PATH))
    pdf_pp = pdfplumber.open(str(PDF_PATH))

    print("=== Baseline(pdfplumber+PyMuPDF) 실행 ===", flush=True)
    t0 = time.perf_counter()
    baseline_preds = run_baseline(doc_fitz, pdf_pp)
    baseline_s = round(time.perf_counter() - t0, 2)
    print(f"baseline 소요: {baseline_s}s", flush=True)

    print("\n=== YOLOv11n 실행 ===", flush=True)
    t0 = time.perf_counter()
    yolo_preds, detections_log, timings = run_yolo(doc_fitz)
    yolo_s = round(time.perf_counter() - t0, 2)
    print(f"YOLOv11n 소요: {yolo_s}s", flush=True)

    pdf_pp.close()
    doc_fitz.close()

    n_table_crops = sum(len(d["table_crops"]) for d in detections_log.values())

    baseline_score = score_against_gt(baseline_preds, gt["pages"])
    yolo_score = score_against_gt(yolo_preds, gt["pages"])

    result = {
        "pdf": str(PDF_PATH.name), "n_pages": len(baseline_preds),
        "baseline_predictions": baseline_preds, "baseline_total_s": baseline_s,
        "baseline_score": baseline_score,
        "yolo_predictions": yolo_preds, "yolo_total_s": yolo_s,
        "yolo_score": yolo_score,
        "yolo_detections_log": detections_log, "yolo_per_page_ms": timings,
        "n_table_crops_found": n_table_crops,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n총 {len(baseline_preds)}페이지, YOLO가 찾은 표 크롭 {n_table_crops}개")
    print(f"\n=== 정답 대비 정확도(30개 판정 = 10페이지 x has_text/has_table/has_image) ===")
    print(f"Baseline: {baseline_score['n_correct']}/{baseline_score['n_total']} "
          f"({baseline_score['accuracy']*100:.1f}%)")
    for m in baseline_score["mismatches"]:
        print(f"  [오답] page{m['page']} {m['field']}: 예측={m['pred']} 정답={m['gt']}")
    print(f"YOLOv11n: {yolo_score['n_correct']}/{yolo_score['n_total']} "
          f"({yolo_score['accuracy']*100:.1f}%)")
    for m in yolo_score["mismatches"]:
        print(f"  [오답] page{m['page']} {m['field']}: 예측={m['pred']} 정답={m['gt']}")
    print(f"\n[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

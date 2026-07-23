"""[2] PyMuPDF Fast Scan + YOLOv11 Crop 벤치마크.

- has_text: pdfplumber(그대로, 이미 100% 정확) 유지
- has_table / has_image: YOLOv11(DocLayNet 학습, Armaggheddon/yolo11-document-layout, nano)로
  페이지 렌더링 이미지에서 레이아웃 감지 → 'Table'/'Picture' 클래스 존재 여부로 판정
- 감지된 Table bbox는 크롭해서 저장(다음 단계 Docling 파싱 입력용)
- ground_truth_pages.json 대비 Accuracy/Precision/Recall/F1 + 지연(ms) 측정
"""

import json
import time
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "20260721_company_279243000.pdf"
OUT_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = OUT_DIR / "ground_truth_pages.json"
RESULT_PATH = OUT_DIR / "result_yolo_crop.json"
YOLO_MODEL_PATH = OUT_DIR / "models" / "yolo11n_doc_layout.pt"
CROP_DIR = OUT_DIR / "table_crops"

CONF_THRESHOLD = 0.25
RENDER_DPI = 150


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main():
    CROP_DIR.mkdir(exist_ok=True)
    gt = {p["page"]: p for p in json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))["pages"]}

    t0 = time.perf_counter()
    model = YOLO(str(YOLO_MODEL_PATH))
    model_load_ms = round((time.perf_counter() - t0) * 1000, 2)
    print(f"[model] YOLOv11n-doc-layout loaded in {model_load_ms}ms", flush=True)

    # 콜드스타트 워밍업(첫 추론은 항상 느림) — 실제 서비스에서도 모델은 한 번만 로드/워밍업하고
    # 이후 요청들을 처리하므로, 지연 측정은 워밍업 이후 정상 상태 기준이 공정함
    warmup_img = Image.new("RGB", (595, 842), (255, 255, 255))
    t0 = time.perf_counter()
    model.predict(warmup_img, conf=CONF_THRESHOLD, verbose=False)
    print(f"[model] warmup inference: {(time.perf_counter()-t0)*1000:.2f}ms (측정 제외)", flush=True)

    doc_fitz = fitz.open(str(PDF_PATH))
    predictions, timings, detections_log = {}, {}, {}

    with pdfplumber.open(str(PDF_PATH)) as pdf:
        for i, (page_pp, page_fz) in enumerate(zip(pdf.pages, doc_fitz), start=1):
            t0 = time.perf_counter()
            text = page_pp.extract_text() or ""
            has_text = len(text.strip()) > 20
            fast_scan_ms = (time.perf_counter() - t0) * 1000

            pix = page_fz.get_pixmap(dpi=RENDER_DPI)
            img_path = OUT_DIR / f"_tmp_page_{i}.png"
            pix.save(str(img_path))
            img = Image.open(img_path).convert("RGB")

            t0 = time.perf_counter()
            results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
            yolo_ms = (time.perf_counter() - t0) * 1000

            names = model.names
            boxes = results.boxes
            classes_found = [names[int(c)] for c in boxes.cls] if boxes is not None else []
            has_table = "Table" in classes_found
            has_image = "Picture" in classes_found

            # Table 크롭 저장 (다음 단계 Docling 입력용)
            table_crops = []
            if boxes is not None:
                for j, (cls_idx, xyxy) in enumerate(zip(boxes.cls, boxes.xyxy)):
                    if names[int(cls_idx)] == "Table":
                        x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
                        crop = img.crop((x1, y1, x2, y2))
                        crop_path = CROP_DIR / f"page_{i}_table_{j}.png"
                        crop.save(crop_path)
                        table_crops.append(str(crop_path.relative_to(ROOT)))
            img_path.unlink(missing_ok=True)

            elapsed_ms = round(fast_scan_ms + yolo_ms, 2)
            pred = {"has_text": has_text, "has_table": has_table, "has_image": has_image}
            predictions[i] = pred
            timings[i] = elapsed_ms
            detections_log[i] = {"classes_found": classes_found, "table_crops": table_crops}
            print(f"page {i}: pred={pred} classes={classes_found} "
                  f"(fast_scan {fast_scan_ms:.2f}ms + yolo {yolo_ms:.2f}ms = {elapsed_ms}ms)", flush=True)
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
        "method": "PyMuPDF Fast Scan + YOLOv11 Crop (Armaggheddon/yolo11-document-layout, nano, DocLayNet)",
        "yolo_model_load_ms": model_load_ms,
        "per_label": per_label,
        "overall_label_accuracy": round(overall_accuracy, 4),
        "exact_match_page_accuracy": round(exact_match_pages / len(gt), 4),
        "avg_latency_ms_per_page": avg_latency_ms,
        "total_latency_ms_6pages": total_latency_ms,
        "predictions": predictions,
        "per_page_latency_ms": timings,
        "detections": detections_log,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 라벨별 성능 ===")
    for label, r in per_label.items():
        print(f"{label}: acc={r['accuracy']:.1%} precision={r['precision']:.1%} recall={r['recall']:.1%} f1={r['f1']:.1%}")
    print(f"\n전체 라벨 정확도: {overall_accuracy:.1%}")
    print(f"페이지 완전일치율: {exact_match_pages}/{len(gt)}")
    print(f"평균 지연: {avg_latency_ms}ms/page (YOLO 모델 로딩 {model_load_ms}ms 별도)")
    print(f"[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

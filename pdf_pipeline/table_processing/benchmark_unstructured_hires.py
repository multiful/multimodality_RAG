"""Unstructured(Hi-Res) 벤치마크 — poppler/tesseract 없이 실행되도록 우회한 버전.

- poppler 우회: partition_pdf() 대신 partition_image()에 PyMuPDF로 미리 렌더링한 페이지 PNG를 입력
- tesseract 우회: OCR_AGENT 환경변수가 이 버전에서 실제로 반영되지 않아(코드 확인 결과 pdf.py가
  env_config.OCR_AGENT를 참조하지 않음), ocr_agent/table_ocr_agent 파라미터를 직접 명시해서
  PaddleOCR(순수 파이썬, Homebrew 불필요)로 강제 지정
- 첫 호출은 PaddleOCR/레이아웃/표구조 모델 다운로드가 겹쳐 매우 느리므로 워밍업 분리 측정
"""

import json
import time
from collections import Counter
from pathlib import Path

from unstructured.partition.image import partition_image
from unstructured.partition.utils.constants import OCR_AGENT_PADDLE

ROOT = Path(__file__).resolve().parent.parent.parent
RENDERED_DIR = ROOT / "pdf_pipeline" / "rendered_pages"
OUT_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = OUT_DIR / "ground_truth_pages.json"
RESULT_PATH = OUT_DIR / "result_unstructured_hires.json"


def classify_from_elements(els) -> dict:
    cats = Counter(e.category for e in els)
    has_table = cats.get("Table", 0) > 0
    has_image = (cats.get("Image", 0) + cats.get("Figure", 0)) > 0
    has_text = (cats.get("NarrativeText", 0) + cats.get("Title", 0)
                + cats.get("UncategorizedText", 0) + cats.get("ListItem", 0)) > 0
    return {"has_text": has_text, "has_table": has_table, "has_image": has_image, "categories": dict(cats)}


def main():
    gt = {p["page"]: p for p in json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))["pages"]}

    # 워밍업(모델 다운로드+로딩, 1회) — page 1로 태우고 측정에서 제외
    print("워밍업(모델 다운로드/로딩) 시작...", flush=True)
    t0 = time.perf_counter()
    warmup_els = partition_image(
        str(RENDERED_DIR / "page_1.png"), strategy="hi_res", infer_table_structure=True,
        ocr_agent=OCR_AGENT_PADDLE, table_ocr_agent=OCR_AGENT_PADDLE,
    )
    warmup_s = round(time.perf_counter() - t0, 3)
    print(f"[timing] 워밍업(모델 로딩 포함): {warmup_s}s", flush=True)

    per_page = []
    # page 1은 워밍업 결과 재사용
    pred = classify_from_elements(warmup_els)
    per_page.append({"page": 1, "elapsed_s": warmup_s, **pred})
    print(f"page 1(워밍업 겸용): {pred}", flush=True)

    for i in range(2, 7):
        t0 = time.perf_counter()
        els = partition_image(
            str(RENDERED_DIR / f"page_{i}.png"), strategy="hi_res", infer_table_structure=True,
            ocr_agent=OCR_AGENT_PADDLE, table_ocr_agent=OCR_AGENT_PADDLE,
        )
        elapsed = round(time.perf_counter() - t0, 3)
        pred = classify_from_elements(els)
        per_page.append({"page": i, "elapsed_s": elapsed, **pred})
        print(f"page {i}: {elapsed}s, {pred}", flush=True)

    labels = ["has_text", "has_table", "has_image"]
    per_label = {}
    correct_total, n_total = 0, 0
    exact_match_pages = 0
    for label in labels:
        tp = fp = fn = tn = 0
        for p in per_page:
            pg = p["page"]
            pred_val, gt_val = p[label], gt[pg][label]
            n_total += 1
            if pred_val == gt_val:
                correct_total += 1
            if pred_val and gt_val:
                tp += 1
            elif pred_val and not gt_val:
                fp += 1
            elif not pred_val and gt_val:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             "accuracy": round((tp + tn) / len(gt), 4),
                             "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    for p in per_page:
        if all(p[l] == gt[p["page"]][l] for l in labels):
            exact_match_pages += 1

    steady_state = per_page[1:]  # page1(워밍업) 제외
    avg_steady_s = round(sum(p["elapsed_s"] for p in steady_state) / len(steady_state), 3) if steady_state else 0.0

    result = {
        "method": "Unstructured Hi-Res (partition_image + PaddleOCR, poppler/tesseract 우회)",
        "warmup_s_incl_model_download": warmup_s,
        "avg_steady_state_s_per_page": avg_steady_s,
        "per_label": per_label,
        "overall_label_accuracy": round(correct_total / n_total, 4),
        "exact_match_page_accuracy": round(exact_match_pages / len(gt), 4),
        "per_page": per_page,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 라벨별 성능 ===")
    for label, r in per_label.items():
        print(f"{label}: acc={r['accuracy']:.1%} precision={r['precision']:.1%} recall={r['recall']:.1%} f1={r['f1']:.1%}")
    print(f"전체 라벨 정확도: {correct_total/n_total:.1%}, 페이지 완전일치: {exact_match_pages}/6")
    print(f"워밍업(모델로딩+다운로드): {warmup_s}s, steady-state 평균: {avg_steady_s}s/page")
    print(f"[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

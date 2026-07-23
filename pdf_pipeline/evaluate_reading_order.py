"""[30] 리딩오더 판별 로직 재검토 — 사용자 질문: "하드 페이지로 분류된 곳은 정말 리딩오더를
복원해야하는 페이지인지? 이 룰베이스 로직이 적절한지? 리딩오더 판별기 모델을 쓸 수 없는지도
검토해서 있다면 적용해서 비교해줘." K-Wave PDF(73페이지) 대상.

방법론:
1. 우리 rule-based 판정(reading_order_router.assess_page_difficulty, columns/overlap 가중 신호 +
   material_overlaps override)을 K-Wave 전체 페이지에 재실행.
2. "리딩오더 판별 전용 모델" 조사 결과 채택: LayoutReader(Microsoft, LayoutLMv3 기반,
   ReadingBank 50만 문서로 사전학습, MinerU 등이 실제로 쓰는 바로 그 모델) —
   HuggingFace `hantian/layoutreader`(CC-BY-NC-SA-4.0)에 실제로 존재해 다운로드해 적용.
   (참고로 Docling도 검토했으나, 자체 GitHub 이슈(#1203, #2067, #3198, Discussion #2791)에서
   다중 컬럼 리딩오더 복원이 아직 불안정하다고 공식적으로 인정하고 있어 비교 대상에서 제외.)
3. LayoutReader의 예측 순서와 "원본 추출 순서"(raw PyMuPDF `get_text("blocks")` 순서, 우리
   whole_page 베이스라인이 실제로 쓰는 순서와 동일) 사이의 정규화된 역전 쌍 비율(normalized
   Kendall-tau distance)을 계산 — 이 값이 임계치를 넘으면 "리딩오더 복원이 실제로 필요했던
   페이지"로 GT 라벨링(LayoutReader 자체가 50만 문서로 학습된 전용 모델이라 이 판단을 GT 삼음).
4. 우리 rule-based hard/easy 분류를 이 GT 대비 정밀도/재현율/F1으로 채점.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import fitz
import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pdf_pipeline"))
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "page_classification"))
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "text_processing"))


def _load_as(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_text_tn = _load_as("_text_processing_text_normalization",
                     ROOT / "pdf_pipeline" / "text_processing" / "text_normalization.py")
sys.modules["text_normalization"] = _text_tn

from page_classifier import classify_pdf  # noqa: E402
from reading_order_router import assess_page_difficulty, NON_TEXT_CLASSES, _is_excluded  # noqa: E402

YOLO_MODEL_PATH = ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "KWave" / "20260721_industry_65157000.pdf"
OUT_PATH = ROOT / "pdf_pipeline" / "result_reading_order_evaluation.json"

DIVERGENCE_THRESHOLD = 0.15  # 정규화된 역전쌍 비율이 이 이상이면 "실제로 리딩오더 복원 필요"로 GT 라벨링

# ---------- LayoutReader 추론 유틸(공식 참조 구현 그대로, ppaanngggg/layoutreader) ----------
MAX_LEN = 510
CLS_TOKEN_ID, UNK_TOKEN_ID, EOS_TOKEN_ID = 0, 3, 2


def boxes2inputs(boxes):
    bbox = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]
    input_ids = [CLS_TOKEN_ID] + [UNK_TOKEN_ID] * len(boxes) + [EOS_TOKEN_ID]
    attention_mask = [1] * len(bbox)
    return {"bbox": torch.tensor([bbox]), "attention_mask": torch.tensor([attention_mask]),
            "input_ids": torch.tensor([input_ids])}


def parse_logits(logits, length):
    from collections import defaultdict
    logits = logits[1:length + 1, :length]
    orders = logits.argsort(descending=False).tolist()
    ret = [o.pop() for o in orders]
    while True:
        order_to_idxes = defaultdict(list)
        for idx, order in enumerate(ret):
            order_to_idxes[order].append(idx)
        order_to_idxes = {k: v for k, v in order_to_idxes.items() if len(v) > 1}
        if not order_to_idxes:
            break
        for order, idxes in order_to_idxes.items():
            idxes_to_logit = {idx: logits[idx, order] for idx in idxes}
            idxes_to_logit = sorted(idxes_to_logit.items(), key=lambda x: x[1], reverse=True)
            for idx, _ in idxes_to_logit[1:]:
                ret[idx] = orders[idx].pop()
    return ret


def predict_reading_order(model, blocks_bbox_pt, page_width, page_height):
    """blocks_bbox_pt: [(x0,y0,x1,y1), ...] in PDF point units. 0~1000 스케일로 정규화 후
    LayoutReader 추론 -> ret[i] = block i에 배정된 예측 순번(0-indexed)."""
    if len(blocks_bbox_pt) > MAX_LEN:
        blocks_bbox_pt = blocks_bbox_pt[:MAX_LEN]  # 이 문서 규모에선 발생 안 하지만 방어적으로 처리
    x_scale, y_scale = 1000.0 / page_width, 1000.0 / page_height
    boxes = []
    for x0, y0, x1, y1 in blocks_bbox_pt:
        boxes.append([
            max(0, min(1000, round(x0 * x_scale))), max(0, min(1000, round(y0 * y_scale))),
            max(0, min(1000, round(x1 * x_scale))), max(0, min(1000, round(y1 * y_scale))),
        ])
    inputs = boxes2inputs(boxes)
    with torch.no_grad():
        logits = model(**inputs).logits.cpu().squeeze(0)
    return parse_logits(logits, len(boxes))


def normalized_kendall_tau(ret: list) -> float:
    """ret(예측 순번 배열)과 항등순열(원본 추출 순서) 사이의 정규화된 역전쌍 비율.
    ret가 항등순열이면(모델이 원본 순서가 이미 맞다고 판단) 0.0, 완전히 뒤집혔으면 1.0에 근접."""
    n = len(ret)
    if n < 2:
        return 0.0
    inversions = sum(1 for i in range(n) for j in range(i + 1, n) if ret[i] > ret[j])
    max_inversions = n * (n - 1) / 2
    return inversions / max_inversions if max_inversions else 0.0


def get_filtered_text_blocks(page: fitz.Page, cached_boxes: list) -> list:
    """reading_order_router.assess_page_difficulty()와 완전히 동일한 필터링(공정 비교를 위해
    그대로 재사용) — [30]에서 고친 면적기반 `_is_excluded()`로 Table/Picture 겹침 블록 제외."""
    exclude_rects = [rect for cls_name, rect in cached_boxes if cls_name in NON_TEXT_CLASSES]
    text_blocks = []
    for b in page.get_text("blocks"):
        if b[6] != 0 or not b[4].strip():
            continue
        bbox = fitz.Rect(b[:4])
        if _is_excluded(bbox, exclude_rects):
            continue
        text_blocks.append(b)
    return text_blocks


def main():
    print("YOLO 모델 로딩 중...", flush=True)
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    warmup = Image.new("RGB", (595, 842), (255, 255, 255))
    yolo_model.predict(warmup, conf=0.25, verbose=False)

    print("LayoutReader(hantian/layoutreader, LayoutLMv3 기반) 로딩 중...", flush=True)
    lr_model = LayoutLMv3ForTokenClassification.from_pretrained("hantian/layoutreader")
    lr_model.eval()

    print("page_classification 중...", flush=True)
    cls_result = classify_pdf(PDF_PATH, yolo_model)
    page_boxes = {p["page"]: p["cached_boxes"] for p in cls_result["pages"]}

    doc_fitz = fitz.open(str(PDF_PATH))
    per_page = []
    t0 = time.perf_counter()
    for i in range(doc_fitz.page_count):
        page = doc_fitz[i]
        cached_boxes = page_boxes.get(i + 1, [])

        # 1) 우리 rule-based 판정(현재 프로덕션 로직 그대로)
        rule_result = assess_page_difficulty(yolo_model, doc_fitz, i, cached_boxes=cached_boxes)

        # 2) LayoutReader 기반 GT
        text_blocks = get_filtered_text_blocks(page, cached_boxes)
        if len(text_blocks) < 2:
            divergence, gt_needed_reorder = 0.0, False
        else:
            bbox_pt = [tuple(b[:4]) for b in text_blocks]
            ret = predict_reading_order(lr_model, bbox_pt, page.rect.width, page.rect.height)
            divergence = normalized_kendall_tau(ret)
            gt_needed_reorder = divergence >= DIVERGENCE_THRESHOLD

        per_page.append({
            "page": i + 1, "n_text_blocks": len(text_blocks),
            "rule_based_difficulty": rule_result.difficulty,
            "rule_based_score": rule_result.difficulty_score,
            "layoutreader_divergence": round(divergence, 4),
            "gt_needed_reorder": gt_needed_reorder,
        })
    elapsed = time.perf_counter() - t0
    doc_fitz.close()

    # ---------- 성능지표: rule-based hard/easy vs LayoutReader 기반 GT ----------
    tp = sum(1 for p in per_page if p["rule_based_difficulty"] == "hard" and p["gt_needed_reorder"])
    fp = sum(1 for p in per_page if p["rule_based_difficulty"] == "hard" and not p["gt_needed_reorder"])
    fn = sum(1 for p in per_page if p["rule_based_difficulty"] == "easy" and p["gt_needed_reorder"])
    tn = sum(1 for p in per_page if p["rule_based_difficulty"] == "easy" and not p["gt_needed_reorder"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(per_page) if per_page else 0.0

    result = {
        "pdf": str(PDF_PATH), "n_pages": len(per_page),
        "divergence_threshold": DIVERGENCE_THRESHOLD,
        "layoutreader_inference_total_s": round(elapsed, 2),
        "n_rule_hard": sum(1 for p in per_page if p["rule_based_difficulty"] == "hard"),
        "n_gt_needed_reorder": sum(1 for p in per_page if p["gt_needed_reorder"]),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "per_page": per_page,
    }
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 결과 ===")
    print(f"rule-based hard 페이지: {result['n_rule_hard']}/{len(per_page)}개")
    print(f"LayoutReader 기준 실제 리딩오더 복원 필요 페이지: {result['n_gt_needed_reorder']}/{len(per_page)}개")
    print(f"Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f} Accuracy={accuracy:.3f}")
    print(f"LayoutReader 추론 총 시간: {elapsed:.1f}s ({elapsed/len(per_page)*1000:.0f}ms/페이지)")
    print(f"\n불일치 사례(FP: rule=hard인데 실제로는 불필요, FN: rule=easy인데 실제로는 필요):")
    for p in per_page:
        is_fp = p["rule_based_difficulty"] == "hard" and not p["gt_needed_reorder"]
        is_fn = p["rule_based_difficulty"] == "easy" and p["gt_needed_reorder"]
        if is_fp or is_fn:
            tag = "FP(과다판정)" if is_fp else "FN(과소판정)"
            print(f"  {tag} page{p['page']}: rule={p['rule_based_difficulty']}(score={p['rule_based_score']}), "
                  f"divergence={p['layoutreader_divergence']}, n_blocks={p['n_text_blocks']}")
    print(f"\n[result] saved to {OUT_PATH}")


if __name__ == "__main__":
    main()

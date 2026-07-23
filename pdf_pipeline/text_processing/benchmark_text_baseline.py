"""[1] 텍스트 파이프라인 베이스라인 기록 — PyMuPDF(+YOLO 라우팅) 현재 방식의 완전성/정합성/리딩오더/
지연을 골든셋(Claude가 렌더링된 페이지 이미지를 직접 육안으로 읽어 작성) 대비로 측정.

비교 두 변형:
(a) whole_page  — 페이지 전체를 PyMuPDF get_text()로 통째로 추출(표/차트 영역 구분 없음)
(b) yolo_filtered — YOLO가 Table/Picture로 분류한 영역은 제외하고, Text/Title/Section-header/
    List-item/Caption 클래스 bbox만 크롭해서 PyMuPDF로 추출(표 파이프라인과 역할 분리)

측정 지표:
- Recall: 골든셋 각 unit(문단/불릿)이 추출 텍스트에 정규화 후 부분문자열로 존재하는 비율
- 리딩오더 위반: 매칭된 unit들의 등장 위치(index)가 골든셋 순서와 다르게 뒤바뀐 쌍의 수
- 순도(purity): 추출 텍스트 중 골든 unit에 해당하지 않는 문자 비율(표/차트 텍스트 오염 proxy)
- 지연: 페이지당 추출 소요시간(ms)
"""

import json
import re
import time
from pathlib import Path

import fitz
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"
RESULT_PATH = OUT_DIR / "result_text_baseline.json"

RENDER_DPI = 150
CONF_THRESHOLD = 0.25
TEXT_CLASSES = {"Text", "Title", "Section-header", "List-item", "Caption"}

PDFS = {
    "LGCNS": {
        "path": ROOT / "pdf_pipeline" / "reference" / "LGCNS" / "20260721_company_279243000.pdf",
        "gt": OUT_DIR / "ground_truth_text_lgcns.json",
    },
    "Construct": {
        "path": ROOT / "pdf_pipeline" / "reference" / "Construct" / "20260721_industry_362851000.pdf",
        "gt": OUT_DIR / "ground_truth_text_construct.json",
    },
}


def normalize(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    return s.strip()


def extract_whole_page(doc_fitz, page_idx: int) -> str:
    return doc_fitz[page_idx].get_text()


PUA_RE = re.compile(r"[-]")  # Private Use Area — 커스텀 불릿/아이콘 폰트가 매핑 실패시 나오는 코드포인트


def detect_pua_artifact(text: str) -> bool:
    """[1] Construct PDF에서 실측: 불릿 아이콘 폰트 글리프가 표준 유니코드로 매핑되지 않고
    Private Use Area 코드포인트(예: \\uf09f)로 그대로 남는 경우 발견 — [19]의 `(cid:\\d+)` 글리프
    매핑 실패(표 파이프라인)와 같은 계열의 문제가 텍스트 파이프라인에도 존재함을 보여줌."""
    return bool(PUA_RE.search(text))


def extract_whole_page_masked(model, doc_fitz, page_idx: int) -> str:
    """[1] whole_page의 리딩오더/완전성 장점은 유지하되, YOLO가 Table/Picture로 잡은 영역과
    겹치는 블록만 빼고 나머지를 이어붙임(포지티브 선택이 아니라 네거티브 마스킹) — yolo_filtered가
    List-item 박스 커버리지 부족으로 recall이 떨어지는 문제를 피하면서, whole_page의 표/차트 오염
    (purity 저하) 문제만 골라서 줄이기 위한 절충안."""
    page = doc_fitz[page_idx]
    pix = page.get_pixmap(dpi=RENDER_DPI)
    tmp = OUT_DIR / f"_tmp_mask_{page_idx}.png"
    pix.save(str(tmp))
    img = Image.open(tmp).convert("RGB")
    res = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
    tmp.unlink(missing_ok=True)
    names = model.names
    boxes = res.boxes
    SCALE = RENDER_DPI / 72
    exclude_rects = []
    if boxes is not None:
        for cls_idx, xyxy in zip(boxes.cls, boxes.xyxy):
            if names[int(cls_idx)] in ("Table", "Picture"):
                x1, y1, x2, y2 = [v / SCALE for v in xyxy.tolist()]
                exclude_rects.append(fitz.Rect(x1, y1, x2, y2))

    blocks = page.get_text("blocks")
    parts = []
    for b in blocks:
        if b[6] != 0 or not b[4].strip():
            continue
        bbox = fitz.Rect(b[:4])
        center = fitz.Point((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)
        if any(r.contains(center) for r in exclude_rects):
            continue
        parts.append(b[4].strip())
    return "\n".join(parts)


def extract_yolo_filtered(model, doc_fitz, page_idx: int) -> str:
    page = doc_fitz[page_idx]
    pix = page.get_pixmap(dpi=RENDER_DPI)
    tmp = OUT_DIR / f"_tmp_yolo_{page_idx}.png"
    pix.save(str(tmp))
    img = Image.open(tmp).convert("RGB")
    res = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
    tmp.unlink(missing_ok=True)
    names = model.names
    boxes = res.boxes
    if boxes is None:
        return ""
    SCALE = RENDER_DPI / 72
    parts = []
    # 세로 위치(y0) 순으로 정렬해서 자연스러운 위->아래 순서로 이어붙임(리딩오더 baseline)
    items = sorted(zip(boxes.cls, boxes.xyxy), key=lambda cb: cb[1][1])
    for cls_idx, xyxy in items:
        cls_name = names[int(cls_idx)]
        if cls_name not in TEXT_CLASSES:
            continue
        x1, y1, x2, y2 = [v / SCALE for v in xyxy.tolist()]
        text = page.get_textbox(fitz.Rect(x1, y1, x2, y2))
        if text.strip():
            parts.append(text.strip())
    return "\n".join(parts)


def score(extracted: str, units: list) -> dict:
    norm_extracted = normalize(extracted)
    matched, positions = [], []
    for u in units:
        norm_u = normalize(u)
        idx = norm_extracted.find(norm_u)
        if idx >= 0:
            matched.append(u)
            positions.append(idx)
        else:
            positions.append(None)

    recall = len(matched) / len(units) if units else 1.0

    # 리딩오더 위반: 매칭된 것들끼리 index가 원래 순서와 다르게 역전된 쌍 개수
    found_positions = [p for p in positions if p is not None]
    order_violations = sum(
        1 for i in range(len(found_positions) - 1) if found_positions[i] > found_positions[i + 1]
    )

    matched_chars = sum(len(normalize(u)) for u in matched)
    purity = matched_chars / len(norm_extracted) if norm_extracted else 0.0

    unmatched = [u for u in units if u not in matched]
    return {
        "recall": round(recall, 4), "n_matched": len(matched), "n_total": len(units),
        "order_violations": order_violations, "purity": round(purity, 4),
        "extracted_char_count": len(norm_extracted),
        "pua_artifact_detected": detect_pua_artifact(extracted),
        "unmatched_units": [u[:60] + ("..." if len(u) > 60 else "") for u in unmatched],
    }


def main():
    model = YOLO(str(YOLO_MODEL_PATH))
    warmup = Image.new("RGB", (595, 842), (255, 255, 255))
    model.predict(warmup, conf=CONF_THRESHOLD, verbose=False)

    all_results = {}
    for pdf_label, cfg in PDFS.items():
        doc_fitz = fitz.open(str(cfg["path"]))
        gt = json.loads(cfg["gt"].read_text(encoding="utf-8"))
        pdf_result = {}
        print(f"\n=== {pdf_label} ===", flush=True)
        for page_str, page_gt in gt["pages"].items():
            page_idx = int(page_str) - 1
            units = page_gt["units"]

            t0 = time.perf_counter()
            wp_text = extract_whole_page(doc_fitz, page_idx)
            wp_ms = round((time.perf_counter() - t0) * 1000, 2)
            wp_score = score(wp_text, units)

            t0 = time.perf_counter()
            yf_text = extract_yolo_filtered(model, doc_fitz, page_idx)
            yf_ms = round((time.perf_counter() - t0) * 1000, 2)
            yf_score = score(yf_text, units)

            t0 = time.perf_counter()
            wm_text = extract_whole_page_masked(model, doc_fitz, page_idx)
            wm_ms = round((time.perf_counter() - t0) * 1000, 2)
            wm_score = score(wm_text, units)

            pdf_result[page_str] = {
                "content_type": page_gt["content_type"],
                "whole_page": {**wp_score, "latency_ms": wp_ms},
                "yolo_filtered": {**yf_score, "latency_ms": yf_ms},
                "whole_page_masked": {**wm_score, "latency_ms": wm_ms},
            }
            print(f"  page{page_str}({page_gt['content_type']}): "
                  f"whole_page recall={wp_score['recall']*100:.1f}% purity={wp_score['purity']*100:.1f}% "
                  f"({wp_ms}ms) | yolo_filtered recall={yf_score['recall']*100:.1f}% "
                  f"purity={yf_score['purity']*100:.1f}% ({yf_ms}ms) | whole_page_masked "
                  f"recall={wm_score['recall']*100:.1f}% purity={wm_score['purity']*100:.1f}% "
                  f"({wm_ms}ms) pua={wm_score['pua_artifact_detected']}", flush=True)
        doc_fitz.close()
        all_results[pdf_label] = pdf_result

    RESULT_PATH.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

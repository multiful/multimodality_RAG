"""[10] 프로덕션 진입점 — 채택된 YOLOv11n 기반 페이지 분류(라벨 정확도 100%, `요약.md` 참고)를
재사용 가능한 함수로 노출. text_processing과 페이지당 YOLO 호출을 공유하기 위해
`pdf_pipeline/yolo_layout.py`의 `run_yolo_layout()`을 그대로 사용 — `classify_pdf()`가 반환하는
`cached_boxes`를 그대로 `text_processing.text_extraction.process_pdf(..., page_boxes=...)`에
넘기면 페이지당 YOLO를 파이프라인 전체에서 **딱 한 번만** 호출하게 된다(사용자 지적 반영 —
이전엔 text_processing 내부 중복만 없앴고, page_classification과의 중복은 남아있었음).
"""

import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yolo_layout import run_yolo_layout  # noqa: E402

TABLE_CLASSES = {"Table"}
IMAGE_CLASSES = {"Picture"}
TEXT_CLASSES = {"Text", "Title", "Section-header", "List-item", "Caption"}


def classify_page(model, page: fitz.Page, page_idx: int, cached_boxes: list = None) -> dict:
    """반환: has_text/has_table/has_image + cached_boxes(다음 단계에서 재사용할 원본 YOLO 결과)."""
    boxes = cached_boxes if cached_boxes is not None else run_yolo_layout(model, page, page_idx)
    classes_found = {cls for cls, _ in boxes}
    return {
        "page": page_idx + 1,
        "has_text": bool(classes_found & TEXT_CLASSES),
        "has_table": bool(classes_found & TABLE_CLASSES),
        "has_image": bool(classes_found & IMAGE_CLASSES),
        "cached_boxes": boxes,
    }


def classify_pdf(pdf_path, model) -> dict:
    """PDF 전체 페이지 분류. 반환의 `pages[i]["cached_boxes"]`를 그대로 text_processing의
    `process_pdf(..., page_boxes={p["page"]: p["cached_boxes"] for p in result["pages"]})`에
    전달하면 YOLO 재추론 없이 이어서 처리 가능."""
    doc_fitz = fitz.open(str(pdf_path))
    pages = [classify_page(model, doc_fitz[i], i) for i in range(doc_fitz.page_count)]
    doc_fitz.close()
    return {"pdf": str(pdf_path), "n_pages": len(pages), "pages": pages}

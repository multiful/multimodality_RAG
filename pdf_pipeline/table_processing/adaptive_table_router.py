"""[13] Adaptive Table Complexity Router — 표마다 구조 복잡도를 판단해
단순 표는 pdfplumber(빠른 규칙 기반 파서)로 끝내고, 복잡한 표만 Docling(TableFormer)로 보낸다.

설계 원칙: [4](표 크기 기반 라우팅, 높이<550px 고정 임계값)는 기각됐었다 — 표 크기와 구조 복잡도가
비례하지 않기 때문(예: page4의 표는 대부분 작지도 않은데 병합 셀 때문에 pdfplumber가 1행만 뽑음,
반면 어떤 큰 표는 격자가 뚜렷해 pdfplumber만으로 충분). 그래서 이번엔 "크기"가 아니라
"pdfplumber가 실제로 그 표를 잘 읽어냈는가"를 직접 신호로 쓴다. 이러면:
1. PDF/DPI가 달라져도 안 깨짐(픽셀/포인트 절대값이 아니라 그 페이지 자체의 글줄 간격 대비 비율)
2. 크기가 작아도 구조가 복잡하면(병합 셀 등) 자동으로 Docling으로 감(= page4류를 올바르게 잡아냄)

복잡도 신호(전부 pdfplumber 1회 실행만으로 계산 — 값싼 신호):
  a) quick_rows: pdfplumber extract_table()이 뽑은 행 수
  b) expected_rows: 표 bbox 높이(pt) / 그 페이지의 실제 글줄 간격 중앙값(median line height, pdfplumber
     extract_words()에서 페이지별로 직접 계산 — 하드코딩 아님, PDF/페이지마다 다르게 적응)
  c) fill_ratio = quick_rows / expected_rows — pdfplumber가 "있어야 할 행"의 몇 %를 실제로 뽑았는가
  d) 컬럼 일관성: 모든 행의 셀 개수가 같은가(다르면 = 병합 셀을 잘못 쪼갠 ragged 결과 → 신뢰 불가)
  e) 비어있지 않은 셀 비율: 너무 낮으면(병합 셀이 빈칸으로 잘못 분리) 신뢰 불가

판정: quick_rows==0 이거나, 컬럼 비일관 이거나, 비어있지 않은 셀 비율이 낮거나,
     fill_ratio가 임계값 미만이면 COMPLEX(Docling) — 그 외엔 SIMPLE(pdfplumber 결과 그대로 채택).
"""

import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import pdfplumber
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # [26] yolo_layout이 pdf_pipeline/에 있음
from yolo_layout import run_yolo_layout  # noqa: E402
from text_cleanup import clean_extracted_text  # noqa: E402 — [36] raw_text가 구조화 출력(LLM) 입력으로도 쓰여서 정규화 필요

ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "LGCNS" / "20260721_company_279243000.pdf"
YOLO_MODEL_PATH = ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"

CONF_THRESHOLD = 0.25
RENDER_DPI = 150
SCALE = RENDER_DPI / 72  # 150dpi 픽셀 -> pt 환산


@dataclass
class RouterThresholds:
    fill_ratio_min: float = 0.6       # 이 미만이면 COMPLEX
    min_quick_rows: int = 2           # 이보다 적으면(0,1행) 구조 파악 불충분 -> COMPLEX
    nonempty_cell_ratio_min: float = 0.5
    fallback_line_height_pt: float = 12.0  # 페이지에 글자가 거의 없어 계산 불가할 때만 사용


def page_median_line_height(page) -> float:
    words = page.extract_words()
    tops = sorted(set(round(w["top"], 1) for w in words))
    diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 1]
    if not diffs:
        return RouterThresholds().fallback_line_height_pt
    return statistics.median(diffs)


def table_to_markdown(table) -> str:
    if not table:
        return ""
    header = [str(c) if c is not None else "" for c in table[0]]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in table[1:]:
        cells = [str(c) if c is not None else "" for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def classify_table(quick_table, bbox_height_pt: float, median_line_height_pt: float,
                    thresholds: RouterThresholds = RouterThresholds()):
    """quick_table: pdfplumber extract_table()의 반환값(list[list[str|None]] or None).
    반환: (complexity: 'simple'|'complex', reason: str, signals: dict)"""
    quick_rows = len(quick_table) if quick_table else 0
    expected_rows = max(bbox_height_pt / median_line_height_pt, 1.0)
    fill_ratio = quick_rows / expected_rows

    col_counts = [len(row) for row in quick_table] if quick_table else []
    consistent_columns = len(set(col_counts)) <= 1 if col_counts else False

    if quick_table:
        total_cells = sum(len(row) for row in quick_table)
        nonempty = sum(1 for row in quick_table for cell in row if cell and str(cell).strip())
        nonempty_ratio = nonempty / total_cells if total_cells else 0.0
    else:
        nonempty_ratio = 0.0

    signals = {
        "quick_rows": quick_rows, "expected_rows": round(expected_rows, 1),
        "fill_ratio": round(fill_ratio, 3), "consistent_columns": consistent_columns,
        "nonempty_ratio": round(nonempty_ratio, 3),
    }

    if quick_rows == 0:
        return "complex", "quick_parse_empty", signals
    if quick_rows < thresholds.min_quick_rows:
        return "complex", "too_few_rows", signals
    if not consistent_columns:
        return "complex", "ragged_columns", signals
    if nonempty_ratio < thresholds.nonempty_cell_ratio_min:
        return "complex", "sparse_cells", signals
    if fill_ratio < thresholds.fill_ratio_min:
        return "complex", "low_fill_ratio", signals
    return "simple", "quick_parse_sufficient", signals


def detect_and_route(thresholds: RouterThresholds = RouterThresholds(), crop_dir: Path = None,
                      yolo_model=None, page_boxes: dict = None, pdf_pp=None):
    """PDF 전체에서 YOLO로 표를 찾고, 표마다 SIMPLE/COMPLEX를 판정한다.
    SIMPLE 표는 pdfplumber 결과(마크다운 포함)까지 바로 채워서 반환하고,
    COMPLEX 표는 크롭 이미지 경로만 반환(Docling은 호출측에서 병렬 처리).

    [26] 사용자 지적 반영: 이 함수가 자체적으로 YOLO 모델을 새로 로드해 페이지당 또 한 번
    추론하던 문제(page_classification/text_processing이 이미 공유 중인 `run_yolo_layout()`과
    별개로 중복 호출) 수정. page_boxes({1-based page: [(cls_name, fitz.Rect), ...]}, 예를 들어
    `page_classification.page_classifier.classify_pdf()`가 반환하는 cached_boxes를 모은 것)를
    넘기면 그 페이지는 YOLO를 아예 다시 안 부른다. page_boxes에 없는 페이지만 `run_yolo_layout()`
    (공유 유틸)로 새로 추론하며, 이때만 yolo_model이 필요(안 주면 여기서 1회 로드).

    [39] pdf_pp: 호출측(build_records())이 이미 같은 PDF를 pdfplumber로 열어뒀으면 그 객체를
    넘겨 재사용 — 이 함수와 build_records()가 각자 pdfplumber.open()을 불러 pdfminer가 페이지당
    콘텐츠 스트림을 두 번 해석하던 것(LGCNS 6페이지 기준 측정 ~3s)이 실측 병목이었음. None이면
    기존처럼 이 함수가 열고 닫는다(하위호환) — 넘겨받은 경우엔 호출측이 lifecycle을 소유하므로
    여기서 닫지 않는다."""
    doc_fitz = fitz.open(str(PDF_PATH))
    if crop_dir:
        crop_dir.mkdir(exist_ok=True, parents=True)

    routed = []
    _owns_pdf_pp = pdf_pp is None
    pdf = pdf_pp if pdf_pp is not None else pdfplumber.open(str(PDF_PATH))
    try:
        for i, (page_pp, page_fz) in enumerate(zip(pdf.pages, doc_fitz), start=1):
            median_lh = page_median_line_height(page_pp)
            # [33] 성능 수정: img는 COMPLEX 표를 크롭 저장할 때(crop_dir 지정 시)만 실제로
            # 쓰이는데, 예전엔 crop_dir 없이 호출하는 현재 프로덕션 경로에서도 매 페이지
            # PNG를 디스크에 썼다가 곧바로 다시 읽어들이고 있었다(순수 낭비 — run_yolo_layout()도
            # 자체적으로 렌더링하므로 여기서 미리 만들어둘 필요가 없음). crop_dir이 실제로
            # 주어졌을 때만 렌더링하도록 지연시키고, 그마저도 디스크 왕복 없이 pix.samples를
            # 바로 PIL 이미지로 변환(yolo_layout.py [33]과 동일 기법).
            img = None
            if crop_dir:
                pix = page_fz.get_pixmap(dpi=RENDER_DPI)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            cached = page_boxes.get(i) if page_boxes else None
            if cached is None:
                if yolo_model is None:
                    from ultralytics import YOLO
                    yolo_model = YOLO(str(YOLO_MODEL_PATH))
                cached = run_yolo_layout(yolo_model, page_fz, i - 1)
            table_rects_pt = [rect for cls_name, rect in cached if cls_name == "Table"]
            if not table_rects_pt:
                continue

            t_idx = 0
            for rect in table_rects_pt:
                t_idx += 1
                bbox_pt = (rect.x0, rect.y0, rect.x1, rect.y1)
                x1, y1, x2, y2 = [v * SCALE for v in bbox_pt]
                height_pt = bbox_pt[3] - bbox_pt[1]
                try:
                    cropped_page = page_pp.crop(bbox_pt)
                    quick_table = cropped_page.extract_table()
                    # [36] raw_text는 classify_table()의 재무 키워드 매칭뿐 아니라 구조화 출력
                    # (extract_table_metadata, [25])의 LLM 입력으로도 그대로 쓰이는데 PUA/구두점
                    # 정규화가 전혀 안 되고 있었음 — text_cleanup.clean_extracted_text()로 통일.
                    raw_text = clean_extracted_text(cropped_page.extract_text() or "")
                except Exception:
                    quick_table = None
                    raw_text = ""

                complexity, reason, signals = classify_table(quick_table, height_pt, median_lh, thresholds)
                entry = {
                    "page": i, "table_idx": t_idx, "complexity": complexity, "reason": reason,
                    "signals": signals, "bbox_px": [x1, y1, x2, y2], "height_px": round(y2 - y1, 1),
                    "median_line_height_pt": round(median_lh, 2),
                    # Docling OCR이 한글 행 라벨을 깨뜨리는 경우가 있어(예: "매출액"->"OH EOH"),
                    # 표 타입 분류 등 "텍스트 내용"이 필요한 용도엔 이 pdfplumber 원문 텍스트를 쓴다
                    # (구조화된 표 데이터 자체는 여전히 Docling/pdfplumber 결과를 그대로 사용).
                    "raw_text": raw_text,
                }
                if complexity == "simple":
                    entry["quick_rows_data"] = quick_table
                    entry["markdown"] = table_to_markdown(quick_table)
                    entry["n_rows"] = len(quick_table)
                else:
                    crop_path = None
                    if crop_dir:
                        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                        crop_path = crop_dir / f"page_{i}_table_{t_idx}.png"
                        crop.save(crop_path)
                    entry["crop_path"] = str(crop_path) if crop_path else None
                routed.append(entry)
    finally:
        if _owns_pdf_pp:
            pdf.close()
        doc_fitz.close()
    return routed

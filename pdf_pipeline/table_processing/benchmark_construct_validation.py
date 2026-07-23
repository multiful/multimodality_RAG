"""[23] 세 번째 섹터 PDF(Construct, 건설 Weekly, 하나증권, 10페이지)로 표 처리 파이프라인
일반화 검증. baseline(pdfplumber만) vs 채택 파이프라인(Adaptive Router: SIMPLE->pdfplumber,
COMPLEX->TATR) 비교 + Canonical Field Metadata 추출(건설 청약/분양 필드가 실제로 매칭되는지) 검증
+ [22] 행 단위 재무항목 필터(is_pure_financial_line_item)까지 반영(표 전체 스킵 없음).
"""

import json
import sys
import time
from pathlib import Path

import fitz
import pdfplumber
from transformers import AutoImageProcessor, AutoModelForObjectDetection

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_table_router  # noqa: E402
from adaptive_table_router import RouterThresholds, detect_and_route, page_median_line_height  # noqa: E402
from table_type_router import classify_table, is_pure_financial_line_item  # noqa: E402
from row_parser import parse_table_adaptive, parse_simple_table_from_words  # noqa: E402
from canonical_field_schema import match_canonical_field, detect_wide_form  # noqa: E402
from text_normalization import fix_hangul_spacing, clean_value_by_type, detect_cid_artifact, is_over_spaced  # noqa: E402
from value_enrichment import extract_unit_and_value  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
CONSTRUCT_PDF = ROOT / "pdf_pipeline" / "reference" / "Construct" / "20260721_industry_362851000.pdf"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / "result_construct_table_validation.json"

TATR_DPI = 300
TATR_TOP_PAD_PT = 35 / (150 / 72)
TATR_SIDE_PAD_PT = 12 / (150 / 72)

# adaptive_table_router 모듈이 참조하는 PDF_PATH를 새 PDF로 교체(함수 내부에서 모듈 전역을
# 참조하므로 이렇게 바꾸면 코드 수정 없이 다른 PDF에 그대로 재사용 가능)
adaptive_table_router.PDF_PATH = CONSTRUCT_PDF


def run_baseline_pdfplumber(routed, pdf_pp):
    """표 위치는 이미 안다고 가정(YOLO 재사용), pdfplumber extract_table()만 적용."""
    total_rows, zero_row = 0, 0
    for r in routed:
        page_pp = pdf_pp.pages[r["page"] - 1]
        x1, y1, x2, y2 = r["bbox_px"]
        SCALE = 150 / 72
        bbox_pt = (x1 / SCALE, y1 / SCALE, x2 / SCALE, y2 / SCALE)
        try:
            table = page_pp.crop(bbox_pt).extract_table()
            n_rows = len(table) if table else 0
        except Exception:
            n_rows = 0
        total_rows += n_rows
        if n_rows == 0:
            zero_row += 1
    return total_rows, zero_row


def run_adaptive_pipeline(routed, pdf_pp, doc_fitz):
    """채택 파이프라인: SIMPLE->word-clustering, COMPLEX->TATR adaptive(컬럼 수 기반 동적 선택)."""
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model.eval()

    median_lh_by_page = {}
    total_rows = 0
    per_table_rows = []
    metadata_records = []
    n_finance_rows_filtered = 0  # [22] 표 전체 스킵이 아니라 행 단위로 걸러낸 순수 재무항목 개수

    for r in routed:
        page_pp = pdf_pp.pages[r["page"] - 1]
        if r["page"] not in median_lh_by_page:
            median_lh_by_page[r["page"]] = page_median_line_height(page_pp)
        median_lh = median_lh_by_page[r["page"]]

        x1, y1, x2, y2 = r["bbox_px"]
        SCALE = 150 / 72
        bbox_pt = (x1 / SCALE, y1 / SCALE, x2 / SCALE, y2 / SCALE)

        if r["complexity"] == "simple":
            parsed_rows = parse_simple_table_from_words(page_pp, bbox_pt, median_lh)
        else:
            parsed_rows = parse_table_adaptive(model, processor, doc_fitz, page_pp, r["page"], bbox_pt,
                                                TATR_DPI, TATR_TOP_PAD_PT, TATR_SIDE_PAD_PT, median_lh)
        total_rows += len(parsed_rows)
        per_table_rows.append({"page": r["page"], "table_idx": r["table_idx"],
                                "complexity": r["complexity"], "n_rows": len(parsed_rows)})

        # ttype은 로깅/참고용으로만 유지(엔티티 추출 라우팅, [14]) — [22] 이후로는 표 전체를
        # finance라고 통째로 스킵하지 않고, 아래에서 행 단위로만 순수 재무항목을 걸러낸다.
        ttype = classify_table(r["raw_text"])
        table_over_spaced = is_over_spaced(r["raw_text"])
        norm_rows = [{"label": fix_hangul_spacing(row["label"], force=table_over_spaced),
                      "cells": [fix_hangul_spacing(c, force=table_over_spaced) for c in row["cells"]]}
                     for row in parsed_rows]

        n_before_filter = len(norm_rows)
        norm_rows = [row for row in norm_rows if not is_pure_financial_line_item(row["label"])]
        n_finance_rows_filtered += n_before_filter - len(norm_rows)
        if not norm_rows:
            continue

        header_fields, data_rows = detect_wide_form(norm_rows)
        if header_fields:
            for row in data_rows:
                row_cells = [row["label"]] + row["cells"]
                for col_idx, cf in enumerate(header_fields):
                    if cf and col_idx < len(row_cells) and row_cells[col_idx]:
                        val = clean_value_by_type(row_cells[col_idx], cf.value_type)
                        metadata_records.append({"page": r["page"], "table_idx": r["table_idx"],
                                                  "canonical_field": cf.key, "value": val})
        else:
            for row in norm_rows:
                cf = match_canonical_field(row["label"])
                if cf:
                    first_val = next((c for c in row["cells"] if c), row["label"])  # [23] 위치 보존 후
                    # 첫 빈 셀 대신 첫 non-empty 셀을 값으로(컬럼 정렬 유지 위해 빈칸도 남기므로)
                    val = clean_value_by_type(first_val, cf.value_type)
                    metadata_records.append({"page": r["page"], "table_idx": r["table_idx"],
                                              "canonical_field": cf.key, "value": val})

    return total_rows, per_table_rows, metadata_records, n_finance_rows_filtered


def main():
    print(f"대상 PDF: {CONSTRUCT_PDF.name}", flush=True)
    t0 = time.perf_counter()
    routed = detect_and_route(RouterThresholds())
    route_s = round(time.perf_counter() - t0, 2)
    n_simple = sum(1 for r in routed if r["complexity"] == "simple")
    n_complex = sum(1 for r in routed if r["complexity"] == "complex")
    print(f"표 탐지+라우팅: {len(routed)}개(SIMPLE {n_simple} / COMPLEX {n_complex}), {route_s}s", flush=True)

    pdf_pp = pdfplumber.open(str(CONSTRUCT_PDF))
    doc_fitz = fitz.open(str(CONSTRUCT_PDF))

    print("\n=== Baseline(pdfplumber만) ===", flush=True)
    t0 = time.perf_counter()
    baseline_rows, baseline_zero = run_baseline_pdfplumber(routed, pdf_pp)
    baseline_s = round(time.perf_counter() - t0, 2)
    print(f"총 {baseline_rows}행, 0행 표 {baseline_zero}개, {baseline_s}s", flush=True)

    print("\n=== 채택 파이프라인(Adaptive Router + TATR) ===", flush=True)
    t0 = time.perf_counter()
    adaptive_rows, per_table, metadata_records, n_finance_rows_filtered = run_adaptive_pipeline(routed, pdf_pp, doc_fitz)
    adaptive_s = round(time.perf_counter() - t0, 2)
    print(f"총 {adaptive_rows}행, {adaptive_s}s (재무항목 {n_finance_rows_filtered}행 필터, 표 단위 스킵 없음)", flush=True)

    pdf_pp.close()
    doc_fitz.close()

    mapped = [m for m in metadata_records if m["canonical_field"]]
    from collections import Counter
    field_counts = dict(Counter(m["canonical_field"] for m in mapped))

    result = {
        "pdf": CONSTRUCT_PDF.name, "n_tables": len(routed), "n_simple": n_simple, "n_complex": n_complex,
        "route_decision_s": route_s,
        "baseline_pdfplumber": {"total_rows": baseline_rows, "zero_row_tables": baseline_zero, "elapsed_s": baseline_s},
        "adaptive_pipeline": {"total_rows": adaptive_rows, "elapsed_s": adaptive_s,
                               "n_finance_rows_filtered": n_finance_rows_filtered, "per_table_rows": per_table},
        "canonical_field_metadata": {"n_records": len(metadata_records), "n_mapped": len(mapped),
                                      "field_counts": field_counts, "records": metadata_records},
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    improvement = adaptive_rows - baseline_rows
    print(f"\n=== 비교 요약 ===")
    print(f"Baseline: {baseline_rows}행 (0행 표 {baseline_zero}개, 전체 {len(routed)}개 중)")
    print(f"채택 파이프라인: {adaptive_rows}행")
    print(f"개선: {improvement:+d}행 ({(adaptive_rows/baseline_rows-1)*100:+.1f}%)" if baseline_rows else "")
    print(f"\nCanonical field 매칭: {len(mapped)}개")
    print(f"필드별 매칭 수: {field_counts}")
    print(f"\n[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

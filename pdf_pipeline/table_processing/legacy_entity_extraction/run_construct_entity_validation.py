"""[23] Construct(건설 Weekly, 하나증권, 10페이지) 엔티티 추출 검증 — 텍스트+표(비재무 행) 기반,
VLM 이미지 캡셔닝은 K-Wave([21])와 동일하게 범위 밖. [22] 행 단위 재무항목 필터를 반영해 표
전체를 스킵하지 않고 순수 재무항목 행만 걸러낸 나머지(청약동향/수주공시/밸류에이션/수급동향 등)를
표 컨텍스트로 LLM에 전달 — "실적테이블처럼 재무+비재무가 섞인 표를 통째로 버리면 안 된다"는
사용자 요청이 엔티티 추출 단계에도 동일하게 적용되는지 확인."""

import json
import re
import sys
import time
from pathlib import Path

import fitz
import pdfplumber
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # [legacy] table_type_router 등은 table_processing/에 그대로 있음
import adaptive_table_router  # noqa: E402
from adaptive_table_router import RouterThresholds, detect_and_route, page_median_line_height  # noqa: E402
from table_type_router import classify_table, is_pure_financial_line_item  # noqa: E402
from row_parser import parse_table_adaptive, parse_simple_table_from_words  # noqa: E402
from canonical_field_schema import detect_wide_form  # noqa: E402
from text_normalization import fix_hangul_spacing, is_over_spaced  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # [legacy] table_processing/legacy_entity_extraction/에서 한 단계 더 이동
CONSTRUCT_PDF = ROOT / "pdf_pipeline" / "reference" / "Construct" / "20260721_industry_362851000.pdf"
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / "result_construct_entity_extraction.json"

TATR_DPI = 300
TATR_TOP_PAD_PT = 35 / (150 / 72)
TATR_SIDE_PAD_PT = 12 / (150 / 72)

adaptive_table_router.PDF_PATH = CONSTRUCT_PDF


def build_page_content():
    """페이지별 [본문 텍스트] + [비재무 표 markdown]만 구성(이미지 캡셔닝 제외, 명시된 범위 축소)."""
    routed = detect_and_route(RouterThresholds())
    print(f"[router] 표 {len(routed)}개(SIMPLE {sum(1 for r in routed if r['complexity']=='simple')} / "
          f"COMPLEX {sum(1 for r in routed if r['complexity']=='complex')})", flush=True)

    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model.eval()
    doc_fitz = fitz.open(str(CONSTRUCT_PDF))
    pdf_pp = pdfplumber.open(str(CONSTRUCT_PDF))

    n_pages = len(pdf_pp.pages)
    page_text = {i: (pdf_pp.pages[i - 1].extract_text() or "") for i in range(1, n_pages + 1)}
    tables_by_page = {i: [] for i in range(1, n_pages + 1)}
    median_lh_by_page = {}
    n_finance_rows_filtered = 0  # [22] 표 전체 스킵이 아니라 행 단위로 걸러낸 순수 재무항목 개수

    for r in routed:
        ttype = classify_table(r["raw_text"])  # 로깅/참고용(엔티티 추출 자체는 행 단위 필터로 처리)
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
        table_over_spaced = is_over_spaced(r["raw_text"])
        norm_rows = [{"label": fix_hangul_spacing(row["label"], force=table_over_spaced),
                      "cells": [fix_hangul_spacing(c, force=table_over_spaced) for c in row["cells"]]}
                     for row in parsed_rows]
        n_before_filter = len(norm_rows)
        norm_rows = [row for row in norm_rows if not is_pure_financial_line_item(row["label"])]
        n_finance_rows_filtered += n_before_filter - len(norm_rows)
        if not norm_rows:
            continue
        md_lines = [f"{row['label']}: {', '.join(row['cells'])}" if row["cells"] else row["label"]
                    for row in norm_rows]
        tables_by_page[r["page"]].append("\n".join(md_lines))

    pdf_pp.close()
    doc_fitz.close()
    return page_text, tables_by_page, n_pages, n_finance_rows_filtered


def load_qwen():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    return model, processor, device


def text_generate(model, processor, device, prompt: str, max_new_tokens: int = 150) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.3)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    result = processor.decode(trimmed, skip_special_tokens=True).strip()
    del inputs, out
    if device == "mps":
        torch.mps.empty_cache()
    return result


def main():
    t0 = time.perf_counter()
    page_text, tables_by_page, n_pages, n_finance_rows_filtered = build_page_content()
    table_stage_s = round(time.perf_counter() - t0, 2)
    print(f"[table] 표 단계 총 소요: {table_stage_s}s (재무항목 {n_finance_rows_filtered}행 필터, 표 단위 스킵 없음)",
          flush=True)

    model, processor, device = load_qwen()
    per_page_entities = {}
    timing = []

    for i in range(1, n_pages + 1):
        parts = []
        if page_text[i].strip():
            parts.append(f"[본문]\n{page_text[i]}")
        for t_idx, md in enumerate(tables_by_page[i], start=1):
            parts.append(f"[표{t_idx}]\n{md}")
        page_context = "\n\n".join(parts)
        if not page_context.strip():
            per_page_entities[i] = ""
            continue
        prompt = (
            "다음은 증권사 산업 리포트 한 페이지에서 추출한 내용입니다(참고: 이미지/차트 설명은 "
            "이번엔 제외됨). 이 안에 등장하는 모든 기업 이름을 빠짐없이 나열하세요. "
            "표 안에서만 언급된 기업도 포함하세요. "
            "한 줄에 하나씩 '기업명 (아는 경우 종목코드)' 형태로만 출력하고 다른 설명은 하지 마세요.\n\n"
            f"{page_context}"
        )
        t0 = time.perf_counter()
        result = text_generate(model, processor, device, prompt)
        elapsed = round(time.perf_counter() - t0, 3)
        timing.append(elapsed)
        per_page_entities[i] = result
        print(f"[p{i}] ({elapsed}s) {result[:150]}", flush=True)

    entity_extract_s = round(sum(timing), 2)
    result = {
        "pdf": CONSTRUCT_PDF.name, "n_pages": n_pages, "n_finance_rows_filtered": n_finance_rows_filtered,
        "table_stage_s": table_stage_s, "entity_extract_s": entity_extract_s,
        "n_llm_calls": len(timing), "per_page_entities": per_page_entities,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n표 단계: {table_stage_s}s, 엔티티추출: {entity_extract_s}s({len(timing)}회 호출)")
    print(f"[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""축4 베이스라인: 순수 PyMuPDF get_text (정제·라우팅·구조인식 없음) + pdfplumber 나이브 표.
compare_baseline_vs_pipeline.py의 baseline 정의와 동일한 관례."""
import sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_exp as C
import fitz, pdfplumber

PDF = Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG/pdf_pipeline/reference/SmartPhone/20260629_industry_47868000.pdf")
OUT = HERE / "out_baseline.json"

def main():
    t0 = time.time()
    doc = fitz.open(str(PDF))
    page_texts = [pg.get_text() for pg in doc]   # 원문 그대로 (PUA/헤더푸터/정규화 전혀 없음)
    parse_time = time.time() - t0

    # 나이브 표: pdfplumber extract_tables (라우팅/TATR/셀정렬 보정 없음)
    t1 = time.time()
    n_tables = 0; n_rows = 0; zero_row = 0
    with pdfplumber.open(str(PDF)) as pdf:
        for pg in pdf.pages:
            for tb in pg.extract_tables():
                n_tables += 1
                rows = len(tb) if tb else 0
                n_rows += rows
                if rows <= 1:
                    zero_row += 1
    table_time = time.time() - t1

    chunks = C.chunk_pages_raw(page_texts)
    full_text = "\n".join(page_texts)
    caps = C.count_captions(full_text)
    out = {
        "axis": "baseline",
        "parse_time_s": round(parse_time, 3),
        "stage_timing": {"text_extract_s": round(parse_time, 3), "table_naive_s": round(table_time, 3)},
        "total_time_s": round(parse_time + table_time, 3),
        "n_chunks": len(chunks),
        "chunks": chunks,
        "full_text": full_text,
        "structure": {
            "n_tables_detected": n_tables,
            "n_table_rows": n_rows,
            "n_zero_or_1row_tables": zero_row,
            "chart_titles_preserved": len(caps["chart_titles"]),
            "table_caps_preserved": len(caps["table_caps"]),
            "note": "표는 pdfplumber 나이브 추출(구조 보정 없음). 청크=페이지 원문(구조 인식 없음)."
        },
        "page_pred": None,   # 페이지 분류 없음
        "routing": None,
    }
    C.dump_json(OUT, out)
    print(f"[baseline] parse {parse_time:.2f}s table {table_time:.2f}s chunks={len(chunks)} "
          f"tables={n_tables} rows={n_rows} zero_row={zero_row} charts={len(caps['chart_titles'])}/93 tabcaps={len(caps['table_caps'])}/11")

if __name__ == "__main__":
    main()

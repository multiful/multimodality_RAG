# -*- coding: utf-8 -*-
"""주간 수주 공시 표(Construct p5) 미추출 진단 — 하이브리드 게이트 3단계를 그대로 재현해
어느 단계에서 구조가 붕괴했는지 증거로 보여준다."""
import os, sys
from pathlib import Path
ROOT = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP = ROOT / "pdf_pipeline"
for d in [str(PP), str(PP / "table_processing")]: sys.path.insert(0, d)
PDF = PP / "reference" / "Construct" / "20260721_industry_362851000.pdf"

import pdfplumber, fitz
import table_processing.adaptive_table_router as art
import table_processing.run_table_metadata_pipeline as rtmp
from table_processing.row_parser import (parse_table_hybrid, parse_simple_table_from_words,
                                          _text_strategy_gate, _word_clustering_looks_flattened)
art.PDF_PATH = PDF; rtmp.PDF_PATH = PDF

def show(tag, rows, n=14):
    print(f"\n--- {tag} ({len(rows)}행) ---")
    for r in rows[:n]:
        print(f"   label={r['label'][:60]!r}  cells={[c[:40] for c in r['cells']][:4]}")

def main():
    pdf_pp = pdfplumber.open(str(PDF)); doc = fitz.open(str(PDF))
    routed = art.detect_and_route(art.RouterThresholds(), pdf_pp=pdf_pp, pdf_path=PDF)
    p5 = [r for r in routed if r["page"] == 5]
    print(f"[router] 전체 표 {len(routed)}개 / p5 표 {len(p5)}개")
    for r in p5:
        print(f"  p5 table complexity={r['complexity']} bbox_px={r['bbox_px']} raw_text앞={r['raw_text'][:50]!r}")
    # 수주공시 표 고르기
    tgt = None
    for r in p5:
        if "수주" in r["raw_text"] or "계약 금액" in r["raw_text"]: tgt = r
    if tgt is None: tgt = p5[0] if p5 else None
    if tgt is None: print("!! p5에 표 없음"); return
    SCALE = 150 / 72
    x1, y1, x2, y2 = tgt["bbox_px"]; bbox_pt = (x1/SCALE, y1/SCALE, x2/SCALE, y2/SCALE)
    page_pp = pdf_pp.pages[4]; mlh = tgt["median_line_height_pt"]
    print(f"\n[대상] complexity={tgt['complexity']} median_line_height={mlh:.2f}")

    # 1단계 text-strategy
    tbl = page_pp.crop(bbox_pt).extract_table({"vertical_strategy": "text", "horizontal_strategy": "text"})
    print(f"\n=== 1단계 text-strategy: {len(tbl) if tbl else 0}행 / 게이트통과={_text_strategy_gate(tbl)}")
    if tbl:
        from collections import Counter
        counts = [sum(1 for c in row if c and str(c).strip()) for row in tbl]
        print(f"   행별 채워진셀수={counts}  최빈={Counter([c for c in counts if c>0]).most_common(3)}")
        for row in tbl[:8]: print("   ", [str(c)[:22] for c in row if c and str(c).strip()])
    # 2단계 word-clustering
    wc = parse_simple_table_from_words(page_pp, bbox_pt, mlh)
    show("2단계 word-clustering", wc)
    print(f"   flattened판정(=TATR로 올릴까)={_word_clustering_looks_flattened(wc)}")
    # 3단계 TATR 강제
    try:
        from table_processing.row_parser import parse_table_adaptive, _get_tatr_model
        m, p = _get_tatr_model()
        tr = parse_table_adaptive(m, p, doc, page_pp, 5, bbox_pt, 300, 35/(150/72), 12/(150/72), mlh)
        show("3단계 TATR(강제 실행)", tr)
    except Exception as e:
        print(f"   TATR 실패: {e}")
    # 실제 채택 경로
    final = parse_table_hybrid(page_pp, bbox_pt, mlh, doc_fitz=doc, page_num=5)
    show("실제 파이프라인 채택 결과", final)
    pdf_pp.close(); doc.close()

if __name__ == "__main__": main()

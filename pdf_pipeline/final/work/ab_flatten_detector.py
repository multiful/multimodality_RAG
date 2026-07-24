# -*- coding: utf-8 -*-
"""탐지기 수정 A/B — _word_clustering_looks_flattened 확장((b) 라벨뭉침, (c) 고아행)이
(1) 실제로 놓치던 표를 TATR로 승격시키는가, (2) 과승격으로 지연을 얼마나 더 쓰는가.
구 탐지기를 로컬에 재구현해 같은 표에 대해 두 판정을 나란히 비교한다."""
import os, sys, time, json, re
from pathlib import Path
ROOT = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP = ROOT / "pdf_pipeline"
for d in [str(PP), str(PP / "table_processing")]: sys.path.insert(0, d)
DOCS = {"Construct": PP/"reference"/"Construct"/"20260721_industry_362851000.pdf",
        "SmartPhone": PP/"reference"/"SmartPhone"/"20260629_industry_47868000.pdf"}
OUT = PP / "final" / "results_flatten_detector_ab.json"

import pdfplumber, fitz
import table_processing.adaptive_table_router as art
from table_processing.row_parser import (parse_simple_table_from_words, _text_strategy_gate,
                                          _word_clustering_looks_flattened, _MULTI_VALUE_RE,
                                          parse_table_adaptive, _get_tatr_model)

def old_detector(rows):
    """수정 전 로직 — cells[0] 숫자 뭉침만 검사."""
    if not rows: return True
    n = sum(1 for r in rows if r["cells"] and len(_MULTI_VALUE_RE.findall(r["cells"][0])) >= 3)
    return n / len(rows) >= 0.3

def quality(rows):
    """구조 품질 대리지표: 평균 셀 개수(컬럼 분해가 됐는가) / 고아행 비율(줄바꿈 조각)."""
    if not rows: return 0.0, 1.0
    avg_cells = sum(len(r["cells"]) for r in rows) / len(rows)
    orphan = sum(1 for r in rows if len((r["label"] or "")) <= 6 and
                 not any(_MULTI_VALUE_RE.search(c) for c in (r["cells"] or []))) / len(rows)
    return round(avg_cells, 2), round(orphan, 2)

def main():
    res = {"docs": {}}
    for name, pdf in DOCS.items():
        art.PDF_PATH = pdf
        pdf_pp = pdfplumber.open(str(pdf)); doc = fitz.open(str(pdf))
        routed = art.detect_and_route(art.RouterThresholds(), pdf_pp=pdf_pp, pdf_path=pdf)
        rows_out, n_new, n_old, tatr_time = [], 0, 0, 0.0
        for r in routed:
            SCALE = 150/72; x1,y1,x2,y2 = r["bbox_px"]
            bbox_pt = (x1/SCALE, y1/SCALE, x2/SCALE, y2/SCALE)
            page_pp = pdf_pp.pages[r["page"]-1]; mlh = r["median_line_height_pt"]
            try:
                tbl = page_pp.crop(bbox_pt).extract_table({"vertical_strategy":"text","horizontal_strategy":"text"})
            except Exception: tbl = None
            if _text_strategy_gate(tbl):
                continue                        # text-strategy 채택 -> TATR 후보 아님
            if r["complexity"] == "simple":
                continue                        # SIMPLE은 애초에 에스컬레이션 대상 제외
            wc = parse_simple_table_from_words(page_pp, bbox_pt, mlh)
            o, n = old_detector(wc), _word_clustering_looks_flattened(wc)
            n_old += int(o); n_new += int(n)
            entry = {"page": r["page"], "head": r["raw_text"][:34].replace("\n"," "),
                     "old_escalate": o, "new_escalate": n, "wc_rows": len(wc),
                     "wc_quality": quality(wc)}
            if n and not o:                     # 새로 승격된 표만 TATR 실행해 개선/비용 측정
                t = time.time()
                try:
                    m, p = _get_tatr_model()
                    tr = parse_table_adaptive(m, p, doc, page_pp, r["page"], bbox_pt,
                                              300, 35/(150/72), 12/(150/72), mlh)
                except Exception as e:
                    tr = []; entry["tatr_error"] = str(e)[:80]
                dt = time.time()-t; tatr_time += dt
                entry.update({"tatr_rows": len(tr), "tatr_quality": quality(tr), "tatr_s": round(dt,2)})
            rows_out.append(entry)
        res["docs"][name] = {"tatr_candidates": len(rows_out), "old_escalated": n_old,
                             "new_escalated": n_new, "added_tatr_s": round(tatr_time,2),
                             "tables": rows_out}
        print(f"\n[{name}] TATR후보 {len(rows_out)}개 / 구탐지기 승격 {n_old} -> 신탐지기 {n_new} "
              f"(추가 TATR 시간 {tatr_time:.2f}s)")
        for e in rows_out:
            mark = "**신규승격**" if e["new_escalate"] and not e["old_escalate"] else ""
            print(f"   p{e['page']:>2} {e['head']!r:38} wc={e['wc_rows']}행 품질(셀평균,고아율)={e['wc_quality']} {mark}")
            if "tatr_rows" in e:
                print(f"        -> TATR {e['tatr_rows']}행 품질={e['tatr_quality']} ({e['tatr_s']}s)")
        pdf_pp.close(); doc.close()
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT.name} 저장")

if __name__ == "__main__": main()

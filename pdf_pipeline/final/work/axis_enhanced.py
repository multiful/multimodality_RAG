# -*- coding: utf-8 -*-
"""축3 고도화 파이프라인: YOLO 페이지분류 -> 난이도 라우팅 -> 텍스트(정제+계층청킹+section_path)
+ 표(Adaptive Router+TATR+canonical 매칭). 구조화출력(OpenAI)은 파싱품질 분리를 위해 OFF."""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG")
PP = ROOT / "pdf_pipeline"
for d in [str(HERE), str(PP), str(PP/"page_classification"), str(PP/"text_processing"), str(PP/"table_processing")]:
    sys.path.insert(0, d)
import common_exp as C

PDF = PP / "reference" / "SmartPhone" / "20260629_industry_47868000.pdf"
OUT = HERE / "out_enhanced.json"
DOC_TITLE = "스마트폰 수요 우려는 예견된 수순 (반도체 및 소부장 Weekly)"

def main():
    from ultralytics import YOLO
    from page_classifier import classify_pdf
    from text_extraction import process_pdf
    import run_table_metadata_pipeline as rtmp
    import adaptive_table_router as atr

    t_load = time.time()
    yolo = YOLO(str(PP/"page_classification"/"models"/"yolo11n_doc_layout.pt"))
    yolo.predict(__import__("numpy").zeros((640,640,3), dtype="uint8"), verbose=False)  # warmup
    load_s = time.time() - t_load

    # --- 1) 페이지 분류 (YOLO 1회/페이지, 결과 공유) ---
    t = time.time()
    cls = classify_pdf(str(PDF), yolo)
    classify_s = time.time() - t
    page_boxes = {p["page"]: p["cached_boxes"] for p in cls["pages"]}
    page_pred = {str(p["page"]): {"text": bool(p["has_text"]), "table": bool(p["has_table"]),
                                   "image": bool(p["has_image"])} for p in cls["pages"]}

    # --- 2) 텍스트 라우팅 (정제+계층청킹+section_path, 난이도 판정) ---
    t = time.time()
    tp = process_pdf(str(PDF), yolo, doc_title=DOC_TITLE, page_boxes=page_boxes,
                     chunk_backend="rulebased", remove_boilerplate=True, add_structured_metadata=False)
    text_s = time.time() - t
    text_chunks = []
    sec_depths = []
    for pg in tp["pages"]:
        for c in pg.get("chunks", []):
            sp = c.get("section_path") or []
            sec_depths.append(len(sp))
            text_chunks.append({"text": c.get("text") or c.get("raw_chunk",""), "kind": "text",
                                 "page": c.get("page"), "section_path": sp})

    # --- 3) 표 라우팅 (Adaptive Router + TATR + canonical 매칭) ---
    atr.PDF_PATH = PDF
    rtmp.PDF_PATH = PDF
    t = time.time()
    records, n_fin_filtered, n_cid = rtmp.build_records("smartphone", add_structured_metadata=False,
                                                        page_boxes=page_boxes, yolo_model=yolo)
    table_s = time.time() - t

    # 표 레코드 -> (page,table_idx) 단위 청크로 조립 (라벨-값 co-occur 보존)
    from collections import OrderedDict
    grp = OrderedDict()
    for r in records:
        k = (r.get("page"), r.get("table_idx"))
        grp.setdefault(k, []).append(r)
    table_chunks = []
    n_rows = len(records)
    n_canon = sum(1 for r in records if r.get("canonical_field"))
    for (pg, tidx), rs in grp.items():
        lines = []
        ttype = rs[0].get("table_type","")
        for r in rs:
            lab = r.get("raw_label","")
            cells = r.get("cells") or []
            cf = r.get("canonical_field")
            line = f"{lab}: " + " | ".join(str(x) for x in cells)
            if cf:
                line += f"  [{cf}]"
            lines.append(line)
        body = f"[표 p{pg} #{tidx} type={ttype}]\n" + "\n".join(lines)
        table_chunks.append({"text": body, "kind": "table", "page": pg, "table_idx": tidx})

    all_chunks = text_chunks + table_chunks
    full_text = "\n".join(c["text"] for c in all_chunks)
    caps = C.count_captions(full_text)

    out = {
        "axis": "enhanced",
        "parse_time_s": round(classify_s + text_s + table_s, 3),
        "stage_timing": {"model_load_warmup_s": round(load_s,3), "page_classify_s": round(classify_s,3),
                          "text_route_s": round(text_s,3), "table_route_s": round(table_s,3)},
        "total_time_s": round(load_s + classify_s + text_s + table_s, 3),
        "n_chunks": len(all_chunks),
        "n_text_chunks": len(text_chunks),
        "n_table_chunks": len(table_chunks),
        "chunks": all_chunks,
        "full_text": full_text,
        "structure": {
            "n_table_records_rows": n_rows,
            "n_canonical_matched": n_canon,
            "canonical_hit_rate": round(n_canon/max(1,n_rows),4),
            "n_finance_rows_filtered": n_fin_filtered,
            "chart_titles_preserved": len(caps["chart_titles"]),
            "table_caps_preserved": len(caps["table_caps"]),
            "avg_section_path_depth": round(sum(sec_depths)/max(1,len(sec_depths)),2),
            "chunks_with_section_path": sum(1 for d in sec_depths if d>0),
        },
        "page_pred": page_pred,
        "routing": {
            "route_to_mineru": tp.get("route_to_mineru"),
            "hard_page_numbers": tp.get("hard_page_numbers"),
            "n_hard_pages": len(tp.get("hard_page_numbers") or []),
        },
    }
    C.dump_json(OUT, out)
    print(f"[enhanced] load {load_s:.1f}s classify {classify_s:.1f}s text {text_s:.1f}s table {table_s:.1f}s")
    print(f"  chunks={len(all_chunks)} (text {len(text_chunks)} + table {len(table_chunks)}) rows={n_rows} canon={n_canon}")
    print(f"  hard_pages={tp.get('hard_page_numbers')} route_to_mineru={tp.get('route_to_mineru')}")
    print(f"  charts={len(caps['chart_titles'])}/93 tabcaps={len(caps['table_caps'])}/11")

if __name__ == "__main__":
    main()

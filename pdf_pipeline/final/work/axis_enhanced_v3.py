# -*- coding: utf-8 -*-
"""축3-v2 고도화 개선판: 지연 유지하며 docling/MinerU 상회 목표.
enhanced 대비 2개 지연중립 개선:
 (1) 표 선형화 수정 — canonical 레코드(wide-form에서 행-엔티티 유실) 대신
     YOLO Table bbox를 pdfplumber로 행단위(회사|값들) 마크다운 재구성(Adaptive Router SIMPLE 경로).
 (2) 캡션 보호 — 원문에서 도표/표 캡션 라인 회수해 정제 과삭제 복구.
페이지분류/텍스트 라우팅은 enhanced와 동일(가중치 재사용)."""
import sys, time, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG")
PP = ROOT/"pdf_pipeline"
for d in [str(HERE), str(PP), str(PP/"page_classification"), str(PP/"text_processing"), str(PP/"table_processing")]:
    sys.path.insert(0, d)
import common_exp as C
import fitz, pdfplumber

PDF = PP/"reference"/"SmartPhone"/"20260629_industry_47868000.pdf"
OUT = HERE/"out_enhanced_v3.json"
DOC_TITLE = "스마트폰 수요 우려는 예견된 수순 (반도체 및 소부장 Weekly)"
CAP_RE = re.compile(r"^\s*(도표|표)\s*\d+\.?\s*.*$")

def main():
    from ultralytics import YOLO
    from page_classifier import classify_pdf
    from text_extraction import process_pdf

    t_load=time.time()
    yolo = YOLO(str(PP/"page_classification"/"models"/"yolo11n_doc_layout.pt"))
    yolo.predict(__import__("numpy").zeros((640,640,3),dtype="uint8"), verbose=False)
    load_s=time.time()-t_load

    t=time.time()
    cls = classify_pdf(str(PDF), yolo)
    classify_s=time.time()-t
    page_boxes = {p["page"]: p["cached_boxes"] for p in cls["pages"]}
    page_pred = {str(p["page"]): {"text":bool(p["has_text"]),"table":bool(p["has_table"]),"image":bool(p["has_image"])} for p in cls["pages"]}

    # 텍스트 라우팅(동일)
    t=time.time()
    tp = process_pdf(str(PDF), yolo, doc_title=DOC_TITLE, page_boxes=page_boxes,
                     chunk_backend="rulebased", remove_boilerplate=True, add_structured_metadata=False)
    text_s=time.time()-t
    text_chunks=[]; sec_depths=[]
    for pg in tp["pages"]:
        for c in pg.get("chunks",[]):
            sp=c.get("section_path") or []; sec_depths.append(len(sp))
            text_chunks.append({"text": c.get("text") or c.get("raw_chunk",""), "kind":"text", "page":c.get("page"), "section_path":sp})

    # (1) 표 재선형화 — 파이프라인 row parser(TATR-grid/word-clustering) 행단위 마크다운.
    #     canonical 붕괴/재무행 필터 없이 모든 행(회사|값들) 보존.
    import adaptive_table_router as atr, run_table_metadata_pipeline as rtmp
    from adaptive_table_router import RouterThresholds, detect_and_route
    from row_parser import parse_table_adaptive, parse_simple_table_from_words
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    atr.PDF_PATH = PDF; rtmp.PDF_PATH = PDF
    doc=fitz.open(str(PDF))
    t=time.time()
    tatr_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition"); tatr_model.eval()
    tatr_proc = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    pdf_pp = pdfplumber.open(str(PDF))
    routed = detect_and_route(RouterThresholds(), yolo_model=yolo, page_boxes=page_boxes, pdf_pp=pdf_pp)
    SCALE=150/72
    table_chunks=[]; n_tables=0; n_rows=0
    for r in routed:
        page_pp=pdf_pp.pages[r["page"]-1]
        median_lh=r["median_line_height_pt"]
        x1,y1,x2,y2=r["bbox_px"]; bbox_pt=(x1/SCALE,y1/SCALE,x2/SCALE,y2/SCALE)
        try:
            if r["complexity"]=="simple":
                rows=parse_simple_table_from_words(page_pp, bbox_pt, median_lh)
            else:
                rows=parse_table_adaptive(tatr_model, tatr_proc, doc, page_pp, r["page"], bbox_pt,
                                          rtmp.TATR_DPI, rtmp.TATR_TOP_PAD_PT, rtmp.TATR_SIDE_PAD_PT, median_lh)
        except Exception:
            rows=[]
        md=[]
        for row in rows:
            lab=(row.get("label") or "").strip()
            cells=[str(c).strip() for c in (row.get("cells") or [])]
            if lab or any(cells):
                md.append("| "+lab+" | "+" | ".join(cells)+" |")
        if not md: continue
        n_tables+=1; n_rows+=len(md)
        table_chunks.append({"text": f"[표 p{r['page']} #{r['table_idx']}]\n"+"\n".join(md),
                              "kind":"table", "page":r["page"], "table_idx":r["table_idx"]})
    table_s=time.time()-t

    # (2) 캡션 보호 — 원문에서 도표/표 캡션 회수
    t=time.time()
    caption_lines=[]
    for i in range(doc.page_count):
        for ln in doc[i].get_text().splitlines():
            if CAP_RE.match(ln): caption_lines.append(ln.strip())
    # dedup 유지
    seen=set(); caps=[]
    for c in caption_lines:
        if c not in seen: seen.add(c); caps.append(c)
    caption_chunk = {"text": "[캡션 목록]\n"+"\n".join(caps), "kind":"caption"}
    cap_s=time.time()-t

    # (3) 완전성 레이어 — 정제된 페이지 원문(행순서·완전)을 청크로 보강.
    #     계층청킹이 흘리는 본문 문장(T10)·TATR이 자르는 표 우측열(F04)을 검색 recall용으로 회수.
    #     구조 청크(table_chunks)는 그대로 두고 "누락 방지" 목적의 별도 레이어.
    t=time.time()
    pagetext_chunks=[]
    for pg in tp["pages"]:
        txt=(pg.get("text") or "").strip()
        if not txt: continue
        for j in range(0,len(txt),1500):
            pagetext_chunks.append({"text":txt[j:j+1500],"kind":"pagetext","page":pg["page"]})
    pagetext_s=time.time()-t

    # (4) 라우팅 오버라이드 — 하드 판정이 하단 컴플라이언스 2단 보일러플레이트 때문이면 취소.
    COMPLI=("Compliance Notice","투자등급 관련사항","투자의견의 유효기간","금융투자상품의 비율")
    raw_hard=list(tp.get("hard_page_numbers") or [])
    kept_hard=[]
    for hp in raw_hard:
        raw=doc[hp-1].get_text()
        is_boiler = sum(1 for m in COMPLI if m in raw) >= 2 and hp >= int(doc.page_count*2/3)
        if not is_boiler: kept_hard.append(hp)
    route_to_mineru = bool(kept_hard)

    all_chunks = text_chunks + table_chunks + pagetext_chunks + [caption_chunk]
    full_text = "\n".join(c["text"] for c in all_chunks)
    capcount = C.count_captions(full_text)

    out = {
        "axis":"enhanced_v3",
        "parse_time_s": round(classify_s+text_s+table_s+cap_s+pagetext_s,3),
        "stage_timing":{"model_load_warmup_s":round(load_s,3),"page_classify_s":round(classify_s,3),
                         "text_route_s":round(text_s,3),"table_pdfplumber_s":round(table_s,3),
                         "caption_recover_s":round(cap_s,3),"pagetext_layer_s":round(pagetext_s,3)},
        "total_time_s": round(load_s+classify_s+text_s+table_s+cap_s+pagetext_s,3),
        "n_chunks": len(all_chunks), "n_text_chunks":len(text_chunks), "n_table_chunks":len(table_chunks),
        "n_pagetext_chunks": len(pagetext_chunks),
        "chunks": all_chunks, "full_text": full_text,
        "structure":{"n_tables_detected":n_tables,"n_table_rows":n_rows,
                      "chart_titles_preserved":len(capcount["chart_titles"]),"table_caps_preserved":len(capcount["table_caps"]),
                      "avg_section_path_depth":round(sum(sec_depths)/max(1,len(sec_depths)),2)},
        "page_pred": page_pred,
        "routing":{"route_to_mineru":route_to_mineru,"hard_page_numbers":kept_hard,
                    "n_hard_pages":len(kept_hard),"raw_hard_before_override":raw_hard,
                    "routing_note":"컴플라이언스 보일러플레이트 하드 판정 오버라이드 적용"},
    }
    C.dump_json(OUT,out)
    print(f"[enhanced_v3] load {load_s:.1f}s classify {classify_s:.1f}s text {text_s:.1f}s "
          f"table {table_s:.1f}s caption {cap_s:.1f}s pagetext {pagetext_s:.1f}s -> parse {out['parse_time_s']}s")
    print(f"  chunks={len(all_chunks)} (text {len(text_chunks)} + table {len(table_chunks)} + pagetext {len(pagetext_chunks)}) tables={n_tables}")
    print(f"  charts={len(capcount['chart_titles'])}/93 tabcaps={len(capcount['table_caps'])}/11 hard(raw)={raw_hard} -> hard(override)={kept_hard}")

if __name__=="__main__":
    main()

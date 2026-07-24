# -*- coding: utf-8 -*-
"""다문서 파싱: 3개 하나증권 보고서 × {baseline, enhanced_v3, docling, mineru}.
모드: python multidoc_parse.py [fast|docling|mineru]  (fast=baseline+enhanced_v3)
출력: multidoc/out_{doc}_{axis}.json"""
import sys, time, os, subprocess, glob, re
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP=ROOT/"pdf_pipeline"
for d in [str(HERE),str(PP),str(PP/"page_classification"),str(PP/"text_processing"),str(PP/"table_processing")]:
    sys.path.insert(0,d)
import common_exp as C
import fitz, pdfplumber
MD=HERE/"multidoc"
DOCS=[("doc1_construction",MD/"doc1_construction.pdf"),
      ("doc2_pharma",MD/"doc2_pharma.pdf"),
      ("doc3_preview",MD/"doc3_preview.pdf"),
      ("doc4_retail",MD/"doc4_retail.pdf"),
      ("doc5_steel",MD/"doc5_steel.pdf"),
      ("doc06_aircraft",MD/"doc06_aircraft.pdf"),
      ("doc07_mart",MD/"doc07_mart.pdf"),
      ("doc08_solar",MD/"doc08_solar.pdf"),
      ("doc09_bank",MD/"doc09_bank.pdf"),
      ("doc10_food",MD/"doc10_food.pdf"),
      ("doc11_delivery",MD/"doc11_delivery.pdf"),
      ("doc12_tourism",MD/"doc12_tourism.pdf"),
      ("doc13_construction2",MD/"doc13_construction2.pdf"),
      ("doc14_semi",MD/"doc14_semi.pdf"),
      ("doc15_earnings",MD/"doc15_earnings.pdf")]
CAP_RE=re.compile(r"^\s*(도표|표)\s*\d+\.?\s*.*$")

def naive_table_stats(pdf):
    n=0;rows=0
    with pdfplumber.open(str(pdf)) as p:
        for pg in p.pages:
            for tb in pg.extract_tables(): n+=1; rows+=len(tb) if tb else 0
    return n,rows

def parse_baseline(pdf):
    t=time.time(); d=fitz.open(str(pdf)); pages=[p.get_text() for p in d]; parse_s=time.time()-t
    full="\n".join(pages); caps=C.count_captions(full)
    nt,nr=naive_table_stats(pdf)
    return {"axis":"baseline","parse_time_s":round(parse_s,3),"chunks":C.chunk_pages_raw(pages),
            "full_text":full,"structure":{"n_tables":nt,"n_rows":nr,
            "chart_titles":len(caps["chart_titles"]),"table_caps":len(caps["table_caps"])}}

def parse_docling(pdf, conv):
    t=time.time(); res=conv.convert(str(pdf)); s=time.time()-t
    md=res.document.export_to_markdown(); caps=C.count_captions(md)
    ntab=len(getattr(res.document,"tables",[]) or [])
    return {"axis":"docling","parse_time_s":round(s,3),"chunks":C.chunk_markdown(md),"full_text":md,
            "structure":{"n_tables":ntab,"chart_titles":len(caps["chart_titles"]),"table_caps":len(caps["table_caps"]),"md_chars":len(md)}}

def parse_mineru(pdf, docid):
    outdir=MD/f"mineru_{docid}"; outdir.mkdir(exist_ok=True)
    env=dict(os.environ); env["MINERU_MODEL_SOURCE"]="huggingface"
    exe=r"c:/Users/wodlf/OneDrive/Desktop/pdfex/demo_venv/Scripts/mineru.exe"
    t=time.time()
    r=subprocess.run([exe,"-p",str(pdf),"-o",str(outdir),"-b","pipeline","-l","korean"],env=env,capture_output=True,text=True)
    s=time.time()-t
    mds=[f for f in glob.glob(str(outdir/"**"/"*.md"),recursive=True) if not f.endswith("_content_list.md")]
    if not mds: return {"axis":"mineru","parse_time_s":round(s,3),"chunks":[],"full_text":"","structure":{"error":"no md","rc":r.returncode}}
    md=Path(max(mds,key=os.path.getsize)).read_text(encoding="utf-8"); caps=C.count_captions(md)
    ntab=md.count("<table")+sum(1 for ln in md.splitlines() if ln.lstrip().startswith("|") and "---" in ln)
    return {"axis":"mineru","parse_time_s":round(s,3),"chunks":C.chunk_markdown(md),"full_text":md,
            "structure":{"n_tables":ntab,"chart_titles":len(caps["chart_titles"]),"table_caps":len(caps["table_caps"]),"md_chars":len(md)}}

_NUM=re.compile(r"^[\d\.,%\-\(\)\s]+$")
def _ts_gate(tbl):
    """text-strategy 출력이 '깨끗한 정형표'인가 판정. clean(정형/수치) -> text-strategy, 불규칙텍스트 -> False."""
    if not tbl or len(tbl)<3: return False
    from collections import Counter
    counts=[sum(1 for c in row if c and str(c).strip()) for row in tbl]; counts=[c for c in counts if c>0]
    if not counts: return False
    modal=Counter(counts).most_common(1)[0][0]; cons=sum(1 for c in counts if c==modal)/len(counts)
    cells=[str(c).strip() for row in tbl for c in row if c and str(c).strip()]
    numr=sum(1 for c in cells if _NUM.match(c))/max(1,len(cells))
    return (cons>=0.5 and modal>=2) or numr>=0.55

def parse_enhanced_v5(pdf, yolo):
    from page_classifier import classify_pdf
    from text_extraction import process_pdf
    import adaptive_table_router as atr, run_table_metadata_pipeline as rtmp
    from row_parser import parse_simple_table_from_words
    atr.PDF_PATH=pdf; rtmp.PDF_PATH=pdf
    t0=time.time()
    cls=classify_pdf(str(pdf),yolo); classify_s=time.time()-t0
    pb={p["page"]:p["cached_boxes"] for p in cls["pages"]}
    t=time.time()
    tp=process_pdf(str(pdf),yolo,page_boxes=pb,chunk_backend="rulebased",remove_boilerplate=True,add_structured_metadata=False)
    text_s=time.time()-t
    text_chunks=[]
    for pg in tp["pages"]:
        for c in pg.get("chunks",[]):
            text_chunks.append({"text":c.get("text") or c.get("raw_chunk",""),"kind":"text","page":c.get("page")})
    # v4 tables via pdfplumber text-strategy (구조화 레코드 완전 + TATR 제거)
    t=time.time(); doc=fitz.open(str(pdf)); pdf_pp=pdfplumber.open(str(pdf))
    TS={"vertical_strategy":"text","horizontal_strategy":"text"}; table_chunks=[]; ntab=0
    for p in cls["pages"]:
        pg=p["page"]; tboxes=[rect for (cn,rect) in pb[pg] if cn=="Table"]
        if not tboxes: continue
        page_pp=pdf_pp.pages[pg-1]
        for ti,rect in enumerate(tboxes,1):
            x0=max(0,rect.x0);y0=max(0,rect.y0);x1=min(page_pp.width,rect.x1);y1=min(page_pp.height,rect.y1)
            if x1-x0<5 or y1-y0<5: continue
            rows_md=[]; used="none"
            try: tbl=page_pp.within_bbox((x0,y0,x1,y1)).extract_table(TS)
            except Exception: tbl=None
            if _ts_gate(tbl):  # 깨끗한 정형/수치표 -> text-strategy(컬럼 완전)
                used="text-strategy"
                for row in tbl:
                    cells=[("" if c is None else str(c).replace("\n"," ").strip()) for c in row]
                    if any(cells): rows_md.append("| "+" | ".join(cells)+" |")
            else:  # 불규칙 표 -> word-clustering(이름 보존)
                used="word-cluster"
                try:
                    for row in parse_simple_table_from_words(page_pp,(x0,y0,x1,y1),12.0):
                        lab=(row.get("label") or "").strip(); cells=[str(c).strip() for c in (row.get("cells") or [])]
                        if lab or any(cells): rows_md.append("| "+lab+" | "+" | ".join(cells)+" |")
                except Exception: pass
                if not rows_md and tbl:  # word-cluster 실패시 text-strategy라도
                    for row in tbl:
                        cells=[("" if c is None else str(c).strip()) for c in row]
                        if any(cells): rows_md.append("| "+" | ".join(cells)+" |")
            if not rows_md: continue
            ntab+=1; table_chunks.append({"text":f"[표 p{pg} #{ti}]\n"+"\n".join(rows_md),"kind":"table","page":pg,"parser":used})
    table_s=time.time()-t
    # captions
    t=time.time(); caplines=[]
    for i in range(doc.page_count):
        for ln in doc[i].get_text().splitlines():
            if CAP_RE.match(ln): caplines.append(ln.strip())
    seen=set(); caps=[c for c in caplines if not (c in seen or seen.add(c))]
    caption_chunk={"text":"[캡션 목록]\n"+"\n".join(caps),"kind":"caption"}
    # pagetext completeness layer
    pagetext_chunks=[]
    for pg in tp["pages"]:
        txt=(pg.get("text") or "").strip()
        for j in range(0,len(txt),1500):
            if txt[j:j+1500].strip(): pagetext_chunks.append({"text":txt[j:j+1500],"kind":"pagetext","page":pg["page"]})
    # routing override
    COMPLI=("Compliance Notice","투자등급 관련사항","투자의견의 유효기간","금융투자상품의 비율")
    raw_hard=list(tp.get("hard_page_numbers") or []); kept=[hp for hp in raw_hard
        if not (sum(1 for m in COMPLI if m in doc[hp-1].get_text())>=2 and hp>=int(doc.page_count*2/3))]
    cap_s=time.time()-t
    all_chunks=text_chunks+table_chunks+pagetext_chunks+[caption_chunk]
    full="\n".join(c["text"] for c in all_chunks); cc=C.count_captions(full)
    return {"axis":"enhanced_v5","parse_time_s":round(classify_s+text_s+table_s+cap_s,3),
            "stage":{"classify":round(classify_s,2),"text":round(text_s,2),"table":round(table_s,2),"cap":round(cap_s,2)},
            "chunks":all_chunks,"full_text":full,
            "structure":{"n_tables":ntab,"chart_titles":len(cc["chart_titles"]),"table_caps":len(cc["table_caps"])},
            "routing":{"raw_hard":raw_hard,"hard":kept,"route_to_mineru":bool(kept)}}

def save(docid,o): C.dump_json(MD/f"out_{docid}_{o['axis']}.json",o)
def exists(docid,axis): return (MD/f"out_{docid}_{axis}.json").exists()

def main():
    mode=sys.argv[1] if len(sys.argv)>1 else "fast"
    if mode=="fast":
        from ultralytics import YOLO
        yolo=YOLO(str(PP/"page_classification"/"models"/"yolo11n_doc_layout.pt"))
        yolo.predict(__import__("numpy").zeros((640,640,3),dtype="uint8"),verbose=False)
        for docid,pdf in DOCS:
            if not exists(docid,"baseline"):
                b=parse_baseline(pdf); save(docid,b); print(f"[{docid}] baseline {b['parse_time_s']}s",flush=True)
            e=parse_enhanced_v5(pdf,yolo); save(docid,e); print(f"[{docid}] enhanced_v5 {e['parse_time_s']}s tables {e['structure']['n_tables']} caps {e['structure']['chart_titles']}+{e['structure']['table_caps']} hard {e['routing']['raw_hard']}->{e['routing']['hard']}",flush=True)
    elif mode=="docling":
        from docling.document_converter import DocumentConverter
        conv=DocumentConverter()
        for docid,pdf in DOCS:
            if exists(docid,"docling"): print(f"[{docid}] docling skip(exists)",flush=True); continue
            d=parse_docling(pdf,conv); save(docid,d); print(f"[{docid}] docling {d['parse_time_s']}s",flush=True)
    elif mode=="mineru":
        for docid,pdf in DOCS:
            if exists(docid,"mineru"): print(f"[{docid}] mineru skip(exists)",flush=True); continue
            m=parse_mineru(pdf,docid); save(docid,m); print(f"[{docid}] mineru {m['parse_time_s']}s",flush=True)

if __name__=="__main__": main()

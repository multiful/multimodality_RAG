import sys
from pathlib import Path
PP=Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG/pdf_pipeline")
for d in [str(PP),str(PP/"page_classification"),str(PP/"text_processing"),str(PP/"table_processing")]: sys.path.insert(0,d)
import fitz, pdfplumber
from ultralytics import YOLO
import adaptive_table_router as atr, run_table_metadata_pipeline as rtmp
from adaptive_table_router import RouterThresholds, detect_and_route
from row_parser import parse_table_adaptive, parse_simple_table_from_words
from transformers import AutoImageProcessor, AutoModelForObjectDetection
PDF=PP/"reference"/"SmartPhone"/"20260629_industry_47868000.pdf"
atr.PDF_PATH=PDF; rtmp.PDF_PATH=PDF
yolo=YOLO(str(PP/"page_classification"/"models"/"yolo11n_doc_layout.pt"))
from page_classifier import classify_pdf
cls=classify_pdf(str(PDF),yolo); pb={p["page"]:p["cached_boxes"] for p in cls["pages"]}
tm=AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition"); tm.eval()
tpq=AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
doc=fitz.open(str(PDF)); pdf_pp=pdfplumber.open(str(PDF))
routed=detect_and_route(RouterThresholds(),yolo_model=yolo,page_boxes=pb,pdf_pp=pdf_pp)
SCALE=150/72
TARGETS={"479.0":"DRAM매출2026F","734.5":"반도체매출2027F","505.1":"Memory매출2026F"}
for r in routed:
    if r["page"]!=23: continue
    page_pp=pdf_pp.pages[22]; x1,y1,x2,y2=r["bbox_px"]; bp=(x1/SCALE,y1/SCALE,x2/SCALE,y2/SCALE)
    print(f"\n== p23 table{r['table_idx']} complexity={r['complexity']} ==")
    # A) 현재 방식(TATR-grid via adaptive)
    rowsA=parse_table_adaptive(tm,tpq,doc,page_pp,23,bp,rtmp.TATR_DPI,rtmp.TATR_TOP_PAD_PT,rtmp.TATR_SIDE_PAD_PT,r["median_line_height_pt"])
    ncolA=max((len(row.get('cells') or []) for row in rowsA), default=0)
    allA=" ".join(str(c) for row in rowsA for c in (row.get('cells') or []))
    # B) word-clustering 대체
    rowsB=parse_simple_table_from_words(page_pp,bp,r["median_line_height_pt"])
    ncolB=max((len(row.get('cells') or []) for row in rowsB), default=0)
    allB=" ".join(str(c) for row in rowsB for c in (row.get('cells') or []))
    # C) pdfplumber text-strategy
    try:
        crop=page_pp.within_bbox((bp[0],bp[1],bp[2],bp[3]))
        tblC=crop.extract_table({"vertical_strategy":"text","horizontal_strategy":"text"})
        allC=" ".join(str(c) for row in (tblC or []) for c in row if c)
        ncolC=max((len(row) for row in (tblC or [])), default=0)
    except Exception as e:
        allC=""; ncolC=0
    print(f"  A TATR-grid:      maxcols={ncolA:2d}  " + " ".join(f"{v}:{'O' if k in allA else 'X'}" for k,v in TARGETS.items()))
    print(f"  B word-cluster:   maxcols={ncolB:2d}  " + " ".join(f"{v}:{'O' if k in allB else 'X'}" for k,v in TARGETS.items()))
    print(f"  C pdfplumber-text:maxcols={ncolC:2d}  " + " ".join(f"{v}:{'O' if k in allC else 'X'}" for k,v in TARGETS.items()))

import sys, re
from pathlib import Path
PP = Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG/pdf_pipeline")
for d in [str(PP),str(PP/"page_classification"),str(PP/"text_processing"),str(PP/"table_processing")]:
    sys.path.insert(0,d)
import fitz, pdfplumber
from ultralytics import YOLO
PDF=PP/"reference"/"SmartPhone"/"20260629_industry_47868000.pdf"
doc=fitz.open(str(PDF)); yolo=YOLO(str(PP/"page_classification"/"models"/"yolo11n_doc_layout.pt"))

# ---- T10: p2 raw vs cleaned ----
print("===== T10: p2 raw text around 'ON Semiconductor' =====")
raw2=doc[1].get_text()
i=raw2.find("ON Semiconductor")
print("RAW has 'ON Semiconductor'? ", i>=0, "| ctx:", repr(raw2[i-8:i+35]) if i>=0 else "-")
from page_classifier import classify_pdf
from text_extraction import process_pdf
cls=classify_pdf(str(PDF),yolo); pb={p["page"]:p["cached_boxes"] for p in cls["pages"]}
tp=process_pdf(str(PDF),yolo,page_boxes=pb,chunk_backend="rulebased",remove_boilerplate=True,add_structured_metadata=False)
p2=[pg for pg in tp["pages"] if pg["page"]==2][0]
p2txt=p2.get("text","")
print("CLEANED p2 'ON Semiconductor'? ", "ON Semiconductor" in p2txt)
print("CLEANED p2 'ON Semi' variants:", [m for m in ["ON Semi","onsemi","ON Semiconductor"] if m in p2txt])
j=p2txt.find("26% 하락")
print("CLEANED ctx near '26% 하락':", repr(p2txt[max(0,j-40):j+10]) if j>=0 else "not found")
# also YOLO boxes on p2 (is thesis text region a Text box or excluded?)
print("p2 YOLO classes:", sorted(set(cn for cn,_ in pb[2])))

# ---- F04: p23 parsed rows (section header structure) ----
print("\n===== F04: p23 parsed table rows =====")
import adaptive_table_router as atr, run_table_metadata_pipeline as rtmp
from adaptive_table_router import RouterThresholds, detect_and_route
from row_parser import parse_table_adaptive, parse_simple_table_from_words
from transformers import AutoImageProcessor, AutoModelForObjectDetection
atr.PDF_PATH=PDF; rtmp.PDF_PATH=PDF
tm=AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition"); tm.eval()
tp2=AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
pdf_pp=pdfplumber.open(str(PDF))
routed=detect_and_route(RouterThresholds(),yolo_model=yolo,page_boxes=pb,pdf_pp=pdf_pp)
SCALE=150/72
for r in routed:
    if r["page"]!=23: continue
    page_pp=pdf_pp.pages[22]; x1,y1,x2,y2=r["bbox_px"]; bp=(x1/SCALE,y1/SCALE,x2/SCALE,y2/SCALE)
    if r["complexity"]=="simple": rows=parse_simple_table_from_words(page_pp,bp,r["median_line_height_pt"])
    else: rows=parse_table_adaptive(tm,tp2,doc,page_pp,23,bp,rtmp.TATR_DPI,rtmp.TATR_TOP_PAD_PT,rtmp.TATR_SIDE_PAD_PT,r["median_line_height_pt"])
    print(f"-- p23 table{r['table_idx']} ({r['complexity']}) {len(rows)} rows --")
    for row in rows[:16]:
        print("   label=%-14r cells=%s" % (row.get('label'), (row.get('cells') or [])[:6]))
    # is 479.0 anywhere?
    allc=" ".join(str(c) for row in rows for c in (row.get('cells') or []))
    print("   479.0 in cells?", "479.0" in allc, "| DRAM label row?", any('DRAM' in (row.get('label') or '') for row in rows))

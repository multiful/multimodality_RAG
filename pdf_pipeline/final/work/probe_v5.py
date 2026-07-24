import sys, glob
from pathlib import Path
from collections import Counter
PP=Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG/pdf_pipeline")
for d in [str(PP),str(PP/"page_classification"),str(PP/"table_processing")]: sys.path.insert(0,d)
import pdfplumber
from ultralytics import YOLO
from page_classifier import classify_pdf
from row_parser import parse_simple_table_from_words
yolo=YOLO(str(PP/"page_classification"/"models"/"yolo11n_doc_layout.pt"))
TS={"vertical_strategy":"text","horizontal_strategy":"text"}
MD=Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG/pdf_pipeline/final/work/multidoc")

def ts_quality(tbl):
    if not tbl or len(tbl)<3: return 0.0,0
    counts=[sum(1 for c in row if c and str(c).strip()) for row in tbl]
    counts=[c for c in counts if c>0]
    if not counts: return 0.0,0
    modal=Counter(counts).most_common(1)[0][0]
    consistency=sum(1 for c in counts if c==modal)/len(counts)
    return consistency, modal

def gate(tbl):
    cons,modal=ts_quality(tbl)
    return (len(tbl or [])>=3 and cons>=0.5 and modal>=2), cons, modal

# test tables: SmartPhone p1/p2/p23 (clean/wide), pharma p7 (irregular)
tests=[("SmartPhone",PP/"reference"/"SmartPhone"/"20260629_industry_47868000.pdf",[1,2,23]),
       ("pharma",MD/"doc2_pharma.pdf",[7])]
import fitz
for name,pdf,pages in tests:
    cls=classify_pdf(str(pdf),yolo); pb={p["page"]:p["cached_boxes"] for p in cls["pages"]}
    ppp=pdfplumber.open(str(pdf))
    for pg in pages:
        for cn,rect in pb[pg]:
            if cn!="Table": continue
            page_pp=ppp.pages[pg-1]
            x0,y0,x1,y1=max(0,rect.x0),max(0,rect.y0),min(page_pp.width,rect.x1),min(page_pp.height,rect.y1)
            if x1-x0<5 or y1-y0<5: continue
            try: tbl=page_pp.within_bbox((x0,y0,x1,y1)).extract_table(TS)
            except: tbl=None
            use_ts,cons,modal=gate(tbl)
            # word-cluster fallback preview
            wc=parse_simple_table_from_words(page_pp,(x0,y0,x1,y1),12.0)
            wctext=" ".join((r.get("label") or "")+" "+" ".join(str(c) for c in (r.get("cells") or [])) for r in wc)
            print(f"[{name} p{pg}] gate={'text-strategy' if use_ts else 'WORD-CLUSTER'} (cons={cons:.2f} modal={modal})")
            if name=="pharma":
                print("   에이비엘바이오 in word-cluster?", "에이비엘바이오" in wctext.replace(' ',''))
            break  # first table per page

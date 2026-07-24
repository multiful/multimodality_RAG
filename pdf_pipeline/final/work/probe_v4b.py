import sys
from pathlib import Path
PP=Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG/pdf_pipeline")
for d in [str(PP),str(PP/"page_classification"),str(PP/"table_processing")]: sys.path.insert(0,d)
import pdfplumber
from ultralytics import YOLO
import adaptive_table_router as atr
from adaptive_table_router import RouterThresholds, detect_and_route
PDF=PP/"reference"/"SmartPhone"/"20260629_industry_47868000.pdf"
atr.PDF_PATH=PDF
yolo=YOLO(str(PP/"page_classification"/"models"/"yolo11n_doc_layout.pt"))
from page_classifier import classify_pdf
cls=classify_pdf(str(PDF),yolo); pb={p["page"]:p["cached_boxes"] for p in cls["pages"]}
pdf_pp=pdfplumber.open(str(PDF)); SCALE=150/72
routed=detect_and_route(RouterThresholds(),yolo_model=yolo,page_boxes=pb,pdf_pp=pdf_pp)
def show(pg,tidx):
    for r in routed:
        if r["page"]==pg and r["table_idx"]==tidx:
            page_pp=pdf_pp.pages[pg-1]; x1,y1,x2,y2=r["bbox_px"]; bp=(x1/SCALE,y1/SCALE,x2/SCALE,y2/SCALE)
            crop=page_pp.within_bbox((bp[0],bp[1],bp[2],bp[3]))
            tbl=crop.extract_table({"vertical_strategy":"text","horizontal_strategy":"text"})
            print(f"\n== p{pg} t{tidx} text-strategy: {len(tbl or [])} rows ==")
            for row in (tbl or [])[:6]:
                print("  |", " | ".join((c or "").replace("\n"," ")[:12] for c in row))
            # SK하이닉스 8.7 & 삼성전자 7.6 정렬 확인
            flat=[[(c or "").strip() for c in row] for row in (tbl or [])]
            for row in flat:
                if any("하이닉스" in c for c in row): print("  SK하이닉스 row:", row)
                if any(c=="삼성전자" for c in row): print("  삼성전자 row:", row)
show(1,1)  # 표1 밸류에이션
show(2,1)  # 표2 필라델피아

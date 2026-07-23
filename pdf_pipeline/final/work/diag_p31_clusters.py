import sys
from pathlib import Path
PP = Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG/pdf_pipeline")
for d in [str(PP), str(PP/"page_classification"), str(PP/"text_processing")]:
    sys.path.insert(0, d)
import fitz
from ultralytics import YOLO
from reading_order_router import _cluster_by_x0, _is_excluded, ColumnRouterThresholds, NON_TEXT_CLASSES, run_yolo_layout
th = ColumnRouterThresholds()
PDF = PP/"reference"/"SmartPhone"/"20260629_industry_47868000.pdf"
doc = fitz.open(str(PDF)); page = doc[30]
model = YOLO(str(PP/"page_classification"/"models"/"yolo11n_doc_layout.pt"))
boxes = run_yolo_layout(model, page, 30)
excl = [r for c,r in boxes if c in NON_TEXT_CLASSES]
tb = []
for b in page.get_text("blocks"):
    if b[6]!=0 or not b[4].strip(): continue
    if _is_excluded(fitz.Rect(b[:4]), excl): continue
    tb.append(b)
clusters = _cluster_by_x0(tb, th.gap_pt)
print("clusters:", len(clusters))
for ci,cl in enumerate(clusters):
    x0s=[round(b[0]) for b in cl]; chars=sum(len(b[4]) for b in cl)
    ys=[round(min(b[1] for b in cl)),round(max(b[3] for b in cl))]
    print(f"\n--- cluster {ci}: {len(cl)} blocks, {chars} chars, x0~{min(x0s)}-{max(x0s)}, y {ys} ---")
    txt=" ".join(b[4].replace(chr(10),' ') for b in cl)
    print("  ", txt[:240].strip())

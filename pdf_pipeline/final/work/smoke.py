import sys, time
from pathlib import Path
ROOT = Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG")
PP = ROOT/"pdf_pipeline"
for d in [PP, PP/"page_classification", PP/"text_processing", PP/"table_processing"]:
    sys.path.insert(0, str(d))
PDF = PP/"reference"/"SmartPhone"/"20260629_industry_47868000.pdf"

from ultralytics import YOLO
t=time.time(); model = YOLO(str(PP/"page_classification"/"models"/"yolo11n_doc_layout.pt")); print("YOLO load %.2fs classes=%s"%(time.time()-t, model.names))

from page_classifier import classify_pdf
t=time.time(); cls = classify_pdf(str(PDF), model); print("classify_pdf %.2fs npages=%d"%(time.time()-t, cls["n_pages"]))
for p in cls["pages"][:8]:
    nb = len(p.get("cached_boxes") or [])
    print("  p%02d text=%s table=%s image=%s boxes=%d"%(p["page"], p["has_text"], p["has_table"], p["has_image"], nb))
# import the heavy stage modules (no run yet)
import importlib
for m in ["text_extraction","run_table_metadata_pipeline","index_text","structured_output"]:
    try:
        importlib.import_module(m); print("import OK:", m)
    except Exception as e:
        print("import FAIL:", m, "->", type(e).__name__, str(e)[:150])

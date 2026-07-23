import sys, json
from pathlib import Path
PP = Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG/pdf_pipeline")
for d in [str(PP), str(PP/"page_classification"), str(PP/"text_processing"), str(PP/"table_processing")]:
    sys.path.insert(0, d)
import fitz
from ultralytics import YOLO
from reading_order_router import assess_page_difficulty, ColumnRouterThresholds
th = ColumnRouterThresholds()
print("THRESHOLDS: min_chars_for_hard=%s min_interleaving_excess=%s gap_pt=%s min_y_overlap_pt=%s"
      % (th.min_chars_for_hard, th.min_interleaving_excess, th.gap_pt, th.min_y_overlap_pt))
PDF = PP/"reference"/"SmartPhone"/"20260629_industry_47868000.pdf"
doc = fitz.open(str(PDF))
model = YOLO(str(PP/"page_classification"/"models"/"yolo11n_doc_layout.pt"))
for pidx in [22,23,26,28,29,30]:  # 0-indexed => p23,24,27,29,30,31
    r = assess_page_difficulty(model, doc, pidx)
    mo = r.material_overlaps
    print(f"\n== PAGE {pidx+1} => {r.difficulty.upper()} ==")
    print(f"  n_text_blocks={r.n_text_blocks} n_clusters={r.n_clusters} "
          f"material_overlaps={len(mo)} interleaving_excess={r.interleaving_excess} score={r.difficulty_score}")
    print(f"  signals={r.signals}")
    for m in mo:
        print(f"    overlap pair {m['cluster_pair']} y_overlap={m['y_overlap_pt']}pt chars={m['chars']}")
    print("  reason:", r.reason)
# render p31
pix = doc[30].get_pixmap(dpi=130)
outp = PP/"final"/"work"/"page31.png"; pix.save(str(outp))
print("\nrendered", outp)

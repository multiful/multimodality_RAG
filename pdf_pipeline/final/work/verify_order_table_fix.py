# -*- coding: utf-8 -*-
"""수주공시 표 수정 end-to-end 검증 — build_records -> from_table_records 직렬화까지."""
import os, sys, time
from pathlib import Path
ROOT = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP = ROOT / "pdf_pipeline"
for d in [str(PP), str(PP/"table_processing")]: sys.path.insert(0, d)
for line in open(ROOT/".env", encoding="utf-8"):
    line=line.strip()
    if line and "=" in line and not line.startswith("#"): k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip())
PDF = PP/"reference"/"Construct"/"20260721_industry_362851000.pdf"
import table_processing.adaptive_table_router as art
import table_processing.run_table_metadata_pipeline as rtmp
import entity_fusion
art.PDF_PATH = PDF; rtmp.PDF_PATH = PDF

t = time.time()
recs, _n_fin, _n_cid = rtmp.build_records("Construct", add_structured_metadata=False)
print(f"\n[build_records] {len(recs)}행 / {time.time()-t:.1f}s")
p5 = [r for r in recs if r.get("page") == 5]
print(f"[p5] {len(p5)}행, column_headers={p5[0].get('column_headers') if p5 else None}")
items = entity_fusion.from_table_records("Construct", p5)
print(f"\n=== p5 evidence content (수정 후) ===")
for it in items: print("  ", it["content"][:150])

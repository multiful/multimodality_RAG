# -*- coding: utf-8 -*-
"""스키마 수정 A/B — Field(description) 부여 + 표 문서컨텍스트(캡션/제목) 주입 + temperature=0이
표 구조화 출력의 실제 채움률을 바꾸는가. 지표: entities_mentioned 비율, time_periods_covered
쓰레기값 비율, table_title 환각(캡션에 없는 제목) 비율."""
import os, sys, json, time
from pathlib import Path
ROOT = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP = ROOT / "pdf_pipeline"
for d in [str(PP), str(PP/"table_processing")]: sys.path.insert(0, d)
for line in open(ROOT/".env", encoding="utf-8"):
    line = line.strip()
    if line and "=" in line and not line.startswith("#"): k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
PDF = PP/"reference"/"Construct"/"20260721_industry_362851000.pdf"
OUT = PP/"final"/"results_schema_fix.json"
import table_processing.adaptive_table_router as art
import table_processing.run_table_metadata_pipeline as rtmp
art.PDF_PATH = PDF; rtmp.PDF_PATH = PDF

t = time.time()
recs, _, _ = rtmp.build_records("Construct", add_structured_metadata=True, sector="건설")
dt = time.time() - t
metas = [r for r in recs if r.get("record_type") == "table_metadata"]
filled = [m for m in metas if m.get("entities_mentioned")]
tp_junk = [m for m in metas if any(len(str(x)) <= 2 for x in (m.get("time_periods_covered") or []))]
res = {"latency_s": round(dt, 1), "n_tables": len(metas),
       "entities_filled": len(filled), "entities_fill_rate": round(len(filled)/max(1, len(metas)), 3),
       "time_period_junk_tables": len(tp_junk),
       "tables": [{"page": m["page"], "title": m.get("table_title"),
                   "type": m.get("table_type_refined"),
                   "entities": m.get("entities_mentioned"),
                   "periods": m.get("time_periods_covered"),
                   "finding": (m.get("notable_finding") or "")[:80]} for m in metas]}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n=== 표 구조화출력 {len(metas)}개 / {dt:.1f}s ===")
print(f"entities_mentioned 채워진 표: {len(filled)}/{len(metas)} ({res['entities_fill_rate']:.0%})")
for m in res["tables"]:
    print(f"  p{m['page']} [{m['type']}] {str(m['title'])[:30]!r} ents={m['entities'][:6]} periods={m['periods'][:4]}")

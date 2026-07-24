# -*- coding: utf-8 -*-
"""수정된 파이프라인(수주표 TATR 승격 + 컬럼헤더 직렬화 + 스키마 description/컨텍스트)으로
Construct 문서를 재인제스트 — 이후 검색/생성 평가는 전부 이 인덱스 위에서 수행."""
import os, sys, time
from pathlib import Path
ROOT = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP = ROOT / "pdf_pipeline"
for d in [str(PP), str(PP/"text_processing")]: sys.path.insert(0, d)
for line in open(ROOT/".env", encoding="utf-8"):
    line = line.strip()
    if line and "=" in line and not line.startswith("#"): k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
PDF = PP/"reference"/"Construct"/"20260721_industry_362851000.pdf"

import psycopg2, entity_fusion
# 기존 Construct evidence 제거(같은 pdf_id로 중복 적재 방지)
c = psycopg2.connect(os.environ["SUPABASE_DIRECT_DB_URL"]); cur = c.cursor()
cur.execute("delete from document_evidence where pdf_id='Construct'")
print(f"[clean] 기존 Construct evidence {cur.rowcount}건 삭제"); c.commit(); c.close()
entity_fusion.invalidate_evidence_cache(pdf_id="Construct")

import run_investment_opinion_demo as demo
t = time.time()
demo.main(pdf_path=PDF, pdf_id="Construct", query="이 리포트의 핵심 투자 인사이트를 도출해줘",
          add_structured_metadata=True, sector="건설", verbose=True, gen_model="gpt-4o-mini")
print(f"\n[재인제스트 완료] {time.time()-t:.1f}s")

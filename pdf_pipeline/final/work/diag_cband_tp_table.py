# -*- coding: utf-8 -*-
"""[재일] C밴드 투자의견 변동 이력표(TP 표) 진단 — 사용자 제보(스텝차트 축 오독으로 TP 추세가
정반대로 조작됨) 후속. 두 가지를 분리해서 확인한다:
  (1) 표 브랜치가 이 표를 아예 못 뽑는가 (추출 실패)
  (2) 뽑기는 하는데 검색 top-k에 못 드는가 (랭킹 실패)
OpenAI 호출 없이 동작한다(구조화 출력 off + 로컬 BGE/BM25만 사용)."""
import os, sys, json
from pathlib import Path
ROOT = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP = ROOT / "pdf_pipeline"
for d in [str(PP), str(PP/"table_processing"), str(PP/"text_processing")]: sys.path.insert(0, d)
for line in open(ROOT/".env", encoding="utf-8"):
    line = line.strip()
    if line and "=" in line and not line.startswith("#"): k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
PDF = PP/"reference"/"C밴드"/"c밴드.pdf"
OUT = PP/"final"/"results_cband_tp_diag.json"

import table_processing.adaptive_table_router as art
import table_processing.run_table_metadata_pipeline as rtmp
import entity_fusion
art.PDF_PATH = PDF; rtmp.PDF_PATH = PDF

def main():
    recs, _, _ = rtmp.build_records("CBand", add_structured_metadata=False)
    rows = [r for r in recs if r.get("record_type") != "table_metadata"]
    print(f"\n[build_records] 전체 {len(rows)}행")
    for p in range(1, 7):
        pr = [r for r in rows if r.get("page") == p]
        hdrs = {tuple(r.get("column_headers") or []) for r in pr}
        print(f"  p{p}: {len(pr)}행, 회수된 컬럼헤더 {[h for h in hdrs if h]}")

    # TP 표 행이 실제로 있는지 — 70,000 / 26.5.19 같은 정답 토큰으로 확인
    print("\n=== 정답 토큰이 담긴 행 ===")
    for tok in ["26.5.19", "70,000", "26.4.14", "25,000"]:
        hit = [r for r in rows if tok in f"{r.get('raw_label','')} {r.get('cells')}"]
        print(f"  '{tok}': {len(hit)}행")
        for r in hit[:3]:
            print(f"      p{r['page']} label={r.get('raw_label')!r} cells={r.get('cells')} hdr={r.get('column_headers')}")

    items = entity_fusion.from_table_records("CBand", rows)
    digests = [i for i in items if (i["metadata"] or {}).get("granularity") == "table_digest"]
    print(f"\n=== evidence 아이템 {len(items)}개 (행 {len(items)-len(digests)} + 표요약 {len(digests)}) ===")
    for i in items:
        c = i["content"]
        if "26.5.19" in c or "26.4.14" in c:
            print(f"  [{(i['metadata'] or {}).get('granularity') or 'row'}/p{i['page']}] {c[:220]}")
    OUT.write_text(json.dumps({
        "n_rows": len(rows), "n_items": len(items), "n_digests": len(digests),
        "sample_tp_items": [i["content"][:400] for i in items if "26.5.19" in i["content"]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT.name} 저장")

if __name__ == "__main__":
    main()

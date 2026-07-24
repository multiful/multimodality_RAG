# -*- coding: utf-8 -*-
"""[재일] C밴드 TP표 '추출 실패 vs 랭킹 실패' 판별 — 표+텍스트+이미지 근거로 인메모리 융합 인덱스를
만들고 회사명 지정 질의를 던진다. 로컬 BGE-m3-ko + BM25만 쓰므로 OpenAI 쿼터와 무관하게 실행된다."""
import os, sys, json
from pathlib import Path
ROOT = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP = ROOT / "pdf_pipeline"
for d in [str(PP), str(PP/"table_processing"), str(PP/"text_processing")]: sys.path.insert(0, d)
for line in open(ROOT/".env", encoding="utf-8"):
    line = line.strip()
    if line and "=" in line and not line.startswith("#"): k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
PDF = PP/"reference"/"C밴드"/"c밴드.pdf"
OUT = PP/"final"/"results_cband_retrieval.json"

import table_processing.adaptive_table_router as art
import table_processing.run_table_metadata_pipeline as rtmp
import entity_fusion
from text_extraction import process_pdf
art.PDF_PATH = PDF; rtmp.PDF_PATH = PDF

QUERIES = [
    ("Q1", "케이엠더블유의 2026년 5월 19일자 목표주가는 얼마야?", ["70,000"]),
    ("Q2", "케이엠더블유 목표주가 변동 이력 알려줘", ["26.5.19", "26.3.17"]),
    ("Q3", "우리넷 목표주가 추이가 어떻게 돼?", ["26.4.14", "26.3.25"]),
    ("Q4", "케이엠더블유 투자의견과 목표주가 괴리율", ["-40.20%", "26.3.17"]),
]

def main():
    recs, _, _ = rtmp.build_records("CBand", add_structured_metadata=False)
    rows = [r for r in recs if r.get("record_type") != "table_metadata"]
    # 텍스트 브랜치 대신, 이미 인제스트된 실제 문서의 text/image 청크를 DB에서 가져와 같은
    # 경쟁 조건을 만든다(차트 카드가 경쟁자로 들어가야 이번 사례가 재현된다).
    import psycopg2
    conn = psycopg2.connect(os.environ["SUPABASE_DIRECT_DB_URL"]); cur = conn.cursor()
    cur.execute("""select source_type,page,content from document_evidence
                   where pdf_id='upload_44b76ed9' and source_type in ('text','image')""")
    others = [{"id": f"o{i}", "pdf_id": "CBand", "source_type": st, "page": pg,
               "content": ct, "weight": 1.0 if st == "text" else 1.1, "metadata": {}}
              for i, (st, pg, ct) in enumerate(cur.fetchall())]
    conn.close()
    items = entity_fusion.from_table_records("CBand", rows) + others
    items, emb = entity_fusion.embed_items(items)
    index = entity_fusion.build_index_from_items("CBand", items, emb)
    print(f"[index] {len(index.chunks)}청크 (text {sum(1 for c in index.chunks if c.get('source_type')=='text')} / "
          f"table {sum(1 for c in index.chunks if c.get('source_type')=='table')})")

    out = {}
    for qid, q, gold in QUERIES:
        hits = entity_fusion.weighted_hybrid_search(index, q, top_k=8)
        rank = next((i+1 for i, h in enumerate(hits)
                     if any(g in (h["chunk"]["content"] or "") for g in gold)), None)
        print(f"\n[{qid}] {q}\n   정답토큰={gold} -> 정답 순위: {rank if rank else '**top-8 밖(실패)**'}")
        for i, h in enumerate(hits[:5], 1):
            c = (h["chunk"]["content"] or "").replace("\n", " ")
            mark = " <<GOLD" if any(g in (h["chunk"]["content"] or "") for g in gold) else ""
            print(f"     {i}. [{h['source_type']}/p{h['chunk'].get('page')}] {h['score']:.3f} {c[:95]}{mark}")
        out[qid] = {"query": q, "gold": gold, "rank": rank,
                    "top5": [{"src": h["source_type"], "page": h["chunk"].get("page"),
                              "score": round(h["score"], 3), "content": (h["chunk"]["content"] or "")[:200]}
                             for h in hits[:5]]}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT.name} 저장")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""축1 docling only -> markdown -> chunk -> RAG."""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_exp as C

PDF = Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG/pdf_pipeline/reference/SmartPhone/20260629_industry_47868000.pdf")
OUT = HERE / "out_docling.json"

def main():
    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()
    t = time.time()
    res = conv.convert(str(PDF))
    convert_s = time.time() - t
    doc = res.document
    md = doc.export_to_markdown()
    (HERE/"docling.md").write_text(md, encoding="utf-8")

    n_tables = len(getattr(doc, "tables", []) or [])
    n_table_rows = 0
    for tb in (getattr(doc, "tables", []) or []):
        try:
            n_table_rows += tb.data.num_rows
        except Exception:
            pass
    chunks = C.chunk_markdown(md)
    caps = C.count_captions(md)
    out = {
        "axis": "docling",
        "parse_time_s": round(convert_s, 3),
        "stage_timing": {"convert_s": round(convert_s, 3)},
        "total_time_s": round(convert_s, 3),
        "n_chunks": len(chunks),
        "chunks": chunks,
        "full_text": md,
        "structure": {
            "n_tables_detected": n_tables,
            "n_table_rows": n_table_rows,
            "chart_titles_preserved": len(caps["chart_titles"]),
            "table_caps_preserved": len(caps["table_caps"]),
            "md_chars": len(md),
        },
        "page_pred": None,
        "routing": None,
    }
    C.dump_json(OUT, out)
    print(f"[docling] convert {convert_s:.1f}s md_chars={len(md)} chunks={len(chunks)} "
          f"tables={n_tables} rows={n_table_rows} charts={len(caps['chart_titles'])}/93 tabcaps={len(caps['table_caps'])}/11")

if __name__ == "__main__":
    main()

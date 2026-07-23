# -*- coding: utf-8 -*-
"""축2 MinerU only -> markdown -> chunk -> RAG. MinerU CLI(subprocess) 사용."""
import sys, time, os, subprocess, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_exp as C

PDF = Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG/pdf_pipeline/reference/SmartPhone/20260629_industry_47868000.pdf")
OUTDIR = HERE / "mineru_out"
OUT = HERE / "out_mineru.json"
MINERU_EXE = r"c:/Users/wodlf/OneDrive/Desktop/pdfex/demo_venv/Scripts/mineru.exe"

def main():
    OUTDIR.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["MINERU_MODEL_SOURCE"] = "huggingface"
    cmd = [MINERU_EXE, "-p", str(PDF), "-o", str(OUTDIR), "-b", "pipeline", "-l", "korean"]
    t = time.time()
    print("[mineru] running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    convert_s = time.time() - t
    print("[mineru] rc=", r.returncode, "elapsed %.1fs" % convert_s, flush=True)
    if r.returncode != 0:
        print("STDERR tail:\n", (r.stderr or "")[-2000:], flush=True)

    md_files = glob.glob(str(OUTDIR / "**" / "*.md"), recursive=True)
    md_files = [f for f in md_files if not f.endswith("_content_list.md")]
    if not md_files:
        print("[mineru] NO markdown produced. stdout tail:\n", (r.stdout or "")[-1500:])
        return
    md_path = max(md_files, key=lambda f: os.path.getsize(f))
    md = Path(md_path).read_text(encoding="utf-8")
    print("[mineru] md:", md_path, "chars=", len(md))

    n_tables = md.count("<table") + sum(1 for ln in md.splitlines() if ln.lstrip().startswith("|") and "---" in ln)
    chunks = C.chunk_markdown(md)
    caps = C.count_captions(md)
    out = {
        "axis": "mineru",
        "parse_time_s": round(convert_s, 3),
        "stage_timing": {"convert_s": round(convert_s, 3)},
        "total_time_s": round(convert_s, 3),
        "n_chunks": len(chunks),
        "chunks": chunks,
        "full_text": md,
        "structure": {
            "n_tables_detected": n_tables,
            "chart_titles_preserved": len(caps["chart_titles"]),
            "table_caps_preserved": len(caps["table_caps"]),
            "md_chars": len(md),
            "md_path": md_path,
        },
        "page_pred": None,
        "routing": None,
    }
    C.dump_json(OUT, out)
    print(f"[mineru] convert {convert_s:.1f}s md_chars={len(md)} chunks={len(chunks)} "
          f"tables~{n_tables} charts={len(caps['chart_titles'])}/93 tabcaps={len(caps['table_caps'])}/11")

if __name__ == "__main__":
    main()

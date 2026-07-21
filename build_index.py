"""Build one Chroma collection per embedder from data/corpus/*.jsonl.

Each corpus file holds one JSON object per line:
    {"id": "AAPL_financial_2026Q1", "text": "...", "ticker": "AAPL", "type": "financial"}

Run once per embedder change, or whenever the corpus is updated:
    python build_index.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from embeddings import BGEEmbedder, BGEKoEmbedder, GPTEmbedder
from vector_store import VectorStore

CORPUS_DIR = Path("data/corpus")


def load_corpus() -> list[dict]:
    files = sorted(CORPUS_DIR.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No corpus files found in {CORPUS_DIR}/. "
            "Add *.jsonl files with {'id', 'text', 'ticker', 'type'} per line first."
        )
    records = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            records.extend(json.loads(line) for line in fh if line.strip())
    return records


def build(embedder, records: list[dict]):
    store = VectorStore(embedder)
    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    metadatas = [{"ticker": r["ticker"], "type": r["type"]} for r in records]
    store.add(ids=ids, texts=texts, metadatas=metadatas)
    print(f"[{embedder.name}] indexed {len(records)} chunks")


def main():
    load_dotenv()
    records = load_corpus()
    for embedder in (BGEEmbedder(), BGEKoEmbedder(), GPTEmbedder()):
        build(embedder, records)


if __name__ == "__main__":
    main()

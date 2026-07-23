"""Run zero-shot RAG generation over data/eval/eval_queries.csv with both generators,
using the same retrieved context and the same RAG-IT-style prompt template for each.

Retrieval uses the bge-m3 index (best MRR in the earlier embedding comparison).
Requires build_index.py to have been run first, and Ollama running for Qwen
(`ollama pull qwen2.5:7b`).

Usage:
    python run_generation.py --top-k 3
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from embeddings import BGEEmbedder
from generation import GPTGenerator, QwenGenerator
from vector_store import VectorStore

EVAL_CSV = "data/eval/eval_queries.csv"
OUT_PATH = Path("data/eval/generation_results.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    load_dotenv()
    df = pd.read_csv(EVAL_CSV)
    store = VectorStore(BGEEmbedder())

    # One batched embedding call for retrieval context across all queries.
    retrieval = store.query_batch(df["query"].tolist(), top_k=args.top_k)
    contexts = ["\n\n".join(docs) for docs in retrieval["documents"]]

    generators = [GPTGenerator(), QwenGenerator()]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for (_, row), context in zip(df.iterrows(), contexts):
            for gen in generators:
                print(f"[{gen.name}] {row['ticker']} / {row['query_type']}")
                answer = gen.generate(row["query"], context)
                record = {
                    "id": int(row["id"]),
                    "ticker": row["ticker"],
                    "query_type": row["query_type"],
                    "query": row["query"],
                    "model": gen.name,
                    "answer": answer,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"saved generation results to {OUT_PATH}")


if __name__ == "__main__":
    main()

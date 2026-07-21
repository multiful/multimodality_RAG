"""Compare BGE-M3 vs BGE-m3-ko vs GPT text-embedding-3-small retrieval quality on data/eval/eval_queries.csv.

Requires:
  1. data/corpus/*.jsonl built and indexed via build_index.py
  2. eval_queries.csv's gold_chunk_id column filled in (rows with it blank are skipped)

All queries for a given embedder are embedded in a single batched call (one embedding
request + one Chroma query), not one call per row.

Usage:
    python evaluate.py --top-k 5
"""

import argparse

import pandas as pd
from dotenv import load_dotenv

from embeddings import BGEEmbedder, BGEKoEmbedder, GPTEmbedder
from vector_store import VectorStore

EVAL_CSV = "data/eval/eval_queries.csv"
RESULTS_CSV = "data/eval/results.csv"


def recall_and_rr(retrieved_ids: list[str], gold_id: str) -> tuple[int, float]:
    hit = 1 if gold_id in retrieved_ids else 0
    rr = 1.0 / (retrieved_ids.index(gold_id) + 1) if hit else 0.0
    return hit, rr


def evaluate(embedder, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    store = VectorStore(embedder)
    result = store.query_batch(df["query"].tolist(), top_k=top_k)

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        retrieved_ids = result["ids"][i]
        hit, rr = recall_and_rr(retrieved_ids, row["gold_chunk_id"])
        rows.append({"query_type": row["query_type"], "hit": hit, "rr": rr})
    return pd.DataFrame(rows)


def summary_rows(embedder_name: str, scored: pd.DataFrame, top_k: int) -> list[dict]:
    rows = [{
        "model": embedder_name,
        "query_type": "overall",
        f"recall@{top_k}": scored["hit"].mean(),
        "mrr": scored["rr"].mean(),
    }]
    for qtype, group in scored.groupby("query_type"):
        rows.append({
            "model": embedder_name,
            "query_type": qtype,
            f"recall@{top_k}": group["hit"].mean(),
            "mrr": group["rr"].mean(),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    top_k = args.top_k

    load_dotenv()
    df = pd.read_csv(EVAL_CSV)
    df = df[df["gold_chunk_id"].notna() & (df["gold_chunk_id"] != "")]
    if df.empty:
        raise ValueError(
            f"No rows with gold_chunk_id in {EVAL_CSV}. Fill in gold labels before evaluating."
        )

    all_rows = []
    for embedder in (BGEEmbedder(), BGEKoEmbedder(), GPTEmbedder()):
        print(f"evaluating {embedder.name} ...")
        scored = evaluate(embedder, df, top_k)
        all_rows.extend(summary_rows(embedder.name, scored, top_k))

    results = pd.DataFrame(all_rows)
    table = results.pivot(index="query_type", columns="model", values=[f"recall@{top_k}", "mrr"])
    print("\n" + table.round(3).to_string())

    results.to_csv(RESULTS_CSV, index=False)
    print(f"\nsaved raw results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

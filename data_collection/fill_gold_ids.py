"""One-off: fill data/eval/eval_queries.csv's gold_chunk_id from ticker + query_type.

Works because the corpus scheme gives exactly one chunk per (ticker, query_type):
  {ticker}_financial, {ticker}_news, {ticker}_price, {ticker}_trend

Usage:
    python data_collection/fill_gold_ids.py
"""

import csv
from pathlib import Path

EVAL_CSV = Path("data/eval/eval_queries.csv")


def main():
    with EVAL_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["gold_chunk_id"] = f"{row['ticker']}_{row['query_type']}"

    with EVAL_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"filled gold_chunk_id for {len(rows)} rows in {EVAL_CSV}")


if __name__ == "__main__":
    main()

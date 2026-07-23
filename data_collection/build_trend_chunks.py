"""Combine financial + price + news chunks per ticker into one trend-summary chunk.

Run after fetch_financials.py, fetch_price.py, fetch_news.py.
Writes data/corpus/trend.jsonl.

Usage:
    python data_collection/build_trend_chunks.py
"""

import json
from pathlib import Path

CORPUS_DIR = Path("data/corpus")


def load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[rec["ticker"]] = rec
    return records


def main():
    financial = load_jsonl(CORPUS_DIR / "financial.jsonl")
    price = load_jsonl(CORPUS_DIR / "price.jsonl")
    news = load_jsonl(CORPUS_DIR / "news.jsonl")

    tickers = sorted(set(financial) | set(price) | set(news))
    chunks = []
    for ticker in tickers:
        parts = [
            records[ticker]["text"]
            for records in (financial, price, news)
            if ticker in records
        ]
        if not parts:
            continue
        chunks.append({
            "id": f"{ticker}_trend",
            "ticker": ticker,
            "type": "trend",
            "text": "\n".join(parts),
        })

    out_path = CORPUS_DIR / "trend.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(chunks)} trend chunks to {out_path}")


if __name__ == "__main__":
    main()

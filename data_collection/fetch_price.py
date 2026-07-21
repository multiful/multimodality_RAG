"""Fetch 1-month stock price trend per ticker via yfinance (free, no API key).

Writes one chunk per ticker to data/corpus/price.jsonl.

Usage:
    python data_collection/fetch_price.py
"""

import json
from pathlib import Path

import yfinance as yf

from tickers import TICKERS

OUT_PATH = Path("data/corpus/price.jsonl")


def build_price_chunk(ticker: str, info: dict) -> dict | None:
    hist = yf.Ticker(ticker).history(period="1mo")
    if hist.empty:
        print(f"  [skip] {ticker}: no price history")
        return None

    start_price = hist["Close"].iloc[0]
    end_price = hist["Close"].iloc[-1]
    pct_change = (end_price - start_price) / start_price * 100
    direction = "상승" if pct_change >= 0 else "하락"
    start_date = hist.index[0].date()
    end_date = hist.index[-1].date()

    text = (
        f"{info['name_ko']}({ticker})의 최근 1개월({start_date}~{end_date}) 주가는 "
        f"${start_price:.2f}에서 ${end_price:.2f}로 {abs(pct_change):.1f}% {direction}했습니다."
    )
    return {"id": f"{ticker}_price", "ticker": ticker, "type": "price", "text": text}


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for ticker, info in TICKERS.items():
        print(f"fetching price: {ticker}")
        chunk = build_price_chunk(ticker, info)
        if chunk:
            chunks.append(chunk)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(chunks)} price chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()

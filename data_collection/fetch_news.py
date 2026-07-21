"""Fetch recent news headlines per ticker via Yahoo Finance RSS (free, no API key).

Digests the latest headlines into one chunk per ticker, written to data/corpus/news.jsonl.

Usage:
    python data_collection/fetch_news.py
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from tickers import TICKERS

OUT_PATH = Path("data/corpus/news.jsonl")
RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; likelion-p2-project/1.0)"}
TOP_N = 5


def fetch_headlines(ticker: str) -> list[dict]:
    resp = requests.get(RSS_URL.format(ticker=ticker), headers=HEADERS, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    items = root.findall("./channel/item")[:TOP_N]
    headlines = []
    for item in items:
        title = (item.findtext("title") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        if title:
            headlines.append({"title": title, "pub_date": pub_date})
    return headlines


def build_news_chunk(ticker: str, info: dict) -> dict | None:
    headlines = fetch_headlines(ticker)
    if not headlines:
        print(f"  [skip] {ticker}: no headlines found")
        return None

    lines = [f"- {h['title']} ({h['pub_date']})" for h in headlines]
    text = f"{info['name_ko']}({ticker}) 관련 최근 뉴스 헤드라인:\n" + "\n".join(lines)
    return {"id": f"{ticker}_news", "ticker": ticker, "type": "news", "text": text}


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for ticker, info in TICKERS.items():
        print(f"fetching news: {ticker}")
        chunk = build_news_chunk(ticker, info)
        if chunk:
            chunks.append(chunk)
        time.sleep(0.3)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(chunks)} news chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()

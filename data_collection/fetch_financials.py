"""Fetch latest-quarter revenue / operating income / net income from SEC EDGAR (free, no API key).

Writes one chunk per ticker to data/corpus/financial.jsonl.

Usage:
    python data_collection/fetch_financials.py
"""

import json
import time
from datetime import date
from pathlib import Path

import requests

from tickers import TICKERS

HEADERS = {"User-Agent": "likelion-p2-project contact@example.com"}
OUT_PATH = Path("data/corpus/financial.jsonl")

# SEC XBRL tags vary by company; try each in order until one has data.
REVENUE_TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"]
OPERATING_INCOME_TAGS = ["OperatingIncomeLoss"]
NET_INCOME_TAGS = ["NetIncomeLoss"]


def latest_quarterly_value(facts_usgaap: dict, tags: list[str]) -> dict | None:
    """Pick the most recent single-quarter (~90 day) 10-Q/10-K datapoint, skipping 6/9-month cumulative ones.

    Companies sometimes switch which XBRL tag they file revenue under over time, so a single tag's
    "most recent" entry can be stale. Pool candidates across all given tags first, then pick globally
    most recent by period end date.
    """
    quarterly = []
    for tag in tags:
        concept = facts_usgaap.get(tag)
        if not concept:
            continue
        items = concept.get("units", {}).get("USD", [])
        for item in items:
            if item.get("form") not in ("10-Q", "10-K"):
                continue
            start = date.fromisoformat(item["start"])
            end = date.fromisoformat(item["end"])
            days = (end - start).days
            if 80 <= days <= 100:
                quarterly.append(item)
    if not quarterly:
        return None
    quarterly.sort(key=lambda x: x["end"], reverse=True)
    return quarterly[0]


def build_financial_chunk(ticker: str, info: dict) -> dict | None:
    cik = str(info["cik"]).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    usgaap = resp.json().get("facts", {}).get("us-gaap", {})

    revenue = latest_quarterly_value(usgaap, REVENUE_TAGS)
    operating_income = latest_quarterly_value(usgaap, OPERATING_INCOME_TAGS)
    net_income = latest_quarterly_value(usgaap, NET_INCOME_TAGS)
    if not revenue:
        print(f"  [skip] {ticker}: no quarterly revenue found")
        return None

    text = (
        f"{info['name_ko']}({ticker})의 {revenue['fy']} {revenue['fp']} 분기 실적: "
        f"매출 ${revenue['val'] / 1e9:.2f}B"
    )
    if operating_income:
        text += f", 영업이익 ${operating_income['val'] / 1e9:.2f}B"
    if net_income:
        text += f", 순이익 ${net_income['val'] / 1e9:.2f}B"
    text += f" (SEC EDGAR 10-Q 기준, {revenue['end']} 마감 분기)."

    return {"id": f"{ticker}_financial", "ticker": ticker, "type": "financial", "text": text}


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for ticker, info in TICKERS.items():
        print(f"fetching financials: {ticker}")
        chunk = build_financial_chunk(ticker, info)
        if chunk:
            chunks.append(chunk)
        time.sleep(0.2)  # stay well under SEC's rate limit

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(chunks)} financial chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()

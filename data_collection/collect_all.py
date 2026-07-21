"""Run the full corpus collection: financials -> price -> news -> trend digest.

Usage:
    python data_collection/collect_all.py
"""

import build_trend_chunks
import fetch_financials
import fetch_news
import fetch_price


def main():
    fetch_financials.main()
    fetch_price.main()
    fetch_news.main()
    build_trend_chunks.main()


if __name__ == "__main__":
    main()

"""KOSPI200 구성종목의 재무제표(손익계산서/대차대조표/현금흐름표)를 yfinance로 가져와
종목별 마크다운 파일로 저장한다.

기존 미국 종목용 파이프라인(src/finance/yahoo_financials.py)을 그대로 재사용하고,
KOSPI200 구성종목 조회(data_collection/kospi200_tickers.py)만 새로 추가한 것이다.
KRX 6자리 종목코드에 ".KS"를 붙이면 yfinance/Yahoo Finance 티커가 된다.

Usage:
    python data_collection/fetch_kospi200_financials.py
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_collection.kospi200_tickers import get_kospi200_tickers
from src.finance.yahoo_financials import YahooFinancialStatementFetcher

OUT_DIR = REPO_ROOT / "output" / "kospi200_financials"


def main():
    tickers = get_kospi200_tickers()
    print(f"KOSPI200 구성종목 {len(tickers)}개 로드")

    saved, skipped = 0, []
    for info in tickers:
        yahoo_ticker = info["yahoo_ticker"]
        print(f"fetching: {yahoo_ticker} ({info['name_ko']})")
        try:
            fetcher = YahooFinancialStatementFetcher(yahoo_ticker)
            fetcher.fetch_and_save_markdown(OUT_DIR)
            saved += 1
        except ValueError as exc:
            print(f"  [skip] {yahoo_ticker}: {exc}")
            skipped.append(yahoo_ticker)
        time.sleep(0.5)  # yfinance 요청 과다 방지

    print(f"완료: {saved}개 저장 -> {OUT_DIR}")
    if skipped:
        print(f"스킵된 {len(skipped)}개 티커 (재무데이터 없음): {', '.join(skipped)}")


if __name__ == "__main__":
    main()

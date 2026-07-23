"""KOSPI200 구성종목의 기업 프로필(Yahoo Finance Profile 탭)을 yfinance로 가져와
종목별 마크다운 파일로 저장한다.

fetch_kospi200_financials.py와 동일한 패턴으로, 종목 조회는
data_collection/kospi200_tickers.py, 프로필 조회/직렬화는
src/finance/yahoo_profile.py를 재사용한다.

Usage:
    python data_collection/fetch_kospi200_profiles.py
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_collection.kospi200_tickers import get_kospi200_tickers
from src.finance.yahoo_profile import YahooProfileFetcher

OUT_DIR = REPO_ROOT / "output" / "kospi200_profiles"


def main():
    tickers = get_kospi200_tickers()
    print(f"KOSPI200 구성종목 {len(tickers)}개 로드")

    saved, skipped = 0, []
    for info in tickers:
        yahoo_ticker = info["yahoo_ticker"]
        print(f"fetching: {yahoo_ticker} ({info['name_ko']})")
        try:
            fetcher = YahooProfileFetcher(yahoo_ticker)
            fetcher.fetch_and_save_markdown(OUT_DIR)
            saved += 1
        except ValueError as exc:
            print(f"  [skip] {yahoo_ticker}: {exc}")
            skipped.append(yahoo_ticker)
        time.sleep(0.5)  # yfinance 요청 과다 방지

    print(f"완료: {saved}개 저장 -> {OUT_DIR}")
    if skipped:
        print(f"스킵된 {len(skipped)}개 티커 (프로필 데이터 없음): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
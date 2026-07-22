"""KOSPI200 구성종목 목록을 pykrx(KRX 공식 데이터)로 가져온다.

pykrx 1.2.x부터는 KRX 데이터 조회에 data.krx.co.kr 회원 로그인이 필요하다.
.env에 KRX_ID, KRX_PW를 설정해야 하며, pykrx 내부 함수들이 이 값을 모듈 임포트
시점에 기본값으로 캐싱하므로 반드시 `from pykrx import stock`보다 먼저
`load_dotenv()`를 호출해야 한다.

Usage:
    python data_collection/kospi200_tickers.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()  # pykrx가 KRX_ID/KRX_PW를 읽기 전에 .env를 먼저 로드해야 한다.

from pykrx import stock as krx  # noqa: E402

KOSPI200_INDEX_NAME = "코스피 200"


def _latest_deposit_file(index_code: str, lookback_days: int = 10) -> list[str]:
    """휴장일에는 당일 구성종목 파일이 비어 있을 수 있어, 최근 영업일까지 거슬러 올라가며 조회한다."""
    day = datetime.now()
    for _ in range(lookback_days):
        date_str = day.strftime("%Y%m%d")
        codes = krx.get_index_portfolio_deposit_file(index_code, date_str)
        if codes:
            return codes
        day -= timedelta(days=1)
    raise RuntimeError(f"최근 {lookback_days}일 내 KOSPI200 구성종목을 조회하지 못했습니다.")


def get_kospi200_tickers() -> list[dict]:
    """KOSPI200 구성종목을 [{code, name_ko, yahoo_ticker}, ...] 형태로 반환한다."""
    index_codes = krx.get_index_ticker_list(market="KOSPI")
    kospi200_code = next(
        (code for code in index_codes if krx.get_index_ticker_name(code) == KOSPI200_INDEX_NAME),
        None,
    )
    if kospi200_code is None:
        raise RuntimeError(f"'{KOSPI200_INDEX_NAME}' 지수 코드를 찾지 못했습니다.")

    stock_codes = _latest_deposit_file(kospi200_code)
    return [
        {"code": code, "name_ko": krx.get_market_ticker_name(code), "yahoo_ticker": f"{code}.KS"}
        for code in stock_codes
    ]


if __name__ == "__main__":
    tickers = get_kospi200_tickers()
    print(f"KOSPI200 구성종목 {len(tickers)}개")
    for t in tickers:
        print(f"{t['yahoo_ticker']}\t{t['name_ko']}")

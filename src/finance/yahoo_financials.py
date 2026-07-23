"""Nasdaq/미국 상장 기업 재무제표를 yfinance로 직접 가져와 마크다운으로 변환한다.

기존 파이프라인(PDF 다운로드 -> 문서 변환 -> 마크다운)은 yfinance가 이미
pandas DataFrame으로 제공하는 손익계산서/대차대조표/현금흐름표를 다시
PDF로 렌더링한 뒤 파싱하는 셈이라 비효율적이다. 이 모듈은 그 중간 단계를
건너뛰고 yfinance의 구조화된 응답을 곧바로 RAG 인덱싱용 마크다운으로
직렬화한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# yfinance Ticker 객체에서 재무제표 3종을 (연간, 분기) 속성명으로 매핑
_STATEMENT_ATTRS: dict[str, dict[str, str]] = {
    "income_statement": {"annual": "income_stmt", "quarterly": "quarterly_income_stmt"},
    "balance_sheet": {"annual": "balance_sheet", "quarterly": "quarterly_balance_sheet"},
    "cash_flow": {"annual": "cashflow", "quarterly": "quarterly_cashflow"},
}

_STATEMENT_TITLES: dict[str, str] = {
    "income_statement": "손익계산서 (Income Statement)",
    "balance_sheet": "대차대조표 (Balance Sheet)",
    "cash_flow": "현금흐름표 (Cash Flow)",
}


def _format_cell(value: object) -> str:
    """과학적 표기법(1.4e+11) 대신 사람이 읽기 쉬운 형태로 변환한다."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.2f}"
    return str(value)


@dataclass
class FinancialStatementBundle:
    """한 종목의 재무제표 3종 x (연간/분기) 데이터를 담는 컨테이너."""

    ticker: str
    fetched_at: datetime
    statements: dict[str, dict[str, pd.DataFrame]]  # {statement_type: {period_type: df}}

    def to_markdown(self) -> str:
        lines = [f"# {self.ticker} 재무제표", f"_fetched: {self.fetched_at.isoformat()}_", ""]

        for stmt_type, periods in self.statements.items():
            lines.append(f"## {_STATEMENT_TITLES[stmt_type]}")
            for period_type, df in periods.items():
                if df.empty:
                    continue
                label = "연간" if period_type == "annual" else "분기"
                display = df.map(_format_cell)
                display.columns = [
                    col.strftime("%Y-%m-%d") if isinstance(col, pd.Timestamp) else str(col)
                    for col in df.columns
                ]
                lines.append(f"### {label}")
                # disable_numparse: tabulate가 콤마 포함 숫자 문자열을 다시 float으로
                # 파싱해 과학적 표기법으로 되돌리는 것을 막는다.
                lines.append(display.to_markdown(disable_numparse=True))
                lines.append("")

        return "\n".join(lines)


class YahooFinancialStatementFetcher:
    """yfinance를 통해 미국/나스닥 상장사 재무제표를 구조화된 형태로 가져온다."""

    def __init__(self, ticker: str):
        self.ticker_symbol = ticker.upper()
        self._ticker = yf.Ticker(self.ticker_symbol)

    def fetch(self, include_quarterly: bool = True) -> FinancialStatementBundle:
        statements: dict[str, dict[str, pd.DataFrame]] = {}

        for stmt_type, attrs in _STATEMENT_ATTRS.items():
            periods = {"annual": getattr(self._ticker, attrs["annual"])}
            if include_quarterly:
                periods["quarterly"] = getattr(self._ticker, attrs["quarterly"])
            statements[stmt_type] = {k: (df if df is not None else pd.DataFrame()) for k, df in periods.items()}

        if all(df.empty for periods in statements.values() for df in periods.values()):
            raise ValueError(f"'{self.ticker_symbol}'에 대한 재무제표를 찾을 수 없습니다. 티커를 확인하세요.")

        return FinancialStatementBundle(
            ticker=self.ticker_symbol,
            fetched_at=datetime.now(),
            statements=statements,
        )

    def fetch_and_save_markdown(self, output_dir: str | Path, include_quarterly: bool = True) -> Path:
        bundle = self.fetch(include_quarterly=include_quarterly)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{self.ticker_symbol}_financials.md"
        out_path.write_text(bundle.to_markdown(), encoding="utf-8")
        return out_path


def fetch_multiple(tickers: list[str], output_dir: str | Path, include_quarterly: bool = True) -> list[Path]:
    """여러 티커의 재무제표를 순회하며 마크다운으로 저장하고 저장 경로 목록을 반환한다."""
    saved_paths = []
    for ticker in tickers:
        fetcher = YahooFinancialStatementFetcher(ticker)
        saved_paths.append(fetcher.fetch_and_save_markdown(output_dir, include_quarterly=include_quarterly))
    return saved_paths


_TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,10}$")


if __name__ == "__main__":
    import sys

    raw_args = sys.argv[1:] or ["AAPL", "MSFT"]
    target_tickers = [t for t in raw_args if _TICKER_PATTERN.match(t.upper())]
    invalid_args = [t for t in raw_args if t not in target_tickers]

    if invalid_args:
        print(f"티커 형식이 아니라 무시한 인자: {invalid_args}")
    if not target_tickers:
        print("사용법: python3 -m src.finance.yahoo_financials AAPL MSFT ...")
        sys.exit(1)

    for ticker in target_tickers:
        try:
            fetcher = YahooFinancialStatementFetcher(ticker)
            saved_path = fetcher.fetch_and_save_markdown("./output/financials")
            print(f"saved: {saved_path}")
        except ValueError as exc:
            print(f"skip {ticker}: {exc}")

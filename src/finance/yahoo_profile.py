"""Yahoo Finance의 Profile 탭(https://finance.yahoo.com/quote/{ticker}/profile/)에
해당하는 기업 개요 정보를 yfinance로 가져와 마크다운으로 변환한다.

yahoo_financials.py(재무제표)와 동일한 패턴으로, yfinance Ticker.info가 제공하는
구조화된 응답을 곧바로 RAG 인덱싱용 마크다운으로 직렬화한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yfinance as yf

_ADDRESS_FIELDS = ["address1", "address2", "address3", "city", "state", "zip", "country"]


def _format_officer_pay(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:,.0f}"
    return str(value)


@dataclass
class CompanyProfileBundle:
    """한 종목의 기업 프로필 정보를 담는 컨테이너."""

    ticker: str
    fetched_at: datetime
    info: dict

    def to_markdown(self) -> str:
        info = self.info
        name = info.get("longName") or info.get("shortName") or self.ticker
        lines = [f"# {name} ({self.ticker}) 기업 프로필", f"_fetched: {self.fetched_at.isoformat()}_", ""]

        address = ", ".join(str(info[f]) for f in _ADDRESS_FIELDS if info.get(f))

        lines.append("## 기본 정보")
        basic_fields = [
            ("섹터", info.get("sector")),
            ("산업", info.get("industry")),
            ("웹사이트", info.get("website")),
            ("주소", address),
            ("전화", info.get("phone")),
            ("직원 수", info.get("fullTimeEmployees")),
        ]
        for label, value in basic_fields:
            if value:
                lines.append(f"- {label}: {value}")
        lines.append("")

        summary = info.get("longBusinessSummary")
        if summary:
            lines.append("## 사업 개요")
            lines.append(summary)
            lines.append("")

        officers = info.get("companyOfficers") or []
        if officers:
            lines.append("## 주요 임원 (Key Executives)")
            lines.append("| 이름 | 직책 | 나이 | 총 보수 |")
            lines.append("|---|---|---|---|")
            for officer in officers:
                name_ = officer.get("name", "")
                title = officer.get("title", "")
                age = officer.get("age", "")
                pay = _format_officer_pay(officer.get("totalPay"))
                lines.append(f"| {name_} | {title} | {age} | {pay} |")
            lines.append("")

        return "\n".join(lines)


class YahooProfileFetcher:
    """yfinance를 통해 기업 프로필(Profile) 정보를 구조화된 형태로 가져온다."""

    def __init__(self, ticker: str):
        self.ticker_symbol = ticker.upper()
        self._ticker = yf.Ticker(self.ticker_symbol)

    def fetch(self) -> CompanyProfileBundle:
        info = self._ticker.info
        if not info or not (info.get("longBusinessSummary") or info.get("sector")):
            raise ValueError(f"'{self.ticker_symbol}'에 대한 프로필 정보를 찾을 수 없습니다. 티커를 확인하세요.")

        return CompanyProfileBundle(ticker=self.ticker_symbol, fetched_at=datetime.now(), info=info)

    def fetch_and_save_markdown(self, output_dir: str | Path) -> Path:
        bundle = self.fetch()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{self.ticker_symbol}_profile.md"
        out_path.write_text(bundle.to_markdown(), encoding="utf-8")
        return out_path


if __name__ == "__main__":
    import sys

    target_tickers = sys.argv[1:] or ["AAPL", "MSFT"]
    for ticker in target_tickers:
        try:
            fetcher = YahooProfileFetcher(ticker)
            saved_path = fetcher.fetch_and_save_markdown("./output/profiles")
            print(f"saved: {saved_path}")
        except ValueError as exc:
            print(f"skip {ticker}: {exc}")
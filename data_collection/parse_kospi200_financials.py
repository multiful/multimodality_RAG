"""output/kospi200_financials/*.md를 종목x재무제표종류x기간 단위 청크로 쪼개
data/corpus/kospi200_financial.jsonl에 저장한다.

마크다운 구조(src/finance/yahoo_financials.py가 생성):
    # {ticker} 재무제표
    ## 손익계산서 (Income Statement)
    ### 연간
    <table>
    ### 분기
    <table>
    ## 대차대조표 (Balance Sheet)
    ...

"## " 아래 "### " 단위(종목당 최대 6개: 3개 재무제표 x 연간/분기)를 하나의 청크로 만든다.

Usage:
    python data_collection/parse_kospi200_financials.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SRC_DIR = REPO_ROOT / "output" / "kospi200_financials"
OUT_FILE = REPO_ROOT / "data" / "corpus" / "kospi200_financial.jsonl"

_STATEMENT_TYPES = {
    "손익계산서": "income_statement",
    "대차대조표": "balance_sheet",
    "현금흐름표": "cash_flow",
}
_PERIOD_TYPES = {"연간": "annual", "분기": "quarterly"}

_TICKER_RE = re.compile(r"^#\s+(\S+)\s+재무제표")
_STMT_RE = re.compile(r"^##\s+([^\s(]+)")
_PERIOD_RE = re.compile(r"^###\s+(연간|분기)")


def parse_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    ticker_match = _TICKER_RE.match(lines[0]) if lines else None
    if not ticker_match:
        raise ValueError(f"{path}: 첫 줄에서 티커를 찾지 못했습니다: {lines[0] if lines else ''!r}")
    ticker = ticker_match.group(1)

    records: list[dict] = []
    stmt_type: str | None = None
    period_type: str | None = None
    buf: list[str] = []

    def flush():
        if stmt_type and period_type and buf:
            body = "\n".join(buf).strip()
            if body:
                records.append(
                    {
                        "id": f"{ticker}_{stmt_type}_{period_type}",
                        "ticker": ticker,
                        "type": "financial",
                        "statement_type": stmt_type,
                        "period_type": period_type,
                        "text": f"{ticker} {_reverse_stmt(stmt_type)} {_reverse_period(period_type)}\n\n{body}",
                    }
                )

    for line in lines[1:]:
        stmt_hit = _STMT_RE.match(line)
        period_hit = _PERIOD_RE.match(line)
        if stmt_hit:
            flush()
            buf = []
            period_type = None
            stmt_type = _STATEMENT_TYPES.get(stmt_hit.group(1))
            continue
        if period_hit:
            flush()
            buf = []
            period_type = _PERIOD_TYPES.get(period_hit.group(1))
            continue
        buf.append(line)
    flush()

    return records


def _reverse_stmt(code: str) -> str:
    return next(k for k, v in _STATEMENT_TYPES.items() if v == code)


def _reverse_period(code: str) -> str:
    return next(k for k, v in _PERIOD_TYPES.items() if v == code)


def main():
    files = sorted(SRC_DIR.glob("*_financials.md"))
    if not files:
        raise FileNotFoundError(f"{SRC_DIR}/ 에서 *_financials.md 파일을 찾지 못했습니다.")

    all_records: list[dict] = []
    for f in files:
        all_records.extend(parse_file(f))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(files)}개 파일 -> {len(all_records)}개 청크 -> {OUT_FILE}")


if __name__ == "__main__":
    main()

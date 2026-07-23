"""KOSPI200 구성종목의 기술적 분석(Layer2)을 실시간 시세로부터 계산해
종목별 마크다운 파일로 저장한다.

investing.com(예: https://www.investing.com/equities/yuhan-technical)을 직접
스크래핑하는 대신, Cloudflare 봇 차단 검증 결과(Playwright/patchright 모두 차단됨)에
따라 동일한 지표·판정 기준을 yfinance 실시간 OHLCV에 직접 적용해 재현한다.
계산 로직은 src/finance/layer2_technical_indicators.py 참고.

Usage:
    python data_collection/layer2_fetch_kospi200_technical.py 000100.KS --name "Yuhan Corporation"
    python data_collection/layer2_fetch_kospi200_technical.py --all
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.finance.layer2_technical_indicators import TechnicalSummary, analyze  # noqa: E402

TECHNICAL_OUT_DIR = REPO_ROOT / "KOSPI200_output" / "kospi200_technical"
LAYER2_OUT_DIR = REPO_ROOT / "KOSPI200_output" / "kospi200_layer2"


def render_technical_markdown(summary: TechnicalSummary, name_ko: str | None = None) -> str:
    code = summary.ticker.split(".")[0]
    title = f"{summary.ticker}" + (f" ({name_ko})" if name_ko else "")
    lines = [
        f"# Technical Analysis — {title}",
        "_source: yfinance 실시간 OHLCV + investing.com 공개 판정 기준 재계산_",
        f"_as of: {summary.as_of.date()} (종가: {summary.close:,.2f})_",
        f"_fetched: {summary.fetched_at.isoformat(timespec='seconds')}_",
        "",
        "## 종합 요약 (Overall Summary)",
        f"- 기술 지표 (Technical Indicators, 집계 대상 {sum(1 for r in summary.indicators if r.counted)}개): "
        f"Buy {sum(1 for r in summary.indicators if r.counted and r.signal == 'Buy')} / "
        f"Sell {sum(1 for r in summary.indicators if r.counted and r.signal == 'Sell')} / "
        f"Neutral {sum(1 for r in summary.indicators if r.counted and r.signal == 'Neutral')}",
        f"- 이동평균 (Moving Averages, {len(summary.mas) * 2}개): "
        f"Buy {sum(1 for m in summary.mas for s in (m.sma_signal, m.ema_signal) if s == 'Buy')} / "
        f"Sell {sum(1 for m in summary.mas for s in (m.sma_signal, m.ema_signal) if s == 'Sell')}",
        "",
        "## 기술 지표 (Technical Indicators)",
        "| 지표 | 값 | 시그널 | 집계 포함 |",
        "|---|---|---|---|",
    ]
    for row in summary.indicators:
        lines.append(f"| {row.name} | {row.value:.4f} | {row.signal} | {'O' if row.counted else 'X (참고용)'} |")

    lines += [
        "",
        "## 이동평균선 (Moving Averages)",
        "| 기간 | SMA | 시그널 | EMA | 시그널 |",
        "|---|---|---|---|---|",
    ]
    for ma in summary.mas:
        lines.append(
            f"| MA{ma.period} | {ma.sma:,.4f} | {ma.sma_signal} | {ma.ema:,.4f} | {ma.ema_signal} |"
        )

    lines += [
        "",
        "## Layer2 스코어링",
        "",
        "$$",
        r"s_{tech} = \frac{N_{buy} - N_{sell}}{N_{buy} + N_{sell} + N_{neutral}}",
        "$$",
        "",
        f"- $N_{{buy}}$ = {summary.n_buy}",
        f"- $N_{{sell}}$ = {summary.n_sell}",
        f"- $N_{{neutral}}$ = {summary.n_neutral}",
        f"- $s_{{tech}}$ = ({summary.n_buy} - {summary.n_sell}) / "
        f"({summary.n_buy} + {summary.n_sell} + {summary.n_neutral}) = **{summary.s_tech:.4f}**",
    ]
    return "\n".join(lines) + "\n"


def render_layer2_markdown(summary: TechnicalSummary, name_ko: str | None = None) -> str:
    title = f"{summary.ticker}" + (f" ({name_ko})" if name_ko else "")
    interpretation = (
        "강한 매수" if summary.s_tech > 0.5
        else "매수 우위" if summary.s_tech > 0.1
        else "강한 매도" if summary.s_tech < -0.5
        else "매도 우위" if summary.s_tech < -0.1
        else "중립"
    )
    lines = [
        f"# {title} — Layer2 기술적 분석 순비율 스코어링",
        f"_generated: {datetime.now().date()}_",
        f"_source: [{summary.ticker.split('.')[0]}_technical_analysis.md]"
        f"(../kospi200_technical/{summary.ticker.split('.')[0]}_technical_analysis.md)_",
        f"_as of: {summary.as_of.date()}_",
        "",
        "## 1. 스코어링 공식 (순비율 정규화 방식)",
        "",
        "$$",
        r"s_{tech} = \frac{N_{buy} - N_{sell}}{N_{buy} + N_{sell} + N_{neutral}}",
        "$$",
        "",
        "- $N_{buy}, N_{sell}, N_{neutral}$ = 기술 지표(8개, RSI/MACD/ADX/CCI/Highs-Lows/"
        "Bull-Bear Power/Ultimate Oscillator/ROC) + 이동평균(12개, SMA·EMA × 5/10/20/50/100/200)"
        " 총 20개 시그널의 집계",
        "- Overbought/Oversold 라벨(STOCH, STOCHRSI, Williams %R)과 변동성 라벨(ATR)은 참고용이며 집계에서 제외",
        "",
        "## 2. 집계 결과",
        "",
        f"| $N_{{buy}}$ | $N_{{sell}}$ | $N_{{neutral}}$ | 합계 |",
        "|---:|---:|---:|---:|",
        f"| {summary.n_buy} | {summary.n_sell} | {summary.n_neutral} | "
        f"{summary.n_buy + summary.n_sell + summary.n_neutral} |",
        "",
        "## 3. 최종 결과",
        "",
        "$$",
        f"s_{{tech}} = \\frac{{{summary.n_buy} - {summary.n_sell}}}"
        f"{{{summary.n_buy} + {summary.n_sell} + {summary.n_neutral}}} "
        f"\\approx {summary.s_tech:.2f}",
        "$$",
        "",
        f"**$s_{{tech}}$ ≈ {summary.s_tech:.2f}** → \"{interpretation}\" 로 해석.",
    ]
    return "\n".join(lines) + "\n"


def fetch_one(ticker: str, name_ko: str | None = None) -> TechnicalSummary:
    print(f"fetching (live yfinance): {ticker}" + (f" ({name_ko})" if name_ko else ""))
    summary = analyze(ticker)

    TECHNICAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    LAYER2_OUT_DIR.mkdir(parents=True, exist_ok=True)

    code = ticker.split(".")[0]
    (TECHNICAL_OUT_DIR / f"{code}_technical_analysis.md").write_text(
        render_technical_markdown(summary, name_ko), encoding="utf-8"
    )
    (LAYER2_OUT_DIR / f"{ticker}_layer2_score.md").write_text(
        render_layer2_markdown(summary, name_ko), encoding="utf-8"
    )
    print(f"  s_tech = {summary.s_tech:.4f}  (Buy {summary.n_buy} / Sell {summary.n_sell} / Neutral {summary.n_neutral})")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", nargs="?", help="예: 000100.KS (미지정 시 --all 필요)")
    parser.add_argument("--name", help="종목명(한글), 출력 파일 표기용")
    parser.add_argument("--all", action="store_true", help="KOSPI200 전 종목 실행")
    args = parser.parse_args()

    if args.all:
        from data_collection.kospi200_tickers import get_kospi200_tickers

        tickers = get_kospi200_tickers()
        print(f"KOSPI200 구성종목 {len(tickers)}개 로드")
        skipped = []
        for info in tickers:
            yahoo_ticker = info["yahoo_ticker"]
            try:
                fetch_one(yahoo_ticker, info.get("name_ko"))
            except ValueError as exc:
                print(f"  [skip] {yahoo_ticker}: {exc}")
                skipped.append(yahoo_ticker)
            time.sleep(0.3)
        if skipped:
            print(f"스킵된 {len(skipped)}개 티커: {', '.join(skipped)}")
    elif args.ticker:
        fetch_one(args.ticker, args.name)
    else:
        parser.error("ticker를 지정하거나 --all을 사용하세요.")


if __name__ == "__main__":
    main()
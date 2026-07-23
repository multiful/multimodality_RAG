"""KOSPI200 구성종목의 Layer1 재무제표 규칙 기반 스코어링(s_fin)을 계산해 종목별
마크다운 파일로 저장한다.

계산 로직은 src/finance/layer1_financial_score.py 참고 (섹터 피어 z-score + tanh).
이미 수집된 KOSPI200_output/kospi200_financials, kospi200_profiles 마크다운을 파싱해서
쓰므로 fetch_kospi200_financials.py / fetch_kospi200_profiles.py를 먼저 실행해둬야 한다.

Usage:
    python data_collection/layer1_score_kospi200_financials.py 000100.KS
    python data_collection/layer1_score_kospi200_financials.py --all
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.finance.layer1_financial_score import (  # noqa: E402
    METRIC_LABELS,
    METRICS,
    Layer1Result,
    compute_layer1_score,
    get_company_name,
    list_available_tickers,
)

OUT_DIR = REPO_ROOT / "KOSPI200_output" / "kospi200_layer1"


def render_markdown(result: Layer1Result) -> str:
    code = result.ticker.split(".")[0]
    name = get_company_name(code) or code
    target = result.target
    n = len(result.peer_metrics)

    lines = [
        f"# {result.ticker} ({name}) — Layer1 재무제표 규칙 기반 스코어링",
        f"_generated: {datetime.now().date()}_",
        f"_source: [{code}.KS_financials.md](../kospi200_financials/{code}.KS_financials.md)_",
        f"_fiscal year: {target.fiscal_year_latest} (vs. {target.fiscal_year_prev})_",
        "",
        "## 1. 스코어링 공식",
        "",
        "$$",
        r"s_{fin} = \frac{1}{|J|} \sum_{j \in J} \tanh\left(\frac{x_j - \mu_j}{\sigma_j}\right) \in [-1, 1]",
        "$$",
        "",
        f"- $J$ = {{{', '.join(METRIC_LABELS.values())}}} (지표 {len(METRICS)}개)",
        f"- $x_j$ = {result.ticker}의 지표 $j$ 값",
        f"- $\\mu_j, \\sigma_j$ = 동일 섹터({result.sector}) KOSPI200 구성종목의 평균/표준편차",
        "- 부채비율은 낮을수록 좋으므로 z-score 부호를 반전(sign-adjust)하여 사용",
        "",
        f"## 2. 섹터 피어 그룹 ({result.sector}, KOSPI200, n={n})",
        "",
        f"재무데이터가 존재하는 KOSPI200 {result.sector} 섹터 종목 {n}개를 표준편차 산정 모집단으로 사용.",
        "",
        "| 티커 | 기업명 | 매출성장률 | 영업이익률 변화 | 부채비율 |",
        "|:---|:---|---:|---:|---:|",
    ]
    for peer_code, m in sorted(result.peer_metrics.items(), key=lambda kv: kv[0] != code):
        peer_name = get_company_name(peer_code) or peer_code
        marker = " (대상)" if peer_code == code else ""
        lines.append(
            f"| {peer_code}.KS{marker} | {peer_name} | {m.revenue_growth:+.2%} | "
            f"{m.opinc_margin_change:+.2%}p | {m.debt_ratio:.3f} |"
        )

    lines += [
        "",
        "- 매출성장률 = (Total Revenue<sub>latest</sub> − Total Revenue<sub>prev</sub>) / Total Revenue<sub>prev</sub>",
        "- 영업이익률 변화 = (Operating Income/Total Revenue)<sub>latest</sub> − (Operating Income/Total Revenue)<sub>prev</sub>",
        "- 부채비율 = Total Liabilities Net Minority Interest / Stockholders Equity (최근 회계연도말 기준)",
        "",
        f"## 3. 섹터 평균(μ) / 표준편차(σ) — 표본표준편차(n-1), n={n}",
        "",
        "| 지표 | μ (평균) | σ (표준편차) |",
        "|:---|---:|---:|",
    ]
    for metric in METRICS:
        mu, sigma = result.mu_sigma[metric]
        if metric == "debt_ratio":
            mu_str = f"{mu:.4f}"
        elif metric == "opinc_margin_change":
            mu_str = f"{mu:+.2%}p"
        else:
            mu_str = f"{mu:+.2%}"
        lines.append(f"| {METRIC_LABELS[metric]} | {mu_str} | {sigma:.4f} |")

    lines += [
        "",
        f"## 4. {result.ticker} 표준화 (Z-score) 및 tanh 압축",
        "",
        "| 지표 | $x_j$ | $z_j = (x_j-\\mu_j)/\\sigma_j$ | 부호조정 | $\\tanh(z_j)$ |",
        "|:---|---:|---:|:---:|---:|",
    ]
    for metric in METRICS:
        x = getattr(target, metric)
        z = result.z_scores[metric]
        adj = "반전(×-1)" if metric == "debt_ratio" else "-"
        x_str = f"{x:.3f}" if metric == "debt_ratio" else f"{x:+.2%}" + ("p" if metric == "opinc_margin_change" else "")
        lines.append(f"| {METRIC_LABELS[metric]} | {x_str} | {z:.3f} | {adj} | {result.tanh_scores[metric]:+.3f} |")

    tanh_sum_str = " + ".join(f"({v:+.3f})" if v < 0 else f"{v:.3f}" for v in result.tanh_scores.values())
    verdict = (
        "뚜렷하게 양호한 펀더멘털" if result.s_fin > 0.3
        else "섹터 평균보다 양호한 펀더멘털" if result.s_fin > 0.1
        else "뚜렷하게 저조한 펀더멘털" if result.s_fin < -0.3
        else "섹터 평균보다 저조한 펀더멘털" if result.s_fin < -0.1
        else "섹터 평균과 거의 동일한 수준의 펀더멘털"
    )
    lines += [
        "",
        "## 5. 최종 결과",
        "",
        "$$",
        f"s_{{fin}} = \\frac{{{tanh_sum_str}}}{{{len(METRICS)}}} \\approx {result.s_fin:.2f}",
        "$$",
        "",
        f"**$s_{{fin}}$ ≈ {result.s_fin:.2f}**",
        "",
        f'→ "{verdict}" 로 해석.',
    ]
    return "\n".join(lines) + "\n"


def score_one(ticker: str) -> Layer1Result | None:
    try:
        result = compute_layer1_score(ticker)
    except ValueError as exc:
        print(f"  [skip] {ticker}: {exc}")
        return None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{ticker}_layer1_score.md"
    out_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"  s_fin = {result.s_fin:.4f}  (섹터: {result.sector}, n={len(result.peer_metrics)})  -> {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker", nargs="?", help="예: 000100.KS (미지정 시 --all 필요)")
    parser.add_argument("--all", action="store_true", help="재무데이터가 있는 KOSPI200 전 종목 실행")
    args = parser.parse_args()

    if args.all:
        codes = list_available_tickers()
        print(f"재무데이터 보유 종목 {len(codes)}개 로드")
        for code in codes:
            score_one(f"{code}.KS")
    elif args.ticker:
        score_one(args.ticker)
    else:
        parser.error("ticker를 지정하거나 --all을 사용하세요.")


if __name__ == "__main__":
    main()

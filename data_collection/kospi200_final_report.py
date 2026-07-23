"""Layer1~4를 한 번에 실행해 종목별 최종 리포트를 마크다운 한 파일로 만든다.

각 레이어의 상세 산출 로직/개별 파일은 그대로 두고(재무 상세는 layer1, 기술 지표 상세는
layer2, 뉴스 랭킹 상세는 layer3 스크립트가 각자 저장), 이 스크립트는 4개 레이어를 한 번에
계산해서 사람이 위에서 아래로 읽기 좋은 요약 리포트 하나로 합친다:
"종합 결론(최종 점수 S) → Layer1 재무 → Layer2 기술적분석 → Layer3 뉴스" 순서.

Usage:
    python data_collection/kospi200_final_report.py \
        --ticker 000100.KS --name "유한양행" --aliases "Yuhan,유한양행" \
        --query "유한양행" --topic "유한양행 실적 및 신약 파이프라인 전망"
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.finance.layer1_financial_score import (  # noqa: E402
    METRIC_LABELS,
    METRICS,
    Layer1Result,
    compute_layer1_score,
)
from src.finance.layer2_technical_indicators import TechnicalSummary, analyze as analyze_technical  # noqa: E402
from src.finance.layer3_news_selection import ScoredArticle, select_news  # noqa: E402
from src.finance.layer3_news_sentiment import SentimentResult, score_news_sentiment  # noqa: E402
from src.finance.layer4_fusion import FusionResult, SourceSignal, fuse  # noqa: E402

LAYER1_DIR = REPO_ROOT / "KOSPI200_output" / "kospi200_layer1"
OUT_DIR = REPO_ROOT / "KOSPI200_output" / "kospi200_final_report"

VERDICT_LABELS = [
    (0.5, "강한 매수"),
    (0.15, "매수 우위"),
    (-0.15, "중립"),
    (-0.5, "매도 우위"),
    (float("-inf"), "강한 매도"),
]


def _verdict(score: float) -> str:
    for threshold, label in VERDICT_LABELS:
        if score > threshold:
            return label
    return VERDICT_LABELS[-1][1]


def _parse_layer1_score(ticker: str) -> tuple[float, float] | None:
    path = LAYER1_DIR / f"{ticker}_layer1_score.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(r"s_\{fin\}\$\s*(?:approx|≈)\s*(-?[0-9.]+)", text)
    if not m:
        return None
    age_days = 0.0
    d = re.search(r"_generated:\s*(\d{4}-\d{2}-\d{2})_", text)
    if d:
        gen_date = datetime.strptime(d.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        age_days = max((datetime.now(timezone.utc) - gen_date).total_seconds() / 86400, 0.0)
    return float(m.group(1)), age_days


def _narrative(fusion: FusionResult) -> str:
    dominant = max(fusion.contributions, key=lambda c: abs(c.weighted_score))
    sentence = f"{dominant.label} 신호가 최종 점수에 가장 크게 기여했습니다({dominant.weighted_score:+.3f})."
    if fusion.conflicts:
        sentence += " 다만 소스 간 신호가 엇갈려 아래 충돌 신호를 함께 확인하는 것을 권합니다."
    elif all(c.score * dominant.score >= 0 for c in fusion.contributions):
        sentence += " 나머지 소스들도 대체로 같은 방향을 가리키고 있습니다."
    return sentence


def render_executive_summary(ticker: str, name_ko: str, sector: str | None, fusion: FusionResult) -> list[str]:
    lines = [
        "## 종합 결론",
        "",
        f"### S ≈ {fusion.S:.3f} — \"{_verdict(fusion.S)}\"",
        "",
        _narrative(fusion),
        "",
        "| 소스 | 점수 | 비중 | 기여분 | 비고 |",
        "|:---|---:|---:|---:|:---|",
    ]
    for c in fusion.contributions:
        lines.append(f"| {c.label} | {c.score:+.2f} | {c.weight_norm:.1%} | {c.weighted_score:+.3f} | {c.note} |")
    if fusion.excluded:
        lines.append("")
        lines.append(f"_제외된 소스: {', '.join(fusion.excluded)}_")
    if fusion.conflicts:
        lines.append("")
        lines.append("**⚠️ 충돌 신호**")
        for conflict in fusion.conflicts:
            lines.append(f"- {conflict}")
    return lines


def render_layer1_section(ticker: str, layer1: Layer1Result | None) -> list[str]:
    code = ticker.split(".")[0]
    lines = ["## Layer1 · 재무제표", ""]
    if layer1 is None:
        lines.append("_재무 데이터가 부족해 계산하지 못했습니다._")
        return lines

    n = len(layer1.peer_metrics)
    t = layer1.target
    lines += [
        f"섹터: **{layer1.sector}** (피어 n={n}) · 기준연도: {t.fiscal_year_latest} (vs {t.fiscal_year_prev})",
        "",
        "| 지표 | 값 | 섹터 평균(μ) | Z-score | tanh |",
        "|:---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        mu, _ = layer1.mu_sigma[metric]
        x = getattr(t, metric)
        if metric == "debt_ratio":
            x_str, mu_str = f"{x:.3f}", f"{mu:.3f}"
        elif metric == "opinc_margin_change":
            x_str, mu_str = f"{x:+.2%}p", f"{mu:+.2%}p"
        else:
            x_str, mu_str = f"{x:+.2%}", f"{mu:+.2%}"
        lines.append(
            f"| {METRIC_LABELS[metric]} | {x_str} | {mu_str} | {layer1.z_scores[metric]:.2f} | "
            f"{layer1.tanh_scores[metric]:+.3f} |"
        )
    lines += [
        "",
        f"**s_fin = {layer1.s_fin:+.3f}**",
        "",
        f"_섹터 피어 {n}개 전체 비교는 [{code}.KS_layer1_score.md](../kospi200_layer1/{code}.KS_layer1_score.md) 참고_",
    ]
    return lines


def render_layer2_section(ticker: str, tech: TechnicalSummary) -> list[str]:
    code = ticker.split(".")[0]
    counted = [r for r in tech.indicators if r.counted]
    lines = [
        "## Layer2 · 기술적 분석",
        "",
        f"기준일: {tech.as_of.date()} · 종가: {tech.close:,.0f}",
        f"기술 지표: Buy {sum(1 for r in counted if r.signal == 'Buy')} / "
        f"Sell {sum(1 for r in counted if r.signal == 'Sell')} / "
        f"Neutral {sum(1 for r in counted if r.signal == 'Neutral')}"
        f" · 이동평균: Buy {sum(1 for m in tech.mas for s in (m.sma_signal, m.ema_signal) if s == 'Buy')} / "
        f"Sell {sum(1 for m in tech.mas for s in (m.sma_signal, m.ema_signal) if s == 'Sell')}",
        "",
        "| 지표 | 시그널 |",
        "|:---|:---:|",
    ]
    for r in counted:
        lines.append(f"| {r.name} | {r.signal} |")
    short_term = [m for m in tech.mas if m.period <= 20]
    long_term = [m for m in tech.mas if m.period > 20]
    st_signal = short_term[0].sma_signal if short_term else "-"
    lt_signal = long_term[-1].sma_signal if long_term else "-"
    lines += [
        f"| 이동평균(단기 MA{short_term[0].period if short_term else '?'}) | {st_signal} |",
        f"| 이동평균(장기 MA{long_term[-1].period if long_term else '?'}) | {lt_signal} |",
        "",
        f"**s_tech = {tech.s_tech:+.3f}**",
        "",
        f"_전체 12개 지표·이동평균 상세는 [{code}_technical_analysis.md]"
        f"(../kospi200_technical/{code}_technical_analysis.md) 참고_",
    ]
    return lines


def render_layer3_section(
    ticker: str, articles: list[ScoredArticle], sentiments: list[SentimentResult], s_news: float
) -> list[str]:
    lines = ["## Layer3 · 뉴스 & 감성 분석", ""]
    if not articles:
        lines.append("_선정된 뉴스가 없습니다._")
        return lines

    for i, (art, sent) in enumerate(zip(articles, sentiments), start=1):
        lines += [
            f"{i}. **[{sent.label}] {art.title}**",
            f"   {sent.reasoning}",
            f"   ({art.pub_date.date()}, [원문]({art.originallink}))",
            "",
        ]
    lines += [
        f"**s_news = {s_news:+.3f}** ({len(articles)}건 감성 평균)",
        "",
        f"_선정 사유·랭킹 점수 상세는 [{ticker}_layer3_news.md](../kospi200_layer3/{ticker}_layer3_news.md) 참고_",
    ]
    return lines


def render_report(
    ticker: str,
    name_ko: str,
    layer1: Layer1Result | None,
    tech: TechnicalSummary,
    articles: list[ScoredArticle],
    sentiments: list[SentimentResult],
    s_news: float,
    fusion: FusionResult,
) -> str:
    sector = layer1.sector if layer1 else None
    lines = [
        f"# {ticker} ({name_ko}) — KOSPI200 Final Report",
        f"_generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
    ]
    if sector:
        lines.append(f"_섹터: {sector}_")
    lines.append("")
    lines += render_executive_summary(ticker, name_ko, sector, fusion)
    lines += ["", "---", ""]
    lines += render_layer1_section(ticker, layer1)
    lines += ["", "---", ""]
    lines += render_layer2_section(ticker, tech)
    lines += ["", "---", ""]
    lines += render_layer3_section(ticker, articles, sentiments, s_news)
    lines += [
        "",
        "---",
        "",
        "## 데이터 출처",
        "- 재무제표: yfinance (fetch_kospi200_financials.py로 수집)",
        "- 시세: yfinance 실시간",
        "- 뉴스: 네이버 검색 API 실시간 + KR-FinBert-SC 감성분류 + Qwen3 reasoning",
        "- 애널리스트 리포트(rep): 수집·분석 파이프라인 미구축 — 최종 점수(S) 융합에서 제외",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", required=True, help="예: 000100.KS")
    parser.add_argument("--name", required=True, help="기업명(한글)")
    parser.add_argument("--aliases", default="", help="쉼표로 구분된 별칭/영문명 (뉴스 선정용)")
    parser.add_argument("--query", help="네이버 검색 쿼리 (기본값: --name)")
    parser.add_argument("--topic", help="리포트 핵심 주제 (기본값: --name)")
    parser.add_argument("--fin-score", type=float, help="재무제표 점수 [-1,1] (미지정 시 Layer1을 직접 계산)")
    parser.add_argument("--no-llm", action="store_true", help="뉴스 선정 LLM 검증 생략")
    args = parser.parse_args()

    aliases = [a for a in args.aliases.split(",") if a.strip()]
    query = args.query or args.name
    topic = args.topic or args.name

    print(f"[Layer1] {args.ticker} 재무제표 스코어링...")
    layer1: Layer1Result | None = None
    fin_signal: SourceSignal | None = None
    if args.fin_score is not None:
        fin_signal = SourceSignal(score=args.fin_score, age_days=0.0, note="수동 입력")
    else:
        try:
            layer1 = compute_layer1_score(args.ticker)
            fin_signal = SourceSignal(score=layer1.s_fin, age_days=0.0, note="Layer1 실시간 계산")
            print(f"  s_fin = {layer1.s_fin:+.3f} (섹터: {layer1.sector}, n={len(layer1.peer_metrics)})")
        except ValueError as exc:
            parsed = _parse_layer1_score(args.ticker)
            if parsed:
                s_fin, age = parsed
                fin_signal = SourceSignal(score=s_fin, age_days=age, note="기존 layer1 파일에서 자동 파싱")
                print(f"  [fallback] 기존 파일에서 파싱: s_fin={s_fin:+.3f}")
            else:
                print(f"  [skip] {exc}")

    print(f"[Layer2] {args.ticker} 기술적 분석...")
    tech = analyze_technical(args.ticker)
    print(f"  s_tech = {tech.s_tech:+.3f} (Buy {tech.n_buy} / Sell {tech.n_sell} / Neutral {tech.n_neutral})")

    print(f"[Layer3] {args.ticker} 뉴스 선정 및 감성 분석...")
    articles = select_news(
        name_ko=args.name, query=query, topic=topic, aliases=aliases,
        top_n=5, use_llm_verification=not args.no_llm,
    )
    s_news, news_age, sentiments = score_news_sentiment(articles, args.name)
    print(f"  s_news = {s_news:+.3f} ({len(articles)}건)")

    print("[Layer4] 최종 융합...")
    signals: dict[str, SourceSignal] = {"tech": SourceSignal(score=tech.s_tech, age_days=0.0, note="Layer2 실시간 계산")}
    if fin_signal is not None:
        signals["fin"] = fin_signal
    signals["news"] = SourceSignal(score=s_news, age_days=news_age, note=f"선정 기사 {len(articles)}건 감성 평균")
    fusion = fuse(signals)
    print(f"  S = {fusion.S:+.3f} -> \"{_verdict(fusion.S)}\"")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.ticker}_final_report.md"
    out_path.write_text(
        render_report(args.ticker, args.name, layer1, tech, articles, sentiments, s_news, fusion),
        encoding="utf-8",
    )
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()

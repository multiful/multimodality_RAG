"""재무제표·기술적 분석·뉴스 감성분석을 한 번에 계산해서, 종목별 최종 리포트를 마크다운
한 파일로 만든다. 다른 파일을 참고하라고 링크만 걸지 않고, 근거가 되는 세부 내용을 전부
이 파일 안에 직접 풀어서 담는다 — 이 리포트 하나만 읽으면 된다.

Usage:
    python data_collection/kospi200_final_report.py \
        --ticker 000100.KS --name "유한양행" --aliases "Yuhan,유한양행" \
        --query "유한양행" --topic "유한양행 실적 및 신약 파이프라인 전망"
"""

from __future__ import annotations

import argparse
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
    get_company_name,
)
from src.finance.layer2_technical_indicators import TechnicalSummary, analyze as analyze_technical  # noqa: E402
from src.finance.layer3_news_selection import ScoredArticle, select_news  # noqa: E402
from src.finance.layer3_news_sentiment import SentimentResult, score_news_sentiment  # noqa: E402
from src.finance.layer4_fusion import FusionResult, SourceSignal, fuse  # noqa: E402

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


def _narrative(fusion: FusionResult) -> str:
    dominant = max(fusion.contributions, key=lambda c: abs(c.weighted_score))
    sentence = f"{dominant.label} 신호가 최종 점수에 가장 크게 영향을 미쳤습니다({dominant.weighted_score:+.3f})."
    if fusion.conflicts:
        sentence += " 다만 아래에서 보듯 소스 간 신호가 서로 엇갈리고 있어 교차 확인이 필요합니다."
    elif all(c.score * dominant.score >= 0 for c in fusion.contributions):
        sentence += " 나머지 지표들도 대체로 같은 방향을 가리키고 있어 신호가 일관됩니다."
    return sentence


def render_executive_summary(fusion: FusionResult) -> list[str]:
    lines = [
        "## 종합 결론",
        "",
        f"### 종합 점수 {fusion.S:+.3f} — \"{_verdict(fusion.S)}\"",
        "",
        "_점수는 -1(매우 부정적)부터 +1(매우 긍정적) 사이이며, 아래 3개 요소를 각자의 비중만큼 "
        "합산해 계산합니다._",
        "",
        _narrative(fusion),
        "",
        "| 요소 | 점수 | 반영 비중 | 최종 점수 기여분 | 이 점수가 의미하는 것 |",
        "|:---|---:|---:|---:|:---|",
    ]
    for c in fusion.contributions:
        lines.append(f"| {c.label} | {c.score:+.2f} | {c.weight_norm:.1%} | {c.weighted_score:+.3f} | {c.note} |")
    if fusion.excluded:
        lines.append("")
        lines.append(f"_아래 항목은 데이터가 없어 이번 계산에서 제외했습니다: {', '.join(fusion.excluded)}_")
    if fusion.conflicts:
        lines.append("")
        lines.append("**⚠️ 서로 엇갈리는 신호**")
        for conflict in fusion.conflicts:
            lines.append(f"- {conflict}")
    return lines


def render_financial_section(layer1: Layer1Result | None) -> list[str]:
    lines = ["## 재무제표 분석", ""]
    if layer1 is None:
        lines.append("_재무 데이터가 부족해 계산하지 못했습니다._")
        return lines

    n = len(layer1.peer_metrics)
    t = layer1.target
    lines += [
        f"같은 업종(**{layer1.sector}**)에 속한 KOSPI200 기업 {n}곳의 재무제표와 비교했습니다 "
        f"(기준연도: {t.fiscal_year_latest}, 전년도 {t.fiscal_year_prev}와 비교).",
        "",
        "**1) 같은 업종 기업들과의 비교**",
        "",
        "| 기업 | 매출성장률 | 영업이익률 변화 | 부채비율 |",
        "|:---|---:|---:|---:|",
    ]
    target_code = layer1.ticker.split(".")[0]
    for peer_code, m in sorted(layer1.peer_metrics.items(), key=lambda kv: kv[0] != target_code):
        peer_name = get_company_name(peer_code) or peer_code
        marker = " **(이 기업)**" if peer_code == target_code else ""
        lines.append(
            f"| {peer_name}{marker} | {m.revenue_growth:+.2%} | {m.opinc_margin_change:+.2%}p | {m.debt_ratio:.3f} |"
        )

    lines += [
        "",
        f"- 매출성장률 = (올해 매출 − 작년 매출) ÷ 작년 매출",
        f"- 영업이익률 변화 = 올해 영업이익률 − 작년 영업이익률 (%p 차이)",
        f"- 부채비율 = 총부채 ÷ 자기자본 (낮을수록 재무구조가 안정적)",
        "",
        "**2) 업종 평균과 비교해 얼마나 벗어나 있는지**",
        "",
        "_업종 평균보다 몇 표준편차만큼 좋거나 나쁜지 계산한 뒤(Z-score), 극단값의 영향을 "
        "줄이기 위해 -1~+1 사이로 압축(tanh)합니다. 부채비율은 낮을수록 좋으므로 부호를 "
        "뒤집어서 계산합니다._",
        "",
        "| 지표 | 이 기업 값 | 업종 평균 | 업종 평균 대비 | 압축 점수(-1~+1) |",
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
        z = layer1.z_scores[metric]
        z_desc = f"{'+' if z >= 0 else ''}{z:.2f}σ"
        lines.append(
            f"| {METRIC_LABELS[metric]} | {x_str} | {mu_str} | {z_desc} | {layer1.tanh_scores[metric]:+.3f} |"
        )

    verdict = (
        "업종 평균보다 뚜렷하게 양호합니다" if layer1.s_fin > 0.3
        else "업종 평균보다 다소 양호합니다" if layer1.s_fin > 0.1
        else "업종 평균보다 뚜렷하게 저조합니다" if layer1.s_fin < -0.3
        else "업종 평균보다 다소 저조합니다" if layer1.s_fin < -0.1
        else "업종 평균과 거의 비슷한 수준입니다"
    )
    lines += [
        "",
        f"**재무제표 종합 점수: {layer1.s_fin:+.3f}** → {verdict}.",
    ]
    return lines


def render_technical_section(tech: TechnicalSummary) -> list[str]:
    n_buy = tech.n_buy
    n_sell = tech.n_sell
    n_neutral = tech.n_neutral
    ma_buy = sum(1 for m in tech.mas for s in (m.sma_signal, m.ema_signal) if s == "Buy")
    ma_sell = sum(1 for m in tech.mas for s in (m.sma_signal, m.ema_signal) if s == "Sell")

    lines = [
        "## 기술적 분석",
        "",
        f"기준일 {tech.as_of.date()}, 종가 {tech.close:,.0f}원 기준으로 주가 매매 지표들을 계산했습니다.",
        "",
        "**1) 매매 지표 (12개)**",
        "",
        "_아래 지표들 중 '매수/매도 판단에 반영'인 것만 최종 신호 집계에 사용됩니다. "
        "'참고용'은 과매수·과매도·변동성처럼 방향성이 아닌 정보라 집계에서는 빠지지만 "
        "참고할 만한 지표입니다._",
        "",
        "| 지표 | 값 | 신호 | 매수/매도 판단 반영 |",
        "|:---|---:|:---:|:---:|",
    ]
    for r in tech.indicators:
        lines.append(f"| {r.name} | {r.value:,.2f} | {r.signal} | {'반영' if r.counted else '참고용'} |")

    lines += [
        "",
        "**2) 이동평균선**",
        "",
        "_주가가 특정 기간 평균 가격보다 위에 있으면 Buy(상승 추세), 아래에 있으면 Sell(하락 추세)로 "
        "판단합니다. 기간이 짧을수록 최근 흐름을, 길수록 장기 추세를 나타냅니다._",
        "",
        "| 기간 | 단순이동평균(SMA) | 신호 | 지수이동평균(EMA) | 신호 |",
        "|:---|---:|:---:|---:|:---:|",
    ]
    for m in tech.mas:
        lines.append(f"| {m.period}일 | {m.sma:,.0f} | {m.sma_signal} | {m.ema:,.0f} | {m.ema_signal} |")

    lines += [
        "",
        f"**3) 종합**: 매매 지표 매수 {n_buy}개 / 매도 {n_sell}개 / 중립 {n_neutral}개, "
        f"이동평균 매수 {ma_buy}개 / 매도 {ma_sell}개",
        "",
        f"**기술적 분석 종합 점수: {tech.s_tech:+.3f}** "
        f"(= (매수 − 매도) ÷ (매수 + 매도 + 중립) = ({n_buy + ma_buy} − {n_sell + ma_sell}) ÷ "
        f"{n_buy + n_sell + n_neutral + ma_buy + ma_sell})",
    ]
    return lines


def _days_ago_str(pub_date: datetime) -> str:
    now = datetime.now(timezone.utc)
    days = max((now - pub_date).total_seconds() / 86400, 0.0)
    if days < 1:
        return f"{days * 24:.0f}시간 전"
    return f"{days:.1f}일 전"


def render_news_section(
    articles: list[ScoredArticle], sentiments: list[SentimentResult], s_news: float
) -> list[str]:
    lines = [
        "## 뉴스 및 시장 심리 분석",
        "",
        "네이버 뉴스에서 이 기업 관련 기사를 실시간으로 검색한 뒤, 기업명 등장 위치·게재일·"
        "제목 중복 여부로 걸러내고, 주제 관련성·최신성·언론사 신뢰도·주요 이벤트 여부를 종합해 "
        "가장 중요한 기사 5건을 선정했습니다. 각 기사는 AI가 긍정/부정 정도를 5단계로 분류했습니다.",
        "",
    ]
    if not articles:
        lines.append("_선정된 뉴스가 없습니다._")
        return lines

    label_desc = {
        "very_positive": "매우 긍정적",
        "positive": "긍정적",
        "neutral": "중립적",
        "negative": "부정적",
        "very_negative": "매우 부정적",
    }
    for i, (art, sent) in enumerate(zip(articles, sentiments), start=1):
        event_desc = "있음" if art.event else "없음"
        lines += [
            f"### {i}. {art.title}",
            f"- 감성 분석: **{label_desc.get(sent.label, sent.label)}** ({sent.score:+.1f}점) — {sent.reasoning}",
            f"- 선정 이유: 주제 관련성 {art.rel:.0%}, {_days_ago_str(art.pub_date)} 게재, "
            f"언론사 신뢰도 {art.src_tier_label}, 실적·공시 등 주요 이벤트 키워드 {event_desc}",
            f"- 게재일: {art.pub_date.date()} · [기사 원문]({art.originallink})",
            "",
        ]
    lines += [
        f"**뉴스 종합 점수: {s_news:+.3f}** (선정된 {len(articles)}건의 감성 점수 평균, "
        f"+1에 가까울수록 긍정적인 뉴스가 많다는 뜻)",
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
        f"# {ticker} ({name_ko}) — 종합 투자 분석 리포트",
        f"_생성 시각: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
    ]
    if sector:
        lines.append(f"_업종: {sector}_")
    lines.append("")
    lines += render_executive_summary(fusion)
    lines += ["", "---", ""]
    lines += render_financial_section(layer1)
    lines += ["", "---", ""]
    lines += render_technical_section(tech)
    lines += ["", "---", ""]
    lines += render_news_section(articles, sentiments, s_news)
    lines += [
        "",
        "---",
        "",
        "## 참고: 데이터 출처",
        "- 재무제표: yfinance",
        "- 주가: yfinance (실시간)",
        "- 뉴스: 네이버 검색 API (실시간) + KR-FinBert-SC(감성 분류) + Qwen3(분류 근거 설명)",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", required=True, help="예: 000100.KS")
    parser.add_argument("--name", required=True, help="기업명(한글)")
    parser.add_argument("--aliases", default="", help="쉼표로 구분된 별칭/영문명 (뉴스 선정용)")
    parser.add_argument("--query", help="네이버 검색 쿼리 (기본값: --name)")
    parser.add_argument("--topic", help="리포트 핵심 주제 (기본값: --name)")
    parser.add_argument("--fin-score", type=float, help="재무제표 점수 [-1,1] (미지정 시 직접 계산)")
    parser.add_argument("--no-llm", action="store_true", help="뉴스 선정 LLM 검증 생략")
    args = parser.parse_args()

    aliases = [a for a in args.aliases.split(",") if a.strip()]
    query = args.query or args.name
    topic = args.topic or args.name

    print(f"[재무제표] {args.ticker} 분석 중...")
    layer1: Layer1Result | None = None
    fin_signal: SourceSignal | None = None
    if args.fin_score is not None:
        fin_signal = SourceSignal(score=args.fin_score, age_days=0.0, note="사용자가 직접 입력한 재무 점수")
    else:
        try:
            layer1 = compute_layer1_score(args.ticker)
            fin_signal = SourceSignal(
                score=layer1.s_fin,
                age_days=0.0,
                note=(
                    f"같은 업종({layer1.sector}) 기업 {len(layer1.peer_metrics)}곳과 비교한 재무 건전성 점수"
                    "(매출성장률·영업이익률 변화·부채비율 종합)"
                ),
            )
            print(f"  s_fin = {layer1.s_fin:+.3f} (업종: {layer1.sector}, 비교 기업 {len(layer1.peer_metrics)}곳)")
        except ValueError as exc:
            print(f"  [skip] {exc}")

    print(f"[기술적 분석] {args.ticker} 계산 중...")
    tech = analyze_technical(args.ticker)
    print(f"  s_tech = {tech.s_tech:+.3f} (매수 {tech.n_buy} / 매도 {tech.n_sell} / 중립 {tech.n_neutral})")

    print(f"[뉴스 분석] {args.ticker} 뉴스 선정 및 감성 분석 중...")
    articles = select_news(
        name_ko=args.name, query=query, topic=topic, aliases=aliases,
        top_n=5, use_llm_verification=not args.no_llm,
    )
    s_news, news_age, sentiments = score_news_sentiment(articles, args.name)
    print(f"  s_news = {s_news:+.3f} ({len(articles)}건)")

    print("[종합] 최종 점수 계산 중...")
    signals: dict[str, SourceSignal] = {
        "tech": SourceSignal(
            score=tech.s_tech,
            age_days=0.0,
            note="최근 주가 흐름과 매매 지표(이동평균·RSI 등)를 종합한 기술적 매수/매도 신호 점수",
        )
    }
    if fin_signal is not None:
        signals["fin"] = fin_signal
    signals["news"] = SourceSignal(
        score=s_news,
        age_days=news_age,
        note=f"최근 뉴스 {len(articles)}건의 감성 분석 평균 점수 (긍정적인 뉴스가 많을수록 +1에 가까움)",
    )
    fusion = fuse(signals)
    print(f"  종합 점수 = {fusion.S:+.3f} -> \"{_verdict(fusion.S)}\"")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.ticker}_final_report.md"
    out_path.write_text(
        render_report(args.ticker, args.name, layer1, tech, articles, sentiments, s_news, fusion),
        encoding="utf-8",
    )
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()

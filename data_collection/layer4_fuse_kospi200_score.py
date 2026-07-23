"""Layer4: rep/fin/news/tech 4개 소스를 시간 감쇠 가중 평균으로 융합해 최종 점수 S를 낸다.

S = Σ_k α_k e^{-λ_k Δt_k} s_k / Σ_k α_k e^{-λ_k Δt_k},  k ∈ {rep, fin, news, tech}
(반감기: rep ~30일, fin ~90일, news ~7일, tech ~1일. 계산 로직은 src/finance/layer4_fusion.py)

소스별 조달 방식:
- tech: src/finance/layer2_technical_indicators.py로 실시간 계산 (Δt≈0)
- news: Layer3(select_news)로 실시간 5건 선정 후 src/finance/layer3_news_sentiment.py로 감성 점수 평균
- fin: KOSPI200_output/kospi200_layer1/{ticker}_layer1_score.md에서 s_fin을 자동 파싱
  (없으면 --fin-score/--fin-age-days로 수동 입력)
- rep: 애널리스트 리포트 수집·분석 파이프라인이 아직 없어 --rep-score/--rep-age-days
  수동 입력만 지원 (미지정 시 융합에서 자동 제외 후 나머지 3개 소스로 재정규화)

Usage:
    python data_collection/layer4_fuse_kospi200_score.py \
        --ticker 000100.KS --name "유한양행" --aliases "Yuhan,유한양행" \
        --query "유한양행" --topic "유한양행 실적 및 신약 파이프라인 전망" \
        [--rep-score 0.3 --rep-age-days 10] [--fin-score -0.02 --fin-age-days 5] [--no-llm]
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

from src.finance.layer4_fusion import ALPHA, FusionResult, SourceSignal, fuse  # noqa: E402
from src.finance.layer3_news_selection import select_news  # noqa: E402
from src.finance.layer3_news_sentiment import SentimentResult, score_news_sentiment  # noqa: E402
from src.finance.layer2_technical_indicators import analyze as analyze_technical  # noqa: E402

LAYER1_DIR = REPO_ROOT / "KOSPI200_output" / "kospi200_layer1"
LAYER4_DIR = REPO_ROOT / "KOSPI200_output" / "kospi200_layer4"


def _parse_layer1_score(ticker: str) -> tuple[float, float] | None:
    """KOSPI200_output/kospi200_layer1/{ticker}_layer1_score.md에서 s_fin과 경과일(Δt)을 읽는다."""
    path = LAYER1_DIR / f"{ticker}_layer1_score.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(r"s_\{fin\}\$\s*(?:approx|≈)\s*(-?[0-9.]+)", text)
    if not m:
        return None
    s_fin = float(m.group(1))

    age_days = 0.0
    d = re.search(r"_generated:\s*(\d{4}-\d{2}-\d{2})_", text)
    if d:
        gen_date = datetime.strptime(d.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        age_days = max((datetime.now(timezone.utc) - gen_date).total_seconds() / 86400, 0.0)
    return s_fin, age_days


def render_markdown(
    ticker: str,
    name_ko: str,
    result: FusionResult,
    news_sentiments: list[SentimentResult],
) -> str:
    verdict = (
        "강한 매수" if result.S > 0.5
        else "매수 우위" if result.S > 0.15
        else "강한 매도" if result.S < -0.5
        else "매도 우위" if result.S < -0.15
        else "중립"
    )
    lines = [
        f"# {ticker} ({name_ko}) — Layer4 최종 가중 융합",
        f"_generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
        "## 스코어링 공식",
        "$$",
        r"S = \frac{\sum_k \alpha_k e^{-\lambda_k \Delta t_k} s_k}{\sum_k \alpha_k e^{-\lambda_k \Delta t_k}},"
        r"\quad k \in \{rep, fin, news, tech\}",
        "$$",
        "",
        "## 소스별 기여도",
        "",
        "| 소스 | $s_k$ | $\\Delta t_k$(일) | 감쇠 | $\\alpha_k$ | 정규화 가중치 | 기여분 | 비고 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in result.contributions:
        lines.append(
            f"| {c.label} | {c.score:+.2f} | {c.age_days:.1f} | {c.decay:.3f} | {ALPHA[c.key]:.2f} "
            f"| {c.weight_norm:.3f} | {c.weighted_score:+.3f} | {c.note} |"
        )
    if result.excluded:
        excluded_labels = ", ".join(result.excluded)
        lines.append("")
        lines.append(f"_제외된 소스(데이터 없음, 재정규화됨): {excluded_labels}_")

    lines += ["", "## 충돌 검사"]
    if result.conflicts:
        for c in result.conflicts:
            lines.append(f"- ⚠️ {c}")
    else:
        lines.append("- 소스 간 뚜렷한 충돌 없음")

    lines += [
        "",
        "## 최종 결과",
        "",
        f"**S ≈ {result.S:.3f}** → \"{verdict}\"",
        "",
    ]

    if news_sentiments:
        lines += ["## 참고: 뉴스 감성 분석 상세 (s_news 산출 근거)", ""]
        for s in news_sentiments:
            lines.append(f"- [{s.label} ({s.score:+.1f})] {s.title} — {s.reasoning}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", required=True, help="예: 000100.KS")
    parser.add_argument("--name", required=True, help="기업명(한글)")
    parser.add_argument("--aliases", default="", help="쉼표로 구분된 별칭/영문명 (뉴스 선정용)")
    parser.add_argument("--query", help="네이버 검색 쿼리 (기본값: --name)")
    parser.add_argument("--topic", help="리포트 핵심 주제 (기본값: --name)")
    parser.add_argument("--rep-score", type=float, help="애널리스트 리포트 점수 [-1,1] (수동 입력, 파이프라인 미구축)")
    parser.add_argument("--rep-age-days", type=float, default=0.0, help="리포트 발행 후 경과일")
    parser.add_argument("--fin-score", type=float, help="재무제표 점수 [-1,1] (미지정 시 layer1 파일에서 자동 파싱)")
    parser.add_argument("--fin-age-days", type=float, help="재무제표 점수 산출 후 경과일 (--fin-score와 함께 사용)")
    parser.add_argument("--no-llm", action="store_true", help="뉴스 선정 LLM 검증 생략")
    args = parser.parse_args()

    aliases = [a for a in args.aliases.split(",") if a.strip()]
    query = args.query or args.name
    topic = args.topic or args.name

    signals: dict[str, SourceSignal] = {}

    if args.rep_score is not None:
        signals["rep"] = SourceSignal(score=args.rep_score, age_days=args.rep_age_days, note="수동 입력")
    else:
        print("[rep] 점수 없음 (리포트 파이프라인 미구축) -> 융합에서 제외")

    if args.fin_score is not None:
        signals["fin"] = SourceSignal(
            score=args.fin_score, age_days=args.fin_age_days or 0.0, note="수동 입력"
        )
    else:
        parsed = _parse_layer1_score(args.ticker)
        if parsed:
            s_fin, age = parsed
            signals["fin"] = SourceSignal(score=s_fin, age_days=age, note=f"{args.ticker}_layer1_score.md 자동 파싱")
            print(f"[fin] layer1 파일에서 자동 파싱: s_fin={s_fin:.3f}, Δt={age:.1f}일")
        else:
            print(f"[fin] {args.ticker}_layer1_score.md 없음, --fin-score 미지정 -> 융합에서 제외")

    print("[tech] 실시간 기술적 분석 계산 중...")
    tech_summary = analyze_technical(args.ticker)
    signals["tech"] = SourceSignal(score=tech_summary.s_tech, age_days=0.0, note="실시간 yfinance 계산")
    print(f"[tech] s_tech={tech_summary.s_tech:.3f}")

    print("[news] 실시간 뉴스 선정 중...")
    articles = select_news(
        name_ko=args.name, query=query, topic=topic, aliases=aliases,
        top_n=5, use_llm_verification=not args.no_llm,
    )
    print(f"[news] {len(articles)}건 선정, 감성 분석 중...")
    s_news, news_age, sentiments = score_news_sentiment(articles, args.name)
    signals["news"] = SourceSignal(score=s_news, age_days=news_age, note=f"선정 기사 {len(articles)}건 감성 평균")
    print(f"[news] s_news={s_news:.3f}, Δt={news_age:.1f}일")

    result = fuse(signals)
    print(f"\n최종 S = {result.S:.3f}")
    for c in result.contributions:
        print(f"  {c.label}: s={c.score:+.2f} weight={c.weight_norm:.3f} contrib={c.weighted_score:+.3f}")
    for conflict in result.conflicts:
        print(f"  [conflict] {conflict}")

    LAYER4_DIR.mkdir(parents=True, exist_ok=True)
    md_path = LAYER4_DIR / f"{args.ticker}_layer4_fusion.md"
    md_path.write_text(render_markdown(args.ticker, args.name, result, sentiments), encoding="utf-8")

    json_path = LAYER4_DIR / f"{args.ticker}_layer4_fusion.json"
    json_path.write_text(__import__("json").dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n저장: {md_path}")
    print(f"저장(structured): {json_path}")


if __name__ == "__main__":
    main()

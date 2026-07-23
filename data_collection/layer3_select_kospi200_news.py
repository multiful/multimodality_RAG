"""KOSPI200 구성종목 관련 뉴스를 네이버 검색 API로 실시간 수집해 Layer3 방식으로 선정한다.

흐름: 네이버 검색 API(실시간) → 하드 필터(기업 매칭·[-2일,present]·제목 임베딩 중복제거)
     → 4요소 가중 랭킹(관련성 0.40·최신성 0.25·소스신뢰도 0.15·이벤트성 0.20)
     → LLM(Qwen3, 기본 0.6B) reasoning 검증 → 최종 5건

.env에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 필요 (.env.example 참고).
계산 로직은 src/finance/layer3_news_selection.py 참고.

Usage:
    python data_collection/layer3_select_kospi200_news.py \
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

from src.finance.layer3_news_selection import ScoredArticle, select_news  # noqa: E402

OUT_DIR = REPO_ROOT / "KOSPI200_output" / "kospi200_layer3"


def render_markdown(
    ticker: str, name_ko: str, topic: str, query: str, articles: list[ScoredArticle]
) -> str:
    lines = [
        f"# {ticker} ({name_ko}) — Layer3 뉴스 선정",
        f"_generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        f"_source: 네이버 검색 API (query=\"{query}\") 실시간 수집_",
        f"_리포트 핵심 주제: {topic}_",
        "",
        "## 선정 방식",
        "1. **하드 필터**: 기업명/별칭이 제목 또는 리드문에 등장 · 게재일 [-2일, present] · "
        "제목 임베딩(BGE-M3-ko) 코사인 유사도 > 0.9인 받아쓰기 기사는 최신 1건만 유지",
        "2. **4요소 가중 랭킹**: "
        "$score_i = 0.40 \\cdot rel_i + 0.25 \\cdot e^{-0.1\\Delta t_i} + 0.15 \\cdot src_i + 0.20 \\cdot event_i$",
        "3. **LLM reasoning 검증**: Qwen3(기본 0.6B)가 랭킹 상위 후보를 재검토해 최종 5건 확정",
        "",
        f"## 최종 선정 {len(articles)}건",
        "",
    ]
    for rank, art in enumerate(articles, start=1):
        lines += [
            f"### {rank}. {art.title}",
            f"- 링크: {art.originallink}",
            f"- 게재일: {art.pub_date.isoformat(timespec='minutes')}",
            f"- 리드문: {art.description}",
            f"- 규칙 점수: {art.score:.3f} "
            f"(rel={art.rel:.2f}, recency_decay={art.recency_decay:.2f}, "
            f"src={art.src}[{art.src_tier_label}], event={art.event})",
            f"- **선정 Reasoning**: {art.reasoning}",
            "",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", required=True, help="예: 000100.KS")
    parser.add_argument("--name", required=True, help="기업명(한글)")
    parser.add_argument("--aliases", default="", help="쉼표로 구분된 별칭/영문명/티커 (예: Yuhan,유한양행)")
    parser.add_argument("--query", help="네이버 검색 쿼리 (기본값: --name)")
    parser.add_argument("--topic", help="리포트 핵심 주제 (기본값: --name, 관련성 rel_i 산정 기준)")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--no-llm", action="store_true", help="LLM reasoning 검증 생략(규칙 기반 상위 N건만)")
    args = parser.parse_args()

    aliases = [a for a in args.aliases.split(",") if a.strip()]
    query = args.query or args.name
    topic = args.topic or args.name

    print(f"뉴스 선정 시작: {args.ticker} ({args.name}) query=\"{query}\" topic=\"{topic}\"")
    articles = select_news(
        name_ko=args.name,
        query=query,
        topic=topic,
        aliases=aliases,
        top_n=args.top_n,
        use_llm_verification=not args.no_llm,
    )
    print(f"최종 선정 {len(articles)}건")
    for a in articles:
        print(f"  [{a.score:.3f}] {a.title}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.ticker}_layer3_news.md"
    out_path.write_text(render_markdown(args.ticker, args.name, topic, query, articles), encoding="utf-8")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
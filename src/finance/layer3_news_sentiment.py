"""Layer3 부속: 선정된 뉴스 기사의 감성 분석 (Qwen3, 5단계 라벨 + Reasoning).

very_positive(+1.0) / positive(+0.5) / neutral(0) / negative(-0.5) / very_negative(-1.0)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from src.finance.layer3_news_selection import ScoredArticle
from src.finance.layer3_qwen3_llm import generate as qwen3_generate

LABEL_SCORES = {
    "very_positive": 1.0,
    "positive": 0.5,
    "neutral": 0.0,
    "negative": -0.5,
    "very_negative": -1.0,
}

LABEL_CRITERIA = """\
- very_positive(+1.0): 펀더멘털을 바꾸는 확정적 호재 (어닝 서프라이즈, 대규모 수주 확정, 승인 획득)
- positive(+0.5): 긍정적이나 규모가 작거나 불확실성 존재 (신제품 출시, 업황 개선 기대)
- neutral(0): 사실 전달, 방향성 없음 (인사 발표, 단순 시황 언급)
- negative(-0.5): 부정적이나 제한적 영향 (실적 소폭 하회, 경쟁 심화 우려)
- very_negative(-1.0): 펀더멘털 훼손급 확정 악재 (어닝 쇼크, 소송 패소, 대형 리콜)
"""


@dataclass
class SentimentResult:
    title: str
    label: str
    score: float
    reasoning: str


def _build_prompt(name_ko: str, article: ScoredArticle) -> str:
    return (
        "너는 금융 뉴스 감성 분석가다. 아래 5단계 라벨 기준으로 기사 하나를 분류하라.\n\n"
        f"{LABEL_CRITERIA}\n"
        f"기업: {name_ko}\n"
        f"제목: {article.title}\n"
        f"리드문: {article.description}\n\n"
        "다음 JSON 형식으로만 답하라 (그 외 설명 없이): "
        '{"label": "very_positive|positive|neutral|negative|very_negative", '
        '"reasoning": "판단 근거 1문장(한국어)"}'
    )


def _parse(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON 객체를 찾지 못했습니다: {raw_text[:200]}")
    return json.loads(match.group(0))


def classify_sentiment(article: ScoredArticle, name_ko: str) -> SentimentResult:
    """기사 1건을 Qwen3로 5단계 감성 라벨 분류한다. 파싱 실패 시 neutral로 안전하게 대체한다."""
    prompt = _build_prompt(name_ko, article)
    try:
        raw = qwen3_generate(prompt, max_new_tokens=200)
        parsed = _parse(raw)
        label = parsed["label"]
        if label not in LABEL_SCORES:
            raise ValueError(f"알 수 없는 라벨: {label}")
        reasoning = parsed.get("reasoning", "")
    except Exception as exc:  # noqa: BLE001 - LLM 호출/파싱 실패 시 neutral로 안전하게 폴백
        print(f"[warn] 감성 분석 실패({exc}), '{article.title[:30]}...' -> neutral로 대체")
        label, reasoning = "neutral", f"(파싱/생성 실패로 기본값 적용: {exc})"
    return SentimentResult(title=article.title, label=label, score=LABEL_SCORES[label], reasoning=reasoning)


def score_news_sentiment(
    articles: list[ScoredArticle], name_ko: str
) -> tuple[float, float, list[SentimentResult]]:
    """선정된 기사들의 감성 점수 평균을 s_news로, 평균 경과일을 Δt_news로 반환한다."""
    if not articles:
        return 0.0, 0.0, []

    results = [classify_sentiment(a, name_ko) for a in articles]
    s_news = sum(r.score for r in results) / len(results)

    now = datetime.now(timezone.utc)
    age_days = sum(max((now - a.pub_date).total_seconds() / 86400, 0.0) for a in articles) / len(articles)
    return s_news, age_days, results

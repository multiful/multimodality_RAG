"""Layer3 부속: 선정된 뉴스 기사의 감성 분석.

역할을 두 모델로 분리한다:
- 분류(라벨 결정): snunlp/KR-FinBert-SC — 한국어 금융 뉴스 감성 분류로 직접 파인튜닝된
  판별 BERT. positive/neutral/negative 3분류 + softmax 확신도를 낸다. 확신도가 높으면
  (기본 0.85 이상) "very_" 단계로 승격해 5단계 스케일(very_positive +1.0 ~ very_negative
  -1.0)에 맞춘다. Qwen3(생성형 LLM)만으로 분류했을 때는 라벨이 한쪽(대부분 very_positive)
  으로 쏠리는 변별력 문제가 있어서, 라벨 결정은 분류 전용 판별 모델로 넘겼다.
- Reasoning(설명 생성): Qwen3-1.7B — FinBERT가 이미 정한 라벨을 왜 그렇게 판단했는지
  자연어 한 문장으로 설명한다. 라벨을 다시 고르게 하지 않고 "이미 정해진 라벨을 설명"만
  시키는 좁은 태스크라, 뉴스 선정 검증에 쓰는 더 가벼운 기본 모델(Qwen3-0.6B)과 별도로
  한 단계 더 큰 모델을 명시적으로 쓴다 (src/finance/layer3_qwen3_llm.py의
  generate_reasoning). Qwen3-4B-Instruct-2507도 검토했지만 이 개발 환경(M2, 통합 메모리
  공유)에서 다운로드+로딩이 지나치게 오래 걸려 1.7B로 낮췄다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.finance.layer3_news_selection import ScoredArticle
from src.finance.layer3_qwen3_llm import generate_reasoning

MODEL_ID = "snunlp/KR-FinBert-SC"
VERY_CONFIDENCE_THRESHOLD = 0.85  # 이 이상 확신하면 very_positive/very_negative로 승격

LABEL_SCORES = {
    "very_positive": 1.0,
    "positive": 0.5,
    "neutral": 0.0,
    "negative": -0.5,
    "very_negative": -1.0,
}

LABEL_DESCRIPTIONS = {
    "very_positive": "펀더멘털을 바꾸는 확정적 호재 (어닝 서프라이즈, 대규모 수주 확정, 승인 획득 등)",
    "positive": "긍정적이나 규모가 작거나 불확실성이 있는 호재",
    "neutral": "사실 전달, 방향성이 뚜렷하지 않음",
    "negative": "부정적이나 영향이 제한적인 악재",
    "very_negative": "펀더멘털을 훼손하는 확정적 악재 (어닝 쇼크, 소송 패소, 대형 리콜 등)",
}

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[finbert] loading {MODEL_ID} on {device}", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
    _model.eval()
    return _model, _tokenizer


def _build_reasoning_prompt(article: ScoredArticle, name_ko: str, label: str) -> str:
    return (
        "아래 금융 뉴스 기사는 이미 감성 라벨이 정해졌다. 그 라벨이 왜 타당한지 "
        "기사 내용에 근거해 한국어 1문장으로 간결하게 설명하라. 라벨을 바꾸거나 다른 라벨을 "
        "제안하지 말고, 설명 문장만 출력하라.\n\n"
        f"기업: {name_ko}\n"
        f"제목: {article.title}\n"
        f"리드문: {article.description}\n"
        f"확정된 라벨: {label} ({LABEL_DESCRIPTIONS[label]})\n\n"
        "설명:"
    )


@dataclass
class SentimentResult:
    title: str
    label: str
    score: float
    reasoning: str
    raw_label: str = ""
    confidence: float = 0.0


def _classify_label(article: ScoredArticle) -> tuple[str, str, float]:
    """KR-FinBert-SC로 positive/neutral/negative + 확신도를 얻어 5단계 라벨로 매핑한다."""
    model, tokenizer = _load_model()
    text = f"{article.title} {article.description}".strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs))
    raw_label = model.config.id2label[idx]
    confidence = float(probs[idx])

    if raw_label == "positive":
        label = "very_positive" if confidence >= VERY_CONFIDENCE_THRESHOLD else "positive"
    elif raw_label == "negative":
        label = "very_negative" if confidence >= VERY_CONFIDENCE_THRESHOLD else "negative"
    else:
        label = "neutral"
    return label, raw_label, confidence


def classify_sentiment(article: ScoredArticle, name_ko: str) -> SentimentResult:
    """분류(KR-FinBert-SC)와 reasoning(Qwen3-4B-Instruct-2507)을 합쳐 최종 결과를 낸다."""
    label, raw_label, confidence = _classify_label(article)

    try:
        reasoning = generate_reasoning(_build_reasoning_prompt(article, name_ko, label))
    except Exception as exc:  # noqa: BLE001 - reasoning 생성 실패해도 분류 결과는 살린다
        print(f"[warn] reasoning 생성 실패({exc}), 템플릿 문장으로 대체")
        reasoning = f"KR-FinBert-SC 분류: {raw_label} (확신도 {confidence:.2f}) -> {label}"

    return SentimentResult(
        title=article.title,
        label=label,
        score=LABEL_SCORES[label],
        reasoning=reasoning,
        raw_label=raw_label,
        confidence=confidence,
    )


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

"""Layer4: 최종 가중 융합 모델.

S = Σ_k α_k * e^{-λ_k Δt_k} * s_k / Σ_k α_k * e^{-λ_k Δt_k},  k ∈ {rep, fin, news, tech}

- 소스별 반감기(half-life)만큼 시간이 지나면 그 소스의 실질 가중치가 절반으로 감쇠한다
  (rep ~30일, fin ~90일, news ~7일, tech ~1일).
- score가 없는 소스(None)는 분자·분모 모두에서 제외되고, 나머지 소스끼리 알파 비율 그대로
  자동 재정규화된다.
- "충돌 검사": 부호가 반대이면서 크게 벌어진(기본 0.5 이상 차이) 소스 쌍을 감지해 경고로 남긴다
  (예: 재무제표는 양호한데 최근 뉴스 감성은 매우 부정적인 경우 -> 교차 검증 필요 신호).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

ALPHA: dict[str, float] = {"rep": 0.40, "fin": 0.25, "news": 0.20, "tech": 0.15}
HALF_LIFE_DAYS: dict[str, float] = {"rep": 30.0, "fin": 90.0, "news": 7.0, "tech": 1.0}
LAMBDA: dict[str, float] = {k: math.log(2) / h for k, h in HALF_LIFE_DAYS.items()}
SOURCE_LABELS: dict[str, str] = {
    "rep": "애널리스트 리포트",
    "fin": "재무제표",
    "news": "뉴스",
    "tech": "기술적 분석",
}
CONFLICT_THRESHOLD = 0.5
SOURCE_KEYS = ("rep", "fin", "news", "tech")


@dataclass
class SourceSignal:
    """소스 하나의 입력. score/age_days 중 하나라도 None이면 융합에서 제외된다."""

    score: float | None = None  # s_k ∈ [-1, 1]
    age_days: float | None = None  # Δt_k (일)
    note: str = ""


@dataclass
class SourceContribution:
    key: str
    label: str
    score: float
    age_days: float
    decay: float
    weight_raw: float
    weight_norm: float
    weighted_score: float
    note: str


@dataclass
class FusionResult:
    S: float
    contributions: list[SourceContribution]
    excluded: list[str]
    conflicts: list[str]

    def to_dict(self) -> dict:
        return {
            "S": self.S,
            "contributions": [c.__dict__ for c in self.contributions],
            "excluded": self.excluded,
            "conflicts": self.conflicts,
        }


def fuse(signals: dict[str, SourceSignal]) -> FusionResult:
    contributions: list[SourceContribution] = []
    excluded: list[str] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for key in SOURCE_KEYS:
        sig = signals.get(key) or SourceSignal()
        if sig.score is None or sig.age_days is None:
            excluded.append(key)
            continue
        decay = math.exp(-LAMBDA[key] * max(sig.age_days, 0.0))
        weight_raw = ALPHA[key] * decay
        weighted_sum += weight_raw * sig.score
        weight_total += weight_raw
        contributions.append(
            SourceContribution(
                key=key,
                label=SOURCE_LABELS[key],
                score=sig.score,
                age_days=sig.age_days,
                decay=decay,
                weight_raw=weight_raw,
                weight_norm=0.0,
                weighted_score=0.0,
                note=sig.note,
            )
        )

    if weight_total == 0:
        raise ValueError("융합할 수 있는 소스가 하나도 없습니다 (모든 소스의 score/age_days가 None).")

    for c in contributions:
        c.weight_norm = c.weight_raw / weight_total
        c.weighted_score = c.weight_norm * c.score

    S = weighted_sum / weight_total

    conflicts = []
    for a, b in combinations(contributions, 2):
        if a.score * b.score < 0 and abs(a.score - b.score) >= CONFLICT_THRESHOLD:
            conflicts.append(
                f"{a.label}({a.score:+.2f})과(와) {b.label}({b.score:+.2f})가 상반된 신호 "
                f"(차이 {abs(a.score - b.score):.2f}) — 교차 검증 필요"
            )

    return FusionResult(S=S, contributions=contributions, excluded=excluded, conflicts=conflicts)

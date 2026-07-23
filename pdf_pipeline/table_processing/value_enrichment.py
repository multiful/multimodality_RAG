"""[20] Value 구조 확장 — 사용자 제안: unit/time/trend/derived events를 값에 같이 저장해서
"12.5"가 "12.5%"인지 "12.5조"인지, 증가인지 감소인지까지 LLM이 바로 알 수 있게 한다.
"""

import re

# 한국 금융 리포트에서 흔한 단위 접미사 -> 표준 단위 코드(전부 동적 정규식 매칭, 하드코딩 문자열
# 매칭이 아니라 "값 뒤에 어떤 단위 패턴이 붙어있는가"를 판단하므로 다른 PDF에도 동일 적용)
UNIT_PATTERNS = [
    (re.compile(r"조\s*원?$"), "KRW_trillion"),
    (re.compile(r"억\s*원$"), "KRW_100M"),
    (re.compile(r"백만\s*원$"), "KRW_1M"),
    (re.compile(r"만\s*원$"), "KRW_10K"),
    (re.compile(r"원$"), "KRW"),
    (re.compile(r"%$|퍼센트$"), "percent"),
    (re.compile(r"배$"), "multiple"),
    (re.compile(r"만\s*주$"), "shares_10K"),
    (re.compile(r"천\s*주$"), "shares_1K"),
    (re.compile(r"bp$|bps$", re.IGNORECASE), "basis_points"),
    (re.compile(r"\$|USD", re.IGNORECASE), "USD"),
]

TIME_PATTERN = re.compile(
    r"\b(20\d{2})\s*([EFAP]|년)?\b|\b([1-4])(Q|분기)\s*(\d{2}|20\d{2})\b|\b([1-2])H\s*(\d{2}|20\d{2})\b"
)


def extract_unit_and_value(raw_value: str, default_unit: str = None) -> dict:
    """"12.5조", "6,516.27p", "5.69%" 같은 문자열에서 숫자와 단위를 분리.
    단위를 못 찾으면 canonical field에 정의된 default_unit을 그대로 사용(값에 명시가 없을 뿐
    필드 자체의 기대 단위는 스키마에 이미 있으므로)."""
    if not raw_value or raw_value.strip() == "-":
        return {"numeric_value": None, "unit": default_unit, "raw": raw_value}

    text = raw_value.strip().replace(",", "")
    unit = default_unit
    for pattern, unit_code in UNIT_PATTERNS:
        if pattern.search(text):
            unit = unit_code
            text = pattern.sub("", text).strip()
            break

    m = re.search(r"-?\d+\.?\d*", text)
    numeric_value = float(m.group()) if m else None
    return {"numeric_value": numeric_value, "unit": unit, "raw": raw_value}


def extract_time_period(text: str):
    """"2026E", "2025F", "3Q25", "1H26" 같은 시점 표기를 탐지해 표준화된 문자열로 반환.
    못 찾으면 None(모든 값이 시계열은 아니므로 — 예: 계약상대방 같은 텍스트 필드)."""
    if not text:
        return None
    m = TIME_PATTERN.search(text)
    if not m:
        return None
    if m.group(1):  # "2026E" 형태
        return f"{m.group(1)}{m.group(2) or ''}".strip()
    if m.group(3):  # "3Q25" 형태
        return f"{m.group(5)}_{m.group(3)}Q"
    if m.group(6):  # "1H26" 형태
        return f"{m.group(7)}_{m.group(6)}H"
    return None


def compute_trend(values: list) -> dict:
    """같은 필드의 연속된 값들(예: 수주잔고 2024/2025/2026F)에서 방향성을 계산.
    values: [{"numeric_value": ..., "time": ...}, ...] 시간순 정렬된 리스트 가정.
    숫자로 해석 가능한 값이 2개 미만이면 트렌드 판단 불가(None)."""
    nums = [v["numeric_value"] for v in values if v.get("numeric_value") is not None]
    if len(nums) < 2:
        return {"trend": None, "first": nums[0] if nums else None, "last": nums[-1] if nums else None}
    first, last = nums[0], nums[-1]
    if last > first * 1.02:  # 2% 초과 변동만 유의미한 추세로 판단(노이즈 방지)
        trend = "up"
    elif last < first * 0.98:
        trend = "down"
    else:
        trend = "flat"
    return {"trend": trend, "first": first, "last": last, "prev": nums[-2] if len(nums) >= 2 else None}


def evaluate_derived_signals(records: list, derived_signals: dict) -> list:
    """추출된 canonical field 레코드들(같은 PDF 내)을 보고 derived_signals 스키마의 트리거 조건과
    대조해 발동된 시그널을 반환. 현재는 단일 필드 트렌드 트리거("field"+"pattern": up/down)만
    지원(다중 필드 조합 트리거는 이 표본 PDF로 검증할 데이터가 없어 프레임만 마련 — 아래 한계 참고)."""
    trend_by_field = {}
    for r in records:
        cf = r.get("canonical_field")
        if not cf or r.get("numeric_value") is None:
            continue
        trend_by_field.setdefault(cf, []).append(r["numeric_value"])

    triggered = []
    for signal_key, spec in derived_signals.items():
        trigger = spec["trigger"]
        if "field" in trigger and trigger.get("pattern") in ("up", "down"):
            field_key = trigger["field"]
            if field_key not in trend_by_field or len(trend_by_field[field_key]) < 2:
                continue
            t = compute_trend([{"numeric_value": v} for v in trend_by_field[field_key]])
            if t["trend"] == trigger["pattern"]:
                triggered.append({"signal": signal_key, "meaning": spec["meaning"],
                                   "description": spec["description"], "based_on_field": field_key})
    return triggered

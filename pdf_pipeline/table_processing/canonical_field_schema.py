"""[19]/[20] Canonical Field Schema — Field-first 설계, YAML 단일 소스로 전환(사용자 설계 반영).

[19]: Sector 먼저 판정하지 않고 필드부터 매칭(표 하나만 보고 섹터를 못 맞추는 문제 — CAPEX는
전자/화학/철강 다 씀). [20]: 여기에 Sector -> TableType -> Field 참고 계층을 추가하되, 실제
추출 게이팅에는 여전히 안 씀(그 원칙은 유지) — TableType은 "이 섹터 리포트에 어떤 표 유형이
나올지"를 문서화/확장하는 참고 레이어일 뿐. 모든 필드/섹터/파생신호 정의는 `sector_schema.yaml`
하나에 모아서, 코드 수정 없이 YAML만 고치면 새 섹터·필드를 추가할 수 있게 했다.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCHEMA_PATH = Path(__file__).resolve().parent / "sector_schema.yaml"


@dataclass
class CanonicalField:
    key: str
    aliases: list
    category: str
    value_type: str = "text"    # numeric_amount|percent|text|date_range — 값 정제 방식 결정
    unit: str = None            # 기본 단위(값 자체에서 못 읽으면 이 기본값 사용)
    relevant_sectors: list = field(default_factory=list)  # 참고용 태그 — 추출 게이팅에는 미사용


def _load_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


_SCHEMA = _load_schema()

# 필드 -> 이 필드를 쓰는 섹터 역색인(참고용 태그 채우기)
_FIELD_TO_SECTORS = {}
for _sector, _sdata in _SCHEMA["sectors"].items():
    for _fields in _sdata["table_types"].values():
        for _fk in _fields:
            _FIELD_TO_SECTORS.setdefault(_fk, set()).add(_sector)

CANONICAL_FIELDS = [
    CanonicalField(
        key=key, aliases=spec["aliases"], category=spec["category"],
        value_type=spec.get("value_type", "text"), unit=spec.get("unit"),
        relevant_sectors=sorted(_FIELD_TO_SECTORS.get(key, [])),
    )
    for key, spec in _SCHEMA["fields"].items()
]
FIELD_BY_KEY = {f.key: f for f in CANONICAL_FIELDS}
SECTOR_TABLE_TYPES = _SCHEMA["sectors"]           # sector -> {table_type: [field_keys]}
DERIVED_SIGNALS = _SCHEMA["derived_signals"]       # signal_key -> {trigger, meaning, description}
INVESTMENT_MEANING = _SCHEMA["investment_meaning"]  # meaning_key -> 설명 문장


def normalize_label(label: str) -> str:
    """라벨 정규화: 공백/괄호(단위 표기)/구두점 제거, 소문자화."""
    s = re.sub(r"\([^)]*\)", "", label)   # "수주잔고(억원)" -> "수주잔고"
    s = re.sub(r"[\s\-_·:]", "", s)        # 공백/하이픈/언더스코어/가운뎃점/콜론 제거
    return s.lower().strip()


def detect_wide_form(parsed_rows: list, min_header_matches: int = 2):
    """표가 narrow-form(라벨: 값들, 예: '수주잔고 | 12조 13조 14조')인지 wide-form(필드명이
    컬럼 헤더로, 레코드가 행으로 나열 — 예: 계약공시표의 '계약일|계약명|계약금액|계약상대...')
    인지 판별. 첫 행(헤더로 추정)의 [label]+cells 중 canonical field에 매칭되는 게
    min_header_matches개 이상이면 wide-form으로 판단하고, (header_fields, data_rows)를 반환.
    아니면 narrow-form으로 보고 (None, parsed_rows) 그대로 반환."""
    if not parsed_rows:
        return None, parsed_rows
    header = parsed_rows[0]
    header_cells = [header["label"]] + header["cells"]
    header_fields = [match_canonical_field(c) for c in header_cells]
    n_matched = sum(1 for f in header_fields if f)
    if n_matched >= min_header_matches:
        return header_fields, parsed_rows[1:]
    return None, parsed_rows


def match_canonical_field(raw_label: str):
    """raw_label(표의 행 라벨)이 어떤 canonical field에 대응하는지 매칭.
    **정방향 substring 매칭만 사용**(별칭이 라벨에 포함되면 매치, 예: 라벨="계약금액(억원)"
    별칭="계약금액") — 리포트마다 "수주잔고(억원)"처럼 단위가 붙거나 "당분기 수주잔고"처럼
    수식어가 붙는 경우까지 커버.

    역방향 매칭(라벨이 별칭에 포함, 예: 라벨="계약금액" 별칭="기술수출계약금액")은 처음엔 최소
    길이 제한(3자)을 두고 같이 썼었는데, 필드 수가 늘수록 오탐이 계속 발견됨 — "계약금액"이
    "기술수출계약금액"에, "성장률"이 "브랜드매출성장률"에, "금융수익"이 "기업금융수익"에,
    "재고자산"이 "재고자산평가손실"에 우연히 포함되는 식으로 3~4자짜리 일반 재무 용어가 계속
    엉뚱한(더 구체적인) 필드로 오매칭됐음. 지금까지 검증된 정상 매칭은 전부 정방향만으로 이미
    충분했다(필요한 짧은 표기는 alias 목록에 직접 추가하는 방식으로 해결) — 그래서 역방향은
    완전히 제거. 매칭되면 CanonicalField, 안 되면 None."""
    norm_label = normalize_label(raw_label)
    if not norm_label:
        return None
    best, best_len = None, 0
    for cf in CANONICAL_FIELDS:
        for alias in cf.aliases:
            norm_alias = normalize_label(alias)
            if norm_alias and norm_alias in norm_label and len(norm_alias) > best_len:
                best, best_len = cf, len(norm_alias)
    return best

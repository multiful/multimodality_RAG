"""[14] Table-aware Entity Extraction 분기: 표를 재무제표(Finance) vs 계약/일반(Contract/General)으로
구분해, 재무제표는 LLM 호출 없이 규칙 기반으로 엔티티 후보만 뽑고(대부분 자사명 반복이라 LLM 불필요),
계약/일반 표만 기존처럼 LLM 프롬프트에 포함시킨다.

동기: 실측 병목은 표 추출(Docling, ~10초)이 아니라 엔티티 추출(LLM, 500~600초) — 특히 page4(재무제표
6개 밀집)가 프롬프트 폭발로 대부분의 시간을 차지. 재무제표는 "매출액/영업이익/자산/부채" 같은 숫자
line-item일 뿐 새로운 회사명이 거의 안 나오므로, 이 표들을 LLM에 안 보내고 규칙 기반으로만 스캔해도
Recall 손실 없이 지연을 크게 줄일 수 있을 것으로 예상.

일반화 원칙: 이 PDF(LG CNS)에 특화된 회사명을 하드코딩하지 않고, 한국 기업/기관명의 일반적인 접미사
패턴(은행/증권/전자/화학/공단/클라우드 등)으로 규칙을 구성 — 다른 PDF에도 적용 가능하도록.
"""

import re

FINANCE_KEYWORDS = [
    "매출액", "매출원가", "매출총이익", "판매비와관리비", "영업이익", "영업외수익", "영업외비용",
    "법인세비용", "당기순이익", "유동자산", "비유동자산", "자산총계", "유동부채", "비유동부채",
    "부채총계", "자본금", "이익잉여금", "자본총계", "현금및현금성자산", "EBITDA", "감가상각비",
    # 투자지표류(밸류에이션 지표) — "주요 투자지표" 표처럼 손익/재무상태표 키워드가 없어도
    # 순수 숫자/지표 나열형 표라 마찬가지로 LLM 없이 규칙 처리가 안전한 케이스
    "투자지표", "PER", "PBR", "EPS", "BPS", "EV/EBITDA", "ROE", "배당수익률", "DPS", "CFPS",
]
FINANCE_MIN_KEYWORD_HITS = 2  # 이 개수 이상 매칭되면 재무제표로 판정

# 한국 기업/기관명의 일반적인 접미사(특정 회사명 하드코딩 아님 — 다른 PDF에도 재사용 가능)
COMPANY_SUFFIX_PATTERN = re.compile(
    r"[가-힣A-Za-z0-9]{1,12}(?:전자|화학|유플러스|은행|증권|카드|생명|화재|해상보험|손해보험|캐피탈|"
    r"공단|클라우드|중공업|건설|자동차|철강|에너지|바이오|제약|통신|물산|상사|지주|이노텍|디스플레이|"
    r"엔지니어링|파트너스|자산운용|투자증권|생명보험|CNS|SDS|C&C)"
)
STOCK_CODE_PATTERN = re.compile(r"\b\d{6}\b")


def classify_table(markdown: str) -> str:
    """표 마크다운 텍스트로 재무제표(finance) 여부 판정. 아니면 contract_or_general."""
    hits = sum(1 for kw in FINANCE_KEYWORDS if kw in markdown)
    return "finance" if hits >= FINANCE_MIN_KEYWORD_HITS else "contract_or_general"


_NUMBERING_PREFIX_RE = re.compile(r"^\s*\d+[\.\-]?\d*\.?\s*")  # "1-1. " / "2. " 같은 번호 접두사


def is_pure_financial_line_item(label: str) -> bool:
    """[22] 사용자 요청: "실적테이블"처럼 재무 기본항목(매출액/영업이익 등, 이미 DB에 있음)과
    세그먼트 정보(음원/음반, 공연 등, DB에 없는 새 정보)가 한 표에 섞여 있을 때, **표 전체를
    통째로 스킵하지 않고 행 단위로만** 순수 재무 항목을 걸러낸다. classify_table()은 표 전체를
    finance/contract로 이진 판정하지만(엔티티 추출 라우팅용, [14]), 이 함수는 그보다 세밀하게
    "이 행 하나"가 재무제표 원초 항목인지만 본다 — "1-1. 음원/음반"처럼 번호가 붙어도, "매출총이익률
    (%)"처럼 비율 표기가 붙어도 매칭되도록 번호 접두사/괄호를 먼저 제거하고 FINANCE_KEYWORDS와
    대조. 매칭되면 True(그 행은 버림 = DB 재무제표와 중복이므로 캐싱 안 함), 아니면 False(그 행은
    세그먼트/비재무 정보이므로 보존 + canonical field 매칭 시도)."""
    norm = _NUMBERING_PREFIX_RE.sub("", label)
    norm = re.sub(r"\([^)]*\)", "", norm).strip()
    if not norm:
        return False
    return any(kw == norm or kw in norm for kw in FINANCE_KEYWORDS)


def rule_extract_entities(markdown: str, anchor_entities: set) -> list:
    """재무제표용 규칙 기반 엔티티 후보 추출(LLM 미사용). 회사 접미사 패턴 + 종목코드 +
    문서 앵커(이미 알고 있는 보고대상 기업)를 결합 — 재무제표에 새 회사명이 드물게라도 등장하면
    잡아내는 안전망 역할(Recall 우선 원칙).

    [27] 이 함수(+ COMPANY_SUFFIX_PATTERN/STOCK_CODE_PATTERN)는 이제 `legacy_entity_extraction/`
    으로 옮긴 구세대 엔티티 추출 스크립트에서만 쓰인다 — 현재 프로덕션(`run_table_metadata_
    pipeline.py`)은 이 파일에서 `classify_table`/`is_pure_financial_line_item`만 가져다 쓰고,
    엔티티 추출 자체는 `[25]` structured_output.py의 entities 필드가 대체한다."""
    found = set(m.group(0) for m in COMPANY_SUFFIX_PATTERN.finditer(markdown))
    found |= set(STOCK_CODE_PATTERN.findall(markdown))
    found |= {a for a in anchor_entities if a in markdown}
    return sorted(found)

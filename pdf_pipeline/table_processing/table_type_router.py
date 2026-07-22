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


def rule_extract_entities(markdown: str, anchor_entities: set) -> list:
    """재무제표용 규칙 기반 엔티티 후보 추출(LLM 미사용). 회사 접미사 패턴 + 종목코드 +
    문서 앵커(이미 알고 있는 보고대상 기업)를 결합 — 재무제표에 새 회사명이 드물게라도 등장하면
    잡아내는 안전망 역할(Recall 우선 원칙)."""
    found = set(m.group(0) for m in COMPANY_SUFFIX_PATTERN.finditer(markdown))
    found |= set(STOCK_CODE_PATTERN.findall(markdown))
    found |= {a for a in anchor_entities if a in markdown}
    return sorted(found)

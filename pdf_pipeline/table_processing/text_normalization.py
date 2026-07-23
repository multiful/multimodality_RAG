"""[19] 텍스트 정규화 유틸 — 전부 "동적 탐지 후 조건부 수정"만 한다(정적/하드코딩 금지, 사용자
요청). 특정 PDF의 특정 문자열을 알고 고치는 게 아니라, "이 문자열이 이 문제를 갖고 있는가"를
런타임에 판단해서 그럴 때만 고친다 — 다른 PDF에도 똑같이 적용 가능해야 하기 때문.

다루는 문제 3가지:
1. 한글 음절 단위 과잉 띄어쓰기("주 식 회 사 우 리 은 행") — pdfplumber가 특정 PDF 폰트에서
   글자 사이마다 공백을 끼워 넣는 현상. "이 문자열이 실제로 이 문제를 겪고 있는가"를 토큰 구성
   비율로 판단한 뒤에만 공백 제거(정상적으로 띄어쓰기 된 한글 문장은 건드리지 않음).
2. 값 타입 기반 정제(예: 금액 필드에 다른 셀 텍스트가 흘러들어온 경우) — 필드의 "기대 타입"에
   맞는 패턴이 시작되는 지점부터만 취해서, 정확히 무엇이 섞여 들어왔는지 몰라도 일반적으로 걸러냄.
3. PDF 폰트 글리프 매핑 실패("(cid:9)" 같은 표기) — 정규식으로 탐지해 해당 레코드에 데이터 품질
   플래그만 남긴다(값을 억지로 복구하지 않음 — OCR 폴백은 실제로 자주 발생하면 그때 추가하는 게
   맞다는 원칙 유지, 지금은 미도입).
"""

import re

HANGUL_RE = re.compile(r"[가-힣]")
CID_ARTIFACT_RE = re.compile(r"\(cid:\d+\)")


def is_over_spaced(text: str, threshold: float = 0.5) -> bool:
    """공백으로 나눈 토큰 중 '한글 1글자짜리'가 threshold 비율 이상이면 음절 단위 과잉 띄어쓰기로
    판단(예: "주식회사 우리은행" 처럼 정상적으로 띄어쓴 문장은 1글자 토큰 비율이 낮음).
    빈 문자열/토큰 1개 이하는 판단 불가로 False."""
    tokens = text.split()
    if len(tokens) < 3:
        return False
    single_hangul = sum(1 for t in tokens if len(t) == 1 and HANGUL_RE.fullmatch(t))
    return (single_hangul / len(tokens)) >= threshold


def fix_hangul_spacing(text: str, force: bool = False) -> str:
    """과잉 띄어쓰기로 판단된 경우에만, 한글 글자 사이의 공백을 제거(한글-비한글 사이 공백은
    보존해서 "Physical AI 인 프 라 계 약" 같은 혼합 문자열에서 "Physical AI"는 안 건드림).

    force=True: 판단을 이 문자열 자체가 아니라 상위(표/페이지) 단위에서 이미 내렸을 때 사용
    — "LG전 자"처럼 토큰이 2개뿐인 짧은 셀 값은 자체적으로 과잉 띄어쓰기인지 판단할 통계량이
    부족한데, 같은 표의 다른 셀들(예: raw_text 전체)에서 이미 과잉 띄어쓰기가 확인됐다면 이
    셀도 같은 표에서 나온 값이니 무조건 적용하는 게 안전(표/페이지 단위 폰트 렌더링 특성이라
    표 안에서 일부 셀만 다를 이유가 없음)."""
    if not text:
        return text
    if not force and not is_over_spaced(text):
        return text
    return re.sub(r"(?<=[가-힣])\s+(?=[가-힣])", "", text)


def clean_value_by_type(value: str, value_type: str) -> str:
    """canonical field의 value_type에 따라 값 앞에 섞여 들어온 문자를 제거.
    "산) 226,178,000,000" 처럼 인접 셀 텍스트가 흘러든 경우, 금액/숫자 값은 항상 숫자나 '-'로
    시작해야 한다는 일반 규칙으로 그 이전 텍스트를 잘라낸다(이 PDF 특정 문자열을 아는 게 아니라
    "숫자 필드는 숫자로 시작해야 한다"는 타입 규칙만 앎 — 다른 PDF의 다른 오염 문자에도 동일 적용)."""
    if value_type == "numeric_amount":
        stripped = value.strip()
        if stripped == "-" or not stripped:
            return stripped
        m = re.search(r"[\d]", stripped)
        if m and m.start() > 0:
            return stripped[m.start():]
        return stripped
    return value


def detect_cid_artifact(value: str) -> bool:
    """PDF 폰트가 특정 글자의 유니코드 매핑을 안 갖고 있을 때 pdfplumber가 남기는 "(cid:9)" 같은
    표기를 탐지 — 가짜로 복구하지 않고 플래그만 남겨서 하위 LLM/사용자가 이 값을 의심하게 함."""
    return bool(CID_ARTIFACT_RE.search(value))

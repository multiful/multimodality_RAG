"""[36] 텍스트 정규화 유틸(PUA/구두점/기호) — 원래 `text_processing/text_normalization.py`에만
있었는데, 사용자 지적("정규화가 일반화가 안 돼?")으로 감사해보니 table_processing의 표 셀
텍스트/raw_text, sector_classifier의 첫 페이지 텍스트 등 **PDF에서 텍스트를 뽑는 다른 모든
지점**은 이 정규화를 전혀 안 거치고 있었음(PUA 문자, 줄임표/대시 표기 혼용, ▲▼ 기호 등이
그대로 임베딩/LLM 입력에 들어감). `pdf_pipeline/` 루트(embedding.py/yolo_layout.py와 같은
위치)로 옮겨서 text_processing/table_processing 양쪽이 같은 함수를 재사용하도록 함 — 이제
`table_processing/text_normalization.py`(Hangul 과잉 띄어쓰기/값 타입 정제/cid 아티팩트 탐지,
표 파싱에만 필요한 별개 관심사)와 이름이 겹치지 않아 `sys.modules` 스왑 우회도 더는 필요 없음.

PUA(Private Use Area, U+E000~U+F8FF) 글리프 매핑 실패는 table_processing의 `(cid:\\d+)` 패턴과
같은 계열의 버그지만 대응이 다르다: `(cid:9)`는 실제 숫자/문자 정보 자체가 유실된 경우라 복구
불가능(그래서 `data_quality: "unmapped_glyph"`로만 플래그하고 값은 원문 그대로 보존, [19]). 반면
Construct PDF에서 실측한 PUA 글리프는 전부 불릿 아이콘 폰트가 표준 유니코드 없이 임베드된
경우로, 문단/불릿 시작 위치에만 나타나고(양쪽 공백으로 둘러싸인 독립 토큰) 뒤따르는 실제 문장
내용에는 전혀 영향이 없음 — 68개 표본(Construct 전체) 전수 확인 완료, 예외 없음. 즉 이 글리프는
장식용 불릿 마커일 뿐 정보 손실이 아니므로, cid:와 달리 **안전하게 제거 가능**하다고 판단.
"""

import re
import unicodedata

PUA_RE = re.compile("[-]")


def detect_pua_artifact(text: str) -> bool:
    """텍스트에 PUA 코드포인트가 있는지 탐지(로깅/모니터링용 — 새 PDF에서 예상 밖의 PUA 사용
    패턴이 생기면 여기서 먼저 걸림, 그러면 strip이 안전한지 재검증 필요)."""
    return bool(PUA_RE.search(text))


def strip_pua_artifacts(text: str) -> str:
    """PUA 코드포인트를 제거하고, 제거로 생긴 중복 공백/줄바꿈을 정리.
    불릿 마커 제거 후에도 문장 자체는 그대로 남아 정보 손실 없음(위 독스트링 근거)."""
    cleaned = PUA_RE.sub("", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    return cleaned


# [4] 구두점 변형 통일 — 사용자 지적: "..."/"…"(줄임표), "-"/"–"/"—"(하이픈/en-dash/em-dash)가
# 리포트마다 섞여 나와 같은 의미의 텍스트가 문자열 레벨에서 다르게 취급됨(예: 임베딩/중복 탐지
# 시 다른 벡터로 계산될 수 있음). 실측(Construct PDF)에서 실제로 같은 문서 안에서도 "재개…도봉"
# (en-dash 줄임표 …) 와 "재개...도봉"(마침표 3개)이 혼용됨을 확인.
ELLIPSIS_RE = re.compile(r"\.{2,}|…")
# en-dash(–)/em-dash(—) -> ASCII 하이픈(-). 단어 구분자든, 숫자 범위든, 음수 부호로 쓰인 경우든
# 전부 글리프만 다를 뿐 의미는 동일해 무조건 변환해도 안전 — 순수 ASCII 하이픈 표기("-15%")는
# 이 정규식이 아예 매칭하지 않으므로 별도 예외처리가 필요 없다(en/em-dash 문자 자체만 대상으로 함).
DASH_RE = re.compile(r"[–—]")


def normalize_punctuation(text: str) -> str:
    """줄임표(.../…)를 "..."로, en-dash/em-dash를 ASCII 하이픈(-)으로 통일."""
    text = ELLIPSIS_RE.sub("...", text)
    text = DASH_RE.sub("-", text)
    return text


# [5] 한국어/금융 특화 구두점·기호 확장.
# ▲/▼(세모)는 장식이 아니라 **의미를 담은 기호**임에 주의 — 한국 금융 리포트에서 숫자 앞의
# ▲/▼는 상승/하락(또는 증가/감소)의 부호 역할을 한다(예: "▲1,200원"=플러스 1,200원,
# "▼0.5%"=마이너스 0.5%p). 그래서 숫자 바로 앞에 오면 +/-로 변환(부호 정보 보존)하고, 숫자가
# 아닌 다른 문맥에 쓰이면(드물게 순수 불릿으로 쓰이는 경우) 그냥 일반 불릿과 동일하게 처리한다.
UP_ARROW_NUM_RE = re.compile(r"▲(?=\s*[\d.])")
DOWN_ARROW_NUM_RE = re.compile(r"▼(?=\s*[\d.])")
GENERIC_BULLET_RE = re.compile(r"[•▶◆■□○●▲▼]\s*")
CIRCLED_DIGIT_MAP = {"①": "1.", "②": "2.", "③": "3.", "④": "4.", "⑤": "5.",
                     "⑥": "6.", "⑦": "7.", "⑧": "8.", "⑨": "9.", "⑩": "10."}
UNICODE_MINUS_RE = re.compile("−")  # U+2212 MINUS SIGN(수학 기호) -> ASCII 하이픈
NBSP_RE = re.compile(" ")            # non-breaking space -> 일반 공백


def normalize_symbols_and_whitespace(text: str) -> str:
    """▲/▼(숫자 앞이면 +/-로, 아니면 일반 불릿으로), 원문자 숫자(①→"1."), 유니코드 마이너스
    기호(U+2212), non-breaking space(U+00A0)를 표준 표기로 통일 + 유니코드 NFC 정규화(한글
    자모 분리(NFD) 현상 방지 — PyMuPDF 버전/폰트에 따라 완성형이 아닌 분리형으로 나오는 경우가
    있어 다운스트림 문자열 비교/임베딩이 깨질 수 있음)."""
    text = unicodedata.normalize("NFC", text)
    text = UNICODE_MINUS_RE.sub("-", text)
    text = NBSP_RE.sub(" ", text)
    text = UP_ARROW_NUM_RE.sub("+", text)
    text = DOWN_ARROW_NUM_RE.sub("-", text)
    for k, v in CIRCLED_DIGIT_MAP.items():
        text = text.replace(k, v)
    text = GENERIC_BULLET_RE.sub("- ", text)
    return text


def clean_extracted_text(text: str) -> str:
    """[36] 위 3개 함수를 이 프로젝트 전체가 공유하는 표준 순서로 묶은 편의 함수 — PUA 감지 시에만
    제거(조건부), 구두점/기호 정규화는 항상 적용(가벼움). 새로 텍스트 추출 지점을 추가할 때 이
    함수 하나만 부르면 정규화 누락을 방지할 수 있음."""
    had_pua = detect_pua_artifact(text)
    cleaned = strip_pua_artifacts(text) if had_pua else text
    cleaned = normalize_punctuation(cleaned)
    cleaned = normalize_symbols_and_whitespace(cleaned)
    return cleaned

"""[4] 헤더/푸터 제거 — 좌표 기반(페이지 상/하단 여백 영역) + 반복 빈도(여러 페이지에 걸쳐
같은 템플릿이 나타나는지) 결합. YOLO의 Page-header/Page-footer 클래스 분류에만 의존하지 않고
좌표+반복 빈도만으로 독립적으로 탐지 — YOLO 분류가 놓치거나(신뢰도 낮음) YOLO 없이도 동작해야
하는 상황을 위한 보완 레이어. "교보증권 2026.07 Page 4"처럼 페이지 번호만 바뀌는 경우를 잡기
위해 숫자를 플레이스홀더로 치환한 뒤 템플릿 동일성을 비교한다.
"""

import re
from collections import Counter

MARGIN_FRACTION = 0.08   # 페이지 상/하단 이 비율 이내 영역을 헤더/푸터 후보로 간주
MIN_PAGE_FRACTION = 0.5  # 전체 페이지 중 이 비율 이상에서 반복돼야 "진짜 헤더/푸터"로 판정
_DIGIT_RE = re.compile(r"\d+")


def _normalize_for_repetition(text: str) -> str:
    """페이지 번호/날짜처럼 페이지마다 바뀌는 숫자를 플레이스홀더로 치환 — "Page 4"와 "Page 5"를
    같은 템플릿으로 인식하기 위함."""
    return _DIGIT_RE.sub("#", text).strip()


def detect_headers_footers(doc_fitz, margin_fraction: float = MARGIN_FRACTION,
                            min_page_fraction: float = MIN_PAGE_FRACTION) -> dict:
    """PDF 전체를 훑어 상/하단 여백에 반복 등장하는 템플릿(헤더/푸터)을 탐지.
    반환: {normalized_template: {"n_pages", "band", "raw_examples", "pages"}}"""
    n_pages = doc_fitz.page_count
    candidates = {}
    for i in range(n_pages):
        page = doc_fitz[i]
        h = page.rect.height
        top_band, bottom_band = h * margin_fraction, h * (1 - margin_fraction)
        for b in page.get_text("blocks"):
            if b[6] != 0 or not b[4].strip():
                continue
            y0, y1 = b[1], b[3]
            if y1 <= top_band:
                band = "header"
            elif y0 >= bottom_band:
                band = "footer"
            else:
                continue
            for raw_line in b[4].strip().split("\n"):
                raw_line = raw_line.strip()
                norm = _normalize_for_repetition(raw_line)
                if norm:
                    candidates.setdefault(norm, []).append((i + 1, raw_line, band))

    min_pages = max(2, int(n_pages * min_page_fraction))
    boilerplate = {}
    for norm, occurrences in candidates.items():
        if len(occurrences) >= min_pages:
            bands = Counter(o[2] for o in occurrences)
            boilerplate[norm] = {
                "n_pages": len(occurrences),
                "band": bands.most_common(1)[0][0],
                "raw_examples": list(dict.fromkeys(o[1] for o in occurrences))[:3],
                "pages": [o[0] for o in occurrences],
            }
    return boilerplate


def strip_headers_footers(text: str, boilerplate_templates: dict) -> str:
    """추출된 페이지 텍스트에서 탐지된 헤더/푸터 템플릿과 일치하는 줄을 제거."""
    templates = set(boilerplate_templates.keys())
    kept = []
    for line in text.split("\n"):
        norm = _normalize_for_repetition(line.strip())
        if norm and norm in templates:
            continue
        kept.append(line)
    return "\n".join(kept)

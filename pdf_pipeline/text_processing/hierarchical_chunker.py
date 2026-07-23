"""[3] 계층적 청킹(Hierarchical Chunking) — 문서 구조(Title/Section-header/Text/List-item)를
그대로 청크 경계로 사용. YOLO가 잡은 블록 클래스를 재사용해 "이 블록이 어느 Section-header 아래에
있는가"로 그룹핑하고, 그룹이 너무 길면 블록 단위(이미 문단/불릿 단위라 의미 경계와 일치)로 다시
쪼갠다. 다른 두 방식과 달리 청크 내용 자체는 원문 그대로 두고, **계층 경로(section_path)를
메타데이터로 붙이는 것**이 이 방식의 핵심 차별점 — 시멘틱 청킹(문장 임베딩 유사도 기반)이나
문맥적 청킹(LLM이 생성한 설명을 본문에 덧붙이는 방식)과 구분됨.

구현 노트(버그 발견·수정): 처음엔 PyMuPDF `get_text("blocks")`로 얻은 블록을 그대로 청크 단위로
썼는데, 이 PDF는 줄바꿈된 각 "시각적 줄"을 별도 block으로 반환해서("출 1.56조원, 영익 1,387억원)
소폭 하회 예상. 클라우드&AI 매출은 AI 인프라 및 유지보" 처럼 문장 중간에서 잘림) 문단 경계와
전혀 안 맞았다. [1]/[2]에서 이미 검증된 `get_textbox(bbox)` 방식(YOLO 박스 하나 = 문단/불릿
하나, 여러 줄을 올바르게 이어붙임)으로 교체 — YOLO 박스 자체가 문단 단위 경계라 더 안전하다.
"""

import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # [10] yolo_layout이 pdf_pipeline/로 이동
from yolo_layout import run_yolo_layout  # noqa: E402 — [8] 공유 YOLO 호출(중복 추론 제거)
from text_cleanup import clean_extracted_text  # noqa: E402 — [36] pdf_pipeline/로 이동(공유)

NON_TEXT_CLASSES = {"Table", "Picture"}
CHROME_CLASSES = {"Page-header", "Page-footer"}  # 페이지 번호/반복 헤더 — 내용도 헤더 로직도 아님
HEADER_CLASSES = {"Title", "Section-header"}
BODY_CLASSES = {"Text", "List-item"}  # next_body 판정용 — Caption/Chrome은 "본문 흐름"이 아니라 제외


def _dedupe_overlapping(items: list, tol_pt: float = 5.0) -> list:
    """YOLO가 같은 텍스트 영역에 대해 서로 다른(또는 같은) 클래스로 거의 겹치는 박스를 중복
    탐지하는 경우 발견(예: "투자의견 Buy 및 목표주가 89,000원 유지"가 Section-header와 Text로
    동시에, 거의 같은 위치에 잡힘) — 같은 텍스트에 x0/y0가 tol_pt 이내로 겹치면 하나만 남기고,
    Title/Section-header 쪽을 우선 채택(구조 신호가 더 유용하므로)."""
    kept = []
    for it in sorted(items, key=lambda t: t["y0"]):
        dup_idx = next((i for i, k in enumerate(kept)
                         if k["text"] == it["text"] and abs(k["x0"] - it["x0"]) < tol_pt
                         and abs(k["y0"] - it["y0"]) < tol_pt), None)
        if dup_idx is None:
            kept.append(it)
        elif it["cls"] in HEADER_CLASSES and kept[dup_idx]["cls"] not in HEADER_CLASSES:
            kept[dup_idx] = it
    return kept


def _rect_containment_ratio(inner: tuple, outer: tuple) -> float:
    """inner 박스 면적 중 outer 박스와 겹치는 비율(0~1) — inner가 outer 안에 완전히 들어있으면 1.0."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    x0, y0 = max(ix0, ox0), max(iy0, oy0)
    x1, y1 = min(ix1, ox1), min(iy1, oy1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    inner_area = (ix1 - ix0) * (iy1 - iy0)
    return inter / inner_area if inner_area > 0 else 0.0


def _dedupe_contained_fragments(items: list, containment_ratio: float = 0.9) -> list:
    """[48] 사용자 지적("실제 pdf에서 텍스트를 잘 뽑아내는지 봐줘")으로 실측 발견 — YOLO가 같은
    텍스트 영역에 대해 "전체를 담은 큰 박스"와 "그 안의 한 줄만 담은 작은 박스"를 동시에
    검출하는 경우가 있음(NMS가 포함관계 박스는 IoU 낮아서 잘 못 잡음). 실측(C밴드.pdf "Top
    Picks" 목록): "쏠리드(050890)\nBUY | TP 30,000원 | CP 9,100원"(전체 박스)과 "쏠리드(050890)"
    (부분 박스), "BUY | TP 30,000원 | CP 9,100원"(부분 박스)이 각각 3개 Text 클래스로 검출됨 —
    _dedupe_overlapping()은 텍스트가 정확히 같아야만 중복으로 보므로(부분 문자열은 안 잡음)
    이 경우를 못 거름. 그 결과 회사명 없이 "BUY | TP..."만 있는 청크가 따로 생겨서, 검색이
    그 청크를 회사명 청크와 별도로 반환하면 LLM이 엉뚱한 회사에 갖다붙이는 실제 오답으로
    이어짐(실험.md [48] 참고).

    한 박스의 텍스트가 다른 박스 텍스트의 부분 문자열이고, 박스 영역도 containment_ratio(0.9)
    이상 겹치면 "같은 내용의 부분집합"으로 보고 작은(부분) 박스를 버리고 큰(전체) 박스만 남긴다."""
    to_drop = set()
    for i, a in enumerate(items):
        if i in to_drop:
            continue
        for j, b in enumerate(items):
            if i == j or j in to_drop or len(b["text"]) >= len(a["text"]):
                continue
            if b["text"] not in a["text"]:
                continue
            if _rect_containment_ratio(b["rect"], a["rect"]) >= containment_ratio:
                to_drop.add(j)
    return [it for i, it in enumerate(items) if i not in to_drop]


def _get_boxes_with_text(model, doc_fitz, page_idx: int, cached_boxes: list = None):
    """YOLO로 페이지의 Text/Title/Section-header/List-item/Caption 박스를 찾고, 각 박스 안의
    텍스트를 get_textbox()로 통째로(여러 줄 자동 결합) 추출한 뒤 `text_cleanup.
    clean_extracted_text()`([36], PUA 제거+구두점/기호 정규화)로 정규화 — Table/Picture/
    Page-header/Page-footer는 제외(표·이미지는 다른 파이프라인 소관, 페이지 크롬은 내용이 아님).
    [35] 사용자 지적("여기 잘 뽑히는지 확인해줘")으로 인덱싱 테스트 중, 실제 임베딩/검색 대상인
    이 청크 텍스트가 `extract_page_text()`(recall 측정용 baseline)와 달리 정규화를 전혀 안 거치고
    있던 버그를 발견해 추가 — [36]에서 정규화 함수 자체를 pdf_pipeline/ 공유 위치로 옮김.

    cached_boxes: [8] 공유 YOLO 호출 결과([(cls_name, fitz.Rect), ...])를 넘기면 이 함수는
    YOLO를 다시 부르지 않고 그 결과를 그대로 재사용 — reading_order_router의 난이도 판정과
    페이지당 YOLO 호출을 공유하기 위함(이전까지는 페이지당 YOLO가 2번 호출되고 있었음)."""
    page = doc_fitz[page_idx]
    yolo_boxes = cached_boxes if cached_boxes is not None else run_yolo_layout(model, page, page_idx)

    items = []
    for cls_name, rect in yolo_boxes:
        if cls_name in NON_TEXT_CLASSES or cls_name in CHROME_CLASSES:
            continue
        text = clean_extracted_text(page.get_textbox(rect).strip())
        if text:
            items.append({"text": text, "y0": rect.y0, "x0": rect.x0, "cls": cls_name,
                          "rect": (rect.x0, rect.y0, rect.x1, rect.y1)})
    items = _dedupe_overlapping(items)
    items = _dedupe_contained_fragments(items)  # [48] 포함관계 부분 박스 제거
    items.sort(key=lambda t: t["y0"])
    return items


def chunk_hierarchical(model, doc_fitz, page_idx: int, max_chars: int = 400,
                        header_x_tolerance_pt: float = 80.0, cached_boxes: list = None) -> list:
    """반환: [{text, section_path, page}, ...]

    header_x_tolerance_pt: 사이드바 오분류 방지용 — LG CNS p1에서 실측한 문제(사이드바의
    "Price & Relative Performance" 캡션이 Section-header로 오분류되고, 단순 y좌표 정렬만 쓰면
    본문 컬럼과 무관한 이 헤더가 뒤따르는 본문 문단들의 section_path로 잘못 붙는 버그 발견).
    헤더 바로 다음에 오는 본문 블록과 x0 위치가 이 값 이상 차이나면(= 다른 컬럼/사이드바 소속으로
    추정) 그 헤더는 section_path 갱신에 쓰지 않고 그냥 본문 블록처럼 취급한다.

    cached_boxes: [8] reading_order_router와 YOLO 호출을 공유하려면 여기에
    `yolo_layout.run_yolo_layout(model, page, page_idx)` 결과를 그대로 전달."""
    tagged = _get_boxes_with_text(model, doc_fitz, page_idx, cached_boxes=cached_boxes)
    chunks = []
    section_path = []  # [Title, Section-header] 스택
    current_group_texts = []

    def flush_group():
        if not current_group_texts:
            return
        joined = "\n".join(current_group_texts)
        if len(joined) <= max_chars:
            chunks.append({"text": joined, "section_path": list(section_path), "page": page_idx + 1})
        else:
            # 그룹이 너무 길면 블록(문단/불릿) 단위로 재분할 — 이미 의미 단위라 자연스러운 경계
            for t in current_group_texts:
                chunks.append({"text": t, "section_path": list(section_path), "page": page_idx + 1})
        current_group_texts.clear()

    for idx, item in enumerate(tagged):
        is_header = item["cls"] in HEADER_CLASSES
        # x0 이탈 검사는 "이미 첫 헤더가 자리잡은 이후"의 Section-header에만 적용 — 페이지 맨 위
        # 타이틀 영역은 YOLO가 Title 대신 Section-header로 분류하는 경우가 실측됐고(문서 제목은
        # 본문보다 왼쪽 여백에서 시작하는 게 정상이라 x0 불일치만으로 사이드바로 오판하면 안 됨),
        # 반대로 "Price & Relative Performance"처럼 본문 흐름 중간에 끼어드는 사이드바 캡션은
        # 이미 첫 섹션이 자리잡은 다음에 나타난다는 점으로 구분한다.
        if item["cls"] == "Section-header" and section_path:
            next_body = next((t for t in tagged[idx + 1:] if t["cls"] in BODY_CLASSES), None)
            if next_body and abs(item["x0"] - next_body["x0"]) > header_x_tolerance_pt:
                current_group_texts.append(item["text"])
                continue
        if item["cls"] == "Title":
            flush_group()
            section_path = [item["text"]]
        elif item["cls"] == "Section-header":
            flush_group()
            section_path = (section_path[:1] if section_path else []) + [item["text"]]
        else:
            current_group_texts.append(item["text"])
    flush_group()
    return chunks

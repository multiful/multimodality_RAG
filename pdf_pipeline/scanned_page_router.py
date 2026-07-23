"""스캔본(텍스트 레이어 없음/희박) 페이지 감지 + MinerU 전용 텍스트 대체.

사용자 요청: "어려운 pdf 페이지는 only MinerU야" — 스캔본으로 판정된 페이지는 우리 자체
파이프라인(PyMuPDF get_text()) 결과를 섞지 않고 MinerU 파싱 결과의 텍스트로 통째로 대체한다.
그리고 이 판정이 발생하면 반드시 로그로 알린다(사용자 요청: "어려운 pdf라고 꼭 말해줘야해").

text_processing/reading_order_router.py의 "hard" 판정과는 다른 문제를 다룬다 — 그쪽은 "텍스트는
있는데 다단 컬럼이라 읽는 순서가 꼬이는가"를 보고, 여기서는 "애초에 텍스트 레이어가 없는가"
(스캔 이미지)를 본다. 두 조건은 독립적이고 겹칠 수 있다(스캔본이면서 다단일 수도 있음).
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz

MIN_TEXT_CHARS_PER_PAGE = 20  # 이 미만이면 "텍스트 레이어 사실상 없음" 후보
MIN_IMAGE_COVERAGE = 0.6      # 페이지 면적의 이 비율 이상을 이미지가 덮으면 "스캔 이미지"로 확정
                               # (텍스트 없는 순수 빈 페이지/구분페이지까지 스캔본으로 오판 방지)


def _image_coverage_ratio(page: fitz.Page) -> float:
    """페이지 전체 면적 대비, 페이지를 덮는 이미지들의 면적 비율(근사치 — 겹침 고려 안 함,
    스캔본은 보통 이미지 1장이 페이지 전체를 덮으므로 이 근사로 충분)."""
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return 0.0
    covered = 0.0
    for img in page.get_images(full=True):
        xref = img[0]
        for rect in page.get_image_rects(xref):
            covered += rect.width * rect.height
    return min(covered / page_area, 1.0)


def detect_scanned_pages(pdf_path) -> list[int]:
    """PDF 전체를 훑어 "텍스트 레이어 사실상 없음(문자 수 < MIN_TEXT_CHARS_PER_PAGE) AND 페이지
    대부분을 이미지가 덮음" 두 조건을 모두 만족하는 페이지 번호(1-based)를 반환."""
    doc = fitz.open(str(pdf_path))
    scanned = []
    for i in range(doc.page_count):
        page = doc[i]
        text_len = len(page.get_text().strip())
        if text_len < MIN_TEXT_CHARS_PER_PAGE and _image_coverage_ratio(page) >= MIN_IMAGE_COVERAGE:
            scanned.append(i + 1)
    doc.close()
    return scanned


def extract_text_via_mineru(doc_id: str, page_numbers: list[int], timeout_sec: int = 1800) -> dict:
    """스캔본 판정된 page_numbers(1-based)의 텍스트를 MinerU 파싱 결과에서만 가져온다(우리 자체
    추출 결과와 섞지 않음 — "only MinerU"). MinerU content_list.json에서 해당 페이지의
    text/title 타입 블록만 모아 페이지별로 이어붙여 반환({page: text}).

    주의: MinerU content_list.json의 본문 텍스트 블록 타입 이름("text"/"title")은 image_processing/
    README(§5 카드 스키마)가 문서화한 chart/image/table 타입 확인만으로 유추한 것 — 이 환경엔
    MinerU가 설치돼 있지 않아 실제 실행으로 검증하지 못했다. MinerU 설치 환경에서 최초 실행 시
    `common.load_content_list(mdir)`의 실제 `type` 값들을 한 번 찍어서 이름이 맞는지 확인할 것."""
    if not page_numbers:
        return {}

    image_processing_dir = str(Path(__file__).resolve().parent / "image_processing")
    if image_processing_dir not in sys.path:
        sys.path.insert(0, image_processing_dir)
    import common as ip_common  # noqa: E402
    from s2_onestop_mineru import ensure_parsed  # noqa: E402

    mdir = ensure_parsed(doc_id, timeout_sec)
    if mdir is None:
        raise RuntimeError(
            f"MinerU 파싱 실패 또는 data/pdfs/metadata.csv에 doc_id={doc_id!r}가 없음 — "
            "image_processing 파이프라인의 문서 등록 절차(README §1)를 먼저 거쳐야 함")

    content = ip_common.load_content_list(mdir)
    target_idx = {p - 1 for p in page_numbers}
    by_page: dict = {p: [] for p in page_numbers}
    for item in content:
        if item.get("page_idx") not in target_idx:
            continue
        if item.get("type") in ("text", "title") and item.get("text"):
            by_page[item["page_idx"] + 1].append(item["text"])

    return {p: "\n".join(texts) for p, texts in by_page.items()}


def route_scanned_pages(pdf_path, doc_id: str, logger=None, timeout_sec: int = 1800) -> dict:
    """PDF에서 스캔본 페이지를 감지하고, 있으면 MinerU로만 텍스트를 대체 추출한다.
    반환: {page_number: mineru_text}. 스캔본이 하나도 없으면 빈 dict(호출측은 기존 파이프라인
    결과를 그대로 쓰면 됨). 감지되면 반드시 로그로 알린다."""
    log = logger.info if logger else print
    scanned_pages = detect_scanned_pages(pdf_path)
    if not scanned_pages:
        return {}

    log(f"[스캔본 감지] {len(scanned_pages)}개 페이지({scanned_pages})가 어려운(스캔) PDF 페이지로 "
        f"판정됨 — MinerU로만 처리(자체 파이프라인 텍스트는 사용 안 함)")
    mineru_texts = extract_text_via_mineru(doc_id, scanned_pages, timeout_sec)
    for p in scanned_pages:
        n_chars = len(mineru_texts.get(p, ""))
        log(f"  page{p}: MinerU 텍스트 {n_chars}자 확보")
    return mineru_texts

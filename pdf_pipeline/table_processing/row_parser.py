"""[19] Row Parser — 사용자 피드백 (3) 반영: "Regex만 쓰는 것은 약하다. Row Parser가 낫다."

TATR로 표의 row bbox뿐 아니라 column bbox까지 탐지하고, 각 (row, column) 교차 영역을 하나의
셀로 취급해 pdfplumber로 텍스트를 채운다. 첫 번째 컬럼(가장 왼쪽)을 라벨로, 나머지 컬럼들을
값 배열(cells)로 분리 — "2026E/2027E처럼 컬럼이 여러 개인 경우"를 label 하나에 값 여러 개로
자연스럽게 표현할 수 있다(계약상대방처럼 텍스트 값도, 수주잔고처럼 연도별 숫자 여러 개도 동일 구조).
"""

import re
import sys
import threading
from collections import Counter
from pathlib import Path

import fitz
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # [36] text_cleanup이 pdf_pipeline/에 있음
from text_cleanup import clean_extracted_text  # noqa: E402


def _boxes_by_label(results, id2label, target_label):
    out = []
    for label_id, box in zip(results["labels"], results["boxes"]):
        if id2label[label_id.item()] == target_label:
            out.append(tuple(v.item() for v in box))
    return out


def _has_overlap(row_box, col_box):
    """행과 열 bbox가 실제로 겹치는 영역이 있는지(폭/높이 둘 다 양수)만 확인.
    주의: 열(column) bbox는 표 전체 높이를 커버하는 게 보통이라, 교집합 면적을 '행 전체 면적'
    대비 비율로 재면 컬럼이 3개 이상일 때 항상 낮게 나와(1/N) 전부 걸러지는 버그가 있었음
    (예: 6컬럼 표면 컬럼당 교집합 비율이 최대 ~17%라 임계값 30%를 절대 못 넘김) — 그래서
    비율 대신 "겹치는 영역이 존재하는가"만 확인하고, 실제 셀 크기는 cell_text()의 로컬 x범위
    교차로 자연스럽게 정해지도록 수정."""
    rx1, ry1, rx2, ry2 = row_box
    cx1, cy1, cx2, cy2 = col_box
    ix1, iy1 = max(rx1, cx1), max(ry1, cy1)
    ix2, iy2 = min(rx2, cx2), min(ry2, cy2)
    return ix2 > ix1 and iy2 > iy1


def _run_tatr(model, processor, doc_fitz, page_num: int, bbox_pt: tuple,
              render_dpi: int, pad_top_pt: float, pad_side_pt: float, conf: float):
    page_fz = doc_fitz[page_num - 1]
    padded_pt = (
        max(0.0, bbox_pt[0] - pad_side_pt), max(0.0, bbox_pt[1] - pad_top_pt),
        min(page_fz.rect.width, bbox_pt[2] + pad_side_pt), min(page_fz.rect.height, bbox_pt[3] + pad_side_pt),
    )
    # [33] 성능 수정: PNG를 디스크에 썼다가 바로 다시 읽던 왕복 제거(yolo_layout.py [33]과 동일
    # 기법) — COMPLEX 표마다(표 개수만큼) 호출되는 함수라 여기 낭비가 누적됨.
    pix = page_fz.get_pixmap(dpi=render_dpi, clip=fitz.Rect(*padded_pt))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    scale = render_dpi / 72
    inputs = processor(images=img, return_tensors="pt")
    # [37] model.to(device)(MPS 등)로 옮겨져 있으면 입력 텐서도 같은 device로 옮겨야 함 —
    # model.device는 파라미터 하나를 보고 판단(모델 전체가 한 device에 있다는 전제, TATR처럼
    # 작은 모델은 항상 그러함).
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([img.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=conf, target_sizes=target_sizes)[0]
    id2label = model.config.id2label

    row_boxes = _boxes_by_label(results, id2label, "table row")
    col_boxes = _boxes_by_label(results, id2label, "table column")
    row_boxes.sort(key=lambda b: b[1])   # 위->아래
    col_boxes.sort(key=lambda b: b[0])   # 왼쪽->오른쪽
    return row_boxes, col_boxes, padded_pt, scale


def _build_grid_rows(page_pp, row_boxes, col_boxes, padded_pt, scale):
    # [33] 성능 수정: 이전엔 셀마다 원본 페이지(`page_pp`)를 직접 crop()했는데, pdfplumber의
    # crop()은 그 페이지의 전체 객체(글자/선 등, 페이지당 수천 개)를 매번 처음부터 다시
    # 필터링한다 — 표 하나에 셀 수십 개가 있으면 "페이지 전체 재스캔"이 셀 수만큼 반복되는
    # 셈. 프로파일링 결과 LGCNS 13개 표만으로 pdfplumber crop 관련 호출이 700만 회 이상
    # 발생해 표 처리 시간의 대부분(13.4초 중 10초 가량)을 차지하는 게 확인됨. 표 영역
    # (padded_pt)으로 딱 한 번만 미리 crop해두면 그 안에 남는 객체 수가 훨씬 적어져서 이후
    # 셀별 crop이 크게 저렴해진다(실측 100셀 기준 528ms -> 192ms, 결과는 완전히 동일함을
    # 별도 검증 — pdfplumber Page.crop()의 bbox는 기본적으로 원본 페이지 기준 절대좌표라
    # 이미 계산해둔 cell_pt를 그대로 재사용해도 안전).
    table_page = page_pp.crop(padded_pt)

    def cell_text(row_box, col_box):
        rx1, ry1, rx2, ry2 = row_box
        cx1, cy1, cx2, cy2 = col_box
        # 셀 = 행의 y범위 x 열의 x범위 교차 영역
        local_box = (max(rx1, cx1), ry1, min(rx2, cx2), ry2)
        if local_box[2] <= local_box[0]:
            return ""
        cell_pt = (padded_pt[0] + local_box[0] / scale, padded_pt[1] + local_box[1] / scale,
                   padded_pt[0] + local_box[2] / scale, padded_pt[1] + local_box[3] / scale)
        raw = (table_page.crop(cell_pt).extract_text() or "").replace("\n", " ").strip()
        return clean_extracted_text(raw)  # [36] PUA/구두점 정규화 — 셀 텍스트도 검색·LLM 입력이 됨

    parsed_rows = []
    for row_box in row_boxes:
        if col_boxes:
            cells = [cell_text(row_box, cb) for cb in col_boxes
                     if _has_overlap(row_box, cb)]
        else:
            cells = []
        # [23] 빈 셀을 통째로 제거하면(예전 방식) 중간 컬럼이 비어 있을 때 뒤 컬럼들이 앞으로
        # 당겨져 위치가 밀린다 — "경쟁률"/"미달"처럼 행마다 둘 중 하나만 값이 있고 나머지는
        # 빈칸인 상호배타적 컬럼 쌍에서 실측(Construct PDF 청약 동향 표)으로 발견: 미달=68인
        # 행의 값이 경쟁률 칸으로 밀려 들어가 헤더-값 매핑이 전부 한 칸씩 어긋남. 빈 셀은
        # 위치를 보존한 채 빈 문자열로 남겨 헤더(wide-form)/라벨(narrow-form) 정렬을 유지한다.
        if not any(cells):
            # 컬럼 탐지 실패(또는 전 셀이 빈칸) 시 폴백: 행 전체를 하나의 텍스트로(라벨=값)
            rx1, ry1, rx2, ry2 = row_box
            row_pt = (padded_pt[0] + rx1 / scale, padded_pt[1] + ry1 / scale,
                      padded_pt[0] + rx2 / scale, padded_pt[1] + ry2 / scale)
            whole = clean_extracted_text((table_page.crop(row_pt).extract_text() or "").replace("\n", " ").strip())
            if not whole:
                continue
            cells = [whole]
        label, values = cells[0], cells[1:]
        parsed_rows.append({"label": label, "cells": values, "row_top_pt": padded_pt[1] + row_box[1] / scale})
    return parsed_rows


def parse_table_rows(model, processor, doc_fitz, page_pp, page_num: int, bbox_pt: tuple,
                      render_dpi: int, pad_top_pt: float, pad_side_pt: float, conf: float = 0.6):
    """표 bbox(pt)를 받아 TATR로 row+column 구조를 탐지하고, pdfplumber로 각 셀 텍스트를 채워
    [{label, cells: [...], row_top_pt}, ...] 형태로 반환(위->아래 순서 정렬)."""
    row_boxes, col_boxes, padded_pt, scale = _run_tatr(
        model, processor, doc_fitz, page_num, bbox_pt, render_dpi, pad_top_pt, pad_side_pt, conf)
    return _build_grid_rows(page_pp, row_boxes, col_boxes, padded_pt, scale)


def parse_simple_table_from_words(page_pp, bbox_pt: tuple, median_line_height_pt: float):
    """SIMPLE 표(작아서 TATR을 안 거치는 표)를 위한 대안 파서 — pdfplumber `extract_table()`을
    바로 신뢰하지 않고, `extract_words()`의 글자 단위 위치로 직접 행/열을 재구성한다.

    이유(발견한 버그): extract_table()의 내부 셀 클러스터링이 촘촘한 2단 정보 박스(예: 회사
    정보 박스)에서 서로 다른 컬럼의 텍스트를 글자 단위로 뒤섞는 경우가 있었음(예: "주요주주"+
    "LG 외 5인"+"국민연금공단"이 "주LG요 외주 주5 인" 처럼 인터리빙됨). 대신 각 단어(word)의
    좌표를 직접 읽어 (a) 그 페이지의 실제 글줄 간격(median_line_height_pt, 페이지마다 다르게
    계산됨 — adaptive_table_router.page_median_line_height() 재사용)로 행을 클러스터링하고,
    (b) 각 행 안에서 단어 사이 x간격이 가장 큰 지점을 라벨/값 경계로 판단해 분리한다 — 특정
    문자열을 하드코딩하지 않고 좌표 기하학만으로 판단하므로 다른 PDF에도 동일하게 적용된다."""
    words = page_pp.crop(bbox_pt).extract_words()
    if not words:
        return []
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    gap_threshold = median_line_height_pt * 0.6
    rows, current_row, current_top = [], [], None
    for w in words:
        if current_top is None or abs(w["top"] - current_top) <= gap_threshold:
            current_row.append(w)
            current_top = w["top"] if current_top is None else current_top
        else:
            rows.append(current_row)
            current_row, current_top = [w], w["top"]
    if current_row:
        rows.append(current_row)

    parsed = []
    for row_words in rows:
        row_words = sorted(row_words, key=lambda w: w["x0"])
        if not row_words:
            continue
        # 단어 사이 x간격이 가장 큰 지점을 라벨/값 경계로("현재가(07/20)" 같은 라벨과
        # "61,800원" 같은 값 사이엔 보통 표 안 다른 단어 사이 간격보다 큰 여백이 있음)
        if len(row_words) > 1:
            gaps = [(row_words[i + 1]["x0"] - row_words[i]["x1"], i) for i in range(len(row_words) - 1)]
            biggest_gap, split_after = max(gaps)
            split_idx = split_after + 1 if biggest_gap > median_line_height_pt * 0.4 else 1
        else:
            split_idx = 1
        label = clean_extracted_text(" ".join(w["text"] for w in row_words[:split_idx]).strip())
        value = clean_extracted_text(" ".join(w["text"] for w in row_words[split_idx:]).strip())
        if not label:
            continue
        parsed.append({"label": label, "cells": [value] if value else [], "row_top_pt": None})
    return parsed


MIN_COLS_FOR_TATR_GRID = 3  # 이 미만이면 word-clustering으로 대체(아래 설명)


def parse_table_adaptive(model, processor, doc_fitz, page_pp, page_num: int, bbox_pt: tuple,
                          render_dpi: int, pad_top_pt: float, pad_side_pt: float,
                          median_line_height_pt: float, conf: float = 0.6):
    """COMPLEX 표 파싱의 최종 진입점 — TATR이 탐지한 컬럼 수로 방법을 동적으로 고른다(발견한 버그
    기반 일반화 규칙, 특정 표를 하드코딩하지 않음):

    실측 결과 TATR의 row+column 격자 방식은 컬럼이 많은(>=3) 표(예: 계약공시표 6컬럼)에서는
    정확했지만, 컬럼이 2개뿐인 촘촘한 정보 박스(예: 회사 정보 박스, 세로로 라벨:값이 빽빽하게
    쌓인 레이아웃)에서는 행 경계 인식이 흔들려 서로 다른 행의 텍스트가 뒤섞이는 문제가 있었다
    (예: "주요주주"+"LG 외 5인"+"국민연금공단"이 뒤엉킴). 이런 좁은 레이아웃은 오히려 pdfplumber
    단어 좌표 기반 클러스터링(parse_simple_table_from_words)이 더 안정적이었다 — 그래서 "TATR이
    이 표에서 몇 개의 컬럼을 찾았는가"를 런타임에 확인해서 방법을 고른다(특정 PDF/표 이름이 아니라
    컬럼 수라는 구조적 특성으로 판단하므로 다른 PDF에도 동일하게 적용됨)."""
    row_boxes, col_boxes, padded_pt, scale = _run_tatr(
        model, processor, doc_fitz, page_num, bbox_pt, render_dpi, pad_top_pt, pad_side_pt, conf)

    if len(col_boxes) >= MIN_COLS_FOR_TATR_GRID:
        return _build_grid_rows(page_pp, row_boxes, col_boxes, padded_pt, scale)
    return parse_simple_table_from_words(page_pp, bbox_pt, median_line_height_pt)


# ---------- [JAEIL, pdf_pipeline/final/실험_4축_비교_스마트폰.md §14-17] 하이브리드 표 파서(v5) ----------
# 15문서 A/B(consensus 387개 pooled)로 검증된 표 파싱 방식. pdfplumber "text" 전략(선/괘선이
# 아니라 글자 좌표 기반이라 무테/와이드 표에도 강함)으로 먼저 뽑아보고, 그 결과가 "깨끗한 정형/
# 수치표"인지 _text_strategy_gate()로 판정 — 맞으면 그대로 채택(구조 완전 + 빠름), 아니면(회사
# 정보 박스처럼 불규칙한 텍스트표) 기존 word-clustering(parse_simple_table_from_words)으로
# 폴백한다. 실측(§17): TATR 대비 표 단계 지연 9.5~11.4배 감소, 엔티티 recall도 baseline 대비
# 유의(p<0.0001) 우세 + docling/MinerU 대비 근소 우세.
#
# [수정 — 3단계 에스컬레이션] 사용자 지적("TATR 강점 부분엔 폴백/라우팅 안 돼?") 반영 — TATR을
# 완전히 들어내는 대신 마지막 안전망으로 남긴다. §15.2에서 word-clustering(당시 v4의 text-
# strategy만 단독으로 씀)이 불규칙 표에서 라벨을 쪼개는 회귀가 있었고, v5는 그걸 "text-strategy
# 게이트 실패시 word-clustering"으로 고쳤지만, word-clustering 자체의 구조적 한계(라벨 뒤 나머지
# 전부를 cells[0] 한 칸에 뭉쳐 담음, 다중 컬럼을 못 나눔 — TATR이 진짜 강점을 가진 지점, 원래
# MIN_COLS_FOR_TATR_GRID 로직이 다루던 것과 동일 문제)는 여전히 남아 있다. 그래서 word-clustering
# 결과가 "값이 여러 개 뭉쳐 있어 보이면"(_word_clustering_looks_flattened) 그때만 TATR로 한 번 더
# 시도한다 — 표 대부분(§16.1 실측: 문서당 text-strategy 다수/word-cluster 소수)은 여전히 TATR을
# 전혀 안 타 속도 이점은 그대로 유지되고, 두 경량 방법이 모두 부실해 보이는 소수의 표에서만 TATR
# 비용을 지불한다. `run_table_metadata_pipeline.build_records()`가 원래 라우터의 complexity
# 신호(SIMPLE=default-strategy로도 이미 깨끗)로 이 3단계 후보 자체를 미리 걸러(SIMPLE 표는 TATR
# 에스컬레이션 대상에서 아예 제외) 불필요한 doc_fitz 사용도 줄인다.

_NUMERIC_CELL_RE = re.compile(r"^[\d.,%\-()\s]+$")
_MULTI_VALUE_RE = re.compile(r"-?\d[\d,]*\.?\d*")

_tatr_model = None
_tatr_processor = None
_tatr_lock = threading.Lock()


def _get_tatr_model():
    """[재도입, 지연 싱글턴] 3단계에서만 드물게 쓰이므로 표마다/문서마다 재로드하지 않도록
    embedding.get_embedding_model()과 동일한 패턴으로 프로세스당 1회만 로드.

    [수정] device를 항상 "cpu"로 고정 — 실측(run_investment_opinion_demo.py의 3브랜치
    ThreadPoolExecutor 동시 실행)으로 발견: TATR을 MPS에 올리면 같은 시점에 text 브랜치의
    BGE-m3-ko 임베딩(embedding.py, 역시 기본적으로 mps:0에 자동 로드됨)이 다른 스레드에서 동시에
    MPS 커널을 제출하면서 TATR의 grid 탐지 결과가 조용히(예외 없이) 틀어짐을 확인했다(같은 문서,
    같은 표(page3 table1)가 격리 실행 시 27개 canonical 매칭 -> 동시 실행 시 2개로 저하, TATR을
    CPU로 고정하니 27개로 복원). PyTorch MPS 백엔드가 멀티스레드 동시 제출에 안전하다는 보장이
    없어 생기는 문제로 보임 — TATR은 이제 드문 안전망 경로일 뿐이라 CPU 고정 비용은 작고(원래
    실측([37])도 MPS가 TATR에 겨우 9% 이득이었음), 매 질의마다 훨씬 자주 쓰이는 임베딩 모델의
    MPS 사용을 방해하지 않는 쪽이 전체적으로 더 안전하다."""
    global _tatr_model, _tatr_processor
    if _tatr_model is None:
        with _tatr_lock:
            if _tatr_model is None:
                from transformers import AutoImageProcessor, AutoModelForObjectDetection
                model = AutoModelForObjectDetection.from_pretrained(
                    "microsoft/table-transformer-structure-recognition")
                processor = AutoImageProcessor.from_pretrained(
                    "microsoft/table-transformer-structure-recognition")
                model.eval()
                _tatr_model, _tatr_processor = model.to("cpu"), processor
    return _tatr_model, _tatr_processor


def _text_strategy_gate(table: list) -> bool:
    """pdfplumber text-strategy 추출 결과가 신뢰할 만한지 판정. True면 text-strategy 결과를 그대로
    쓰고, False면 word-clustering(그다음 필요하면 TATR)으로 폴백. 표마다 동적으로 판단(특정 표/
    PDF 하드코딩 아님):
    (a) 행별 비어있지 않은 셀 개수의 최빈값 기준 일관성이 50% 이상이고 최빈값이 2열 이상이거나,
    (b) 전체 셀 중 숫자로만 이뤄진 셀 비율이 55% 이상이면(숫자 위주 재무표는 컬럼이 안 맞아도
    text-strategy가 값 자체는 항상 올바르게 뽑음) 신뢰."""
    if not table or len(table) < 3:
        return False
    counts = [sum(1 for c in row if c and str(c).strip()) for row in table]
    counts = [c for c in counts if c > 0]
    if not counts:
        return False
    modal = Counter(counts).most_common(1)[0][0]
    consistency = sum(1 for c in counts if c == modal) / len(counts)
    cells = [str(c).strip() for row in table for c in row if c and str(c).strip()]
    numeric_ratio = sum(1 for c in cells if _NUMERIC_CELL_RE.match(c)) / max(1, len(cells))
    return (consistency >= 0.5 and modal >= 2) or numeric_ratio >= 0.55


def _word_clustering_looks_flattened(parsed_rows: list) -> bool:
    """word-clustering은 라벨 뒤 나머지를 통째로 한 셀(cells[0])에 담으므로, 그 안에 숫자 토큰이
    여러 개(예: "479.0 512.3 734.5"처럼 연도별 값이 한 셀에 뭉침) 있으면 실제로는 다중 컬럼 구조가
    한 셀로 뭉개진 것 — TATR의 강점(행+열 격자 인식)이 필요한 신호. 행이 아예 없어도(구조 파악
    실패) 에스컬레이션 대상."""
    if not parsed_rows:
        return True
    flattened = sum(
        1 for row in parsed_rows
        if row["cells"] and len(_MULTI_VALUE_RE.findall(row["cells"][0])) >= 3
    )
    return flattened / len(parsed_rows) >= 0.3


def parse_table_hybrid(page_pp, bbox_pt: tuple, median_line_height_pt: float,
                        doc_fitz=None, page_num: int = None, tatr_render_dpi: int = 300,
                        tatr_pad_top_pt: float = 35 / (150 / 72), tatr_pad_side_pt: float = 12 / (150 / 72),
                        tatr_conf: float = 0.6):
    """[JAEIL v5 + 3단계 에스컬레이션] 표 파싱 진입점. 1) pdfplumber text-strategy 시도 —
    _text_strategy_gate()가 신뢰하면 즉시 채택(가장 흔한 경로, TATR 전혀 안 씀). 2) 신뢰 못 하면
    word-clustering(parse_simple_table_from_words) — 대부분의 불규칙 표는 여기서 해결. 3) doc_fitz/
    page_num이 주어졌고(호출측이 이 표를 TATR 후보로 판단한 경우만 — 보통 라우터의 complexity=
    "complex"인 표만) word-clustering 결과가 _word_clustering_looks_flattened()로 부실해 보이면,
    그때만 TATR(parse_table_adaptive)로 마지막 시도. TATR도 실패하면(예외) word-clustering 결과로
    안전하게 되돌아간다. 모든 경로가 동일한 [{label, cells, row_top_pt}] 스키마를 반환하므로 호출측
    (canonical field 매칭 등)은 무수정."""
    try:
        table = page_pp.crop(bbox_pt).extract_table(
            {"vertical_strategy": "text", "horizontal_strategy": "text"})
    except Exception:
        table = None

    if _text_strategy_gate(table):
        parsed = []
        for row in table:
            cells = [clean_extracted_text(str(c).replace("\n", " ").strip()) if c else "" for c in row]
            if not any(cells):
                continue
            label, values = cells[0], cells[1:]
            parsed.append({"label": label, "cells": values, "row_top_pt": None})
        if parsed:
            return parsed

    word_clustered = parse_simple_table_from_words(page_pp, bbox_pt, median_line_height_pt)

    if doc_fitz is not None and page_num is not None and _word_clustering_looks_flattened(word_clustered):
        try:
            model, processor = _get_tatr_model()
            tatr_rows = parse_table_adaptive(model, processor, doc_fitz, page_pp, page_num, bbox_pt,
                                              tatr_render_dpi, tatr_pad_top_pt, tatr_pad_side_pt,
                                              median_line_height_pt, conf=tatr_conf)
            if tatr_rows:
                return tatr_rows
        except Exception:
            pass  # TATR 자체가 실패해도 이미 있는 word-clustering 결과로 안전하게 폴백

    return word_clustered

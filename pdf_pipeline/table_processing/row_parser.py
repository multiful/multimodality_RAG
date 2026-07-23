"""[19] Row Parser — 사용자 피드백 (3) 반영: "Regex만 쓰는 것은 약하다. Row Parser가 낫다."

TATR로 표의 row bbox뿐 아니라 column bbox까지 탐지하고, 각 (row, column) 교차 영역을 하나의
셀로 취급해 pdfplumber로 텍스트를 채운다. 첫 번째 컬럼(가장 왼쪽)을 라벨로, 나머지 컬럼들을
값 배열(cells)로 분리 — "2026E/2027E처럼 컬럼이 여러 개인 경우"를 label 하나에 값 여러 개로
자연스럽게 표현할 수 있다(계약상대방처럼 텍스트 값도, 수주잔고처럼 연도별 숫자 여러 개도 동일 구조).
"""

import fitz
from PIL import Image
import torch


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
        return (table_page.crop(cell_pt).extract_text() or "").replace("\n", " ").strip()

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
            whole = (table_page.crop(row_pt).extract_text() or "").replace("\n", " ").strip()
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
        label = " ".join(w["text"] for w in row_words[:split_idx]).strip()
        value = " ".join(w["text"] for w in row_words[split_idx:]).strip()
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

"""[8] 페이지당 YOLO 호출을 한 번으로 통합하기 위한 공유 유틸.

사용자 지적 반영: `reading_order_router.py`(난이도 판정)와 `hierarchical_chunker.py`(문단
경계+section_path)가 각자 자기 몫의 YOLO 추론을 따로 호출하고 있었음(페이지당 2회) — 렌더링+
추론이 이 파이프라인에서 가장 비싼 연산이라 이중 호출은 그대로 지연 낭비. 이 모듈의
`run_yolo_layout()` 하나만 페이지당 1번 호출하고, 그 결과([(cls_name, fitz.Rect), ...])를
난이도 판정과 계층적 청킹 양쪽에 그대로 넘겨 재사용한다.
"""

import fitz
from PIL import Image

RENDER_DPI = 150
CONF_THRESHOLD = 0.25


def run_yolo_layout(model, page: fitz.Page, page_idx: int) -> list:
    """페이지를 렌더링 -> YOLO 예측 -> [(cls_name, fitz.Rect(pt 단위)), ...] 반환(필터링 없음 —
    호출자가 필요한 클래스만 골라 쓴다).

    [33] 성능 수정: 이전엔 렌더링한 이미지를 PNG로 디스크에 저장했다가 곧바로 다시 읽어들였다
    (인코딩+디스크 write+디스크 read+디코딩 왕복) — `pix.samples`(원본 RGB 바이트)를
    `Image.frombytes()`로 바로 PIL 이미지로 변환하면 디스크 왕복이 통째로 사라진다. 실측
    50회 기준 21.9ms/회 -> 0.9ms/회(약 25배). 이 함수는 page_classification/text_processing/
    table_processing 전부가 공유하는 유틸이라 여기 한 곳만 고쳐도 세 파이프라인 전부에 적용됨."""
    pix = page.get_pixmap(dpi=RENDER_DPI)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    res = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]

    names = model.names
    boxes = res.boxes
    SCALE = RENDER_DPI / 72
    result = []
    if boxes is not None:
        for cls_idx, xyxy in zip(boxes.cls, boxes.xyxy):
            cls_name = names[int(cls_idx)]
            x1, y1, x2, y2 = [v / SCALE for v in xyxy.tolist()]
            result.append((cls_name, fitz.Rect(x1, y1, x2, y2)))
    return result

"""[2]/[4] PDF 난이도 판별 라우터 — 우리 파이프라인(whole_page PyMuPDF)으로 처리 가능한 "쉬운" PDF와
리딩오더 복원이 필요한 "어려운" PDF(MinerU 등으로 위임)를 자동으로 구분.

[2]까지는 "컬럼 겹침이 있느냐 없느냐"만 보는 이진 판정이었는데, 사용자 피드백(단일 신호로는
약하다)을 반영해 **가중합 난이도 스코어**로 고도화:

    difficulty_score = w1*columns + w2*overlap + w3*rotation + w4*font_variance + w5*figure_density

각 항은 0~1로 정규화. columns/overlap은 [1]/[2]에서 이미 실측 검증된 신호라 가중치를 가장 높게
주고(0.30/0.25), rotation/font_variance/figure_density는 이번에 새로 추가한 신호라 상대적으로
낮은 가중치(0.15씩)로 시작 — "본문 컬럼이 실제로 겹친다"는 확증된 신호가 여전히 판정을 지배하되,
컬럼 겹침이 애매한 경계 사례에서 다른 레이아웃 복잡도 신호가 보조적으로 거들도록 설계했다.

- **columns**: 겹치는 컬럼 쌍이 하나라도 material(각 200자 이상)하면 1.0, 아니면 클러스터 수 기반 약한 신호
- **overlap**: 겹치는 컬럼 쌍의 y범위 겹침 정도를 페이지 높이 대비 비율로
- **rotation**: PyMuPDF `get_text("dict")`의 line.dir가 수평(1,0)이 아닌 라인의 비율
- **font_variance**: 페이지 내 폰트 크기의 변동계수(std/mean) — 잡지/브로슈어처럼 제목·본문·캡션
  크기가 들쭉날쭉하면 커짐(사용자가 언급한 "레이아웃 복잡도"의 정량화)
- **figure_density**: YOLO Table+Picture 박스가 차지하는 페이지 면적 비율(그림/표 밀도)

한계(정직하게 기록): 실제 hard 문서가 없어 rotation/font_variance/figure_density 가중치는 아직
실측 캘리브레이션이 안 된 초기값 — columns/overlap만으로 검증된 [2]의 결론(easy/hard 양쪽 분기
모두 확인)은 유지되나, 새로 추가한 3개 신호가 실제 판정에 미치는 영향은 그런 레이아웃(회전된
스캔본, 폰트가 들쭉날쭉한 브로슈어, 그림이 빽빽한 잡지)을 가진 PDF가 생기면 재검증 필요.
"""

import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # [10] yolo_layout이 pdf_pipeline/로 이동
from yolo_layout import run_yolo_layout  # noqa: E402 — [8] 공유 YOLO 호출(중복 추론 제거, [10]에서 page_classification과도 공유하도록 위치 이동)

CONF_THRESHOLD = 0.25
NON_TEXT_CLASSES = {"Table", "Picture"}
TEXT_CLASSES = {"Text", "Title", "Section-header", "List-item", "Caption"}


@dataclass
class ColumnRouterThresholds:
    gap_pt: float = 60.0            # 이 이상 벌어지면 새 x클러스터 후보([1]에서 쓴 값 그대로)
    min_y_overlap_pt: float = 30.0  # 두 클러스터의 y범위가 이 이상 겹쳐야 "진짜 나란히 배치"로 판단
    min_chars_for_hard: int = 200   # 겹치는 두 클러스터 각각의 글자 수가 이 이상이어야 "본문 컬럼"


@dataclass
class DifficultyWeights:
    w_columns: float = 0.30
    w_overlap: float = 0.25
    w_rotation: float = 0.15
    w_font_variance: float = 0.15
    w_figure_density: float = 0.15
    hard_threshold: float = 0.40


@dataclass
class PageDifficultyResult:
    page: int
    difficulty: str                 # "easy" | "hard"
    reason: str
    difficulty_score: float
    signals: dict                   # 정규화된 개별 신호값(디버깅/재캘리브레이션용)
    n_text_blocks: int
    n_clusters: int
    material_overlaps: list = field(default_factory=list)   # 근거로 쓰인 겹치는 클러스터 쌍 정보
    layout_class_diversity: int = 0  # 참고용 신호(제목/본문/캡션/사이드바 등 블록 타입 다양성)


def _cluster_by_x0(blocks: list, gap_pt: float) -> list:
    sb = sorted(blocks, key=lambda b: b[0])
    clusters = [[sb[0]]]
    for b in sb[1:]:
        if b[0] - clusters[-1][-1][0] > gap_pt:
            clusters.append([b])
        else:
            clusters[-1].append(b)
    return clusters


def _yrange(cluster: list):
    return min(b[1] for b in cluster), max(b[3] for b in cluster)


def _rotation_fraction(page: fitz.Page) -> float:
    """텍스트 라인 중 수평(dir=(1,0))이 아닌 비율 — 회전된 스캔본/세로쓰기 감지."""
    d = page.get_text("dict")
    total, rotated = 0, 0
    for block in d["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            total += 1
            dx, dy = line["dir"]
            if abs(dy) > 0.1:
                rotated += 1
    return rotated / total if total else 0.0


def _font_size_cv(page: fitz.Page) -> float:
    """페이지 내 폰트 크기의 변동계수(std/mean), 1.0으로 클립 — 레이아웃 복잡도(제목/본문/캡션
    크기가 들쭉날쭉한 잡지·브로슈어형)의 정량화."""
    d = page.get_text("dict")
    sizes = [span["size"] for block in d["blocks"] if block["type"] == 0
             for line in block["lines"] for span in line["spans"]]
    if len(sizes) < 2:
        return 0.0
    mean = statistics.mean(sizes)
    if mean == 0:
        return 0.0
    cv = statistics.pstdev(sizes) / mean
    return min(cv, 1.0)


def _is_excluded(bbox: fitz.Rect, exclude_rects: list, overlap_threshold: float = 0.5) -> bool:
    """[30] 사용자 지적("hard 페이지가 정말 리딩오더 복원이 필요한 페이지인지") 검증 중 발견한 버그
    수정: 기존엔 블록 "중심점"이 exclude_rect 안에 있는지만 봤는데, PyMuPDF가 나란히 붙은 두 YOLO
    Table 박스(예: 재무상태표+손익계산서가 별개 박스로 검출된 표)를 하나의 거대한 텍스트 블록으로
    합쳐 반환하면, 그 블록의 중심점이 두 Table 박스 "사이 틈"에 떨어져 어느 쪽에도 안 걸리는 경우가
    실측(K-Wave PDF)에서 발견됨 — 결과적으로 표 전체가 리딩오더 텍스트 블록으로 잘못 남아 난이도
    판정을 오염시킴(실제로는 표 영역이라 이 페이지의 텍스트 블록 수가 비정상적으로 적어짐, 예:
    헤더+표덩어리+푸터 3개뿐). 중심점 포함 대신 "블록 면적 대비 exclude_rects와 겹치는 면적 비율"로
    바꿔서, 여러 Table 박스에 걸쳐 있어도 합산 겹침 비율이 높으면(기본 50%) 제외되도록 수정."""
    if bbox.width * bbox.height <= 0:
        return False
    total_overlap = 0.0
    for r in exclude_rects:
        ix0, iy0 = max(bbox.x0, r.x0), max(bbox.y0, r.y0)
        ix1, iy1 = min(bbox.x1, r.x1), min(bbox.y1, r.y1)
        if ix1 > ix0 and iy1 > iy0:
            total_overlap += (ix1 - ix0) * (iy1 - iy0)
    return (total_overlap / (bbox.width * bbox.height)) > overlap_threshold


def _figure_density(page: fitz.Page, exclude_rects: list) -> float:
    """YOLO Table+Picture 박스 면적 / 페이지 전체 면적."""
    page_area = page.rect.width * page.rect.height
    if page_area == 0:
        return 0.0
    fig_area = sum(r.width * r.height for r in exclude_rects)
    return min(fig_area / page_area, 1.0)


def assess_page_difficulty(model, doc_fitz, page_idx: int,
                            thresholds: ColumnRouterThresholds = None,
                            weights: DifficultyWeights = None,
                            cached_boxes: list = None) -> PageDifficultyResult:
    """한 페이지의 리딩오더 복원 필요 여부를 가중합 난이도 스코어로 판정.

    cached_boxes: [5] 사용자 피드백 반영 — page_classification 단계가 이미 같은 페이지에 대해
    YOLO를 돌린 결과가 있으면 [(cls_name, fitz.Rect), ...] 형태로 여기에 넘겨서 중복 추론을
    피할 수 있음(비용 큰 게 렌더링+YOLO 추론이라, 재사용시 이 페이지 처리는 사실상 무료가 됨).
    안 주면 기존처럼 직접 렌더링+예측."""
    th = thresholds or ColumnRouterThresholds()
    w = weights or DifficultyWeights()
    page = doc_fitz[page_idx]

    yolo_boxes = cached_boxes if cached_boxes is not None else run_yolo_layout(model, page, page_idx)

    exclude_rects, text_classes_found = [], set()
    for cls_name, rect in yolo_boxes:
        if cls_name in NON_TEXT_CLASSES:
            exclude_rects.append(rect)
        elif cls_name in TEXT_CLASSES:
            text_classes_found.add(cls_name)

    all_blocks = page.get_text("blocks")
    text_blocks = []
    for b in all_blocks:
        if b[6] != 0 or not b[4].strip():
            continue
        bbox = fitz.Rect(b[:4])
        if _is_excluded(bbox, exclude_rects):
            continue
        text_blocks.append(b)

    # --- 보조 신호(컬럼/겹침과 무관하게 항상 계산) ---
    rotation_signal = _rotation_fraction(page)
    font_variance_signal = _font_size_cv(page)
    figure_density_signal = _figure_density(page, exclude_rects)

    def _finalize(difficulty, reason, columns_signal, overlap_signal, n_clusters, material_overlaps):
        score = (w.w_columns * columns_signal + w.w_overlap * overlap_signal
                 + w.w_rotation * rotation_signal + w.w_font_variance * font_variance_signal
                 + w.w_figure_density * figure_density_signal)
        # material_overlaps(본문 분량 컬럼이 실제로 나란히 겹침)는 [1]/[2]에서 유일하게 실측
        # 검증된 확증 신호라 가중합 점수와 무관하게 항상 hard로 확정한다 — 가중합에만 맡기면
        # 다른 신호(rotation/font_variance/figure_density)가 전부 0인 "딱 2컬럼만 있고 나머지는
        # 평범한" 문서에서 컬럼 가중치(0.30)만으로는 threshold(0.40)를 못 넘어 easy로 오판되는
        # 회귀를 합성 2단 PDF 재검증 중 실측으로 발견 — 그래서 확증 신호는 override로 분리.
        # 가중합 점수는 material_overlaps가 없을 때(회전/폰트/그림밀도 등 다른 이유로 어려운
        # 문서일 수 있는 경계 사례)만 보조적으로 사용.
        final_difficulty = "hard" if material_overlaps or score >= w.hard_threshold else difficulty
        return PageDifficultyResult(
            page=page_idx + 1, difficulty=final_difficulty, reason=reason,
            difficulty_score=round(score, 4),
            signals={"columns": round(columns_signal, 4), "overlap": round(overlap_signal, 4),
                     "rotation": round(rotation_signal, 4), "font_variance": round(font_variance_signal, 4),
                     "figure_density": round(figure_density_signal, 4)},
            n_text_blocks=len(text_blocks), n_clusters=n_clusters,
            material_overlaps=material_overlaps, layout_class_diversity=len(text_classes_found),
        )

    if len(text_blocks) < 2:
        return _finalize("easy", "텍스트 블록 1개 이하(컬럼 자체가 성립 안 함)", 0.0, 0.0, 0, [])

    clusters = _cluster_by_x0(text_blocks, th.gap_pt)
    if len(clusters) < 2:
        return _finalize("easy", "x클러스터 1개(단일 컬럼)", 0.0, 0.0, 1, [])

    material_overlaps, max_overlap_ratio = [], 0.0
    page_height = page.rect.height
    for ci in range(len(clusters)):
        for cj in range(ci + 1, len(clusters)):
            y1a, y2a = _yrange(clusters[ci])
            y1b, y2b = _yrange(clusters[cj])
            ov = max(0, min(y2a, y2b) - max(y1a, y1b))
            if ov <= th.min_y_overlap_pt:
                continue
            max_overlap_ratio = max(max_overlap_ratio, ov / page_height if page_height else 0.0)
            chars_a = sum(len(b[4]) for b in clusters[ci])
            chars_b = sum(len(b[4]) for b in clusters[cj])
            if chars_a >= th.min_chars_for_hard and chars_b >= th.min_chars_for_hard:
                material_overlaps.append({
                    "cluster_pair": (ci, cj), "y_overlap_pt": round(ov, 1),
                    "chars": (chars_a, chars_b),
                })

    if material_overlaps:
        columns_signal, overlap_signal = 1.0, max_overlap_ratio
        reason = (f"본문 분량의 텍스트 컬럼 {len(material_overlaps)}쌍이 나란히 겹쳐 있음 "
                  f"(각 {th.min_chars_for_hard}자 이상) — 리딩오더 복원 권장")
    else:
        columns_signal = min((len(clusters) - 1) / 3, 1.0)
        overlap_signal = max_overlap_ratio
        reason = (f"x클러스터 {len(clusters)}개 있으나 겹치는 쌍이 없거나 전부 소량(boilerplate 수준)"
                  f" — 다른 레이아웃 신호로 최종 판정")

    return _finalize("easy", reason, columns_signal, overlap_signal, len(clusters), material_overlaps)


def assess_pdf(pdf_path, model=None, thresholds: ColumnRouterThresholds = None,
               weights: DifficultyWeights = None, page_boxes: dict = None) -> dict:
    """PDF 전체를 페이지별로 판정하고 요약 반환. model을 안 주면 기본 YOLOv11n을 로드.

    page_boxes: [5] 사용자 피드백 반영 — {page_idx(0-based): [(cls_name, fitz.Rect), ...]} 형태로
    다른 단계(예: page_classification)에서 이미 계산한 YOLO 결과를 넘기면 페이지별 재추론을
    건너뛴다. 실측(아래 [5] 참고): PDF당 YOLO 재호출을 생략하면 페이지당 렌더링+추론 비용
    (~100~300ms/페이지)이 전부 제거됨."""
    if model is None and page_boxes is None:
        model_path = (Path(__file__).resolve().parent.parent / "page_classification"
                      / "models" / "yolo11n_doc_layout.pt")
        model = YOLO(str(model_path))
        warmup = Image.new("RGB", (595, 842), (255, 255, 255))
        model.predict(warmup, conf=CONF_THRESHOLD, verbose=False)

    doc_fitz = fitz.open(str(pdf_path))
    page_results = [
        assess_page_difficulty(model, doc_fitz, i, thresholds, weights,
                                cached_boxes=(page_boxes.get(i) if page_boxes else None))
        for i in range(doc_fitz.page_count)
    ]
    doc_fitz.close()

    n_hard = sum(1 for r in page_results if r.difficulty == "hard")
    return {
        "pdf": str(pdf_path), "n_pages": len(page_results), "n_hard_pages": n_hard,
        "hard_page_numbers": [r.page for r in page_results if r.difficulty == "hard"],
        "route_to_mineru": n_hard > 0,
        "pages": [
            {"page": r.page, "difficulty": r.difficulty, "reason": r.reason,
             "difficulty_score": r.difficulty_score, "signals": r.signals,
             "n_text_blocks": r.n_text_blocks, "n_clusters": r.n_clusters,
             "material_overlaps": r.material_overlaps,
             "layout_class_diversity": r.layout_class_diversity}
            for r in page_results
        ],
    }

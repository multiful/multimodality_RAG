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

[38] 위 한계가 실제로 문제를 일으킨 게 확인됨 — 클로드가 K-Wave 73페이지를 전수 검토(`[34]`)한
결과 rotation/font_variance/figure_density가 밀어올린 가중합 점수로 hard 판정된 33페이지가
전부 "실제로는 리딩오더 문제 없음"이었다(차트/표 중심 리포트에서 이 신호들은 난이도가 아니라
그냥 "차트가 많다"만 의미했음). 그래서 **hard 판정 자체는 더 이상 가중합 점수로 내리지 않고**,
`material_overlaps`(컬럼이 실제로 겹침) + `_interleaving_excess()`(원문 추출 순서가 그 겹치는
컬럼 사이를 실제로 오가는가, 그냥 컬럼별로 그룹화돼서 나오면 안 헷갈림) 두 조건을 **동시에**
만족할 때만 hard로 확정한다. 가중합 점수/개별 신호는 `signals`에 참고용으로만 남겨둠(디버깅/
재캘리브레이션 대비) — 이걸로 K-Wave 34개(모두 오탐) -> 0개(클로드 판정과 정확히 일치)로 수정.
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
    min_interleaving_excess: int = 2  # [38] 이 이상 "초과 전환"이 있어야 실제 인터리빙으로 판단
    # (1 정도는 제목 하나 위치가 어긋나는 정도의 노이즈로 취급 — [34] page3 사례처럼 컬럼별로
    # 그룹화는 됐는데 제목 하나만 어긋난 경우까지 hard로 잡으면 또 과다판정이 됨)


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
    interleaving_excess: int = 0     # [38] 컬럼 사이 실제 인터리빙 정도(0=그룹화됨, 클수록 심함)


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


def _interleaving_excess(text_blocks: list, clusters: list) -> int:
    """[38] 사용자 지적("클로드 판정을 기준으로 거의 비슷하게 분류되도록") 반영 — `[34]`에서
    클로드가 K-Wave 73페이지를 전수 검토한 결과, "컬럼이 겹친다"(현재 material_overlaps 신호)와
    "실제로 읽었을 때 혼동된다"는 별개였다: 여러 컬럼이 있어도 원문 추출 순서가 **컬럼별로
    통째로 그룹화**돼 있으면(컬럼1 전체 -> 컬럼2 전체 -> ...) 사람이/LLM이 읽어도 문제없고,
    실제로 순서가 컬럼 사이를 왔다갔다(인터리빙)해야 진짜 혼동이 생긴다.

    text_blocks(페이지에서 원문 그대로 추출된 순서)를 그대로 순회하면서 각 블록이 어느
    x클러스터에 속하는지 시퀀스로 만들고, "클러스터마다 한 번씩만 몰아서 방문"하는 이상적인
    경우의 최소 전환 횟수(클러스터 수-1) 대비 실제 전환 횟수가 얼마나 초과하는지를 반환한다
    (0=완전히 그룹화됨=인터리빙 없음, 클수록 컬럼 사이를 자주 오간다는 뜻)."""
    block_to_cluster = {}
    for ci, cluster in enumerate(clusters):
        for b in cluster:
            block_to_cluster[id(b)] = ci
    seq = [block_to_cluster[id(b)] for b in text_blocks if id(b) in block_to_cluster]
    if len(seq) < 2:
        return 0
    transitions = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    ideal = len(set(seq)) - 1
    return max(0, transitions - ideal)


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

    def _finalize(difficulty, reason, columns_signal, overlap_signal, n_clusters, material_overlaps,
                  interleaving_excess=0):
        score = (w.w_columns * columns_signal + w.w_overlap * overlap_signal
                 + w.w_rotation * rotation_signal + w.w_font_variance * font_variance_signal
                 + w.w_figure_density * figure_density_signal)
        # [38] 사용자 지적("클로드 판정 기준으로 재분류")으로 K-Wave 73페이지를 다시 돌려보니,
        # material_overlaps override를 `_interleaving_excess()` 조건으로 고친 뒤에도 나머지
        # 33페이지가 **가중합 점수(score >= hard_threshold) 경로**로 여전히 hard 판정되고 있었음
        # — 전부 rotation/font_variance/figure_density(애초에 "실측 미검증 초기값"이라고 [4]에
        # 정직하게 적어뒀던 신호들)가 밀어올린 점수였다. 클로드가 이 33페이지를 전부 "리딩오더
        # 문제 없음"으로 판정했으니(`[34]`), 이 신호들이 이 문서 유형(차트/표 중심 증권 리포트)
        # 에서는 "리딩오더 필요성"과 상관관계가 없다는 게 실측으로 확인된 셈 — 애초에 검증 안 된
        # 채로 hard 판정을 단독으로 내릴 수 있게 해둔 게 설계 결함이었음. 이제 가중합 점수는
        # signals에 참고용으로만 남기고, **hard 판정은 material_overlaps+interleaving 하나로만
        # 결정**한다(rotation/font_variance/figure_density가 단독으로 hard를 강제하지 못하게).
        material_and_interleaved = bool(material_overlaps) and interleaving_excess >= th.min_interleaving_excess
        final_difficulty = "hard" if material_and_interleaved else difficulty
        return PageDifficultyResult(
            page=page_idx + 1, difficulty=final_difficulty, reason=reason,
            difficulty_score=round(score, 4),
            signals={"columns": round(columns_signal, 4), "overlap": round(overlap_signal, 4),
                     "rotation": round(rotation_signal, 4), "font_variance": round(font_variance_signal, 4),
                     "figure_density": round(figure_density_signal, 4)},
            n_text_blocks=len(text_blocks), n_clusters=n_clusters,
            material_overlaps=material_overlaps, layout_class_diversity=len(text_classes_found),
            interleaving_excess=interleaving_excess,
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

    interleaving_excess = _interleaving_excess(text_blocks, clusters) if material_overlaps else 0

    if material_overlaps and interleaving_excess >= th.min_interleaving_excess:
        columns_signal, overlap_signal = 1.0, max_overlap_ratio
        reason = (f"본문 분량의 텍스트 컬럼 {len(material_overlaps)}쌍이 나란히 겹쳐 있고, 추출 순서도 "
                  f"실제로 컬럼 사이를 오감(초과 전환 {interleaving_excess}회) — 리딩오더 복원 권장")
    elif material_overlaps:
        columns_signal, overlap_signal = 1.0, max_overlap_ratio
        reason = (f"텍스트 컬럼 {len(material_overlaps)}쌍이 겹치지만 추출 순서는 컬럼별로 그룹화돼 "
                  f"있음(초과 전환 {interleaving_excess}회 < 임계 {th.min_interleaving_excess}) — "
                  f"[38] 실제로는 안 헷갈리는 것으로 판단, 가중합 점수로만 보조 판정")
    else:
        columns_signal = min((len(clusters) - 1) / 3, 1.0)
        overlap_signal = max_overlap_ratio
        reason = (f"x클러스터 {len(clusters)}개 있으나 겹치는 쌍이 없거나 전부 소량(boilerplate 수준)"
                  f" — 다른 레이아웃 신호로 최종 판정")

    return _finalize("easy", reason, columns_signal, overlap_signal, len(clusters), material_overlaps,
                     interleaving_excess)


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

"""[3] 표 크롭 고도화 — Adaptive Padding + Caption 공간매핑 + 부분 고해상도 렌더링.

1) Adaptive Padding: 상단 +35px(캡션 포함 목적, 150dpi 픽셀 기준) / 좌우·하단 +12px(타이트, 밀집 표 침범 방지)
2) Spatial Nearest Caption Mapping: 'Caption'/'Section-header' bbox 중 표 바로 위에서 가장 가까운 것을
   유클리드 거리로 1:1 매칭해 표별 제목 메타데이터로 저장(LLM 프롬프트용 로컬 앵커)
3) DPI Sub-rendering: YOLO 탐지는 150dpi(빠름) 그대로, 최종 크롭은 PDF 원본에서 300dpi로 해당
   bbox만 부분 재렌더링(PyMuPDF clip)해서 Docling엔 고해상도 이미지를 공급

YOLO11n/YOLO26n 둘 다에 동일 로직 적용 가능(모델 경로만 인자로 받음).
"""

import json
import math
import sys
from pathlib import Path

import fitz
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "20260721_company_279243000.pdf"
OUT_DIR = Path(__file__).resolve().parent

CONF_THRESHOLD = 0.25
DETECT_DPI = 150          # YOLO 탐지용(빠름)
RENDER_DPI = 300          # 최종 크롭용(고해상도)
DETECT_SCALE = DETECT_DPI / 72
RENDER_SCALE = RENDER_DPI / 72

PAD_TOP_PX = 35    # 150dpi 픽셀 기준 상단 패딩(캡션 포함)
PAD_SIDE_PX = 12   # 150dpi 픽셀 기준 좌우/하단 패딩(타이트)

CAPTION_CLASSES = {"Caption", "Section-header"}
MAX_CAPTION_DIST_PX = 200  # 이보다 멀면 "이 표 전용 캡션 없음"으로 판단(억지 매칭 방지)


def nearest_caption_above(table_box, caption_boxes):
    """table_box: (x1,y1,x2,y2). caption_boxes: [(x1,y1,x2,y2), ...] 같은 페이지의 Caption/Section-header."""
    tx1, ty1, tx2, ty2 = table_box
    t_top_center = ((tx1 + tx2) / 2, ty1)
    best, best_dist = None, None
    for cb in caption_boxes:
        cx1, cy1, cx2, cy2 = cb
        if cy2 > ty1 + 5:  # 표보다 아래/많이 안쪽이면 "위 캡션" 후보에서 제외
            continue
        c_bottom_center = ((cx1 + cx2) / 2, cy2)
        dist = math.hypot(t_top_center[0] - c_bottom_center[0], t_top_center[1] - c_bottom_center[1])
        if best_dist is None or dist < best_dist:
            best, best_dist = cb, dist
    if best_dist is not None and best_dist > MAX_CAPTION_DIST_PX:
        return None, best_dist  # 너무 멀면 매칭 포기(거리 값은 기록용으로 반환)
    return best, best_dist


def main(model_name: str, model_path: Path, crop_dir_name: str):
    crop_dir = OUT_DIR / crop_dir_name
    crop_dir.mkdir(exist_ok=True)
    model = YOLO(str(model_path))
    doc_fitz = fitz.open(str(PDF_PATH))

    meta = {}
    total_tables = 0
    for i, page_fz in enumerate(doc_fitz, start=1):
        pix = page_fz.get_pixmap(dpi=DETECT_DPI)
        tmp_path = OUT_DIR / f"_tmp_enh_p{i}.png"
        pix.save(str(tmp_path))
        img = Image.open(tmp_path).convert("RGB")
        page_w_px, page_h_px = img.size

        results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
        names = model.names
        boxes = results.boxes
        tmp_path.unlink(missing_ok=True)
        if boxes is None:
            continue

        table_boxes, caption_boxes = [], []
        for cls_idx, xyxy in zip(boxes.cls, boxes.xyxy):
            cls_name = names[int(cls_idx)]
            box = tuple(float(v) for v in xyxy.tolist())
            if cls_name == "Table":
                table_boxes.append(box)
            elif cls_name in CAPTION_CLASSES:
                caption_boxes.append(box)

        for t_idx, tb in enumerate(table_boxes, start=1):
            total_tables += 1
            x1, y1, x2, y2 = tb
            # 1) Adaptive Padding (150dpi 픽셀 좌표 기준), 페이지 경계 안으로 clamp
            px1 = max(0, x1 - PAD_SIDE_PX)
            py1 = max(0, y1 - PAD_TOP_PX)
            px2 = min(page_w_px, x2 + PAD_SIDE_PX)
            py2 = min(page_h_px, y2 + PAD_SIDE_PX)

            # 2) Spatial Nearest Caption Mapping (패딩 전 원본 표 bbox 기준으로 매칭)
            cap_box, cap_dist = nearest_caption_above(tb, caption_boxes)
            cap_text = None  # 실제 텍스트는 pdfplumber로 별도 추출(아래)

            # 3) DPI Sub-rendering: 패딩된 bbox를 pt 좌표로 환산해 300dpi로 부분 재렌더링
            rect_pt = fitz.Rect(px1 / DETECT_SCALE, py1 / DETECT_SCALE, px2 / DETECT_SCALE, py2 / DETECT_SCALE)
            hi_res_pix = page_fz.get_pixmap(dpi=RENDER_DPI, clip=rect_pt)
            crop_path = crop_dir / f"page_{i}_table_{t_idx}.png"
            hi_res_pix.save(str(crop_path))

            meta[str(crop_path.relative_to(ROOT))] = {
                "page": i, "table_idx": t_idx,
                "table_bbox_150dpi_px": tb,
                "padded_bbox_150dpi_px": [px1, py1, px2, py2],
                "caption_bbox_150dpi_px": list(cap_box) if cap_box else None,
                "caption_distance_px": round(cap_dist, 1) if cap_dist is not None else None,
                "render_dpi": RENDER_DPI,
            }
            if cap_box:
                print(f"page{i} table{t_idx}: 캡션매칭=O(거리{cap_dist:.0f}px)", flush=True)
            else:
                print(f"page{i} table{t_idx}: 캡션매칭=X", flush=True)
    doc_fitz.close()

    meta_path = crop_dir.parent / f"{crop_dir_name}_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[{model_name}] 총 {total_tables}개 표 크롭(고도화: 비대칭 패딩+캡션매핑+300dpi) 생성 완료")
    print(f"[meta] saved to {meta_path}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "yolo11"
    if target == "yolo11":
        main("YOLOv11n", OUT_DIR / "models" / "yolo11n_doc_layout.pt", "table_crops_enhanced_yolo11")
    elif target == "yolo26":
        main("YOLOv26n", OUT_DIR / "models" / "yolo26n_doc_layout.pt", "table_crops_enhanced_yolo26")
    else:
        raise SystemExit("usage: benchmark_enhanced_crop.py [yolo11|yolo26]")

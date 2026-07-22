"""[3-ablation] 기본 패딩(균일 여백)만 단독 적용 — Caption 매핑/DPI 재렌더링 없이
150dpi 그대로 사방 동일 패딩만 추가했을 때의 효과를 격리해서 측정.

[3](adaptive padding + caption 매핑 + 300dpi 재렌더링 결합)에서 YOLOv11n이 오히려
악화(240→225행, 10.65→25.56초)된 원인이 '패딩 자체'인지 'DPI 업샘플링'인지 분리하기 위한 대조군.
"""

import json
import sys
from pathlib import Path

import fitz
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "20260721_company_279243000.pdf"
OUT_DIR = Path(__file__).resolve().parent

CONF_THRESHOLD = 0.25
RENDER_DPI = 150       # 탐지/크롭 전부 150dpi 그대로(DPI 변경 없음 — 순수 패딩 효과만 측정)
PAD_PX = 12            # 사방 동일 패딩(기본 패딩, 비대칭 없음)


def main(model_name: str, model_path: Path, crop_dir_name: str):
    crop_dir = OUT_DIR / crop_dir_name
    crop_dir.mkdir(exist_ok=True)
    model = YOLO(str(model_path))
    doc_fitz = fitz.open(str(PDF_PATH))

    total_tables = 0
    for i, page_fz in enumerate(doc_fitz, start=1):
        pix = page_fz.get_pixmap(dpi=RENDER_DPI)
        tmp_path = OUT_DIR / f"_tmp_pad_p{i}.png"
        pix.save(str(tmp_path))
        img = Image.open(tmp_path).convert("RGB")
        page_w_px, page_h_px = img.size

        results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
        names = model.names
        boxes = results.boxes
        tmp_path.unlink(missing_ok=True)
        if boxes is None:
            continue

        t_idx = 0
        for cls_idx, xyxy in zip(boxes.cls, boxes.xyxy):
            if names[int(cls_idx)] != "Table":
                continue
            t_idx += 1
            total_tables += 1
            x1, y1, x2, y2 = xyxy.tolist()
            px1 = max(0, x1 - PAD_PX)
            py1 = max(0, y1 - PAD_PX)
            px2 = min(page_w_px, x2 + PAD_PX)
            py2 = min(page_h_px, y2 + PAD_PX)
            crop = img.crop((int(px1), int(py1), int(px2), int(py2)))
            crop_path = crop_dir / f"page_{i}_table_{t_idx}.png"
            crop.save(crop_path)
    doc_fitz.close()

    print(f"[{model_name}] 총 {total_tables}개 표 크롭(기본 패딩 {PAD_PX}px 사방 균일, 150dpi 유지) 생성 완료 -> {crop_dir}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "yolo11"
    if target == "yolo11":
        main("YOLOv11n", OUT_DIR / "models" / "yolo11n_doc_layout.pt", "table_crops_padonly_yolo11")
    elif target == "yolo26":
        main("YOLOv26n", OUT_DIR / "models" / "yolo26n_doc_layout.pt", "table_crops_padonly_yolo26")
    else:
        raise SystemExit("usage: benchmark_padding_only_crop.py [yolo11|yolo26]")

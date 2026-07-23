"""[16] Table Transformer(TATR, Microsoft, CVPR 2022) vs Docling(TableFormer) — 사용자가 지적한
"TSR 모델 비교가 없다"는 공백을 메우는 실험. TATR은 순수 구조 인식(row/column/spanning-cell bbox
탐지)만 하고 텍스트 추출은 별도(OCR/pdfplumber) — 그래서 "행 수"는 TATR이 탐지한 'table row' 개수로,
Docling의 "행 수"(export_to_dataframe 행 수)와 동일 선상에서 비교한다(완전한 텍스트 추출 파이프라인
비교는 아니고, 순수 구조 인식(TSR) 성능만 비교 — 사용자가 요청한 "OCR -> Detection -> TSR" 3단 분리 중
TSR 단계만 격리 비교).
"""

import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

CROP_DIR = Path(__file__).resolve().parent / "table_crops_router_complex"
CONF_THRESHOLD = 0.6


def load_tatr():
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model.eval()
    return model, processor


def run():
    model, processor = load_tatr()
    crop_paths = sorted(CROP_DIR.glob("*.png"))

    # 워밍업(측정 제외)
    img0 = Image.open(crop_paths[0]).convert("RGB")
    inputs0 = processor(images=img0, return_tensors="pt")
    with torch.no_grad():
        model(**inputs0)

    total_rows = total_cols = total_spans = 0
    t0 = time.perf_counter()
    for p in crop_paths:
        img = Image.open(p).convert("RGB")
        tf0 = time.perf_counter()
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, threshold=CONF_THRESHOLD, target_sizes=target_sizes
        )[0]
        labels = [model.config.id2label[l.item()] for l in results["labels"]]
        n_rows = labels.count("table row")
        n_cols = labels.count("table column")
        n_spans = labels.count("table spanning cell")
        elapsed = round(time.perf_counter() - tf0, 3)
        total_rows += n_rows
        total_cols += n_cols
        total_spans += n_spans
        print(f"  [TATR] {p.name}: {elapsed}s -> row={n_rows} col={n_cols} span={n_spans}", flush=True)
    total_s = round(time.perf_counter() - t0, 3)

    print(f"\n[TATR] 순차 처리 {len(crop_paths)}개 총 {total_s}s")
    print(f"[TATR] 총 detected rows={total_rows}, cols={total_cols}, spanning_cells={total_spans}")
    print(f"[비교] Docling(TableFormer) 동일 12개 크롭: 228행(export_to_dataframe 기준), 20.24s(CPU순차)")


if __name__ == "__main__":
    run()

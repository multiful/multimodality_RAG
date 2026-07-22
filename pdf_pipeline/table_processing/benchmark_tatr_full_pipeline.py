"""[16] TATR 구조 인식 + pdfplumber 텍스트 매핑 — Docling과 "텍스트까지 채운 최종 표"
기준으로 공정 비교(순수 구조 탐지 개수 비교였던 benchmark_tatr_comparison.py의 후속 검증).

TATR은 row/column bbox만 뱉으므로, 각 detected row bbox를 크롭의 페이지 내 오프셋 + 150dpi->pt
환산을 거쳐 pdfplumber로 그 구간의 실제 텍스트를 뽑아 채운다 — 이래야 "빈 bbox만 예쁘게 그린 것"이
아니라 실제 사용 가능한 표 데이터인지 확인 가능.
"""

import sys
import time
from pathlib import Path

import pdfplumber
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_table_router import RouterThresholds, detect_and_route  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "20260721_company_279243000.pdf"
CONF_THRESHOLD = 0.6
DPI = 150
SCALE = DPI / 72


def main():
    # bbox_px와 실제 크롭 이미지가 반드시 "같은 detect_and_route 호출"에서 나와야 함 — 별도 실행에서
    # 저장된 기존 크롭 파일을 재사용하면 표 인덱싱(t_idx)이 실행마다 미세하게 달라져 bbox와 이미지가
    # 어긋날 수 있음(실제로 이 버그로 page_3_table_2가 0행으로 잘못 나온 적 있음 — 수정).
    crop_dir = Path(__file__).resolve().parent / "table_crops_tatr_fresh"
    routed = detect_and_route(RouterThresholds(), crop_dir=crop_dir)
    complex_ = [r for r in routed if r["complexity"] == "complex"]
    print(f"대상 COMPLEX 표 {len(complex_)}개", flush=True)

    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model.eval()

    pdf = pdfplumber.open(str(PDF_PATH))

    # 워밍업
    warm_img = Image.new("RGB", (400, 300), (255, 255, 255))
    with torch.no_grad():
        model(**processor(images=warm_img, return_tensors="pt"))

    total_usable_rows = 0
    total_s = 0.0
    per_table = []
    for r in complex_:
        page_num = r["page"]
        x1, y1, x2, y2 = r["bbox_px"]
        crop_path = Path(r["crop_path"])
        if not crop_path.exists():
            continue
        img = Image.open(crop_path).convert("RGB")

        t0 = time.perf_counter()
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, threshold=CONF_THRESHOLD, target_sizes=target_sizes
        )[0]

        page_pp = pdf.pages[page_num - 1]
        usable = 0
        for label_id, box in zip(results["labels"], results["boxes"]):
            if model.config.id2label[label_id.item()] != "table row":
                continue
            rx1, ry1, rx2, ry2 = [v.item() for v in box]
            # 크롭 로컬 좌표 -> 페이지 150dpi 픽셀 좌표 -> pt 좌표
            page_px = (x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2)
            row_pt = tuple(v / SCALE for v in page_px)
            try:
                text = page_pp.crop(row_pt).extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                usable += 1
        elapsed = round(time.perf_counter() - t0, 3)
        total_s += elapsed
        total_usable_rows += usable
        per_table.append((crop_path.name, elapsed, usable))
        print(f"  {crop_path.name}: {elapsed}s -> usable_rows(텍스트 있음)={usable}", flush=True)
    pdf.close()

    print(f"\n[TATR+pdfplumber] 총 {round(total_s,3)}s, 실사용 가능 행수(텍스트 채워짐) 합계={total_usable_rows}")
    print(f"[비교] Docling(TableFormer) 동일 12개: 228행, 20.24s(CPU 순차)")


if __name__ == "__main__":
    main()

"""[3-batch] 같은 페이지의 YOLO 표 크롭들을 세로로 합쳐 하나의 이미지로 만들어
Docling을 크롭당이 아니라 페이지당 1회만 호출 — 호출 횟수를 줄여 지연을 낮출 수 있는지 검증.

기존 benchmark_docling_on_crops.py(크롭 14개 → 호출 14번, 27.6초)와 비교.
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path

from docling.document_converter import DocumentConverter
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
CROP_DIR = ROOT / "pdf_pipeline" / "page_classification" / "table_crops"
BATCH_DIR = OUT_DIR / "table_crops_batched"
RESULT_PATH = OUT_DIR / "result_docling_batched.json"


def group_by_page(crop_paths):
    groups = defaultdict(list)
    for p in crop_paths:
        m = re.match(r"page_(\d+)_table_\d+\.png", p.name)
        if m:
            groups[int(m.group(1))].append(p)
    return dict(sorted(groups.items()))


def stack_vertical(paths, gap=20):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    width = max(im.width for im in imgs)
    height = sum(im.height for im in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height + gap
    return canvas


def main():
    BATCH_DIR.mkdir(exist_ok=True)
    crop_paths = sorted(CROP_DIR.glob("*.png"))
    groups = group_by_page(crop_paths)

    converter = DocumentConverter()
    # 워밍업(모델 로딩) — 크롭 파싱 실험 때 이미 다운로드된 모델 재사용, 별도로 한 번 태워서 측정 분리
    warm_path = OUT_DIR / "_warmup.png"
    Image.new("RGB", (200, 200), (255, 255, 255)).save(warm_path)
    t0 = time.perf_counter()
    converter.convert(str(warm_path))
    model_ready_s = round(time.perf_counter() - t0, 3)
    warm_path.unlink(missing_ok=True)
    print(f"[timing] 모델 준비(워밍업): {model_ready_s}s", flush=True)

    per_page = []
    total_rows = 0
    for page, paths in groups.items():
        batch_img = stack_vertical(paths)
        batch_path = BATCH_DIR / f"page_{page}_batch.png"
        batch_img.save(batch_path)

        t0 = time.perf_counter()
        res = converter.convert(str(batch_path))
        elapsed = round(time.perf_counter() - t0, 3)
        n_tables = len(res.document.tables)
        n_rows = sum(t.export_to_dataframe(res.document).shape[0] for t in res.document.tables)
        total_rows += n_rows
        per_page.append({
            "page": page, "n_crops_batched": len(paths), "elapsed_s": elapsed,
            "n_tables_detected": n_tables, "n_rows": n_rows,
        })
        print(f"page {page}: {len(paths)}개 크롭 배치 -> {elapsed}s, tables={n_tables}, rows={n_rows}", flush=True)

    total_s = round(sum(p["elapsed_s"] for p in per_page), 3)

    # 기존 크롭별 개별 호출 결과와 비교
    prev_path = OUT_DIR / "result_docling_on_crops.json"
    prev_total_s, prev_rows = None, None
    if prev_path.exists():
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        prev_total_s = prev["total_docling_time_s"]
        prev_rows = prev["total_rows_extracted_docling"]

    result = {
        "method": "PyMuPDF Fast Scan + YOLOv11 Crop + Docling(TableFormer), 페이지 단위 배치(크롭 세로 결합)",
        "model_ready_s": model_ready_s,
        "per_page": per_page,
        "total_docling_time_s": total_s,
        "total_rows_extracted": total_rows,
        "n_docling_calls": len(groups),
        "comparison_vs_per_crop": {
            "per_crop_total_s": prev_total_s,
            "per_crop_n_calls": 14,
            "per_crop_rows": prev_rows,
            "batched_total_s": total_s,
            "batched_n_calls": len(groups),
            "batched_rows": total_rows,
            "latency_reduction_s": round(prev_total_s - total_s, 3) if prev_total_s else None,
            "latency_reduction_pct": round((1 - total_s / prev_total_s) * 100, 1) if prev_total_s else None,
        },
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n총 {total_s}s ({len(groups)}회 호출), 총 행수 {total_rows}")
    if prev_total_s:
        print(f"기존(크롭별 14회 호출): {prev_total_s}s, {prev_rows}행")
        print(f"지연 감소: {result['comparison_vs_per_crop']['latency_reduction_s']}s "
              f"({result['comparison_vs_per_crop']['latency_reduction_pct']}%)")
    print(f"[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

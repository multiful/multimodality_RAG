"""YOLOv26n 크롭에 대해 Docling(동적 워커 병렬) 파싱 — YOLOv11n과 비교용."""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
CROP_DIR = ROOT / "pdf_pipeline" / "page_classification" / "table_crops_yolo26"
RESULT_PATH = OUT_DIR / "result_docling_yolo26.json"
MAX_WORKERS = 8

_converter = None


def _init_worker():
    global _converter
    import torch
    torch.set_num_threads(1)
    from docling.document_converter import DocumentConverter
    _converter = DocumentConverter()
    import tempfile
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        Image.new("RGB", (100, 100), (255, 255, 255)).save(f.name)
        _converter.convert(f.name)


def _noop():
    return True


def parse_one(crop_path_str: str):
    import time as _time
    t0 = _time.perf_counter()
    res = _converter.convert(crop_path_str)
    n_rows = sum(t.export_to_dataframe(res.document).shape[0] for t in res.document.tables)
    return {"file": Path(crop_path_str).name, "elapsed_s": round(_time.perf_counter() - t0, 3), "n_rows": n_rows}


def main():
    crop_paths = sorted(str(p) for p in CROP_DIR.glob("*.png"))
    n_workers = min(len(crop_paths), os.cpu_count(), MAX_WORKERS)
    print(f"대상 크롭 {len(crop_paths)}개, 워커 {n_workers}개", flush=True)

    ex = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker)
    for f in [ex.submit(_noop) for _ in range(n_workers)]:
        f.result()
    print("워밍업 완료", flush=True)

    t0 = time.perf_counter()
    results = []
    futures = {ex.submit(parse_one, p): p for p in crop_paths}
    for fut in as_completed(futures):
        r = fut.result()
        results.append(r)
        print(f"{r['file']}: {r['elapsed_s']}s, rows={r['n_rows']}", flush=True)
    wall_clock_s = round(time.perf_counter() - t0, 3)
    ex.shutdown()

    total_rows = sum(r["n_rows"] for r in results)
    zero_row_count = sum(1 for r in results if r["n_rows"] == 0)

    result = {
        "n_crops": len(crop_paths), "n_workers": n_workers,
        "wall_clock_s": wall_clock_s, "total_rows": total_rows,
        "zero_row_count": zero_row_count, "per_crop": results,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n총 소요: {wall_clock_s}s, 총 행수: {total_rows}, 0-row 크롭: {zero_row_count}")
    print(f"[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

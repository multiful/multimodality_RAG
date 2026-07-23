"""[12] 표 크롭 고도화(Adaptive Padding + Caption 공간매핑 + 300dpi 부분 재렌더링) 크롭에
Docling(TableFormer)을 프로세스 병렬로 적용 — YOLOv11n/YOLOv26n 각각의 일반 크롭 대비
행수 회복 여부를 측정(비교 기준: 일반 크롭 240행/YOLOv11n, 149행/YOLOv26n).

크롭 생성: page_classification/benchmark_enhanced_crop.py
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
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
    elapsed = round(_time.perf_counter() - t0, 3)
    return {"file": Path(crop_path_str).name, "elapsed_s": elapsed, "n_rows": n_rows,
            "n_tables": len(res.document.tables)}


def main(model_name: str, crop_dir: Path, baseline_rows: int, baseline_latency_s: float, result_name: str,
         method_label: str = "Adaptive Padding + Caption 공간매핑 + 300dpi 부분 재렌더링"):
    crop_paths = sorted(str(p) for p in crop_dir.glob("*.png"))
    n_workers = min(len(crop_paths), os.cpu_count(), MAX_WORKERS)
    print(f"[{model_name}] 대상 크롭 {len(crop_paths)}개, 워커 {n_workers}개", flush=True)

    ex = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker)
    warmup_futs = [ex.submit(_noop) for _ in range(n_workers)]
    for f in warmup_futs:
        f.result()
    print("풀 워밍업 완료", flush=True)

    t0 = time.perf_counter()
    results = []
    futures = {ex.submit(parse_one, p): p for p in crop_paths}
    for fut in as_completed(futures):
        r = fut.result()
        results.append(r)
        print(f"{r['file']}: {r['elapsed_s']}s, tables={r['n_tables']}, rows={r['n_rows']}", flush=True)
    wall_clock_s = round(time.perf_counter() - t0, 3)
    ex.shutdown()

    total_rows = sum(r["n_rows"] for r in results)
    result = {
        "method": f"{model_name} + {method_label} + Docling(프로세스 병렬)",
        "n_workers": n_workers,
        "n_crops": len(crop_paths),
        "wall_clock_s_incl_pool_startup": wall_clock_s,
        "total_rows_extracted": total_rows,
        "per_crop": results,
        "comparison_vs_plain_crop": {
            "plain_crop_rows": baseline_rows,
            "plain_crop_latency_s": baseline_latency_s,
            "enhanced_crop_rows": total_rows,
            "enhanced_crop_latency_s": wall_clock_s,
            "row_delta": total_rows - baseline_rows,
            "row_delta_pct": round((total_rows / baseline_rows - 1) * 100, 1) if baseline_rows else None,
        },
    }
    result_path = OUT_DIR / result_name
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{model_name}] 벽시계 시간: {wall_clock_s}s, 총 추출 행수: {total_rows}")
    print(f"일반 크롭 대비: {baseline_rows}행({baseline_latency_s}s) -> {total_rows}행({wall_clock_s}s), "
          f"행수 변화 {result['comparison_vs_plain_crop']['row_delta']:+d} "
          f"({result['comparison_vs_plain_crop']['row_delta_pct']:+.1f}%)")
    print(f"[result] saved to {result_path}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "yolo11"
    if target == "yolo11":
        # 일반 YOLOv11n 크롭 기준: table_processing/result_docling_parallel.json (240행, 10.65s)
        main(
            "YOLOv11n",
            ROOT / "pdf_pipeline" / "page_classification" / "table_crops_enhanced_yolo11",
            baseline_rows=240,
            baseline_latency_s=10.65,
            result_name="result_docling_enhanced_yolo11.json",
        )
    elif target == "yolo26":
        # 일반 YOLOv26n 크롭 기준: table_processing/result_docling_yolo26.json (149행, 5.638s)
        main(
            "YOLOv26n",
            ROOT / "pdf_pipeline" / "page_classification" / "table_crops_enhanced_yolo26",
            baseline_rows=149,
            baseline_latency_s=5.638,
            result_name="result_docling_enhanced_yolo26.json",
        )
    elif target == "yolo11_padonly":
        # 대조군: 기본 패딩(12px 사방 균일)만, DPI 변경 없음(150dpi 유지), 캡션 매핑 없음
        main(
            "YOLOv11n",
            ROOT / "pdf_pipeline" / "page_classification" / "table_crops_padonly_yolo11",
            baseline_rows=240,
            baseline_latency_s=10.65,
            result_name="result_docling_padonly_yolo11.json",
            method_label="기본 패딩(12px 사방 균일, 150dpi 유지, DPI 변경 없음)",
        )
    elif target == "yolo26_padonly":
        main(
            "YOLOv26n",
            ROOT / "pdf_pipeline" / "page_classification" / "table_crops_padonly_yolo26",
            baseline_rows=149,
            baseline_latency_s=5.638,
            result_name="result_docling_padonly_yolo26.json",
            method_label="기본 패딩(12px 사방 균일, 150dpi 유지, DPI 변경 없음)",
        )
    else:
        raise SystemExit("usage: benchmark_docling_enhanced_crops.py [yolo11|yolo26|yolo11_padonly|yolo26_padonly]")

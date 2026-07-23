"""[3-parallel] YOLO 크롭 14개를 프로세스 병렬로 Docling 파싱 — 크롭을 합치지 않으므로
(배치 처리와 달리) 품질 손실 없이 지연만 줄이는 게 목표.

1차 시도(태스크마다 컨버터 새로 생성)는 워커별 모델 재로딩 비용이 병렬 이득을 다 잡아먹어서
오히려 느려짐(29.1s > 순차 27.6s). initializer로 워커 프로세스당 Docling을 딱 1번만 로드하고
재사용하도록 수정한 버전.
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
CROP_DIR = ROOT / "pdf_pipeline" / "page_classification" / "table_crops"
RESULT_PATH = OUT_DIR / "result_docling_parallel.json"
MAX_WORKERS = 8  # 코어를 다 쓰면 다른 파이프라인 단계(YOLO 등)와 경합하므로 상한선

_converter = None  # 워커 프로세스별 전역 — initializer에서 한 번만 채움


def _init_worker():
    global _converter
    import torch
    torch.set_num_threads(1)  # 프로세스 병렬화 시 워커별 PyTorch 스레드 오버섭스크립션 방지
    from docling.document_converter import DocumentConverter
    _converter = DocumentConverter()
    # 워밍업 1회(모델 실제 로딩 트리거) — 이 시간은 풀 준비 시간에 포함, 크롭 처리 시간엔 미포함
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


def main(n_workers: int = None):
    crop_paths = sorted(str(p) for p in CROP_DIR.glob("*.png"))
    if n_workers is None:
        # PDF마다 표 개수가 다르므로 워커 수를 고정하지 않고 매번 동적으로 산정
        n_workers = min(len(crop_paths), os.cpu_count(), MAX_WORKERS)
        print(f"동적 워커 산정: min(표 {len(crop_paths)}개, CPU {os.cpu_count()}코어, 상한 {MAX_WORKERS}) = {n_workers}", flush=True)
    print(f"대상 크롭 {len(crop_paths)}개, 워커 {n_workers}개 (워커당 모델 1회 로딩 후 재사용)", flush=True)

    ex = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker)
    # 풀을 미리 띄워서 워커들이 전부 모델 로딩을 끝내도록 유도(더미 태스크로 대기) —
    # 실제 서비스에서는 워커가 상시 대기 상태이므로 이 시점부터 측정하는 게 공정함
    warmup_futs = [ex.submit(_noop) for _ in range(n_workers)]
    for f in warmup_futs:
        f.result()
    print("풀 워밍업 완료(전 워커 모델 로딩 끝)", flush=True)

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

    prev_path = OUT_DIR / "result_docling_on_crops.json"
    prev = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else None

    result = {
        "method": f"PyMuPDF Fast Scan + YOLOv11 Crop + Docling(TableFormer), 프로세스 병렬"
                   f"(worker={n_workers}, 워커당 모델 1회 로딩+재사용)",
        "n_workers": n_workers,
        "wall_clock_s_incl_pool_startup": wall_clock_s,
        "total_rows_extracted": total_rows,
        "per_crop": results,
        "comparison_vs_sequential": {
            "sequential_total_s": prev["total_docling_time_s"] if prev else None,
            "sequential_rows": prev["total_rows_extracted_docling"] if prev else None,
            "parallel_wall_clock_s": wall_clock_s,
            "parallel_rows": total_rows,
            "latency_reduction_s": round(prev["total_docling_time_s"] - wall_clock_s, 3) if prev else None,
            "latency_reduction_pct": round((1 - wall_clock_s / prev["total_docling_time_s"]) * 100, 1) if prev else None,
        },
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n벽시계 시간(병렬, worker={n_workers}, 풀 초기화+처리 전체): {wall_clock_s}s")
    print(f"총 추출 행수: {total_rows} (순차 처리와 동일해야 함 — 크롭을 안 합쳤으므로 품질 손실 없음)")
    if prev:
        print(f"순차 처리: {prev['total_docling_time_s']}s, {prev['total_rows_extracted_docling']}행")
        print(f"지연 감소: {result['comparison_vs_sequential']['latency_reduction_s']}s "
              f"({result['comparison_vs_sequential']['latency_reduction_pct']}%)")
    print(f"[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

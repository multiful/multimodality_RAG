"""[13] Adaptive Table Complexity Router — 표 추출 품질/지연만 측정(엔티티 추출 제외).
SIMPLE 표는 pdfplumber(무료, 이미 계산됨), COMPLEX 표만 Docling 프로세스 병렬로 처리.

비교 대상:
- [3] 전부 Docling 병렬: 240행, 10.65s (채택된 현재 파이프라인)
- [4] 표 크기 기반 라우팅(높이<550px): 192행(-20%), 9.42s(-11.5%) — 기각됨
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_table_router import RouterThresholds, detect_and_route  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
MAX_WORKERS = 8
# round2: 성능 코어 수(10) 근처까지 허용 — 효율 코어(4)는 다른 단계(YOLO 등)에 남겨둠.
# 라우터가 표 일부를 pdfplumber로 덜어내므로 남은 COMPLEX 작업 수가 원래(14)보다 보통 적어져,
# 상한을 높여도 평소엔 실제 워커 수(min(len(complex), cap))가 크게 안 늘어남 — 필요할 때만 더 씀.
MAX_WORKERS_ROUND2 = 10

_converter = None


def _init_docling_worker():
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


def _docling_parse(crop_path_str: str):
    import time as _time
    t0 = _time.perf_counter()
    res = _converter.convert(crop_path_str)
    n_rows = sum(t.export_to_dataframe(res.document).shape[0] for t in res.document.tables)
    return {"file": Path(crop_path_str).name, "elapsed_s": round(_time.perf_counter() - t0, 3), "n_rows": n_rows}


def run(thresholds: RouterThresholds, version_label: str, result_name: str,
        overlap_warmup: bool = False, max_workers_cap: int = MAX_WORKERS):
    crop_dir = OUT_DIR / "table_crops_router_complex"

    ex = None
    warmup_futs = []
    pool_n_workers = min(max_workers_cap, os.cpu_count())
    if overlap_warmup:
        # round2 개선: Docling 워커 풀 기동(프로세스 생성+모델 로딩)을 라우팅 판단(YOLO+pdfplumber)과
        # 동시에 시작 — 서로 독립적인 두 작업을 순차가 아닌 병렬로 겹쳐서 둘 중 더 긴 쪽만큼만 대기.
        # n_complex를 아직 몰라 워커 수는 상한(max_workers_cap)으로 우선 기동, 실제 태스크는 라우팅 후 제출.
        ex = ProcessPoolExecutor(max_workers=pool_n_workers, initializer=_init_docling_worker)
        warmup_futs = [ex.submit(_noop) for _ in range(pool_n_workers)]

    t0 = time.perf_counter()
    routed = detect_and_route(thresholds, crop_dir=crop_dir)
    route_decision_s = round(time.perf_counter() - t0, 3)

    simple = [r for r in routed if r["complexity"] == "simple"]
    complex_ = [r for r in routed if r["complexity"] == "complex"]
    print(f"[{version_label}] SIMPLE(pdfplumber) {len(simple)}개 / COMPLEX(Docling) {len(complex_)}개", flush=True)
    for r in routed:
        tag = "simple" if r["complexity"] == "simple" else "complex"
        print(f"  page{r['page']} table{r['table_idx']}: {tag}({r['reason']}) {r['signals']}", flush=True)

    simple_rows = sum(r["n_rows"] for r in simple)

    docling_results = []
    docling_elapsed = 0.0
    n_workers = 0
    if complex_:
        n_workers = min(len(complex_), os.cpu_count(), max_workers_cap)
        if not overlap_warmup:
            ex = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_docling_worker)
            warmup_futs = [ex.submit(_noop) for _ in range(n_workers)]
        for f in warmup_futs:
            f.result()
        t0 = time.perf_counter()
        futures = {ex.submit(_docling_parse, r["crop_path"]): r for r in complex_}
        for fut in as_completed(futures):
            res = fut.result()
            docling_results.append(res)
            print(f"  [docling] {res['file']}: {res['elapsed_s']}s -> {res['n_rows']}행", flush=True)
        docling_elapsed = round(time.perf_counter() - t0, 3)
        ex.shutdown()
    elif ex is not None:
        ex.shutdown()

    docling_rows = sum(r["n_rows"] for r in docling_results)
    total_rows = simple_rows + docling_rows
    # overlap_warmup=True면 route_decision_s와 풀 기동이 겹쳐 실행됐으므로 단순 합산이 아니라
    # "라우팅과 풀기동 중 더 오래 걸린 쪽 + docling 처리 시간"이 실제 벽시계에 가깝다.
    # 우리는 라우팅을 main 스레드에서 동기 실행했으니 route_decision_s 자체가 이미 그 겹침을 포함한 실측값.
    total_elapsed = round(route_decision_s + docling_elapsed, 3)

    result = {
        "version": version_label,
        "thresholds": vars(thresholds),
        "overlap_warmup": overlap_warmup, "max_workers_cap": max_workers_cap,
        "n_simple": len(simple), "n_complex": len(complex_),
        "simple_rows": simple_rows, "docling_rows": docling_rows, "total_rows_extracted": total_rows,
        "route_decision_s": route_decision_s,  # YOLO탐지+pdfplumber 신호계산(overlap시 풀기동과 겹쳐 실행됨)
        "docling_elapsed_s": docling_elapsed, "n_docling_workers": n_workers,
        "total_elapsed_s_docling_stage_only": total_elapsed,
        "routed_detail": [{k: v for k, v in r.items() if k != "quick_rows_data"} for r in routed],
        "comparison": {
            "all_docling_parallel": {"rows": 240, "docling_elapsed_s": 10.65},
            "naive_height_router_v4_rejected": {"rows": 192, "elapsed_s": 9.42},
            "this_version": {"rows": total_rows, "docling_elapsed_s": docling_elapsed,
                              "total_elapsed_s": total_elapsed},
        },
    }
    result_path = OUT_DIR / result_name
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{version_label}] SIMPLE {len(simple)}개(pdfplumber, {simple_rows}행) + "
          f"COMPLEX {len(complex_)}개(Docling, {docling_rows}행) = 총 {total_rows}행")
    print(f"라우팅 판단(YOLO+pdfplumber): {route_decision_s}s{'(풀 기동과 겹침)' if overlap_warmup else ''}, "
          f"Docling 처리: {docling_elapsed}s (워커 {n_workers}개), 총: {total_elapsed}s")
    print(f"[result] saved to {result_path}")
    return result


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "v1"
    if target == "v1":
        run(RouterThresholds(), "round1_initial", "result_adaptive_router_v1.json")
    elif target == "v2":
        run(RouterThresholds(), "round2_overlap_warmup", "result_adaptive_router_v2.json",
            overlap_warmup=True, max_workers_cap=MAX_WORKERS_ROUND2)
    elif target == "v3":
        # round2(겹침+워커10)는 오히려 악화(14.29s>13.12s) -> 겹침 없이 워커 캡만 12로 올려
        # (n_complex=12와 정확히 일치, 단일 웨이브) 원인이 '겹침 경쟁'인지 '워커 수 자체'인지 분리 검증
        run(RouterThresholds(), "round3_no_overlap_workers12", "result_adaptive_router_v3.json",
            overlap_warmup=False, max_workers_cap=12)
    else:
        raise SystemExit("usage: benchmark_adaptive_router.py [v1|v2|v3]")

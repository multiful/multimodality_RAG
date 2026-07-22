"""[3] PyMuPDF Fast Scan + YOLOv11 Crop + Docling(TableFormer) 벤치마크.

- 입력: benchmark_yolo_crop.py가 저장한 table_crops/*.png (YOLO가 감지한 Table 영역만)
- 각 크롭을 Docling DocumentConverter로 파싱해 표 구조(행/열) 복원
- pdfplumber baseline(표당 1행만 추출됐던 문제)과 직접 비교: 총 추출 행 수
- 지연: 모델 최초 로딩(1회) vs 크롭당 steady-state 변환 시간 분리 측정
"""

import json
import time
from pathlib import Path

from docling.document_converter import DocumentConverter

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
CROP_DIR = ROOT / "pdf_pipeline" / "page_classification" / "table_crops"
RESULT_PATH = OUT_DIR / "result_docling_on_crops.json"
BASELINE_RESULT_PATH = OUT_DIR.parent / "memory_store.json"  # pdfplumber baseline 행수 비교용


def main():
    crop_paths = sorted(CROP_DIR.glob("*.png"))
    print(f"대상 크롭 {len(crop_paths)}개", flush=True)

    converter = DocumentConverter()

    # 최초 1회 호출(모델 로딩 포함) 별도 측정
    t0 = time.perf_counter()
    first_res = converter.convert(str(crop_paths[0]))
    first_call_s = round(time.perf_counter() - t0, 3)
    print(f"[timing] 첫 변환(모델 로딩 포함): {first_call_s}s", flush=True)

    per_crop = []
    total_rows = 0
    # 첫 크롭 결과 재사용
    n_rows_first = sum(t.export_to_dataframe(first_res.document).shape[0] for t in first_res.document.tables)
    per_crop.append({"file": crop_paths[0].name, "elapsed_s": first_call_s, "n_tables": len(first_res.document.tables), "n_rows": n_rows_first})
    total_rows += n_rows_first

    for p in crop_paths[1:]:
        t0 = time.perf_counter()
        res = converter.convert(str(p))
        elapsed = round(time.perf_counter() - t0, 3)
        n_rows = sum(t.export_to_dataframe(res.document).shape[0] for t in res.document.tables)
        per_crop.append({"file": p.name, "elapsed_s": elapsed, "n_tables": len(res.document.tables), "n_rows": n_rows})
        total_rows += n_rows
        print(f"{p.name}: {elapsed}s, tables={len(res.document.tables)}, rows={n_rows}", flush=True)

    steady_state = per_crop[1:]  # 첫 크롭(모델 로딩 포함) 제외
    avg_steady_s = round(sum(c["elapsed_s"] for c in steady_state) / len(steady_state), 3) if steady_state else 0.0

    # pdfplumber baseline 총 행수 (memory_store.json의 tables_markdown에서 라인 수로 근사)
    baseline_total_rows = 0
    if BASELINE_RESULT_PATH.exists():
        memory = json.loads(BASELINE_RESULT_PATH.read_text(encoding="utf-8"))
        for pg in memory["pages"]:
            for md in pg["tables_markdown"]:
                # 마크다운 표: 헤더+구분선 2줄 제외하고 데이터 행 수
                lines = [l for l in md.split("\n") if l.strip()]
                baseline_total_rows += max(0, len(lines) - 2)

    result = {
        "method": "PyMuPDF Fast Scan + YOLOv11 Crop + Docling(TableFormer) on crops",
        "n_crops": len(crop_paths),
        "first_call_s_incl_model_load": first_call_s,
        "avg_steady_state_s_per_crop": avg_steady_s,
        "total_docling_time_s": round(sum(c["elapsed_s"] for c in per_crop), 3),
        "total_rows_extracted_docling": total_rows,
        "total_rows_extracted_pdfplumber_baseline": baseline_total_rows,
        "per_crop": per_crop,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n총 크롭 {len(crop_paths)}개")
    print(f"첫 변환(모델로딩포함): {first_call_s}s / steady-state 평균: {avg_steady_s}s/crop")
    print(f"총 소요: {result['total_docling_time_s']}s")
    print(f"총 추출 행수 — Docling(on YOLO crop): {total_rows}  vs  pdfplumber baseline: {baseline_total_rows}")
    print(f"[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

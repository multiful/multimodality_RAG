"""[4] 표 크기 기반 라우팅: 작은 표는 pdfplumber, 큰 표만 Docling(동적 워커 병렬).

- YOLOv11로 페이지별 Table bbox 재검출(픽셀 좌표) 후, bbox 높이 기준으로 small/large 분류
  - small (< SMALL_HEIGHT_PX): 픽셀 bbox를 PDF 포인트 좌표로 환산(150dpi -> 72pt 스케일)해서
    pdfplumber page.crop().extract_table()로 직접 추출 (Docling 호출 없음, 사실상 무료)
  - large (>= SMALL_HEIGHT_PX): 크롭 이미지를 Docling(TableFormer)으로 파싱, 프로세스 병렬
    workers = min(작업 개수, os.cpu_count(), MAX_WORKERS) — PDF마다 표 개수가 다르므로 동적 산정
- [3-parallel](순수 Docling 병렬)과 결과 비교
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz
import pdfplumber
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "20260721_company_279243000.pdf"
OUT_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"
RESULT_PATH = OUT_DIR / "result_hybrid_table_routing.json"

CONF_THRESHOLD = 0.25
RENDER_DPI = 150
PDF_POINTS_PER_INCH = 72
SCALE = RENDER_DPI / PDF_POINTS_PER_INCH  # 픽셀 -> pt 환산

SMALL_HEIGHT_PX = 550  # 이 미만이면 "작은 표" -> pdfplumber, 이상이면 "큰 표" -> Docling
MAX_WORKERS = 8

_converter = None


def _init_docling_worker():
    global _converter
    import torch
    torch.set_num_threads(1)
    from docling.document_converter import DocumentConverter
    _converter = DocumentConverter()
    import tempfile
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


def main():
    model = YOLO(str(YOLO_MODEL_PATH))
    doc_fitz = fitz.open(str(PDF_PATH))

    small_tables, large_table_crops = [], []
    crop_dir = OUT_DIR / "table_crops_v2"
    crop_dir.mkdir(exist_ok=True)

    with pdfplumber.open(str(PDF_PATH)) as pdf:
        for i, (page_pp, page_fz) in enumerate(zip(pdf.pages, doc_fitz), start=1):
            pix = page_fz.get_pixmap(dpi=RENDER_DPI)
            tmp_path = OUT_DIR / f"_tmp_p{i}.png"
            pix.save(str(tmp_path))
            img = Image.open(tmp_path).convert("RGB")

            results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
            names = model.names
            boxes = results.boxes
            if boxes is None:
                tmp_path.unlink(missing_ok=True)
                continue

            for j, (cls_idx, xyxy) in enumerate(zip(boxes.cls, boxes.xyxy)):
                if names[int(cls_idx)] != "Table":
                    continue
                x1, y1, x2, y2 = [float(v) for v in xyxy.tolist()]
                height_px = y2 - y1
                if height_px < SMALL_HEIGHT_PX:
                    # pdfplumber로 직접: 픽셀 bbox -> pt bbox
                    bbox_pt = (x1 / SCALE, y1 / SCALE, x2 / SCALE, y2 / SCALE)
                    small_tables.append({"page": i, "bbox_px": [x1, y1, x2, y2], "height_px": height_px,
                                          "bbox_pt": bbox_pt})
                else:
                    crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                    crop_path = crop_dir / f"page_{i}_table_{j}.png"
                    crop.save(crop_path)
                    large_table_crops.append({"page": i, "path": str(crop_path), "height_px": height_px})
            tmp_path.unlink(missing_ok=True)
    doc_fitz.close()

    print(f"작은 표(pdfplumber 라우팅): {len(small_tables)}개, 높이<{SMALL_HEIGHT_PX}px", flush=True)
    print(f"큰 표(Docling 라우팅): {len(large_table_crops)}개, 높이>={SMALL_HEIGHT_PX}px", flush=True)

    # --- 작은 표: pdfplumber ---
    t0 = time.perf_counter()
    small_results = []
    with pdfplumber.open(str(PDF_PATH)) as pdf:
        for t in small_tables:
            page = pdf.pages[t["page"] - 1]
            cropped_page = page.crop(t["bbox_pt"])
            table = cropped_page.extract_table()
            n_rows = len(table) if table else 0
            small_results.append({"page": t["page"], "height_px": t["height_px"], "n_rows": n_rows})
            print(f"  [small] page{t['page']} height={t['height_px']:.0f}px -> {n_rows}행 (pdfplumber)", flush=True)
    small_elapsed = round(time.perf_counter() - t0, 3)

    # --- 큰 표: Docling, 동적 워커 수로 병렬 ---
    n_workers = min(len(large_table_crops), os.cpu_count(), MAX_WORKERS) if large_table_crops else 0
    print(f"\nDocling 워커 수 = min({len(large_table_crops)}, {os.cpu_count()}, {MAX_WORKERS}) = {n_workers}", flush=True)

    large_results = []
    docling_elapsed = 0.0
    if large_table_crops:
        ex = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_docling_worker)
        warmup_futs = [ex.submit(_noop) for _ in range(n_workers)]
        for f in warmup_futs:
            f.result()
        print("Docling 워커 풀 워밍업 완료", flush=True)

        t0 = time.perf_counter()
        futures = {ex.submit(_docling_parse, t["path"]): t for t in large_table_crops}
        for fut in as_completed(futures):
            r = fut.result()
            large_results.append(r)
            print(f"  [large] {r['file']}: {r['elapsed_s']}s -> {r['n_rows']}행 (Docling)", flush=True)
        docling_elapsed = round(time.perf_counter() - t0, 3)
        ex.shutdown()

    total_rows = sum(r["n_rows"] for r in small_results) + sum(r["n_rows"] for r in large_results)
    total_elapsed = round(small_elapsed + docling_elapsed, 3)

    prev_parallel_path = OUT_DIR / "result_docling_parallel.json"
    prev_parallel = json.loads(prev_parallel_path.read_text(encoding="utf-8")) if prev_parallel_path.exists() else None

    result = {
        "method": f"YOLOv11 Crop + 표 크기 라우팅(<{SMALL_HEIGHT_PX}px pdfplumber / >= Docling 동적병렬)",
        "small_table_count": len(small_tables),
        "large_table_count": len(large_table_crops),
        "n_docling_workers_used": n_workers,
        "small_pdfplumber_elapsed_s": small_elapsed,
        "large_docling_elapsed_s": docling_elapsed,
        "total_elapsed_s": total_elapsed,
        "total_rows_extracted": total_rows,
        "small_results": small_results,
        "large_results": large_results,
        "comparison_vs_all_docling_parallel": {
            "all_docling_parallel_s": prev_parallel["wall_clock_s_incl_pool_startup"] if prev_parallel else None,
            "all_docling_parallel_rows": prev_parallel["total_rows_extracted"] if prev_parallel else None,
            "hybrid_s": total_elapsed,
            "hybrid_rows": total_rows,
        },
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n총 소요: {total_elapsed}s (작은표 pdfplumber {small_elapsed}s + 큰표 Docling {docling_elapsed}s)")
    print(f"총 추출 행수: {total_rows}")
    if prev_parallel:
        print(f"[3-parallel](전부 Docling): {prev_parallel['wall_clock_s_incl_pool_startup']}s, "
              f"{prev_parallel['total_rows_extracted']}행")
    print(f"[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()

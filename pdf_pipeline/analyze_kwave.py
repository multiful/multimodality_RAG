"""K-Wave 산업 리포트(73페이지, 표 104개) 추가 분석 — 사용자 요청: "이건 실적이 잘 나오면
좋아. 메타데이터로." 텍스트 골든셋이 없어 recall은 측정 못 하고(솔직히 이 스크립트 범위 밖으로
남김), 표 쪽에 집중: (1) 현재 코드(YOLO 공유/[26], row-level 재무필터/[22] 포함)로 baseline vs
current를 신선하게 재측정하고, (2) finance로 분류된 표에 구조화 출력을 켜서 "실적" 관련 정보가
canonical field 매칭을 벗어난 세그먼트 단위로도 메타데이터에 잘 잡히는지 확인한다.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import pdfplumber
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pdf_pipeline"))
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "page_classification"))
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "text_processing"))
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "table_processing"))


def _load_as(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_text_tn = _load_as("_text_processing_text_normalization",
                     ROOT / "pdf_pipeline" / "text_processing" / "text_normalization.py")
_table_tn = _load_as("_table_processing_text_normalization",
                      ROOT / "pdf_pipeline" / "table_processing" / "text_normalization.py")

sys.modules["text_normalization"] = _text_tn
from page_classifier import classify_pdf  # noqa: E402
from text_extraction import process_pdf  # noqa: E402
import hierarchical_chunker  # noqa: E402,F401 — [35] _text_tn 활성 상태에서 미리 로드(자세한 이유는 compare_baseline_vs_pipeline.py 참고)

sys.modules["text_normalization"] = _table_tn
import adaptive_table_router as atr  # noqa: E402
import run_table_metadata_pipeline as rtmp  # noqa: E402
from table_type_router import classify_table  # noqa: E402

YOLO_MODEL_PATH = ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "KWave" / "20260721_industry_65157000.pdf"
OUT_PATH = ROOT / "pdf_pipeline" / "result_kwave_analysis.json"


def table_baseline(pdf_path, page_boxes):
    pdf_pp = pdfplumber.open(str(pdf_path))
    total_rows, zero_row, n_tables = 0, 0, 0
    t0 = time.perf_counter()
    for page_num, boxes in page_boxes.items():
        page_pp = pdf_pp.pages[page_num - 1]
        for cls_name, rect in boxes:
            if cls_name != "Table":
                continue
            n_tables += 1
            bbox_pt = (rect.x0, rect.y0, rect.x1, rect.y1)
            try:
                table = page_pp.crop(bbox_pt).extract_table()
                n_rows = len(table) if table else 0
            except Exception:
                n_rows = 0
            total_rows += n_rows
            if n_rows == 0:
                zero_row += 1
    elapsed = time.perf_counter() - t0
    pdf_pp.close()
    return {"n_tables": n_tables, "total_rows": total_rows, "zero_row_tables": zero_row,
            "elapsed_s": round(elapsed, 3)}


def main():
    print("YOLO 모델 로딩 중...", flush=True)
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    warmup = Image.new("RGB", (595, 842), (255, 255, 255))
    yolo_model.predict(warmup, conf=0.25, verbose=False)

    print(f"page_classification 중 (73페이지, 시간 좀 걸림)...", flush=True)
    t0 = time.perf_counter()
    cls_result = classify_pdf(PDF_PATH, yolo_model)
    page_boxes = {p["page"]: p["cached_boxes"] for p in cls_result["pages"]}
    cls_s = time.perf_counter() - t0
    print(f"page_classification: {cls_result['n_pages']}페이지, {cls_s:.1f}s", flush=True)

    # ---------- 텍스트: 골든셋 없어 recall은 스킵, 처리량/지연만 측정(structured metadata off) ----------
    t0 = time.perf_counter()
    text_result = process_pdf(PDF_PATH, yolo_model, page_boxes=page_boxes,
                               chunk_backend="rulebased", remove_boilerplate=True,
                               add_structured_metadata=False)
    text_s = time.perf_counter() - t0
    n_chunks = sum(len(p["chunks"]) for p in text_result["pages"])
    n_hard = len(text_result["hard_page_numbers"])
    print(f"[텍스트] {n_chunks}청크, hard페이지 {n_hard}개, {text_s:.1f}s (골든셋 없어 recall 미측정)", flush=True)

    # ---------- 표: baseline vs current(신선하게 재측정, [22]/[26] 포함된 현재 코드) ----------
    base_table = table_baseline(PDF_PATH, page_boxes)
    print(f"[표] baseline: {base_table['n_tables']}개 표, {base_table['total_rows']}행, "
          f"0행 표 {base_table['zero_row_tables']}개 ({base_table['elapsed_s']}s)", flush=True)

    atr.PDF_PATH = PDF_PATH
    rtmp.PDF_PATH = PDF_PATH
    t0 = time.perf_counter()
    records, n_finance_filtered, n_cid = rtmp.build_records("KWave", page_boxes=page_boxes, yolo_model=yolo_model)
    current_table_s = time.perf_counter() - t0
    row_records = [r for r in records if r.get("record_type") != "table_metadata"]
    mapped = [r for r in row_records if r.get("canonical_field")]
    hit_rate = round(len(mapped) / len(row_records), 4) if row_records else 0.0
    from collections import Counter
    field_counts = dict(Counter(r["canonical_field"] for r in mapped))
    print(f"[표] current: {len(row_records)}행, canonical 매칭 {len(mapped)}개(hit_rate={hit_rate*100:.1f}%), "
          f"재무항목필터 {n_finance_filtered}행 ({current_table_s:.1f}s)", flush=True)
    print(f"  필드별 매칭: {field_counts}", flush=True)

    # ---------- 구조화 출력: finance로 분류된 표만 켜서 "실적" 요약 품질 확인 ----------
    finance_tables = {}  # (page, table_idx) -> raw_text, for re-fetching table_text
    routed = atr.detect_and_route(atr.RouterThresholds(), yolo_model=yolo_model, page_boxes=page_boxes)
    for r in routed:
        ttype = classify_table(r["raw_text"])
        if ttype == "finance":
            finance_tables[(r["page"], r["table_idx"])] = r

    print(f"\nfinance 분류 표 {len(finance_tables)}개에 구조화 출력 실행 중...", flush=True)
    from structured_output import extract_table_metadata
    finance_meta_samples = []
    t0 = time.perf_counter()
    for (page, tidx), r in list(finance_tables.items()):
        table_records = [rec for rec in row_records if rec["page"] == page and rec["table_idx"] == tidx]
        mapped_t = [x for x in table_records if x["canonical_field"]]
        unmapped_labels_t = [x["raw_label"] for x in table_records if not x["canonical_field"]]
        table_text = r.get("markdown") or r["raw_text"]
        meta = extract_table_metadata(table_text, mapped_t, unmapped_labels_t)
        finance_meta_samples.append({"page": page, "table_idx": tidx, **meta})
        print(f"  page{page} table{tidx}: {meta['table_title']} | {meta['table_type_refined']} | "
              f"notable={meta['notable_finding']}", flush=True)
    structured_s = time.perf_counter() - t0
    print(f"finance 표 구조화 출력: {len(finance_tables)}개, {structured_s:.1f}s "
          f"(표당 평균 {structured_s/max(len(finance_tables),1):.2f}s)", flush=True)

    result = {
        "n_pages": cls_result["n_pages"], "page_classification_s": round(cls_s, 2),
        "text": {"elapsed_s": round(text_s, 2), "n_chunks": n_chunks, "n_hard_pages": n_hard},
        "table": {
            "baseline": base_table,
            "current": {"total_rows": len(row_records), "n_mapped": len(mapped), "hit_rate": hit_rate,
                        "n_finance_rows_filtered": n_finance_filtered, "elapsed_s": round(current_table_s, 2),
                        "field_counts": field_counts},
            "n_finance_classified_tables": len(finance_tables),
            "structured_output_finance_tables": {"elapsed_s": round(structured_s, 2), "samples": finance_meta_samples},
        },
    }
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[result] saved to {OUT_PATH}")


if __name__ == "__main__":
    main()

"""[25]/[11] 구조화 출력 실측 — 사용자 질문("병목 없는거야? 잘 나오는지 측정 못해?")에 답하기
위해, 지금까지의 스모크 테스트(임의로 만든 짧은 발췌 2~3개)가 아니라 LGCNS+Construct 실제
문서 전체에 `add_structured_metadata=True`를 켜서 (a) 추가되는 latency, (b) 실제 출력 품질을
확인한다. 표는 이미 canonical 매칭이 안 된 표에 한해서만 (원래 목적이 rule-based가 놓친 것
보완이므로) 구조화 출력을 호출하도록도 확인.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

from ultralytics import YOLO
from PIL import Image

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

YOLO_MODEL_PATH = ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"
OUT_PATH = ROOT / "pdf_pipeline" / "result_structured_output_measurement.json"

PDFS = {
    "LGCNS": ROOT / "pdf_pipeline" / "reference" / "LGCNS" / "20260721_company_279243000.pdf",
    "Construct": ROOT / "pdf_pipeline" / "reference" / "Construct" / "20260721_industry_362851000.pdf",
}


def run_for_pdf(pdf_key, pdf_path, yolo_model):
    print(f"\n{'='*60}\n{pdf_key}\n{'='*60}", flush=True)
    cls_result = classify_pdf(pdf_path, yolo_model)
    page_boxes = {p["page"]: p["cached_boxes"] for p in cls_result["pages"]}

    # ---------- 텍스트: structured metadata ON ----------
    t0 = time.perf_counter()
    text_result = process_pdf(pdf_path, yolo_model, page_boxes=page_boxes,
                               chunk_backend="rulebased", remove_boilerplate=True,
                               add_structured_metadata=True)
    text_s_on = time.perf_counter() - t0

    t0 = time.perf_counter()
    text_result_off = process_pdf(pdf_path, yolo_model, page_boxes=page_boxes,
                                   chunk_backend="rulebased", remove_boilerplate=True,
                                   add_structured_metadata=False)
    text_s_off = time.perf_counter() - t0

    all_chunks = [c for p in text_result["pages"] for c in p["chunks"]]
    n_chunks = len(all_chunks)
    n_pages_with_chunks = sum(1 for p in text_result["pages"] if p["chunks"])
    print(f"[텍스트] add_structured_metadata off={text_s_off:.2f}s -> on={text_s_on:.2f}s "
          f"(+{text_s_on - text_s_off:.2f}s, {n_chunks}청크/{n_pages_with_chunks}페이지, "
          f"청크당 평균 +{(text_s_on - text_s_off) / max(n_chunks,1)*1000:.0f}ms)", flush=True)

    sample_chunks = [
        {"page": c["page"], "raw_chunk": c["raw_chunk"][:80], "structured_metadata": c["structured_metadata"]}
        for c in all_chunks[:3]
    ]

    # ---------- 표: structured metadata ON ----------
    atr.PDF_PATH = pdf_path
    rtmp.PDF_PATH = pdf_path
    t0 = time.perf_counter()
    records_on, _, _ = rtmp.build_records(pdf_key, page_boxes=page_boxes, yolo_model=yolo_model,
                                           add_structured_metadata=True)
    table_s_on = time.perf_counter() - t0

    t0 = time.perf_counter()
    records_off, _, _ = rtmp.build_records(pdf_key, page_boxes=page_boxes, yolo_model=yolo_model,
                                            add_structured_metadata=False)
    table_s_off = time.perf_counter() - t0

    table_meta_records = [r for r in records_on if r.get("record_type") == "table_metadata"]
    n_tables = len(table_meta_records)
    print(f"[표] add_structured_metadata off={table_s_off:.2f}s -> on={table_s_on:.2f}s "
          f"(+{table_s_on - table_s_off:.2f}s, {n_tables}개 표, "
          f"표당 평균 +{(table_s_on - table_s_off) / max(n_tables,1):.2f}s)", flush=True)

    sample_tables = table_meta_records[:3]

    return {
        "text": {"elapsed_off_s": round(text_s_off, 3), "elapsed_on_s": round(text_s_on, 3),
                 "added_s": round(text_s_on - text_s_off, 3), "n_chunks": n_chunks,
                 "added_ms_per_chunk": round((text_s_on - text_s_off) / max(n_chunks, 1) * 1000, 1),
                 "sample": sample_chunks},
        "table": {"elapsed_off_s": round(table_s_off, 3), "elapsed_on_s": round(table_s_on, 3),
                  "added_s": round(table_s_on - table_s_off, 3), "n_tables": n_tables,
                  "added_s_per_table": round((table_s_on - table_s_off) / max(n_tables, 1), 3),
                  "sample": sample_tables},
    }


def main():
    print("YOLO 모델 로딩 중...", flush=True)
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    warmup = Image.new("RGB", (595, 842), (255, 255, 255))
    yolo_model.predict(warmup, conf=0.25, verbose=False)

    all_results = {}
    for pdf_key, pdf_path in PDFS.items():
        all_results[pdf_key] = run_for_pdf(pdf_key, pdf_path, yolo_model)

    OUT_PATH.write_text(json.dumps(all_results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[result] saved to {OUT_PATH}")


if __name__ == "__main__":
    main()

"""전체 파이프라인 vs 베이스라인 비교 — LG CNS, Construct 2개 PDF(사용자가 최초에 보낸 레퍼런스
PDF) 기준. page_classification/text_processing/table_processing 세 모듈을 실제로 연결해서 돌리고,
"라우팅/정제/청킹/canonical 매칭이 전혀 없는 순수 추출"과 얼마나 차이 나는지 측정한다.
(이미지 파트는 팀원 모듈 통합 예정이라 이번 비교에서는 제외)

베이스라인 정의:
- 텍스트: PyMuPDF `page.get_text()` 그대로(PUA 제거/헤더푸터 제거/구두점 정규화조차 없음)
- 표: YOLO로 찾은 Table bbox에 pdfplumber `extract_table()`만(Adaptive Router/TATR/canonical
  매칭 없음) — 병합 셀이 많은 표에서 대개 1행짜리 쓰레기가 나온다는 게 핵심 포인트라, canonical
  매칭 자체를 시도하지 않음(benchmark_construct_validation.py [23]와 동일 관례)

현재 파이프라인: page_classification(공유 YOLO, 페이지당 1회) -> text_processing.process_pdf()
-> table_processing 표 라우팅(Adaptive Router+TATR)+canonical field 매칭. structured_output
([11]/[25])은 유료 API라 이 비교에서는 기본 off(따로 스모크 테스트로 이미 검증됨).
"""

import json
import re
import sys
import time
from pathlib import Path

import fitz
import pdfplumber
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pdf_pipeline"))
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "page_classification"))
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "text_processing"))
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "table_processing"))

# [36] 예전엔 text_processing/table_processing 양쪽에 이름이 같은 text_normalization.py가 있어
# sys.modules 스왑으로 우회해야 했는데, PUA/구두점 정규화 함수를 pdf_pipeline/text_cleanup.py로
# 옮기면서(text_processing 쪽 text_normalization.py는 삭제) 이름 충돌 자체가 사라져 더는 필요 없음.
from page_classifier import classify_pdf  # noqa: E402
from text_extraction import process_pdf  # noqa: E402
import adaptive_table_router as atr  # noqa: E402
import run_table_metadata_pipeline as rtmp  # noqa: E402

YOLO_MODEL_PATH = ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"
OUT_PATH = ROOT / "pdf_pipeline" / "result_baseline_vs_pipeline_comparison.json"

PDFS = {
    "LGCNS": {
        "path": ROOT / "pdf_pipeline" / "reference" / "LGCNS" / "20260721_company_279243000.pdf",
        "text_gt": ROOT / "pdf_pipeline" / "text_processing" / "ground_truth_text_lgcns.json",
    },
    "Construct": {
        "path": ROOT / "pdf_pipeline" / "reference" / "Construct" / "20260721_industry_362851000.pdf",
        "text_gt": ROOT / "pdf_pipeline" / "text_processing" / "ground_truth_text_construct.json",
    },
}


def normalize(s: str) -> str:
    """공백 제거 + ellipsis 통일("…"/"..."→"...") — 파이프라인의 `normalize_punctuation()`이
    "…"(U+2026)를 "..."로 바꾸는데, 골든셋은 렌더링된 페이지를 육안으로 옮겨 적어 원래 문자
    "…"를 그대로 쓰고 있어서 이걸 안 맞춰주면 실제로는 존재하는 문장이 recall miss로
    잘못 집계된다(Construct p5에서 실제로 발견 — 뉴스 헤드라인 3건이 정보 손실 없이 그대로
    있는데도 ellipsis 문자만 달라 recall 93.6%로 잘못 나왔던 원인)."""
    s = re.sub(r"\.{3}|…", "...", s or "")
    return re.sub(r"\s+", "", s)


def score_text_recall(pages_text: dict, gt: dict) -> dict:
    total_matched, total_units, per_page = 0, 0, {}
    for page_str, page_gt in gt["pages"].items():
        page = int(page_str)
        norm_extracted = normalize(pages_text.get(page, ""))
        units = page_gt["units"]
        matched = sum(1 for u in units if normalize(u) in norm_extracted)
        per_page[page] = {"matched": matched, "total": len(units)}
        total_matched += matched
        total_units += len(units)
    return {"matched": total_matched, "total": total_units,
            "recall": round(total_matched / total_units, 4) if total_units else None,
            "per_page": per_page}


def table_baseline(pdf_path, page_boxes: dict) -> dict:
    """표 위치는 이미 안다고 가정(공유 YOLO 재사용), pdfplumber extract_table()만 순수 적용."""
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


def run_for_pdf(pdf_key: str, cfg: dict, yolo_model) -> dict:
    pdf_path = cfg["path"]
    print(f"\n{'='*60}\n{pdf_key}\n{'='*60}", flush=True)

    t0 = time.perf_counter()
    cls_result = classify_pdf(pdf_path, yolo_model)
    page_boxes = {p["page"]: p["cached_boxes"] for p in cls_result["pages"]}
    page_classification_s = time.perf_counter() - t0
    print(f"page_classification: {cls_result['n_pages']}페이지, YOLO {cls_result['n_pages']}회, "
          f"{page_classification_s:.2f}s", flush=True)

    # ---------- 텍스트 ----------
    gt = json.loads(cfg["text_gt"].read_text(encoding="utf-8"))

    doc = fitz.open(str(pdf_path))
    t0 = time.perf_counter()
    baseline_pages = {i + 1: doc[i].get_text() for i in range(doc.page_count)}
    baseline_text_s = time.perf_counter() - t0
    doc.close()
    baseline_text_score = score_text_recall(baseline_pages, gt)

    t0 = time.perf_counter()
    current_result = process_pdf(pdf_path, yolo_model, page_boxes=page_boxes,
                                  chunk_backend="rulebased", remove_boilerplate=True,
                                  add_structured_metadata=False)
    current_text_s = time.perf_counter() - t0
    current_pages_text = {p["page"]: p["text"] for p in current_result["pages"]}
    current_text_score = score_text_recall(current_pages_text, gt)
    n_chunks = sum(len(p["chunks"]) for p in current_result["pages"])
    n_hard = len(current_result["hard_page_numbers"])

    print(f"[텍스트] baseline recall={baseline_text_score['recall']*100:.1f}%"
          f"({baseline_text_score['matched']}/{baseline_text_score['total']}, {baseline_text_s*1000:.1f}ms) "
          f"vs current recall={current_text_score['recall']*100:.1f}%"
          f"({current_text_score['matched']}/{current_text_score['total']}, {current_text_s:.2f}s, "
          f"{n_chunks}청크, hard페이지 {n_hard}개)", flush=True)

    # ---------- 표 ----------
    base_table = table_baseline(pdf_path, page_boxes)
    print(f"[표] baseline: {base_table['n_tables']}개 표, {base_table['total_rows']}행, "
          f"0행 표 {base_table['zero_row_tables']}개 ({base_table['elapsed_s']}s)", flush=True)

    atr.PDF_PATH = pdf_path
    rtmp.PDF_PATH = pdf_path
    t0 = time.perf_counter()
    records, n_finance_filtered, n_cid = rtmp.build_records(pdf_key, page_boxes=page_boxes, yolo_model=yolo_model)
    current_table_s = time.perf_counter() - t0
    row_records = [r for r in records if r.get("record_type") != "table_metadata"]
    mapped = [r for r in row_records if r.get("canonical_field")]
    hit_rate = round(len(mapped) / len(row_records), 4) if row_records else 0.0
    n_tables_current = len(set((r["page"], r["table_idx"]) for r in row_records))

    print(f"[표] current: {n_tables_current}개 표, {len(row_records)}행, "
          f"canonical 매칭 {len(mapped)}개(hit_rate={hit_rate*100:.1f}%), "
          f"재무항목필터 {n_finance_filtered}행 ({current_table_s:.2f}s)", flush=True)

    return {
        "page_classification_s": round(page_classification_s, 3),
        "text": {
            "baseline": {"elapsed_s": round(baseline_text_s, 4), **baseline_text_score},
            "current": {"elapsed_s": round(current_text_s, 3), **current_text_score,
                        "n_chunks": n_chunks, "n_hard_pages": n_hard},
        },
        "table": {
            "baseline": base_table,
            "current": {"n_tables": n_tables_current, "total_rows": len(row_records),
                        "n_mapped": len(mapped), "hit_rate": hit_rate,
                        "n_finance_rows_filtered": n_finance_filtered,
                        "elapsed_s": round(current_table_s, 3)},
        },
    }


def main():
    print("YOLO 모델 로딩 중...", flush=True)
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    warmup = Image.new("RGB", (595, 842), (255, 255, 255))
    yolo_model.predict(warmup, conf=0.25, verbose=False)

    all_results = {}
    for pdf_key, cfg in PDFS.items():
        all_results[pdf_key] = run_for_pdf(pdf_key, cfg, yolo_model)

    OUT_PATH.write_text(json.dumps(all_results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\n\n{'='*60}\n종합 요약\n{'='*60}")
    for pdf_key, r in all_results.items():
        t = r["text"]
        tb = r["table"]
        print(f"\n[{pdf_key}]")
        print(f"  텍스트 recall: baseline {t['baseline']['recall']*100:.1f}% -> "
              f"current {t['current']['recall']*100:.1f}% "
              f"(청크 {t['current']['n_chunks']}개, hard페이지 {t['current']['n_hard_pages']}개)")
        print(f"  표 추출: baseline {tb['baseline']['total_rows']}행(0행표 {tb['baseline']['zero_row_tables']}개) -> "
              f"current {tb['current']['total_rows']}행, canonical hit_rate {tb['current']['hit_rate']*100:.1f}%")
    print(f"\n[result] saved to {OUT_PATH}")


if __name__ == "__main__":
    main()

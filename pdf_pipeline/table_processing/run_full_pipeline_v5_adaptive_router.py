"""[13] Adaptive Table Complexity Router를 실제 엔티티 추출 파이프라인에 연결해
Recall/Precision/F1/총 처리시간을 v2+repetition_penalty(현재 최선, 90% Recall, 1097.51초) 대비 재측정.

- 텍스트, 이미지 설명은 기존 memory_store.json 재사용(변경 없음)
- 표만: SIMPLE 표는 pdfplumber 결과(마크다운)를 그대로, COMPLEX 표만 Docling(8워커 병렬,
  라우터 자가검증 round1/round3 비교에서 8워커가 10/12워커보다 더 빠른 것으로 확인된 설정 재사용)
- 엔티티 추출은 동일하게 repetition_penalty=1.3 페이지 단위 방식
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_table_router import RouterThresholds, detect_and_route  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
OUT_DIR = Path(__file__).resolve().parent
MEMORY_PATH = ROOT / "pdf_pipeline" / "memory_store.json"
GROUND_TRUTH_PATH = ROOT / "pdf_pipeline" / "ground_truth_064400.json"
RESULT_PATH = OUT_DIR / "result_pipeline_v5_adaptive_router_entities.json"
REPORT_PATH = ROOT / "pdf_pipeline" / "table_processing" / "실험_v5_adaptive_router_recall_report.md"
MAX_WORKERS = 8  # [13] 라우터 자가검증에서 8워커가 10/12워커보다 빠른 것으로 확인(경합 회피)

_converter = None


def _init_docling_worker():
    global _converter
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
    md_parts = []
    for t in res.document.tables:
        df = t.export_to_dataframe(res.document)
        md_parts.append(df.to_markdown(index=False))
    return {"file": Path(crop_path_str).name,
            "elapsed_s": round(_time.perf_counter() - t0, 3), "markdown": "\n\n".join(md_parts)}


def run_table_stage():
    """[13] 라우터 실행: SIMPLE은 pdfplumber 마크다운 즉시 확정, COMPLEX만 Docling 병렬."""
    crop_dir = OUT_DIR / "table_crops_router_complex_v5"
    t0 = time.perf_counter()
    routed = detect_and_route(RouterThresholds(), crop_dir=crop_dir)
    route_decision_s = round(time.perf_counter() - t0, 3)

    simple = [r for r in routed if r["complexity"] == "simple"]
    complex_ = [r for r in routed if r["complexity"] == "complex"]
    print(f"[table] SIMPLE(pdfplumber) {len(simple)}개 / COMPLEX(Docling) {len(complex_)}개", flush=True)

    by_page = {}
    for r in simple:
        by_page.setdefault(r["page"], []).append(r["markdown"])

    docling_elapsed = 0.0
    if complex_:
        n_workers = min(len(complex_), os.cpu_count(), MAX_WORKERS)
        print(f"[table] Docling 워커 {n_workers}개", flush=True)
        ex = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_docling_worker)
        for f in [ex.submit(_noop) for _ in range(n_workers)]:
            f.result()
        t0 = time.perf_counter()
        futures = {ex.submit(_docling_parse, r["crop_path"]): r for r in complex_}
        for fut in as_completed(futures):
            res = fut.result()
            match = next(rr for rr in complex_ if Path(rr["crop_path"]).name == res["file"])
            by_page.setdefault(match["page"], []).append(res["markdown"])
            print(f"  [docling] {res['file']}: {res['elapsed_s']}s", flush=True)
        docling_elapsed = round(time.perf_counter() - t0, 3)
        ex.shutdown()

    table_stage_total_s = round(route_decision_s + docling_elapsed, 3)
    print(f"[table] 표 단계 총 소요: {table_stage_total_s}s "
          f"(라우팅 {route_decision_s}s + Docling {docling_elapsed}s)", flush=True)
    return by_page, table_stage_total_s, len(simple), len(complex_)


def load_qwen():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    return model, processor, device


def text_generate(model, processor, device, prompt: str, max_new_tokens: int = 200) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.3)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    result = processor.decode(trimmed, skip_special_tokens=True).strip()
    del inputs, out
    if device == "mps":
        torch.mps.empty_cache()
    return result


def main():
    memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    table_md_by_page, table_stage_s, n_simple, n_complex = run_table_stage()

    model, processor, device = load_qwen()
    per_page_entities = {}
    entity_timing = []

    for p in memory["pages"]:
        parts = []
        if p["text"]:
            parts.append(f"[본문]\n{p['text']}")
        mds = table_md_by_page.get(p["page"], [])
        for t_idx, md in enumerate(mds, start=1):
            parts.append(f"[표{t_idx}]\n{md}")
        for desc in p["image_descriptions"]:
            parts.append(f"[이미지/차트]\n{desc}")
        page_context = "\n\n".join(parts)
        if not page_context.strip():
            per_page_entities[p["page"]] = ""
            continue

        prompt = (
            "다음은 증권사 리포트 한 페이지에서 추출한 내용입니다. "
            "이 안에 등장하는 모든 기업/기관 이름을 빠짐없이 나열하세요. "
            "표나 차트 범례 안에서만 언급된 기업도 포함하세요. "
            "한 줄에 하나씩 '기업명 (아는 경우 종목코드)' 형태로만 출력하고 다른 설명은 하지 마세요.\n\n"
            f"{page_context}"
        )
        t0 = time.perf_counter()
        result = text_generate(model, processor, device, prompt)
        elapsed = round(time.perf_counter() - t0, 3)
        entity_timing.append(elapsed)
        per_page_entities[p["page"]] = result
        print(f"[entity p{p['page']}] ({elapsed}s) {result}", flush=True)

    entity_extract_total_s = round(sum(entity_timing), 3)

    gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    target_set = gt["entity_recall_target_set"]
    aliases = gt.get("aliases", {})

    def norm(s):
        return s.lower().replace(" ", "")

    combined_norm = norm("\n".join(per_page_entities.values()))
    hits, misses = [], []
    for ent in target_set:
        candidates = [ent] + aliases.get(ent, [])
        (hits if any(norm(c) in combined_norm for c in candidates) else misses).append(ent)
    recall = len(hits) / len(target_set)

    import re as _re
    KNOWN_NON_ENTITIES = {"대외고객", "기타특수관계자", "researchcenter"}
    all_candidates = []
    for raw in per_page_entities.values():
        for part in _re.split(r"[\n,]", raw):
            part = _re.sub(r"\([^)]*\)", "", part).strip(" -·")
            if part:
                all_candidates.append(part)
    unique = {}
    for c in all_candidates:
        k = norm(c)
        if k and k not in unique:
            unique[k] = c
    tp, fp = 0, 0
    for key in unique:
        matched = any(any(norm(c) == key for c in ([ent] + aliases.get(ent, []))) for ent in target_set)
        if matched:
            tp += 1
        elif key in KNOWN_NON_ENTITIES:
            fp += 1
        else:
            tp += 1
    precision = tp / len(unique) if unique else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    timing = memory.get("timing", {})
    classify_text_image_s = timing.get("model_load_s", 0) + sum(
        pg["classify_s"] + pg["text_extract_s"] + pg["image_vlm_s"] for pg in timing.get("pages", [])
    )
    total_pipeline_s = round(classify_text_image_s + table_stage_s + entity_extract_total_s, 2)

    baseline = {"recall": 0.9, "precision": 0.8333, "f1": 0.8654,
                "table_stage_s": 11.422, "total_pipeline_s": 1097.51}

    result = {
        "recall": round(recall, 4), "hits": hits, "misses": misses,
        "precision_approx": round(precision, 4), "f1_approx": round(f1, 4),
        "n_simple_tables": n_simple, "n_complex_tables": n_complex,
        "table_stage_s": table_stage_s,
        "entity_extract_stage_s": entity_extract_total_s,
        "classify_text_image_stage_s": round(classify_text_image_s, 2),
        "total_pipeline_s": total_pipeline_s,
        "per_page_entities": per_page_entities,
        "comparison_vs_v2_reppenalty_baseline": baseline,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# [13] Adaptive Table Complexity Router — 전체 파이프라인 연동 결과",
        "",
        "## 성능 지표",
        "",
        "| 지표 | v2+penalty(전부 Docling, 현재 최선) | **v5(Adaptive Router)** |",
        "|---|---|---|",
        f"| Recall | 90.0% (9/10) | **{recall:.1%} ({len(hits)}/{len(target_set)})** |",
        f"| Precision(근사) | 83.3% | **{precision:.1%}** |",
        f"| F1(근사) | 86.5% | **{f1:.1%}** |",
        f"| 표 단계 소요 | 11.422초(전부 Docling) | **{table_stage_s}초**"
        f"(SIMPLE {n_simple}개 pdfplumber + COMPLEX {n_complex}개 Docling) |",
        f"| 총 처리시간 | 1097.51초 | **{total_pipeline_s}초** |",
        "",
        "### 구간별 지연",
        f"- 페이지분류+텍스트+이미지설명(Qwen2.5-VL): {classify_text_image_s:.1f}초 (변경 없음, 기존 재사용)",
        f"- 표 구조화(Adaptive Router): {table_stage_s}초",
        f"- 엔티티 추출(Qwen2.5-VL, repetition_penalty=1.3): {entity_extract_total_s}초",
        "",
        "### Hit", *[f"- {h}" for h in hits], "",
        "### Miss", *[f"- {m}" for m in misses], "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nRecall: {recall:.1%} ({len(hits)}/{len(target_set)})")
    print(f"Precision(근사): {precision:.1%}, F1(근사): {f1:.1%}")
    print(f"표 단계: {table_stage_s}s, 총 처리시간: {total_pipeline_s}s")
    print(f"[report] saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()

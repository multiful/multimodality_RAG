"""[8] 표 단위 분리 엔티티 추출 — 반복 유발(및 page4류 극단적 지연) 구조적 차단.

기존 v2/v2_reppenalty 문제: 텍스트+표(전부)+이미지설명을 한 프롬프트로 합쳐서 넣다 보니
표가 많은 페이지(page4)는 프롬프트가 거대해져 prefill 자체가 느려지고(614초, 반복도 아님),
다른 페이지는 컨텍스트 과부하로 degenerate 반복이 유발됨.

해결: 프롬프트를 쪼갠다.
  A) 페이지 레벨 호출: 본문 텍스트 + 이미지/차트 설명만(표 제외) -> 항상 짧음
  B) 표 레벨 호출: 표 1개씩 개별적으로 엔티티 추출 -> 표가 몇 개든 각 호출은 항상 짧음
  최종 엔티티 = A ∪ B (페이지별로 합산 후 전체 dedup)

YOLOv11n 크롭(가장 recall 좋았던 조합, table_crops/) + Docling 동적병렬은 그대로 재사용.
repetition_penalty=1.3은 안전장치로 유지(짧은 프롬프트에서도 비용 없음).
"""

import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # [legacy] table_processing/legacy_entity_extraction/에서 한 단계 더 이동
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
PDF_DIR = Path(__file__).resolve().parent
CROP_DIR = ROOT / "pdf_pipeline" / "page_classification" / "table_crops"
MEMORY_PATH = ROOT / "pdf_pipeline" / "memory_store.json"
GROUND_TRUTH_PATH = ROOT / "pdf_pipeline" / "ground_truth_064400.json"
RESULT_PATH = PDF_DIR / "result_pipeline_v3_entities.json"
REPORT_PATH = ROOT / "pdf_pipeline" / "table_processing" / "실험_v3_split_entity_recall_report.md"
MAX_WORKERS = 8

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
    md_parts = [t.export_to_dataframe(res.document).to_markdown(index=False) for t in res.document.tables]
    m = re.match(r"page_(\d+)_table_\d+\.png", Path(crop_path_str).name)
    page = int(m.group(1)) if m else None
    return {"file": Path(crop_path_str).name, "page": page,
            "elapsed_s": round(_time.perf_counter() - t0, 3), "markdown": "\n\n".join(md_parts)}


def run_docling_table_stage():
    crop_paths = sorted(str(p) for p in CROP_DIR.glob("*.png"))
    n_workers = min(len(crop_paths), os.cpu_count(), MAX_WORKERS)
    print(f"[table] Docling 동적 워커: min({len(crop_paths)}, {os.cpu_count()}, {MAX_WORKERS}) = {n_workers}", flush=True)

    ex = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_docling_worker)
    for f in [ex.submit(_noop) for _ in range(n_workers)]:
        f.result()
    print("[table] Docling 워커 풀 워밍업 완료", flush=True)

    t0 = time.perf_counter()
    by_page = {}
    futures = {ex.submit(_docling_parse, p): p for p in crop_paths}
    for fut in as_completed(futures):
        r = fut.result()
        by_page.setdefault(r["page"], []).append(r["markdown"])
        print(f"  {r['file']}: {r['elapsed_s']}s", flush=True)
    docling_elapsed = round(time.perf_counter() - t0, 3)
    ex.shutdown()
    print(f"[table] Docling 표 구조화 총 소요: {docling_elapsed}s", flush=True)
    return by_page, docling_elapsed


def load_qwen():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    return model, processor, device


def text_generate(model, processor, device, prompt: str, max_new_tokens: int = 150) -> str:
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


ENTITY_PROMPT_TMPL = (
    "다음은 증권사 리포트에서 추출한 내용입니다. 이 안에 등장하는 모든 기업/기관 이름을 빠짐없이 나열하세요. "
    "한 줄에 하나씩 '기업명 (아는 경우 종목코드)' 형태로만 출력하고 다른 설명은 하지 마세요.\n\n{context}"
)


def main():
    memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    docling_tables_by_page, docling_elapsed = run_docling_table_stage()

    model, processor, device = load_qwen()
    per_page_entities = {}
    page_level_timing = []
    table_level_timing = []

    for p in memory["pages"]:
        # A) 페이지 레벨: 본문 + 이미지설명만 (표 제외 -> 항상 짧음)
        parts = []
        if p["text"]:
            parts.append(f"[본문]\n{p['text']}")
        for desc in p["image_descriptions"]:
            parts.append(f"[이미지/차트]\n{desc}")
        page_context = "\n\n".join(parts)

        page_entities = ""
        if page_context.strip():
            t0 = time.perf_counter()
            page_entities = text_generate(model, processor, device, ENTITY_PROMPT_TMPL.format(context=page_context))
            elapsed = round(time.perf_counter() - t0, 3)
            page_level_timing.append(elapsed)
            print(f"[page-level p{p['page']}] ({elapsed}s, {len(page_context)}자) {page_entities}", flush=True)

        # B) 표 레벨: 표 1개씩 개별 호출 (표가 몇 개든 각 호출은 항상 짧음)
        table_entities_list = []
        for t_idx, md in enumerate(docling_tables_by_page.get(p["page"], []), start=1):
            t0 = time.perf_counter()
            te = text_generate(model, processor, device, ENTITY_PROMPT_TMPL.format(context=f"[표]\n{md}"),
                                max_new_tokens=100)
            elapsed = round(time.perf_counter() - t0, 3)
            table_level_timing.append(elapsed)
            table_entities_list.append(te)
            print(f"  [table-level p{p['page']} t{t_idx}] ({elapsed}s, {len(md)}자) {te}", flush=True)

        per_page_entities[p["page"]] = "\n".join([page_entities] + table_entities_list)

    page_level_total_s = round(sum(page_level_timing), 3)
    table_level_total_s = round(sum(table_level_timing), 3)
    entity_extract_total_s = round(page_level_total_s + table_level_total_s, 3)

    # --- Recall/Precision (기존과 동일 방식) ---
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

    KNOWN_NON_ENTITIES = {"대외고객", "기타특수관계자", "researchcenter"}
    all_candidates = []
    for raw in per_page_entities.values():
        for part in re.split(r"[\n,]", raw):
            part = re.sub(r"\([^)]*\)", "", part).strip(" -·")
            if part:
                all_candidates.append(part)
    unique = {}
    for c in all_candidates:
        k = norm(c)
        if k and k not in unique:
            unique[k] = c
    tp = fp = 0
    for key in unique:
        matched = any(any(norm(c) == key for c in ([ent] + aliases.get(ent, []))) for ent in target_set)
        if matched or key not in KNOWN_NON_ENTITIES:
            tp += 1
        else:
            fp += 1
    precision = tp / len(unique) if unique else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    timing = memory.get("timing", {})
    classify_text_image_s = timing.get("model_load_s", 0) + sum(
        pg["classify_s"] + pg["text_extract_s"] + pg["image_vlm_s"] for pg in timing.get("pages", [])
    )
    total_pipeline_s = round(classify_text_image_s + docling_elapsed + entity_extract_total_s, 2)

    result = {
        "recall": round(recall, 4), "hits": hits, "misses": misses,
        "precision_approx": round(precision, 4), "f1_approx": round(f1, 4),
        "docling_table_stage_s": docling_elapsed,
        "page_level_entity_s": page_level_total_s,
        "table_level_entity_s": table_level_total_s,
        "entity_extract_stage_s": entity_extract_total_s,
        "classify_text_image_stage_s": round(classify_text_image_s, 2),
        "total_pipeline_s": total_pipeline_s,
        "per_page_entities": per_page_entities,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# [8] 표 단위 분리 엔티티 추출 — 반복 유발 구조적 차단",
        "",
        "## 성능 지표",
        "",
        "| 지표 | v2+penalty(통짜 프롬프트) | **v3(표 분리)** |",
        "|---|---|---|",
        f"| Recall | 90.0% (9/10) | **{recall:.1%} ({len(hits)}/{len(target_set)})** |",
        f"| Precision(근사) | 83.3% | **{precision:.1%}** |",
        f"| F1(근사) | 86.5% | **{f1:.1%}** |",
        f"| 총 처리시간 | 1097.5초 | **{total_pipeline_s}초** |",
        "",
        "### 구간별 지연",
        f"- 페이지분류+텍스트+이미지설명(Qwen2.5-VL): {classify_text_image_s:.1f}초 (변경 없음)",
        f"- 표 구조화(YOLOv11n Crop + Docling 동적병렬): {docling_elapsed}초",
        f"- 엔티티 추출(페이지 레벨, 표 제외): {page_level_total_s}초",
        f"- 엔티티 추출(표 레벨, 개별): {table_level_total_s}초",
        "",
        "### Hit", *[f"- {h}" for h in hits], "",
        "### Miss", *[f"- {m}" for m in misses], "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nRecall: {recall:.1%} ({len(hits)}/{len(target_set)})")
    print(f"Precision(근사): {precision:.1%}, F1(근사): {f1:.1%}")
    print(f"총 처리시간: {total_pipeline_s}s (페이지레벨 {page_level_total_s}s + 표레벨 {table_level_total_s}s)")
    print(f"[report] saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()

"""[14] Table-aware Entity Extraction 분기 — 진짜 병목(표 추출 10초가 아니라 엔티티 추출 500~600초)을
직접 겨냥한 고도화. 표를 재무제표(Finance)/계약·일반(Contract·General)으로 나눠 재무제표는 LLM 호출 없이
규칙 기반으로만 처리하고, 계약/일반 표만 LLM에 보낸다. 표 위치→구조화는 [13] Adaptive Router(round1,
8워커) 그대로 재사용 — 이번 고도화의 대상은 오직 "표 구조화 결과를 어떻게 엔티티 추출에 넘기느냐"뿐.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_table_router import RouterThresholds, detect_and_route  # noqa: E402
from table_type_router import classify_table, rule_extract_entities  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
OUT_DIR = Path(__file__).resolve().parent
MEMORY_PATH = ROOT / "pdf_pipeline" / "memory_store.json"
GROUND_TRUTH_PATH = ROOT / "pdf_pipeline" / "ground_truth_064400.json"
RESULT_PATH = OUT_DIR / "result_pipeline_v6_table_aware_entities.json"
REPORT_PATH = ROOT / "pdf_pipeline" / "table_processing" / "실험_v6_table_aware_recall_report.md"
MAX_WORKERS = 8

HEADER_PATTERN = re.compile(r"([가-힣A-Za-z]+(?:\s[가-힣A-Za-z]+)?)\s*\[(\d{6})\]")

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
    """[13] Adaptive Router(round1 설정: 순차, 8워커) 재사용 — 표마다 개별 마크다운 리스트 반환."""
    crop_dir = OUT_DIR / "table_crops_router_complex_v6"
    t0 = time.perf_counter()
    routed = detect_and_route(RouterThresholds(), crop_dir=crop_dir)
    route_decision_s = round(time.perf_counter() - t0, 3)

    simple = [r for r in routed if r["complexity"] == "simple"]
    complex_ = [r for r in routed if r["complexity"] == "complex"]
    print(f"[table] SIMPLE(pdfplumber) {len(simple)}개 / COMPLEX(Docling) {len(complex_)}개", flush=True)

    # by_page: [(markdown, raw_text), ...] — markdown은 LLM에 넘길 구조화 표 데이터,
    # raw_text는 pdfplumber 원문(표 타입 분류처럼 "정확한 한글 텍스트"가 필요한 용도 전용
    # — Docling OCR이 한글 행 라벨을 깨뜨리는 경우가 있어 분류 신호로는 markdown을 못 씀)
    by_page = {}
    for r in simple:
        by_page.setdefault(r["page"], []).append((r["markdown"], r["raw_text"]))

    docling_elapsed = 0.0
    if complex_:
        n_workers = min(len(complex_), os.cpu_count(), MAX_WORKERS)
        ex = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_docling_worker)
        for f in [ex.submit(_noop) for _ in range(n_workers)]:
            f.result()
        t0 = time.perf_counter()
        futures = {ex.submit(_docling_parse, r["crop_path"]): r for r in complex_}
        for fut in as_completed(futures):
            res = fut.result()
            match = next(rr for rr in complex_ if Path(rr["crop_path"]).name == res["file"])
            by_page.setdefault(match["page"], []).append((res["markdown"], match["raw_text"]))
            print(f"  [docling] {res['file']}: {res['elapsed_s']}s", flush=True)
        docling_elapsed = round(time.perf_counter() - t0, 3)
        ex.shutdown()

    table_stage_total_s = round(route_decision_s + docling_elapsed, 3)
    print(f"[table] 표 단계 총 소요: {table_stage_total_s}s", flush=True)
    return by_page, table_stage_total_s, len(simple), len(complex_)


def derive_document_anchor(memory) -> set:
    """LLM 호출 없이 페이지 텍스트에서 '회사명 [종목코드]' 헤더 패턴으로 문서 앵커를 자동 추출
    (하드코딩 아님 — 다른 한국 증권사 리포트 헤더 포맷에도 일반적으로 적용 가능)."""
    counts = {}
    for p in memory["pages"]:
        for name, code in HEADER_PATTERN.findall(p.get("text", "")):
            key = (name.strip(), code)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return set()
    (name, code), _ = max(counts.items(), key=lambda kv: kv[1])
    return {name, code}


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
    anchor_entities = derive_document_anchor(memory)
    print(f"[anchor] 자동 도출된 문서 앵커: {anchor_entities}", flush=True)

    # --- 표 타입 분류: finance는 규칙 기반, contract/general만 LLM 프롬프트에 포함 ---
    llm_tables_by_page = {}
    rule_entities_by_page = {}
    table_type_log = []
    for page, entries in table_md_by_page.items():
        for t_idx, (md, raw_text) in enumerate(entries, start=1):
            # 분류는 pdfplumber 원문(raw_text)으로 — Docling markdown은 OCR 오류로 한글 라벨이
            # 깨질 수 있어 분류 신호로 신뢰 불가(예: "매출액"->"OH EOH"). 다운스트림 LLM에 넘길
            # 내용은 여전히 markdown(구조화된 표) 그대로 사용.
            ttype = classify_table(raw_text)
            table_type_log.append({"page": page, "table_idx": t_idx, "type": ttype})
            if ttype == "finance":
                found = rule_extract_entities(raw_text, anchor_entities)
                rule_entities_by_page.setdefault(page, []).extend(found)
                print(f"[table_type] page{page} table{t_idx}: FINANCE(규칙) -> {found}", flush=True)
            else:
                llm_tables_by_page.setdefault(page, []).append(md)
                print(f"[table_type] page{page} table{t_idx}: CONTRACT/GENERAL(LLM)", flush=True)

    model, processor, device = load_qwen()
    per_page_entities = {}
    entity_timing = []
    llm_calls_made = 0
    llm_calls_skipped = 0

    for p in memory["pages"]:
        parts = []
        if p["text"]:
            parts.append(f"[본문]\n{p['text']}")
        mds = llm_tables_by_page.get(p["page"], [])
        for t_idx, md in enumerate(mds, start=1):
            parts.append(f"[표{t_idx}]\n{md}")
        for desc in p["image_descriptions"]:
            parts.append(f"[이미지/차트]\n{desc}")
        page_context = "\n\n".join(parts)

        rule_entities = rule_entities_by_page.get(p["page"], [])
        if not page_context.strip():
            per_page_entities[p["page"]] = ", ".join(rule_entities)
            llm_calls_skipped += 1
            print(f"[entity p{p['page']}] LLM 호출 생략(재무제표만 존재) -> 규칙추출 {rule_entities}", flush=True)
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
        llm_calls_made += 1
        combined = result + ("\n" + ", ".join(rule_entities) if rule_entities else "")
        per_page_entities[p["page"]] = combined
        print(f"[entity p{p['page']}] ({elapsed}s) LLM={result} | 규칙추가={rule_entities}", flush=True)

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

    baseline_v2 = {"recall": 0.9, "precision": 0.8333, "f1": 0.8654, "total_pipeline_s": 1097.51}
    baseline_v5 = {"recall": 0.9, "precision": 0.8421, "f1": 0.8701, "total_pipeline_s": 992.74}

    result = {
        "recall": round(recall, 4), "hits": hits, "misses": misses,
        "precision_approx": round(precision, 4), "f1_approx": round(f1, 4),
        "n_simple_tables": n_simple, "n_complex_tables": n_complex,
        "table_type_log": table_type_log,
        "llm_calls_made": llm_calls_made, "llm_calls_skipped": llm_calls_skipped,
        "anchor_entities": sorted(anchor_entities),
        "table_stage_s": table_stage_s,
        "entity_extract_stage_s": entity_extract_total_s,
        "classify_text_image_stage_s": round(classify_text_image_s, 2),
        "total_pipeline_s": total_pipeline_s,
        "per_page_entities": per_page_entities,
        "comparison_vs_v2_baseline": baseline_v2,
        "comparison_vs_v5_adaptive_router": baseline_v5,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# [14] Table-aware Entity Extraction 분기 — 결과",
        "",
        "## 성능 지표",
        "",
        "| 지표 | v2+penalty(전부 LLM) | v5(Adaptive Router) | **v6(Table-aware 분기)** |",
        "|---|---|---|---|",
        f"| Recall | 90.0%(9/10) | 90.0%(9/10) | **{recall:.1%}({len(hits)}/{len(target_set)})** |",
        f"| Precision(근사) | 83.3% | 84.2% | **{precision:.1%}** |",
        f"| F1(근사) | 86.5% | 87.0% | **{f1:.1%}** |",
        f"| 표 단계 소요 | 11.4초 | 12.4초 | **{table_stage_s}초** |",
        f"| 엔티티추출 LLM 호출 수 | 6/6페이지 | 6/6페이지 | **{llm_calls_made}/6페이지({llm_calls_skipped}개 생략)** |",
        f"| 엔티티추출 단계 소요 | 695.5초 | 589.7초 | **{entity_extract_total_s}초** |",
        f"| 총 처리시간 | 1097.51초 | 992.74초 | **{total_pipeline_s}초** |",
        "",
        f"자동 도출 문서 앵커: {sorted(anchor_entities)}",
        "",
        "### 표 타입 분류 결과", *[f"- page{t['page']} table{t['table_idx']}: {t['type']}" for t in table_type_log], "",
        "### Hit", *[f"- {h}" for h in hits], "",
        "### Miss", *[f"- {m}" for m in misses], "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nRecall: {recall:.1%} ({len(hits)}/{len(target_set)})")
    print(f"Precision(근사): {precision:.1%}, F1(근사): {f1:.1%}")
    print(f"LLM 호출: {llm_calls_made}/6 (생략 {llm_calls_skipped}개)")
    print(f"표 단계: {table_stage_s}s, 엔티티추출: {entity_extract_total_s}s, 총 처리시간: {total_pipeline_s}s")
    print(f"[report] saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()

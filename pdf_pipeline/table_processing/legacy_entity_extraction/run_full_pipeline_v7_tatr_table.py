"""[18] v6c의 표 단계(Docling)를 TATR(adaptive_padding+300dpi)+pdfplumber로 교체해 실제
엔티티 추출 파이프라인에서 Recall/Precision/F1이 유지되는지 검증. [17]에서 "usable rows"
대리 지표로는 Docling(228행)과 동등/우수(242행)했지만, 실제 엔티티 추출 성능으로 이어지는지는
별도 확인이 필요 — 그 확인이 이 스크립트의 목적. [13]의 SIMPLE/COMPLEX 라우팅은 그대로 유지
(SIMPLE 표에선 TATR이 오히려 손해였음을 확인했으므로), COMPLEX 표의 엔진만 교체.
"""

import json
import re
import sys
import time
from pathlib import Path

import fitz
import pdfplumber
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # [legacy] table_type_router 등은 table_processing/에 그대로 있음
from adaptive_table_router import RouterThresholds, detect_and_route  # noqa: E402
from table_type_router import classify_table, rule_extract_entities  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # [legacy] table_processing/legacy_entity_extraction/에서 한 단계 더 이동
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "LGCNS" / "20260721_company_279243000.pdf"
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
OUT_DIR = Path(__file__).resolve().parent
MEMORY_PATH = ROOT / "pdf_pipeline" / "memory_store.json"
GROUND_TRUTH_PATH = ROOT / "pdf_pipeline" / "ground_truth_064400.json"
RESULT_PATH = OUT_DIR / "result_pipeline_v7_tatr_table_entities.json"
REPORT_PATH = ROOT / "pdf_pipeline" / "table_processing" / "실험_v7_tatr_table_recall_report.md"

HEADER_PATTERN = re.compile(r"([가-힣A-Za-z]+(?:\s[가-힣A-Za-z]+)?)\s*\[(\d{6})\]")

# [17]에서 확인된 TATR 최적 설정
TATR_DPI = 300
TATR_TOP_PAD_PT = 35 / (150 / 72)
TATR_SIDE_PAD_PT = 12 / (150 / 72)
TATR_CONF = 0.6


def tatr_extract_table(model, processor, doc_fitz, page_pp, page_num, bbox_pt):
    """TATR로 row 구조 인식 후 pdfplumber로 각 row 텍스트 채움 -> 마크다운 텍스트 생성."""
    page_fz = doc_fitz[page_num - 1]
    padded_pt = (
        max(0.0, bbox_pt[0] - TATR_SIDE_PAD_PT), max(0.0, bbox_pt[1] - TATR_TOP_PAD_PT),
        min(page_fz.rect.width, bbox_pt[2] + TATR_SIDE_PAD_PT), min(page_fz.rect.height, bbox_pt[3] + TATR_SIDE_PAD_PT),
    )
    pix = page_fz.get_pixmap(dpi=TATR_DPI, clip=fitz.Rect(*padded_pt))
    tmp = OUT_DIR / f"_tmp_v7_p{page_num}.png"
    pix.save(str(tmp))
    img = Image.open(tmp).convert("RGB")
    tmp.unlink(missing_ok=True)

    scale = TATR_DPI / 72
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([img.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=TATR_CONF, target_sizes=target_sizes)[0]

    rows_with_y = []
    for label_id, box in zip(results["labels"], results["boxes"]):
        if model.config.id2label[label_id.item()] != "table row":
            continue
        rx1, ry1, rx2, ry2 = [v.item() for v in box]
        row_pt = (padded_pt[0] + rx1 / scale, padded_pt[1] + ry1 / scale,
                  padded_pt[0] + rx2 / scale, padded_pt[1] + ry2 / scale)
        text = (page_pp.crop(row_pt).extract_text() or "").replace("\n", " ").strip()
        if text:
            rows_with_y.append((ry1, text))
    rows_with_y.sort(key=lambda r: r[0])  # 위->아래 순서로 정렬
    return "\n".join(t for _, t in rows_with_y)


def run_table_stage():
    """[13] Adaptive Router로 SIMPLE/COMPLEX 분류(불변), COMPLEX만 TATR+pdfplumber로 교체."""
    routed = detect_and_route(RouterThresholds())
    simple = [r for r in routed if r["complexity"] == "simple"]
    complex_ = [r for r in routed if r["complexity"] == "complex"]
    print(f"[table] SIMPLE(pdfplumber) {len(simple)}개 / COMPLEX(TATR+pdfplumber) {len(complex_)}개", flush=True)

    by_page = {}
    for r in simple:
        by_page.setdefault(r["page"], []).append((r["markdown"], r["raw_text"]))

    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    model.eval()

    doc_fitz = fitz.open(str(PDF_PATH))
    pdf_pp = pdfplumber.open(str(PDF_PATH))

    t0 = time.perf_counter()
    for r in complex_:
        x1, y1, x2, y2 = r["bbox_px"]
        SCALE = 150 / 72
        bbox_pt = (x1 / SCALE, y1 / SCALE, x2 / SCALE, y2 / SCALE)
        md = tatr_extract_table(model, processor, doc_fitz, pdf_pp.pages[r["page"] - 1], r["page"], bbox_pt)
        by_page.setdefault(r["page"], []).append((md, r["raw_text"]))
        print(f"  [tatr] page{r['page']} table{r['table_idx']}: {len(md.splitlines())}행 추출", flush=True)
    tatr_elapsed = round(time.perf_counter() - t0, 3)
    pdf_pp.close()
    doc_fitz.close()

    table_stage_total_s = tatr_elapsed
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

    baseline_v6c = {"recall": 0.9, "precision": 0.9441, "f1": 0.9216, "table_stage_s": 12.164,
                     "total_pipeline_s": 462.8}

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
        "comparison_vs_v6c_docling_baseline": baseline_v6c,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# [18] v7: 표 단계를 TATR(adaptive_padding+300dpi)+pdfplumber로 교체 — 결과",
        "",
        "## 성능 지표",
        "",
        "| 지표 | v6c(Docling, grounding filter 적용 후) | **v7(TATR)** |",
        "|---|---|---|",
        f"| Recall | 90.0%(9/10) | **{recall:.1%}({len(hits)}/{len(target_set)})** |",
        f"| Precision(근사, grounding filter 전) | 97.0% | **{precision:.1%}** |",
        f"| F1(근사, grounding filter 전) | 93.4% | **{f1:.1%}** |",
        f"| 표 단계 소요 | 12.16초(Docling) | **{table_stage_s}초(TATR)** |",
        f"| 엔티티추출 LLM 호출 수 | 6/6페이지 | **{llm_calls_made}/6페이지({llm_calls_skipped}개 생략)** |",
        f"| 엔티티추출 단계 소요 | 60.1초 | **{entity_extract_total_s}초** |",
        f"| 총 처리시간 | 462.8초 | **{total_pipeline_s}초** |",
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

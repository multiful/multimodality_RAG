"""memory_store.json(페이지별 텍스트/표/이미지 추출 결과)을 읽어 페이지별로 엔티티를
추출(짧은 컨텍스트라 MPS OOM 없음)하고, 합쳐서 ground_truth 대비 recall을 계산한다.

run_baseline.py의 마지막 단계(전체 16K자 컨텍스트 1회 호출)가 MPS OOM으로 실패해서
페이지 단위로 쪼갠 버전으로 대체.
"""

import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
OUT_DIR = Path(__file__).resolve().parent
MEMORY_PATH = OUT_DIR / "memory_store.json"
GROUND_TRUTH_PATH = OUT_DIR / "ground_truth_064400.json"
ENTITIES_PATH = OUT_DIR / "extracted_entities.json"
REPORT_PATH = OUT_DIR / "recall_report.md"


def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[model] loading Qwen2.5-VL-7B-Instruct on {device}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    return model, processor, device


def text_generate(model, processor, device, prompt: str, max_new_tokens: int = 300) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    result = processor.decode(trimmed, skip_special_tokens=True).strip()
    del inputs, out
    if device == "mps":
        torch.mps.empty_cache()
    return result


def main():
    memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    model, processor, device = load_model()

    per_page_entities = {}
    for p in memory["pages"]:
        parts = []
        if p["text"]:
            parts.append(f"[본문]\n{p['text']}")
        for t_idx, md in enumerate(p["tables_markdown"], start=1):
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
        print(f"\n[page {p['page']}] context {len(page_context)} chars -> extracting entities", flush=True)
        result = text_generate(model, processor, device, prompt, max_new_tokens=200)
        print(f"[page {p['page']}] {result}", flush=True)
        per_page_entities[p["page"]] = result

    ENTITIES_PATH.write_text(json.dumps(per_page_entities, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[entities] saved to {ENTITIES_PATH}", flush=True)

    # Recall 평가: 페이지별 추출 결과를 전부 합쳐서 문자열 포함 여부로 매칭
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    target_set = ground_truth["entity_recall_target_set"]
    combined = "\n".join(per_page_entities.values()).lower().replace(" ", "")

    hits, misses = [], []
    for ent in target_set:
        norm = ent.lower().replace(" ", "")
        if norm in combined:
            hits.append(ent)
        else:
            misses.append(ent)

    recall = len(hits) / len(target_set)
    lines = [
        "# Baseline 엔티티 Recall 리포트 (페이지 단위 추출)",
        "",
        f"- 대상 문서: 20260721_company_279243000.pdf",
        f"- 정답 엔티티 수: {len(target_set)}",
        f"- 추출 성공(hit): {len(hits)}",
        f"- 누락(miss): {len(misses)}",
        f"- **Recall: {recall:.1%}**",
        "",
        "## Hit", *[f"- {h}" for h in hits], "",
        "## Miss", *[f"- {m}" for m in misses], "",
        "## 페이지별 추출 원본",
    ]
    for pg, ents in per_page_entities.items():
        lines += [f"### page {pg}", "```", ents or "(내용 없음)", "```"]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[recall] {recall:.1%} ({len(hits)}/{len(target_set)})", flush=True)
    print(f"[report] saved to {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()

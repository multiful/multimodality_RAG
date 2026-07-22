"""memory_store.json(페이지별 텍스트/표/이미지 추출 결과)을 읽어 페이지별로 엔티티를
추출한다(짧은 컨텍스트라 MPS OOM 없음 — run_baseline.py의 원래 방식인 전체 16K자
컨텍스트 1회 호출은 MPS OOM으로 실패해서 페이지 단위로 쪼갬).

recall/precision/F1/latency 집계는 evaluate.py에서 이 파일의 출력(extracted_entities.json,
entity_extract_timing.json)과 run_baseline.py의 memory_store.json을 합쳐서 수행한다.
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
OUT_DIR = Path(__file__).resolve().parent
MEMORY_PATH = OUT_DIR / "memory_store.json"
ENTITIES_PATH = OUT_DIR / "extracted_entities.json"
ENTITY_TIMING_PATH = OUT_DIR / "entity_extract_timing.json"


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

    t0 = time.time()
    model, processor, device = load_model()
    model_load_s = round(time.time() - t0, 2)
    print(f"[timing] model load: {model_load_s}s", flush=True)

    per_page_entities = {}
    per_page_timing = []
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
            per_page_timing.append({"page": p["page"], "entity_extract_s": 0.0, "context_chars": 0})
            continue

        prompt = (
            "다음은 증권사 리포트 한 페이지에서 추출한 내용입니다. "
            "이 안에 등장하는 모든 기업/기관 이름을 빠짐없이 나열하세요. "
            "표나 차트 범례 안에서만 언급된 기업도 포함하세요. "
            "한 줄에 하나씩 '기업명 (아는 경우 종목코드)' 형태로만 출력하고 다른 설명은 하지 마세요.\n\n"
            f"{page_context}"
        )
        print(f"\n[page {p['page']}] context {len(page_context)} chars -> extracting entities", flush=True)
        t = time.time()
        result = text_generate(model, processor, device, prompt, max_new_tokens=200)
        elapsed = round(time.time() - t, 3)
        print(f"[page {p['page']}] ({elapsed}s) {result}", flush=True)
        per_page_entities[p["page"]] = result
        per_page_timing.append({"page": p["page"], "entity_extract_s": elapsed, "context_chars": len(page_context)})

    ENTITIES_PATH.write_text(json.dumps(per_page_entities, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[entities] saved to {ENTITIES_PATH}", flush=True)

    timing_out = {
        "model_load_s": model_load_s,
        "pages": per_page_timing,
        "total_entity_extract_s": round(sum(pt["entity_extract_s"] for pt in per_page_timing), 3),
    }
    ENTITY_TIMING_PATH.write_text(json.dumps(timing_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[timing] saved to {ENTITY_TIMING_PATH}", flush=True)


if __name__ == "__main__":
    main()

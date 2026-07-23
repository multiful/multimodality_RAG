"""[7] 사용자 제안 A/B — 계층적 청킹의 실제 문단 경계에 gpt-4o-mini로 컨텍스트를 생성하면
rule-based(현재 채택)보다 나은지 검증. [5]의 openai 백엔드는 naive base_chars 분할에 적용했었는데,
rule-based가 정적 placeholder([5])에서 계층적 경계의 실제 section_path([6])로 바꿨을 때 recall이
크게 개선됐던 것과 같은 효과가 openai에도 있는지 확인.
"""

import json
import os
import sys
import time
from pathlib import Path

import fitz
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contextual_chunker import chunk_contextual_production  # noqa: E402
from text_extraction import extract_page_text  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
YOLO_MODEL_PATH = ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"


def extract_entities_facts(chunk_text, qwen_model, qwen_processor, device, max_new_tokens=120):
    prompt = (
        "다음은 증권사 리포트의 한 조각(chunk)입니다. 이 조각 하나만 보고 판단하세요(다른 "
        "맥락은 없다고 가정). 여기 등장하는 모든 회사/기관명과, 숫자가 포함된 구체적 사실을 "
        "전부 나열하세요. 없으면 '없음'이라고만 쓰세요. 다른 설명 없이 나열만 하세요.\n\n"
        f"{chunk_text}"
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.3)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    result = qwen_processor.decode(trimmed, skip_special_tokens=True).strip()
    del inputs, out
    if device == "mps":
        torch.mps.empty_cache()
    return result


def score_recall(extracted_union, gt_doc):
    entity_hits = [e["name"] for e in gt_doc["entities"] if e["anchor"] in extracted_union]
    fact_hits = [f["fact"] for f in gt_doc["key_facts"] if f["anchor"] in extracted_union]
    return {"entity_recall": round(len(entity_hits) / len(gt_doc["entities"]), 4),
            "fact_recall": round(len(fact_hits) / len(gt_doc["key_facts"]), 4)}


def main():
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY 환경변수가 설정되어 있지 않음"

    print("모델 로딩 중...", flush=True)
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    warmup = Image.new("RGB", (595, 842), (255, 255, 255))
    yolo_model.predict(warmup, conf=0.25, verbose=False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    qwen_processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    print(f"로딩 완료 (device={device})", flush=True)

    gt = json.loads((OUT_DIR / "ground_truth_chunking_eval.json").read_text(encoding="utf-8"))
    doc_titles = {"lgcns_p1": "LG CNS 기업분석 리포트", "construct_p1": "Construct 건설 Weekly 리포트",
                  "construct_p5": "Construct 건설 Weekly 리포트"}

    all_results = {}
    for doc_key, gt_doc in gt["documents"].items():
        print(f"\n=== {doc_key} ===", flush=True)
        pdf_path = ROOT / "pdf_pipeline" / gt_doc["pdf"]
        page_idx = gt_doc["page"] - 1
        doc_fitz = fitz.open(str(pdf_path))
        full_doc_text = extract_page_text(doc_fitz, page_idx)["text"]

        t0 = time.perf_counter()
        chunks = chunk_contextual_production(yolo_model, doc_fitz, page_idx,
                                              doc_title=doc_titles[doc_key],
                                              backend="openai", full_doc_text=full_doc_text,
                                              model_name="gpt-4o-mini")
        construct_s = time.perf_counter() - t0
        doc_fitz.close()

        t0 = time.perf_counter()
        extraction_outputs = []
        for c in chunks:
            if len(c["raw_chunk"]) < 15:
                extraction_outputs.append("")
                continue
            extraction_outputs.append(extract_entities_facts(c["text"], qwen_model, qwen_processor, device))
        extraction_s = time.perf_counter() - t0

        union_text = "\n".join(extraction_outputs)
        score = score_recall(union_text, gt_doc)
        result = {
            "n_chunks": len(chunks), "chunk_construction_s": round(construct_s, 3),
            "extraction_total_s": round(extraction_s, 3),
            "total_latency_s": round(construct_s + extraction_s, 3),
            **score,
            "sample_contexts": [c["context_prefix"] for c in chunks[:3]],
        }
        all_results[doc_key] = result
        print(f"  hierarchical+openai: {len(chunks)}청크, entity={score['entity_recall']*100:.0f}%, "
              f"fact={score['fact_recall']*100:.0f}%, 지연={result['total_latency_s']}s "
              f"(구성{construct_s:.2f}s+추출{extraction_s:.2f}s)", flush=True)

    (OUT_DIR / "result_hierarchical_openai.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[result] saved to {OUT_DIR / 'result_hierarchical_openai.json'}")


if __name__ == "__main__":
    main()

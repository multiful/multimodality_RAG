"""[5] 문맥적 청킹 컨텍스트 생성 백엔드 3종 A/B — qwen(로컬 VLM) vs openai(경량 API 모델) vs
rulebased(LLM 호출 없음). 청크 경계(base_chars=350, 문장분리)는 동일하게 고정하고 컨텍스트
생성 방식만 바꿔서 비교 — 지연(컨텍스트 생성 자체)과 하류 엔티티/사실 추출 recall을 함께 측정.
추출 단계는 세 백엔드 모두 로컬 Qwen으로 통일(추출 모델 차이가 비교에 섞이지 않도록 — 순수하게
"컨텍스트 생성 방식"만의 효과를 보기 위함).
"""

import json
import sys
import time
from pathlib import Path

import fitz
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contextual_chunker import chunk_contextual  # noqa: E402
from hierarchical_chunker import _get_boxes_with_text  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"

CLEAN_TEXT_SOURCE = {
    "lgcns_p1": ("ground_truth_text_lgcns.json", "1"),
    "construct_p1": ("ground_truth_text_construct.json", "1"),
    "construct_p5": ("ground_truth_text_construct.json", "5"),
}


def _load_clean_narrative_text(doc_key: str) -> str:
    fname, page_str = CLEAN_TEXT_SOURCE[doc_key]
    gt = json.loads((OUT_DIR / fname).read_text(encoding="utf-8"))
    return "\n".join(gt["pages"][page_str]["units"])


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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Qwen 로딩 중 (device={device})...", flush=True)
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    qwen_processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    print("로딩 완료", flush=True)

    gt = json.loads((OUT_DIR / "ground_truth_chunking_eval.json").read_text(encoding="utf-8"))

    all_results = {}
    for doc_key, gt_doc in gt["documents"].items():
        print(f"\n=== {doc_key} ===", flush=True)
        full_text = _load_clean_narrative_text(doc_key)
        pdf_path = ROOT / "pdf_pipeline" / gt_doc["pdf"]
        page_idx = gt_doc["page"] - 1

        # 계층적 청킹의 section_path(rulebased 백엔드용) 재사용
        doc_fitz = fitz.open(str(pdf_path))
        # YOLO 없이 rulebased 데모용 최소 컨텍스트만 필요하므로 문서 타이틀만 사용
        doc_fitz.close()
        doc_title = gt_doc["pdf"].split("/")[-1]

        doc_result = {}
        for backend in ("qwen", "openai", "rulebased"):
            kwargs = {}
            if backend == "qwen":
                kwargs = {"model": qwen_model, "processor": qwen_processor, "device": device}
            elif backend == "openai":
                kwargs = {"model_name": "gpt-4o-mini"}
            elif backend == "rulebased":
                kwargs = {"section_path": [gt_doc.get("page_title", "")], "doc_title": doc_title}

            t0 = time.perf_counter()
            try:
                chunks = chunk_contextual(full_text, full_text, page=gt_doc["page"], base_chars=350,
                                           backend=backend, **kwargs)
            except Exception as e:
                print(f"  {backend}: 실패 - {e}", flush=True)
                continue
            construct_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            extraction_outputs = []
            for c in chunks:
                out = extract_entities_facts(c["text"], qwen_model, qwen_processor, device)
                extraction_outputs.append(out)
            extraction_s = time.perf_counter() - t0

            union_text = "\n".join(extraction_outputs)
            score = score_recall(union_text, gt_doc)
            doc_result[backend] = {
                "n_chunks": len(chunks), "chunk_construction_s": round(construct_s, 3),
                "extraction_total_s": round(extraction_s, 3),
                "total_latency_s": round(construct_s + extraction_s, 3),
                **score,
                "sample_context_prefixes": [c["context_prefix"] for c in chunks[:3]],
            }
            print(f"  {backend}: {len(chunks)}청크, entity_recall={score['entity_recall']*100:.0f}%, "
                  f"fact_recall={score['fact_recall']*100:.0f}%, 컨텍스트생성={construct_s:.2f}s, "
                  f"총지연={doc_result[backend]['total_latency_s']:.2f}s", flush=True)

        all_results[doc_key] = doc_result

    (OUT_DIR / "result_contextual_backends.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[result] saved to {OUT_DIR / 'result_contextual_backends.json'}")


if __name__ == "__main__":
    main()

"""[3] 청킹 전략 3종 비교 — 계층적(hierarchical) / 시멘틱(semantic) / 문맥적(contextual).
평가 방법: 각 청크를 "독립적으로"(실제 RAG에서 청크 하나만 검색되는 상황 재현) LLM(로컬
Qwen2.5-VL-7B-Instruct, 이 프로젝트의 기존 엔티티 추출과 동일 모델)에 주고 엔티티+핵심 사실을
추출시킨 뒤, 모든 청크의 추출 결과를 합쳐서 골든셋(ground_truth_chunking_eval.json) 대비 recall을
계산. 청크 경계가 "엔티티+숫자" 짝을 갈라놓으면 개별 청크 단위 추출에서 놓치는 게 드러난다.
지연은 (1) 청킹 자체 구성 시간 + (2) 청크별 추출 LLM 호출 시간 합계로 측정.
"""

import json
import sys
import time
from pathlib import Path

import fitz
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from ultralytics import YOLO
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hierarchical_chunker import chunk_hierarchical  # noqa: E402
from semantic_chunker import chunk_semantic  # noqa: E402
from contextual_chunker import chunk_contextual  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
YOLO_MODEL_PATH = ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"

MIN_CHARS_FOR_EXTRACTION = 15  # 이보다 짧은 청크(페이지 장식 조각)는 LLM 추출 대상에서 제외


def load_models():
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    warmup = Image.new("RGB", (595, 842), (255, 255, 255))
    yolo_model.predict(warmup, conf=0.25, verbose=False)

    embed_model = SentenceTransformer("BAAI/bge-m3")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    qwen_processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    return yolo_model, embed_model, qwen_model, qwen_processor, device


def extract_entities_facts(chunk_text: str, qwen_model, qwen_processor, device,
                            max_new_tokens: int = 120) -> str:
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


def score_recall(extracted_union: str, gt_doc: dict) -> dict:
    entity_hits = [e["name"] for e in gt_doc["entities"] if e["anchor"] in extracted_union]
    fact_hits = [f["fact"] for f in gt_doc["key_facts"] if f["anchor"] in extracted_union]
    return {
        "entity_recall": round(len(entity_hits) / len(gt_doc["entities"]), 4),
        "fact_recall": round(len(fact_hits) / len(gt_doc["key_facts"]), 4),
        "entity_hits": entity_hits, "fact_hits": fact_hits,
        "entity_misses": [e["name"] for e in gt_doc["entities"] if e["name"] not in entity_hits],
        "fact_misses": [f["fact"] for f in gt_doc["key_facts"] if f["fact"] not in fact_hits],
    }


CLEAN_TEXT_SOURCE = {  # doc_key -> ([1]에서 검증된 골든셋 파일, 페이지)
    "lgcns_p1": ("ground_truth_text_lgcns.json", "1"),
    "construct_p1": ("ground_truth_text_construct.json", "1"),
    "construct_p5": ("ground_truth_text_construct.json", "5"),
}


def _load_clean_narrative_text(doc_key: str) -> str:
    """시멘틱/문맥적 청킹용 입력 — [1]에서 이미 검증된 골든셋 문단(표/차트 오염 없는 서술형 본문만)을
    그대로 사용. whole_page 원문을 그대로 쓰면 사이드바/차트축 숫자 같은 노이즈([1]에서 순도
    67.3%로 측정된 바로 그 오염)가 섞여, "청킹 전략 자체의 경계 배치 품질"이 아니라 "입력 정제
    수준"을 비교하게 돼버림 — 이 비교의 목적과 맞지 않아 분리."""
    fname, page_str = CLEAN_TEXT_SOURCE[doc_key]
    gt = json.loads((OUT_DIR / fname).read_text(encoding="utf-8"))
    return "\n".join(gt["pages"][page_str]["units"])


def run_doc(doc_key: str, gt_doc: dict, yolo_model, embed_model, qwen_model, qwen_processor, device) -> dict:
    pdf_path = ROOT / "pdf_pipeline" / gt_doc["pdf"]
    page_idx = gt_doc["page"] - 1
    doc_fitz = fitz.open(str(pdf_path))
    # 계층적 청킹은 PDF+YOLO 구조를 그대로 활용(표/차트 배제가 이 방식의 정당한 강점이라 그대로
    # 둠) — 시멘틱/문맥적은 순수 텍스트 방식이라 [1]에서 검증된 정제된 서술형 본문을 입력으로 사용
    full_text = _load_clean_narrative_text(doc_key)

    result = {"doc": doc_key, "methods": {}}

    # --- 계층적 청킹 ---
    t0 = time.perf_counter()
    hier_chunks = chunk_hierarchical(yolo_model, doc_fitz, page_idx)
    hier_construct_s = time.perf_counter() - t0
    result["methods"]["hierarchical"] = _eval_chunks(
        [{"text": c["text"], "meta": {"section_path": c["section_path"]}} for c in hier_chunks],
        gt_doc, qwen_model, qwen_processor, device, hier_construct_s)

    # --- 시멘틱 청킹 ---
    t0 = time.perf_counter()
    sem_chunks = chunk_semantic(full_text, embed_model, page=gt_doc["page"])
    sem_construct_s = time.perf_counter() - t0
    result["methods"]["semantic"] = _eval_chunks(
        [{"text": c["text"], "meta": {"n_sentences": c["n_sentences"]}} for c in sem_chunks],
        gt_doc, qwen_model, qwen_processor, device, sem_construct_s)

    # --- 문맥적 청킹(기본: 로컬 Qwen 백엔드 — [5]에서 openai/rulebased 백엔드 A/B 비교 추가) ---
    t0 = time.perf_counter()
    ctx_chunks = chunk_contextual(full_text, full_text, page=gt_doc["page"], base_chars=350,
                                   backend="qwen", model=qwen_model, processor=qwen_processor, device=device)
    ctx_construct_s = time.perf_counter() - t0
    result["methods"]["contextual"] = _eval_chunks(
        [{"text": c["text"], "meta": {"context_prefix": c["context_prefix"], "raw_chunk": c["raw_chunk"]}}
         for c in ctx_chunks],
        gt_doc, qwen_model, qwen_processor, device, ctx_construct_s)

    doc_fitz.close()
    return result


def _eval_chunks(chunks: list, gt_doc: dict, qwen_model, qwen_processor, device, construct_s: float) -> dict:
    extraction_outputs, extraction_s_total = [], 0.0
    for c in chunks:
        if len(c["text"]) < MIN_CHARS_FOR_EXTRACTION:
            extraction_outputs.append("")
            continue
        t0 = time.perf_counter()
        out = extract_entities_facts(c["text"], qwen_model, qwen_processor, device)
        extraction_s_total += time.perf_counter() - t0
        extraction_outputs.append(out)

    union_text = "\n".join(extraction_outputs)
    score = score_recall(union_text, gt_doc)
    return {
        "n_chunks": len(chunks),
        "chunk_construction_s": round(construct_s, 3),
        "extraction_total_s": round(extraction_s_total, 3),
        "total_latency_s": round(construct_s + extraction_s_total, 3),
        **score,
        "chunks": [{"text": c["text"], "meta": c["meta"], "extracted": ex}
                   for c, ex in zip(chunks, extraction_outputs)],
    }


def main():
    print("모델 로딩 중...", flush=True)
    yolo_model, embed_model, qwen_model, qwen_processor, device = load_models()
    print(f"로딩 완료 (device={device})", flush=True)

    gt = json.loads((OUT_DIR / "ground_truth_chunking_eval.json").read_text(encoding="utf-8"))
    all_results = {}
    for doc_key, gt_doc in gt["documents"].items():
        print(f"\n=== {doc_key} ===", flush=True)
        t0 = time.perf_counter()
        result = run_doc(doc_key, gt_doc, yolo_model, embed_model, qwen_model, qwen_processor, device)
        print(f"  ({round(time.perf_counter()-t0,1)}s)", flush=True)
        for method, m in result["methods"].items():
            print(f"  {method}: {m['n_chunks']}청크, entity_recall={m['entity_recall']*100:.0f}%, "
                  f"fact_recall={m['fact_recall']*100:.0f}%, 지연={m['total_latency_s']}s "
                  f"(구성 {m['chunk_construction_s']}s + 추출 {m['extraction_total_s']}s)", flush=True)
        all_results[doc_key] = result

    (OUT_DIR / "result_chunking_eval.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 방법별 청크 실물을 json/md로 따로 저장(사용자가 실제 출력물을 보고 싶어함)
    for method in ("hierarchical", "semantic", "contextual"):
        method_dump = {doc_key: all_results[doc_key]["methods"][method]["chunks"] for doc_key in gt["documents"]}
        (OUT_DIR / f"chunks_{method}.json").write_text(
            json.dumps(method_dump, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[result] saved to {OUT_DIR / 'result_chunking_eval.json'}")
    print(f"[chunks] saved to chunks_hierarchical.json / chunks_semantic.json / chunks_contextual.json")


if __name__ == "__main__":
    main()

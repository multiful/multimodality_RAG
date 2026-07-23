"""[6] 최종 프로덕션 파이프라인 검증 — 사용자 지시("맞는 파이프라인 적용, 텍스트 정확도 재측정")
반영. 채택 파이프라인: whole_page 추출([1]) + PUA/헤더푸터/구두점·기호 정규화([2][4][5]) +
`chunk_contextual_production`(계층적 청킹 경계 + rule-based 컨텍스트 주입, [5]에서 지연 우선
결정에 따라 채택) — 전부 LLM 호출 없이 무료. 이 최종 조합으로 (a) [1]의 whole_page 텍스트 추출
정확도(골든셋 recall)와 (b) [3]/[5]의 청킹 entity/fact recall을 둘 다 재측정.
"""

import json
import sys
import time
from pathlib import Path

import fitz
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_extraction import extract_pdf_text  # noqa: E402
from contextual_chunker import chunk_contextual_production  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
YOLO_MODEL_PATH = ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"


def normalize(s: str) -> str:
    import re
    return re.sub(r"\s+", "", s)


def measure_extraction_accuracy(yolo_model):
    """[1] whole_page 텍스트 추출 골든셋 recall 재측정 — [2]/[4]/[5]의 모든 정규화(PUA/헤더푸터/
    구두점/기호/NFC)가 누적 적용된 최종 text_extraction.py 기준."""
    results = {}
    for pdf_key, gt_file, pdf_rel in [
        ("LGCNS", "ground_truth_text_lgcns.json", "pdf_pipeline/reference/LGCNS/20260721_company_279243000.pdf"),
        ("Construct", "ground_truth_text_construct.json",
         "pdf_pipeline/reference/Construct/20260721_industry_362851000.pdf"),
    ]:
        gt = json.loads((OUT_DIR / gt_file).read_text(encoding="utf-8"))
        t0 = time.perf_counter()
        result = extract_pdf_text(str(ROOT / pdf_rel), model=yolo_model)
        elapsed = time.perf_counter() - t0
        text_by_page = {p["page"]: p["text"] for p in result["pages"]}
        page_scores = {}
        for page_str, page_gt in gt["pages"].items():
            page = int(page_str)
            norm_extracted = normalize(text_by_page[page])
            units = page_gt["units"]
            matched = sum(1 for u in units if normalize(u) in norm_extracted)
            page_scores[page] = {"matched": matched, "total": len(units)}
        results[pdf_key] = {"elapsed_s": round(elapsed, 3), "pages": page_scores}
    return results


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
            "fact_recall": round(len(fact_hits) / len(gt_doc["key_facts"]), 4),
            "entity_misses": [e["name"] for e in gt_doc["entities"] if e["name"] not in entity_hits],
            "fact_misses": [f["fact"] for f in gt_doc["key_facts"] if f["fact"] not in fact_hits]}


def measure_chunking_recall(yolo_model, qwen_model, qwen_processor, device):
    """chunk_contextual_production(계층적 경계 + rule-based 컨텍스트)의 entity/fact recall +
    지연(청킹 구성 + 추출) 측정 — [3]/[5]와 동일한 3개 문서/골든셋으로 비교 가능하게."""
    gt = json.loads((OUT_DIR / "ground_truth_chunking_eval.json").read_text(encoding="utf-8"))
    doc_titles = {"lgcns_p1": "LG CNS 기업분석 리포트", "construct_p1": "Construct 건설 Weekly 리포트",
                  "construct_p5": "Construct 건설 Weekly 리포트"}

    results = {}
    for doc_key, gt_doc in gt["documents"].items():
        pdf_path = ROOT / "pdf_pipeline" / gt_doc["pdf"]
        page_idx = gt_doc["page"] - 1
        doc_fitz = fitz.open(str(pdf_path))

        t0 = time.perf_counter()
        chunks = chunk_contextual_production(yolo_model, doc_fitz, page_idx, doc_title=doc_titles[doc_key])
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
        results[doc_key] = {
            "n_chunks": len(chunks), "chunk_construction_s": round(construct_s, 3),
            "extraction_total_s": round(extraction_s, 3),
            "total_latency_s": round(construct_s + extraction_s, 3),
            **score,
        }
        print(f"  {doc_key}: {len(chunks)}청크, entity={score['entity_recall']*100:.0f}%, "
              f"fact={score['fact_recall']*100:.0f}%, 지연={results[doc_key]['total_latency_s']}s "
              f"(구성{construct_s:.2f}s+추출{extraction_s:.2f}s)", flush=True)
    return results


def main():
    print("모델 로딩 중...", flush=True)
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    warmup = Image.new("RGB", (595, 842), (255, 255, 255))
    yolo_model.predict(warmup, conf=0.25, verbose=False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    qwen_processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    print(f"로딩 완료 (device={device})", flush=True)

    print("\n=== [1] 텍스트 추출 정확도 재측정 ===", flush=True)
    extraction_result = measure_extraction_accuracy(yolo_model)
    for pdf_key, r in extraction_result.items():
        total_matched = sum(p["matched"] for p in r["pages"].values())
        total_units = sum(p["total"] for p in r["pages"].values())
        print(f"  {pdf_key}: {total_matched}/{total_units} matched ({r['elapsed_s']}s), "
              f"페이지별={r['pages']}", flush=True)

    print("\n=== 청킹 파이프라인(계층적 경계 + rule-based 컨텍스트) recall 재측정 ===", flush=True)
    chunking_result = measure_chunking_recall(yolo_model, qwen_model, qwen_processor, device)

    (OUT_DIR / "result_production_pipeline.json").write_text(
        json.dumps({"extraction_accuracy": extraction_result, "chunking_recall": chunking_result},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[result] saved to {OUT_DIR / 'result_production_pipeline.json'}")


if __name__ == "__main__":
    main()

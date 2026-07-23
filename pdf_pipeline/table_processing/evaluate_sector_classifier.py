"""[11] 섹터 분류 방식 2종(임베딩 vs LLM zero-shot) A/B — golden set(실제 PDF 2건 + 합성 10건)
기준 정확도(Accuracy) + 지연 비교.
"""

import json
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from sector_classifier import embedding_classify, llm_classify, SECTOR_DESCRIPTIONS

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"


def main():
    gt = json.loads((OUT_DIR / "ground_truth_sector_classification.json").read_text(encoding="utf-8"))
    cases = gt["cases"]

    print("BGE-m3-ko 로딩 중...", flush=True)
    embed_model = SentenceTransformer("dragonkue/BGE-m3-ko")
    sector_embs = embed_model.encode(list(SECTOR_DESCRIPTIONS.values()), normalize_embeddings=True)

    print("Qwen2.5-VL 로딩 중...", flush=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    qwen_processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    print(f"로딩 완료 (device={device})", flush=True)

    emb_correct, llm_correct = 0, 0
    emb_time_total, llm_time_total = 0.0, 0.0
    details = []

    for case in cases:
        t0 = time.perf_counter()
        emb_result = embedding_classify(case["text"], embed_model, sector_embs=sector_embs)
        emb_time = time.perf_counter() - t0
        emb_time_total += emb_time

        t0 = time.perf_counter()
        llm_result = llm_classify(case["text"], qwen_model, qwen_processor, device)
        llm_time = time.perf_counter() - t0
        llm_time_total += llm_time

        emb_ok = emb_result["sector"] == case["sector"]
        llm_ok = llm_result["sector"] == case["sector"]
        emb_correct += emb_ok
        llm_correct += llm_ok
        details.append({
            "source": case["source"], "gt_sector": case["sector"],
            "embedding": {"pred": emb_result["sector"], "correct": emb_ok, "confidence": round(emb_result["confidence"], 4)},
            "llm": {"pred": llm_result["sector"], "correct": llm_ok, "raw": llm_result["raw_output"]},
        })
        print(f"  정답={case['sector']:8s} | 임베딩={emb_result['sector']:8s}({'O' if emb_ok else 'X'}) "
              f"| LLM={str(llm_result['sector']):8s}({'O' if llm_ok else 'X'}, raw={llm_result['raw_output'][:20]!r})",
              flush=True)

    n = len(cases)
    result = {
        "n_cases": n,
        "embedding": {"accuracy": round(emb_correct / n, 4), "total_time_s": round(emb_time_total, 3),
                      "avg_time_s": round(emb_time_total / n, 4)},
        "llm": {"accuracy": round(llm_correct / n, 4), "total_time_s": round(llm_time_total, 3),
                "avg_time_s": round(llm_time_total / n, 4)},
        "details": details,
    }
    (OUT_DIR / "result_sector_classification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n임베딩 방식: accuracy={result['embedding']['accuracy']*100:.0f}% "
          f"평균 {result['embedding']['avg_time_s']*1000:.1f}ms/건")
    print(f"LLM 방식: accuracy={result['llm']['accuracy']*100:.0f}% "
          f"평균 {result['llm']['avg_time_s']*1000:.1f}ms/건")
    print(f"[result] saved to {OUT_DIR / 'result_sector_classification.json'}")


if __name__ == "__main__":
    main()

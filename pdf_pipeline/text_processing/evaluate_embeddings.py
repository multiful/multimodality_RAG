"""[9] 임베딩 모델 3종 비교 — BGE-M3(다국어 베이스, 이미 시멘틱청킹/boilerplate탐지에 재사용 중)
vs dragonkue/BGE-m3-ko(커뮤니티 한국어 파인튜닝, HuggingFace 다운로드 49만+로 가장 널리 쓰이는
한국어 BGE-M3 변형) vs OpenAI text-embedding-3-small(API).

평가 방법: [6]/[8]에서 검증된 최종 파이프라인(`chunk_contextual_production`)으로 3개 문서의
실제 청크를 코퍼스로 만들고, Claude가 작성한 자연스러운 한국어 질의 15개(정답 청크를 anchor
부분문자열로 식별)에 대해 코사인 유사도 기반 검색 정확도(Recall@1/@3, MRR)를 측정 — 표준
정보검색(IR) 평가 방식.
"""

import json
import os
import sys
import time
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contextual_chunker import chunk_contextual_production  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"

TEST_DOCS = {
    "lgcns_p1": (ROOT / "pdf_pipeline/reference/LGCNS/20260721_company_279243000.pdf", 0,
                 "LG CNS 기업분석 리포트"),
    "construct_p1": (ROOT / "pdf_pipeline/reference/Construct/20260721_industry_362851000.pdf", 0,
                      "Construct 건설 Weekly 리포트"),
    "construct_p5": (ROOT / "pdf_pipeline/reference/Construct/20260721_industry_362851000.pdf", 4,
                      "Construct 건설 Weekly 리포트"),
}
MIN_CHUNK_CHARS = 15


def build_corpus(yolo_model) -> list:
    corpus = []
    for doc_key, (pdf_path, page_idx, doc_title) in TEST_DOCS.items():
        doc_fitz = fitz.open(str(pdf_path))
        chunks = chunk_contextual_production(yolo_model, doc_fitz, page_idx, doc_title=doc_title)
        doc_fitz.close()
        for c in chunks:
            if len(c["raw_chunk"]) < MIN_CHUNK_CHARS:
                continue
            corpus.append({"doc": doc_key, "text": c["text"], "raw_chunk": c["raw_chunk"]})
    return corpus


def assign_ground_truth(corpus: list, queries: list) -> list:
    """anchor는 실제로 임베딩되는 c["text"](컨텍스트 접두어 + 본문) 기준으로 찾는다 — 일부
    사실(예: 뉴스 헤드라인의 수치)은 rule-based 컨텍스트 주입([6]/[8])으로만 청크에 들어오고
    raw_chunk(본문만)에는 없는 경우가 실측으로 확인됨(Section-header 텍스트는 section_path
    메타데이터로만 쓰이고 본문에는 안 들어감) — c["text"] 기준으로 찾아야 실제 검색 대상과 일치."""
    for q in queries:
        q["correct_indices"] = [i for i, c in enumerate(corpus) if q["anchor"] in c["text"]]
    return queries


def embed_bge(texts: list, model) -> np.ndarray:
    return np.array(model.encode(texts, normalize_embeddings=True))


def embed_openai(texts: list, client, model_name="text-embedding-3-small", batch_size=50) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model_name, input=batch)
        all_embs.extend([d.embedding for d in resp.data])
    embs = np.array(all_embs)
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)


def evaluate(name: str, corpus_embs: np.ndarray, query_embs: np.ndarray, queries: list,
             corpus_s: float, query_s: float) -> dict:
    recall_1, recall_3, mrr_sum, n_scored = 0, 0, 0.0, 0
    details = []
    for qi, q in enumerate(queries):
        correct = set(q["correct_indices"])
        if not correct:
            continue
        n_scored += 1
        sims = corpus_embs @ query_embs[qi]
        ranking = np.argsort(-sims)
        rank = next((r + 1 for r, idx in enumerate(ranking) if idx in correct), None)
        if rank == 1:
            recall_1 += 1
        if rank and rank <= 3:
            recall_3 += 1
        if rank:
            mrr_sum += 1.0 / rank
        details.append({"query": q["query"], "rank": rank})
    return {
        "model": name,
        "recall_at_1": round(recall_1 / n_scored, 4) if n_scored else 0.0,
        "recall_at_3": round(recall_3 / n_scored, 4) if n_scored else 0.0,
        "mrr": round(mrr_sum / n_scored, 4) if n_scored else 0.0,
        "n_queries_scored": n_scored,
        "corpus_embed_s": round(corpus_s, 3), "query_embed_s": round(query_s, 3),
        "details": details,
    }


def main():
    print("YOLO 로딩 + 코퍼스 구성 중...", flush=True)
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    warmup = Image.new("RGB", (595, 842), (255, 255, 255))
    yolo_model.predict(warmup, conf=0.25, verbose=False)
    corpus = build_corpus(yolo_model)
    print(f"코퍼스: {len(corpus)}개 청크(3문서)", flush=True)

    gt = json.loads((OUT_DIR / "ground_truth_embedding_eval.json").read_text(encoding="utf-8"))
    queries = assign_ground_truth(corpus, gt["queries"])
    unmatched = [q["query"] for q in queries if not q["correct_indices"]]
    if unmatched:
        print(f"경고: anchor가 코퍼스에서 안 잡힌 질의 {len(unmatched)}개: {unmatched}", flush=True)

    corpus_texts = [c["text"] for c in corpus]
    query_texts = [q["query"] for q in queries]

    results = []

    print("\n=== BGE-M3(다국어 베이스) ===", flush=True)
    from sentence_transformers import SentenceTransformer
    bge_m3 = SentenceTransformer("BAAI/bge-m3")
    t0 = time.perf_counter(); corpus_embs = embed_bge(corpus_texts, bge_m3); corpus_s = time.perf_counter() - t0
    t0 = time.perf_counter(); query_embs = embed_bge(query_texts, bge_m3); query_s = time.perf_counter() - t0
    r = evaluate("BGE-M3", corpus_embs, query_embs, queries, corpus_s, query_s)
    results.append(r)
    print(f"  recall@1={r['recall_at_1']*100:.0f}% recall@3={r['recall_at_3']*100:.0f}% "
          f"mrr={r['mrr']:.3f} (코퍼스 임베딩 {corpus_s:.2f}s)", flush=True)
    del bge_m3

    print("\n=== dragonkue/BGE-m3-ko(한국어 파인튜닝) ===", flush=True)
    bge_m3_ko = SentenceTransformer("dragonkue/BGE-m3-ko")
    t0 = time.perf_counter(); corpus_embs = embed_bge(corpus_texts, bge_m3_ko); corpus_s = time.perf_counter() - t0
    t0 = time.perf_counter(); query_embs = embed_bge(query_texts, bge_m3_ko); query_s = time.perf_counter() - t0
    r = evaluate("BGE-m3-ko", corpus_embs, query_embs, queries, corpus_s, query_s)
    results.append(r)
    print(f"  recall@1={r['recall_at_1']*100:.0f}% recall@3={r['recall_at_3']*100:.0f}% "
          f"mrr={r['mrr']:.3f} (코퍼스 임베딩 {corpus_s:.2f}s)", flush=True)
    del bge_m3_ko

    if os.environ.get("OPENAI_API_KEY"):
        print("\n=== OpenAI text-embedding-3-small ===", flush=True)
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        t0 = time.perf_counter(); corpus_embs = embed_openai(corpus_texts, client); corpus_s = time.perf_counter() - t0
        t0 = time.perf_counter(); query_embs = embed_openai(query_texts, client); query_s = time.perf_counter() - t0
        r = evaluate("OpenAI-text-embedding-3-small", corpus_embs, query_embs, queries, corpus_s, query_s)
        results.append(r)
        print(f"  recall@1={r['recall_at_1']*100:.0f}% recall@3={r['recall_at_3']*100:.0f}% "
              f"mrr={r['mrr']:.3f} (코퍼스 임베딩 {corpus_s:.2f}s)", flush=True)
    else:
        print("\nOPENAI_API_KEY 없어 OpenAI 임베딩 스킵", flush=True)

    (OUT_DIR / "result_embedding_eval.json").write_text(
        json.dumps({"n_corpus_chunks": len(corpus), "n_queries": len(queries), "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[result] saved to {OUT_DIR / 'result_embedding_eval.json'}")


if __name__ == "__main__":
    main()

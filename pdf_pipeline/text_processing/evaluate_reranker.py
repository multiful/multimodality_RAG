"""[50] 사용자 질문("리랭커 도입 안 해도 돼? cohere 넣을까") 검증 — Cohere Rerank API는 이 환경에
키가 없어 직접 못 쓰지만, 같은 원리(cross-encoder로 query-document 쌍을 함께 인코딩해 재정렬,
bi-encoder 기반 dense 검색보다 정밀하지만 느림)의 로컬 모델 BAAI/bge-reranker-v2-m3(다국어,
BGE 계열이라 이미 쓰는 dragonkue/BGE-m3-ko와 궁합 좋음)로 "reranking이 실제로 이득이 있는지"를
먼저 실측 검증한다 — 있으면 Cohere(유료 API) 도입 근거가 되고, 없으면 오버엔지니어링 방지.

방법: route_search()의 top-10 후보를 cross-encoder로 재정렬해서 원래 순위 대비
NDCG/MAP/MRR/Recall/Precision/F1이 개선되는지, 그리고 재정렬 자체의 지연은 얼마인지 측정."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "page_classification"))

from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=str(ROOT / ".env"))

from openai import OpenAI
from index_text import TextIndex, route_search, precompute_entity_count, _tokenize
from rank_bm25 import BM25Okapi
from evaluate_retrieval_ab import resolve_gold_indices, score_ranking, GOLD_PATH, CHUNK_CACHE_DIR

client = OpenAI()
gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
query = gold["query"]
TOP_K = 10

print("cross-encoder 로딩 중(BAAI/bge-reranker-v2-m3)...")
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
print("로딩 완료\n")

for doc_id, doc_info in gold["docs"].items():
    chunks = json.loads((CHUNK_CACHE_DIR / f"{doc_id}_chunks.json").read_text(encoding="utf-8"))
    from embedding import embed_texts
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    chunk_ids = [f"{doc_id}_p{c['page']}_{i}" for i, c in enumerate(chunks)]
    index = TextIndex(pdf_id=doc_id, chunk_ids=chunk_ids, chunks=chunks, embeddings=embeddings, bm25=bm25)
    precompute_entity_count(index, client=client)  # pdf_path 없어 LLM 폴백일 수 있음(참고용 문서라 무방)
    gold_idx = resolve_gold_indices(index, doc_info["gold_anchors"])
    id_to_idx = {cid: i for i, cid in enumerate(index.chunk_ids)}

    t0 = time.perf_counter()
    hits, qtype = route_search(index, query, client=client, top_k=TOP_K)
    t_route = time.perf_counter() - t0
    original_order = [h["chunk_id"] for h in hits]
    m_before = score_ranking(original_order, id_to_idx, gold_idx)

    t0 = time.perf_counter()
    pairs = [(query, h["chunk"]["text"]) for h in hits]
    rerank_scores = reranker.predict(pairs)
    reranked = [hits[i]["chunk_id"] for i in (-rerank_scores).argsort()]
    t_rerank = time.perf_counter() - t0
    m_after = score_ranking(reranked, id_to_idx, gold_idx)

    print(f"=== {doc_id} (entity_count={index.entity_count}, qtype={qtype}) ===")
    print(f"  기존 순위(route_search만, {t_route:.2f}s)      : ndcg={m_before['ndcg@k']:.3f} "
          f"map={m_before['map@k']:.3f} mrr={m_before['mrr']:.3f} recall={m_before['recall@k']:.3f} "
          f"({m_before['n_gold_found_in_top_k']}/{m_before['n_gold_total']})")
    print(f"  리랭킹 후(+cross-encoder, +{t_rerank:.2f}s)     : ndcg={m_after['ndcg@k']:.3f} "
          f"map={m_after['map@k']:.3f} mrr={m_after['mrr']:.3f} recall={m_after['recall@k']:.3f} "
          f"({m_after['n_gold_found_in_top_k']}/{m_after['n_gold_total']})")
    print()

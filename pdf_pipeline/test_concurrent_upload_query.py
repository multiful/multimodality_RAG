"""[51] 사용자 질문("업로드와 쿼리가 같이 주어지는데.. 9.1초 기다리고 쿼리를 받는게 좋나?") 검증 —
질의 분류(+MQE 하위질의 생성)는 질의 텍스트만 있으면 되고 문서/인덱스가 전혀 필요 없다(순수 함수:
_classify_and_expand(query, client)). 엔티티 카운트도 하나증권 포맷이면 원본 PDF 텍스트만 있으면
되고(정규식, YOLO/청킹/임베딩 불필요) 인덱싱과 무관하게 즉시 계산 가능하다. 즉 "인덱싱 9.1초"와
"질의 분류+하위질의 생성(~1.5~2.5s)"는 서로 의존관계가 없으므로 동시에 돌리면 그 시간만큼
벽시계 지연을 숨길 수 있다 — 순차 실행이 이 부분을 낭비하고 있었는지 실측으로 확인한다.

다기업(C밴드, entity_count=11 -> MQE)과 단일기업(LGCNS, entity_count=1 -> HyDE) 둘 다 테스트 —
MQE 경로는 분류 호출 자체가 이미 하위질의까지 만들어서 오버랩으로 전부 커버되지만, HyDE 경로는
분류 호출만으론 가상 문단 생성까지 안 끝나서(엔티티 수를 알아야 HyDE로 갈지 결정되므로) 오버랩
이득이 부분적이라는 것까지 함께 확인한다."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pdf_pipeline"))
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "page_classification"))
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "text_processing"))

from page_classifier import classify_pdf  # noqa: E402
from text_extraction import process_pdf_streaming  # noqa: E402
from index_text import (TextIndex, _classify_and_expand, _fuse_multi_query, hyde_search,
                         count_document_entities_hana, hybrid_search, _tokenize)  # noqa: E402
import entity_fusion  # noqa: E402
from rank_bm25 import BM25Okapi

load_dotenv(ROOT / ".env")
QUERY = "이 PDF에 나오는 기업에 대한 투자 인사이트를 도출해줘"


def build_index_fast(pdf_path, yolo_model, doc_title, page_boxes):
    pages_chunks = [page["chunks"] for page in
                    process_pdf_streaming(pdf_path, yolo_model, doc_title=doc_title, page_boxes=page_boxes)]
    flat = [c for pc in pages_chunks for c in pc]
    from embedding import embed_texts
    texts = [c["text"] for c in flat]
    embeddings = embed_texts(texts)
    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    chunk_ids = [f"{doc_title}_p{c['page']}_{i}" for i, c in enumerate(flat)]
    return TextIndex(pdf_id=doc_title, chunk_ids=chunk_ids, chunks=flat, embeddings=embeddings, bm25=bm25)


def run_sequential(pdf_path, yolo_model, doc_title, page_boxes, client):
    t0 = time.perf_counter()
    index = build_index_fast(pdf_path, yolo_model, doc_title, page_boxes)
    t_indexed = time.perf_counter() - t0
    qtype, sub_queries = _classify_and_expand(QUERY, client=client)
    t_classified = time.perf_counter() - t0
    full_text = "\n".join(page.get_text() for page in __import__("fitz").open(str(pdf_path)))
    entity_count = count_document_entities_hana(full_text) or 1
    if qtype == "abstract" and entity_count <= 1:
        hits, _ = hyde_search(index, QUERY, client=client, top_k=8, use_bm25=False)
    elif qtype == "abstract":
        hits = _fuse_multi_query(index, [QUERY] + sub_queries, top_k=8, use_bm25=False)
    else:
        hits = hybrid_search(index, QUERY, top_k=8, fusion="rrf")
    t_total = time.perf_counter() - t0
    return {"indexed": t_indexed, "classified": t_classified, "total": t_total,
            "qtype": qtype, "entity_count": entity_count}


def run_concurrent(pdf_path, yolo_model, doc_title, page_boxes, client):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_index = ex.submit(build_index_fast, pdf_path, yolo_model, doc_title, page_boxes)
        f_classify = ex.submit(_classify_and_expand, QUERY, client)
        f_entity = ex.submit(lambda: count_document_entities_hana(
            "\n".join(page.get_text() for page in __import__("fitz").open(str(pdf_path)))) or 1)
        index = f_index.result()
        t_indexed = time.perf_counter() - t0
        qtype, sub_queries = f_classify.result()
        t_classified = time.perf_counter() - t0
        entity_count = f_entity.result()

    if qtype == "abstract" and entity_count <= 1:
        hits, _ = hyde_search(index, QUERY, client=client, top_k=8, use_bm25=False)
    elif qtype == "abstract":
        hits = _fuse_multi_query(index, [QUERY] + sub_queries, top_k=8, use_bm25=False)
    else:
        hits = hybrid_search(index, QUERY, top_k=8, fusion="rrf")
    t_total = time.perf_counter() - t0
    return {"indexed": t_indexed, "classified": t_classified, "total": t_total,
            "qtype": qtype, "entity_count": entity_count}


def main():
    from openai import OpenAI
    client = OpenAI()
    yolo = YOLO(str(ROOT / "pdf_pipeline/page_classification/models/yolo11n_doc_layout.pt"))
    yolo.predict(Image.new("RGB", (595, 842), (255, 255, 255)), conf=0.25, verbose=False)

    # [51] 측정 오염 방지 — embedding 모델 싱글턴 콜드로드(~6.8s)를 두 실행이 공평하게 겪도록
    # 미리 한 번 워밍업(안 그러면 순차 실행에서 워밍업 비용을 떠안고, 같은 프로세스 안의 동시
    # 실행이 그 혜택만 거저 가져가서 비교가 왜곡됨 — 처음 이 버그로 동시 실행이 비정상적으로
    # 빠르게 나온 걸 실측 중 발견).
    print("임베딩 모델 워밍업...")
    from embedding import get_embedding_model
    get_embedding_model()

    for name, path in [
        ("C밴드(다기업->MQE)", ROOT / "pdf_pipeline/reference/C밴드/c밴드.pdf"),
        ("LGCNS(단일기업->HyDE)", ROOT / "pdf_pipeline/reference/LGCNS/20260721_company_279243000.pdf"),
    ]:
        cls_result = classify_pdf(path, yolo)
        page_boxes = {p["page"]: p["cached_boxes"] for p in cls_result["pages"]}

        print(f"=== {name} ===")
        r_seq = run_sequential(path, yolo, f"{name}_seq", page_boxes, client)
        print(f"  순차: 인덱싱={r_seq['indexed']:.2f}s 분류후={r_seq['classified']:.2f}s "
              f"총={r_seq['total']:.2f}s (qtype={r_seq['qtype']}, entity_count={r_seq['entity_count']})")

        r_con = run_concurrent(path, yolo, f"{name}_con", page_boxes, client)
        print(f"  동시: 인덱싱완료시점={r_con['indexed']:.2f}s(그안에 분류 이미 끝남) "
              f"총={r_con['total']:.2f}s (qtype={r_con['qtype']}, entity_count={r_con['entity_count']})")
        print(f"  절감: {r_seq['total']-r_con['total']:.2f}s\n")


if __name__ == "__main__":
    main()

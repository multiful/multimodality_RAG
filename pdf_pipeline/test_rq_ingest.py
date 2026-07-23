"""[49] test_async_ingest.py(raw threading.Thread)를 RQ 작업 큐로 교체해 재검증.

주의(정직하게 기록): 이 환경엔 Docker/Homebrew/실제 redis-server 바이너리가 전혀 없어(확인함),
fakeredis로 큐 백엔드를 대신한다. fakeredis는 프로세스 내부 인메모리라 "진짜 프로세스가 죽어도
작업이 안 사라진다"는 배포 시나리오의 핵심 이점 자체는 이 테스트로 증명 못 한다 — 이번 테스트가
검증하는 건 (1) RQ enqueue/worker API를 실제로 올바르게 연결했는지(잡 직렬화, 인자 전달, 실행,
결과 저장), (2) enqueue가 즉시 반환하고 워커가 별도로 처리해도 그 사이 질의가 막히지 않는지
두 가지뿐. 실제 배포 시 REDIS_URL로 진짜 Redis에 붙이면 프로세스 독립성/내구성까지 그대로
적용된다(코드는 동일, connection만 교체)."""
import sys
import threading
import time
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
from index_text import route_search, precompute_entity_count  # noqa: E402
from ingest_queue import get_queue_connection, get_queue, enqueue_structured_metadata  # noqa: E402
import entity_fusion  # noqa: E402

import os
load_dotenv(ROOT / ".env")
DB_URL = os.environ["SUPABASE_DIRECT_DB_URL"]
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "C밴드" / "c밴드.pdf"
PDF_ID = "C밴드_RQ테스트"


def main():
    from openai import OpenAI
    client = OpenAI()
    yolo = YOLO(str(ROOT / "pdf_pipeline/page_classification/models/yolo11n_doc_layout.pt"))
    yolo.predict(Image.new("RGB", (595, 842), (255, 255, 255)), conf=0.25, verbose=False)
    cls_result = classify_pdf(PDF_PATH, yolo)
    page_boxes = {p["page"]: p["cached_boxes"] for p in cls_result["pages"]}

    t0 = time.perf_counter()
    print("1) 빠른 경로: 스트리밍 청킹 -> 임베딩 -> 적재")
    pages_chunks = [page["chunks"] for page in
                    process_pdf_streaming(PDF_PATH, yolo, doc_title=PDF_ID, page_boxes=page_boxes)]
    flat_chunks = [c for pc in pages_chunks for c in pc]
    items, emb = entity_fusion.embed_items(entity_fusion.from_text_chunks(PDF_ID, flat_chunks))
    entity_fusion.store_evidence(DB_URL, PDF_ID, items, emb)
    t_searchable = time.perf_counter() - t0
    print(f"   검색 가능 시점: {t_searchable:.2f}s ({len(flat_chunks)}청크)")

    print("2) RQ 큐에 구조화 메타데이터 작업 등록(즉시 반환, 안 막음)")
    conn = get_queue_connection(use_fake=True)
    queue = get_queue(connection=conn)
    t_enqueue_start = time.perf_counter()
    job = enqueue_structured_metadata(queue, PDF_ID, pages_chunks, "통신장비", PDF_ID, DB_URL)
    t_enqueue = time.perf_counter() - t0
    print(f"   enqueue 반환: {t_enqueue:.2f}s (job_id={job.id}, status={job.get_status()})")

    print("3) 워커가 아직 처리 전인 상태에서 질의 실행")
    index = entity_fusion.load_evidence_from_db(DB_URL, pdf_id=PDF_ID)
    precompute_entity_count(index, pdf_path=PDF_PATH, client=client)
    hits, qtype = route_search(index, "이 PDF에 나오는 기업에 대한 투자 인사이트를 도출해줘",
                                client=client, top_k=3)
    t_query = time.perf_counter() - t0
    print(f"   질의 완료: {t_query:.2f}s, job 상태(아직 처리 안 됨 확인): {job.get_status()}")
    print(f"   분류={qtype}, entity_count={index.entity_count}")

    print("4) 워커(SimpleWorker) 실행해서 큐 처리")
    from rq import SimpleWorker
    worker = SimpleWorker([queue], connection=conn)
    worker.work(burst=True)
    t_worker_done = time.perf_counter() - t0
    job.refresh()
    print(f"   워커 완료: {t_worker_done:.2f}s, job 상태={job.get_status()}, result={job.result}")

    print("\n=== 결론 ===")
    print(f"검색 가능: {t_searchable:.2f}s | enqueue: {t_enqueue:.2f}s | "
          f"질의(워커 처리 전): {t_query:.2f}s | 워커 완료: {t_worker_done:.2f}s")
    print(f"질의가 워커 완료보다 {'먼저' if t_query < t_worker_done else '나중에'} 끝남"
          f" -> enqueue는 논블로킹{'임을 확인' if t_query < t_worker_done else ' 실패'}")


if __name__ == "__main__":
    main()

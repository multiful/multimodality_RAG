"""[49] 사용자 지적("병렬로 진행되는거지? 동적할당? 문제 없는거지?") 검증 — process_pdf_streaming()
의 설계 의도(구조화 출력은 느린 백그라운드 잡으로 분리)가 실제로 "질의 가능 시점을 안 막는다"는
게 맞는지 진짜로 테스트한다(이전엔 docstring의 의도를 확인 없이 그대로 보고했음 — 검증 안 된
주장이었음).

시나리오: (1) 빠른 경로로 청킹+임베딩+적재(구조화 출력 없음) -> 이 시점부터 "검색 가능"으로 본다.
(2) 그 직후 백그라운드 스레드에서 구조화 출력을 돌리기 시작한다(블로킹 안 함). (3) 백그라운드가
아직 끝나기 전에 실제 질의를 던져서 응답이 오는지, 그 응답이 (2)의 완료를 기다리지 않는지 시간을
직접 재서 확인한다. (4) 백그라운드가 끝나면 같은 id로 재적재(upsert)해서 구조화 메타데이터가
나중에 채워지는지 확인한다."""
import os
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
from structured_output import extract_text_chunk_metadata  # noqa: E402
from index_text import route_search, precompute_entity_count  # noqa: E402
import entity_fusion  # noqa: E402

load_dotenv(ROOT / ".env")
DB_URL = os.environ["SUPABASE_DIRECT_DB_URL"]
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "C밴드" / "c밴드.pdf"
PDF_ID = "C밴드_비동기테스트"

background_done = threading.Event()
background_finish_time = [None]


def background_metadata_job(chunks, client, t0):
    """[49] 진짜 백그라운드 스레드 — 메인 흐름을 블로킹하지 않고 구조화 출력만 계산."""
    for page_chunks in chunks:
        if page_chunks:
            metas = extract_text_chunk_metadata(page_chunks, doc_title=PDF_ID, client=client, sector="통신장비")
            for c, m in zip(page_chunks, metas):
                c["structured_metadata"] = m
    background_finish_time[0] = time.perf_counter() - t0
    background_done.set()
    # 백그라운드 완료 후 같은 id로 재적재(upsert) — 구조화 메타데이터가 나중에 채워짐
    flat = [c for page_chunks in chunks for c in page_chunks]
    items, emb = entity_fusion.embed_items(entity_fusion.from_text_chunks(PDF_ID, flat))
    entity_fusion.store_evidence(DB_URL, PDF_ID, items, emb)
    print(f"   [백그라운드] 구조화 메타데이터 완료 + 재적재 (t={background_finish_time[0]:.2f}s)")


def main():
    from openai import OpenAI
    client = OpenAI()
    yolo = YOLO(str(ROOT / "pdf_pipeline/page_classification/models/yolo11n_doc_layout.pt"))
    yolo.predict(Image.new("RGB", (595, 842), (255, 255, 255)), conf=0.25, verbose=False)
    cls_result = classify_pdf(PDF_PATH, yolo)
    page_boxes = {p["page"]: p["cached_boxes"] for p in cls_result["pages"]}

    t0 = time.perf_counter()
    print("1) 빠른 경로: process_pdf_streaming (구조화 출력 없음)")
    pages_chunks = []
    for page in process_pdf_streaming(PDF_PATH, yolo, doc_title=PDF_ID, page_boxes=page_boxes):
        pages_chunks.append(page["chunks"])
    t_stream = time.perf_counter() - t0
    flat_chunks = [c for pc in pages_chunks for c in pc]
    print(f"   스트리밍 완료: {t_stream:.2f}s, 청크 {len(flat_chunks)}개")

    items, emb = entity_fusion.embed_items(entity_fusion.from_text_chunks(PDF_ID, flat_chunks))
    n = entity_fusion.store_evidence(DB_URL, PDF_ID, items, emb)
    t_searchable = time.perf_counter() - t0
    print(f"   적재 완료 -> 검색 가능 시점: {t_searchable:.2f}s ({n}건)")

    print("2) 백그라운드 스레드에서 구조화 출력 시작 (블로킹 안 함)")
    bg_thread = threading.Thread(target=background_metadata_job, args=(pages_chunks, client, t0), daemon=True)
    bg_thread.start()

    print("3) 백그라운드가 끝나기 전에 실제 질의 던지기")
    t_query_start = time.perf_counter() - t0
    index = entity_fusion.load_evidence_from_db(DB_URL, pdf_id=PDF_ID)
    precompute_entity_count(index, pdf_path=PDF_PATH, client=client)
    hits, qtype = route_search(index, "이 PDF에 나오는 기업에 대한 투자 인사이트를 도출해줘",
                                client=client, top_k=5)
    t_query_done = time.perf_counter() - t0
    still_running = not background_done.is_set()
    print(f"   질의 시작={t_query_start:.2f}s, 완료={t_query_done:.2f}s "
          f"(이 시점 백그라운드 아직 진행 중: {still_running})")
    print(f"   분류={qtype}, entity_count={index.entity_count}, 상위 결과 {len(hits)}건")
    for h in hits[:2]:
        has_meta = bool(h["chunk"].get("metadata", {}).get("structured_metadata"))
        print(f"     구조화메타 있음={has_meta}: {h['chunk']['content'][:50]!r}")

    bg_thread.join()
    print(f"\n결론: 질의 응답이 백그라운드 완료({background_finish_time[0]:.2f}s)보다 "
          f"{'먼저' if t_query_done < background_finish_time[0] else '나중에'} 끝남 "
          f"(질의={t_query_done:.2f}s vs 백그라운드완료={background_finish_time[0]:.2f}s)")


if __name__ == "__main__":
    main()

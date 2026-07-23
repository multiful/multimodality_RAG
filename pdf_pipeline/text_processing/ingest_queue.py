"""[49] 사용자 요청("작업 큐 도입해야지... 배포가 가능하면서 속도는 빨라야해") — 구조화 메타데이터
계산을 raw `threading.Thread`(test_async_ingest.py에서 검증한 프로토타입)에서 RQ(Redis Queue)
기반 작업 큐로 옮긴다.

작업 큐를 넣는 이유는 "한 문서 처리 속도"가 빨라지기 때문이 아니다(LLM API 호출 지연이 바닥이라
그대로) — (1) 요청을 처리한 프로세스가 죽거나 재시작돼도 작업이 안 사라짐(threading.Thread는
그 프로세스가 끝나면 같이 죽음), (2) 여러 사용자가 동시에 PDF를 올려도 워커 프로세스 수만큼만
동시 실행돼 GIL/리소스 경합 없이 안정적(무제한으로 스레드가 늘어나지 않음 — 이게 "부하가 걸려도
느려지지 않는다"는 의미의 속도), (3) 워커를 API 서버와 별도 프로세스/컨테이너로 배포 가능
(`rq worker` 명령으로 몇 대든 띄울 수 있음).

RQ를 Celery보다 채택한 이유: 이 프로젝트가 필요한 건 "인제스트 후 구조화 메타데이터를 비동기로
계산해 upsert" 하나뿐이라(복잡한 워크플로우/주기 작업/체이닝 불필요), Celery의 다중 기능이
오버엔지니어링에 가깝다고 판단 — RQ는 큐 하나, 워커 하나로 이 요구를 그대로 충족.
metadata_cache.py가 이미 확립한 "fakeredis로 로컬 검증, use_fake=False로 실제 Redis 전환"
관례를 그대로 따름.
"""

import os


def get_queue_connection(use_fake: bool = True):
    """[49] metadata_cache.get_client()와 동일한 관례. use_fake=True(기본)는 fakeredis(로컬
    검증용 — 실제 크로스 프로세스 내구성은 검증 못 함, 아래 module docstring/실험.md [49] 참고).
    use_fake=False면 REDIS_URL 환경변수로 실제 Redis에 연결(프로덕션)."""
    if use_fake:
        import fakeredis
        return fakeredis.FakeStrictRedis()
    import redis
    return redis.Redis.from_url(os.environ["REDIS_URL"])


def get_queue(connection=None, use_fake: bool = True):
    from rq import Queue
    return Queue(connection=connection or get_queue_connection(use_fake))


def run_structured_metadata_job(pdf_id: str, pages_chunks: list, sector: str,
                                 doc_title: str, db_url: str) -> dict:
    """[49] RQ 워커가 실제로 실행하는 작업 — 직렬화를 위해 client/모델 객체를 인자로 안 받고
    함수 내부에서 새로 만든다(RQ 잡은 pickle로 큐에 저장되므로 인자는 단순 직렬화 가능 타입만:
    str/list/dict). pages_chunks: process_pdf_streaming()이 페이지별로 yield한 chunks를 모은
    리스트의 리스트(빠른 경로에서 이미 임베딩+적재까지 끝난 청크들 — 여기서는 구조화 메타데이터만
    추가로 계산해 같은 id로 upsert).

    반환: {"pdf_id", "n_pages_processed", "n_chunks_with_metadata"} — RQ가 job.result로 보존해
    호출측이 완료 여부/결과를 나중에 조회할 수 있음(job.fetch(job_id).result)."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "pdf_pipeline"))
    sys.path.insert(0, str(ROOT / "pdf_pipeline" / "text_processing"))

    from openai import OpenAI
    from structured_output import extract_text_chunk_metadata
    import entity_fusion

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    n_with_meta = 0
    for page_chunks in pages_chunks:
        if not page_chunks:
            continue
        metas = extract_text_chunk_metadata(page_chunks, doc_title=doc_title, client=client, sector=sector)
        for c, m in zip(page_chunks, metas):
            c["structured_metadata"] = m
            if m:
                n_with_meta += 1

    flat = [c for page_chunks in pages_chunks for c in page_chunks]
    items, emb = entity_fusion.embed_items(entity_fusion.from_text_chunks(pdf_id, flat))
    entity_fusion.store_evidence(db_url, pdf_id, items, emb)

    return {"pdf_id": pdf_id, "n_pages_processed": len(pages_chunks), "n_chunks_with_metadata": n_with_meta}


def enqueue_structured_metadata(queue, pdf_id: str, pages_chunks: list, sector: str,
                                 doc_title: str, db_url: str):
    """[49] 빠른 경로(스트리밍 인입) 직후 호출 — 큐에 작업만 등록하고 즉시 반환(안 막음). 실제
    실행은 별도 `rq worker`(또는 테스트에서는 SimpleWorker)가 담당."""
    return queue.enqueue(run_structured_metadata_job, pdf_id, pages_chunks, sector, doc_title, db_url)

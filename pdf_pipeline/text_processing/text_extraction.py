"""[2]/[4]/[8] 텍스트 파이프라인 프로덕션 진입점 — [1]에서 검증된 whole_page(PyMuPDF `get_text()`)
베이스라인에 PUA 글리프 제거 + 난이도 라우팅(reading_order_router)을 결합. [4]에서 헤더/푸터
제거 + 구두점 정규화를 추가(항상 적용, 가벼움) — boilerplate 제거(임베딩 모델 로드가 필요해
상대적으로 무거움)는 옵션으로 분리해 호출자가 필요할 때만 켜도록 함.

[8] `process_pdf()` 추가 — 사용자 지적("YOLO 두 번 쓰면 지연 늘잖아") 반영: 이전에는 난이도
라우팅(`assess_pdf`)과 계층적 청킹(`chunk_hierarchical`, contextual_chunker 경유)이 페이지당
YOLO를 각자 따로 호출했음. `process_pdf()`는 페이지당 YOLO를 **한 번만** 호출(`run_yolo_layout`)
하고 그 결과를 난이도 판정과 청킹 양쪽에 그대로 재사용한다 — [5]에서 검증만 해뒀던 7.6배 개선을
실제로 배선.
"""

import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # [10] yolo_layout이 pdf_pipeline/로 이동
from reading_order_router import assess_pdf, assess_page_difficulty  # noqa: E402
from text_cleanup import (detect_pua_artifact, strip_pua_artifacts,  # noqa: E402 — [36] pdf_pipeline/로 이동(공유)
                            normalize_punctuation, normalize_symbols_and_whitespace)
from header_footer_remover import detect_headers_footers, strip_headers_footers  # noqa: E402
from yolo_layout import run_yolo_layout  # noqa: E402
from boilerplate_remover import detect_boilerplate_paragraphs_fast  # noqa: E402
from structured_output import extract_text_chunk_metadata  # noqa: E402


def extract_page_text(doc_fitz: fitz.Document, page_idx: int, difficulty: str = "easy",
                       header_footer_templates: dict = None) -> dict:
    """단일 페이지 텍스트 추출: whole_page 베이스라인([1] 검증, recall 100%/9.5ms) + PUA 제거 +
    헤더/푸터 제거(header_footer_templates가 주어지면) + 구두점/기호/공백 정규화. difficulty="hard"
    인 페이지는 우리 파이프라인 결과를 참고용으로만 채우고 `needs_external_reader=True`로 표시 —
    실제 MinerU 호출은 이 함수 범위 밖(별도 연동 필요)."""
    raw = doc_fitz[page_idx].get_text()
    had_pua = detect_pua_artifact(raw)
    cleaned = strip_pua_artifacts(raw) if had_pua else raw
    if header_footer_templates:
        cleaned = strip_headers_footers(cleaned, header_footer_templates)
    cleaned = normalize_punctuation(cleaned)
    cleaned = normalize_symbols_and_whitespace(cleaned)
    return {
        "page": page_idx + 1,
        "text": cleaned,
        "had_pua_artifact": had_pua,
        "difficulty": difficulty,
        "needs_external_reader": difficulty == "hard",
    }


def extract_pdf_text(pdf_path, model=None, thresholds=None, strip_headers=True) -> dict:
    """PDF 전체: reading_order_router로 페이지별 난이도부터 판정한 뒤, 페이지별로 텍스트 추출.
    hard 페이지가 있으면 결과에 `route_to_mineru=True`와 해당 페이지 번호를 같이 반환해,
    호출자가 그 페이지만 MinerU 등 외부 리더로 넘기도록 유도(문서 전체를 넘기지 않음 — 표/차트
    처리는 그대로 우리 파이프라인이 담당하므로 페이지 단위 라우팅이 더 효율적, table_processing의
    Adaptive Router와 같은 설계 철학). strip_headers=True면 좌표+반복빈도 기반 헤더/푸터 탐지를
    PDF 전체에 대해 한 번 실행해 각 페이지 텍스트에서 제거([4]).

    boilerplate(Compliance Notice 등) 제거는 여기 포함하지 않음 — 임베딩 모델(BGE-M3) 로드가
    필요해 상대적으로 무거워서, 필요한 호출자가 `boilerplate_remover.strip_boilerplate()`를
    문단 단위로 별도 호출하도록 분리(청킹 이후 문단 단위에서 적용하는 게 더 적합 — 페이지
    전체를 통째로 boilerplate 판정하면 안 되고, 문단 단위로 판정해야 함)."""
    router_result = assess_pdf(pdf_path, model=model, thresholds=thresholds)
    difficulty_by_page = {p["page"]: p["difficulty"] for p in router_result["pages"]}

    doc_fitz = fitz.open(str(pdf_path))
    header_footer_templates = detect_headers_footers(doc_fitz) if strip_headers else None
    pages = [
        extract_page_text(doc_fitz, i, difficulty=difficulty_by_page.get(i + 1, "easy"),
                          header_footer_templates=header_footer_templates)
        for i in range(doc_fitz.page_count)
    ]
    doc_fitz.close()

    n_pua = sum(1 for p in pages if p["had_pua_artifact"])
    return {
        "pdf": str(pdf_path), "n_pages": len(pages),
        "route_to_mineru": router_result["route_to_mineru"],
        "hard_page_numbers": router_result["hard_page_numbers"],
        "n_pages_with_pua_artifact": n_pua,
        "headers_footers_detected": header_footer_templates or {},
        "pages": pages,
    }


def process_pdf(pdf_path, model, doc_title: str = None, strip_headers: bool = True,
                 chunk_backend: str = "rulebased", remove_boilerplate: bool = True,
                 embed_model=None, page_boxes: dict = None,
                 add_structured_metadata: bool = False, openai_client=None,
                 structured_metadata_workers: int = 8, sector: str = None) -> dict:
    """[8]/[10] 최종 통합 진입점 — 페이지당 YOLO를 한 번만 호출해 난이도 판정과 계층적 청킹(+컨텍스트
    주입) 양쪽에 재사용. easy 페이지만 청킹까지 진행하고, hard 페이지는 텍스트만 채우고
    `needs_external_reader=True`로 표시(청킹은 생략 — MinerU 등 외부 리더가 처리할 몫).

    page_boxes: [10] page_classification 단계가 이미 같은 PDF에 대해 YOLO를 돌린 결과가 있으면
    `{page_number(1-based): [(cls_name, fitz.Rect), ...]}` 형태로 여기에 넘겨서 파이프라인 전체
    (page_classification + text_processing) 기준으로도 페이지당 YOLO를 1번만 쓰게 만들 수 있음
    — `page_classification.page_classifier.classify_pdf()`가 반환하는 `cached_boxes`를 그대로
    모아서 전달하면 됨(사용자 지적 반영: [8]은 text_processing 내부 중복만 없앴고, 이 앞단
    page_classification과의 중복은 남아있었음). 안 주면 이 함수가 페이지당 새로 1번만 호출.

    remove_boilerplate=True(기본값)면 각 청크의 raw_chunk를 Sentence Embedding 기반 boilerplate
    탐지([4]/[5])에 통과시켜, Compliance Notice류로 판정된 청크는 `is_boilerplate=True`로 표시하고
    최종 chunks 리스트에서 제외한다 — [10] 사용자 지적 반영: 이전엔 이 단계가 `process_pdf()`에
    안 묶여 있어서 호출자가 깜빡하면 법적고지 문구가 그대로 인덱싱될 위험이 있었음. embed_model을
    안 주면 [9]에서 채택한 `embedding.get_embedding_model()`(BGE-m3-ko)을 재사용 — 인덱싱에 쓸
    모델과 같은 모델이라 별도로 무거운 모델을 새로 로드하지 않음.

    add_structured_metadata=True면 [25] 텍스트 라우팅의 마지막 단계로 페이지당 한 번씩 OpenAI
    Structured Output(`structured_output.extract_text_chunk_metadata`, gpt-4o-mini)을 호출해 각
    청크에 "structured_metadata"(entities/sector_mentioned/topic/sentiment 등, 초안 스키마 —
    사용자가 필드를 가감할 예정) 키를 채운다. 기본값 False — 유료 API 호출이라 명시적으로 켜야
    함(boilerplate 제거처럼 자동 on으로 두지 않음). openai_client를 넘기면 그 클라이언트를 재사용
    (페이지마다 새 클라이언트를 만들지 않도록).

    [29] 사용자 지적("병목 없는거야?" — 실측 결과 페이지당 순차 호출 시 6페이지 문서에서도 +24초)
    반영: 각 페이지의 API 호출은 서로 독립적(I/O 바운드)이라 `concurrent.futures.ThreadPoolExecutor`
    로 병렬 디스패치한다 — 모든 페이지의 청크/난이도 판정(로컬 연산, YOLO 포함)을 먼저 순차로 끝낸
    뒤, structured_metadata 호출만 마지막에 한꺼번에 동시 실행. `structured_metadata_workers`로
    동시 호출 수 조절(기본 8 — OpenAI 기본 레이트리밋 대비 안전한 수준)."""
    from contextual_chunker import chunk_contextual_production

    if remove_boilerplate and embed_model is None:
        from embedding import get_embedding_model
        embed_model = get_embedding_model()

    if add_structured_metadata and openai_client is None:
        import os
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    doc_fitz = fitz.open(str(pdf_path))
    header_footer_templates = detect_headers_footers(doc_fitz) if strip_headers else None
    n_pages = doc_fitz.page_count

    # [35] 사용자 지시("스트럭처 아웃풋은 청킹과 병렬로 진행해") 반영 — 이전엔 전체 문서 청킹이
    # 다 끝난 뒤에야 구조화 출력 API 호출을 몰아서 디스패치했음(청킹 자체는 빨라서 큰 손해는
    # 아니지만, 진짜 파이프라이닝이 아니었음). 이제 페이지 청킹이 끝나는 즉시 그 페이지의 구조화
    # 출력 호출을 바로 스레드풀에 제출 — 다음 페이지 청킹(CPU)이 진행되는 동안 이전 페이지의
    # API 호출(I/O)이 함께 진행되도록 실제로 겹치게 함.
    executor = None
    futures = {}
    if add_structured_metadata:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=structured_metadata_workers)

    try:
        pages = []
        for i in range(n_pages):
            page = doc_fitz[i]
            cached_boxes = (page_boxes.get(i + 1) if page_boxes else None)
            if cached_boxes is None:
                cached_boxes = run_yolo_layout(model, page, i)  # page_boxes 없을 때만 여기서 호출

            difficulty_result = assess_page_difficulty(model, doc_fitz, i, cached_boxes=cached_boxes)
            difficulty = difficulty_result.difficulty

            page_text = extract_page_text(doc_fitz, i, difficulty=difficulty,
                                           header_footer_templates=header_footer_templates)

            chunks = []
            if difficulty == "easy":
                chunks = chunk_contextual_production(model, doc_fitz, i, doc_title=doc_title,
                                                      backend=chunk_backend, cached_boxes=cached_boxes)
                # boilerplate(Compliance Notice 등)는 실측([4]) 문서 마지막 페이지 근처에만 등장 —
                # 위치 사전필터로 그 구간 밖의 페이지는 임베딩 호출 자체를 건너뛴다. detect_boilerplate_
                # paragraphs_fast()의 last_n_pages_only는 "한 문서 전체 배치" 호출을 전제로 최대
                # 페이지번호를 스스로 계산하므로, 여기처럼 페이지 하나씩 순회하며 부르면 그 계산이
                # 깨진다 — 그래서 페이지 단위 게이팅은 여기서 직접 계산하고, 함수 자체는 필터 없이 호출.
                near_last_pages = i + 1 > n_pages - max(2, n_pages // 3)
                if remove_boilerplate and chunks and near_last_pages:
                    raw_texts = [c["raw_chunk"] for c in chunks]
                    scored = detect_boilerplate_paragraphs_fast(raw_texts, embed_model)
                    chunks = [c for c, s in zip(chunks, scored) if not s["is_boilerplate"]]

            page_entry = {**page_text, "difficulty_score": difficulty_result.difficulty_score,
                          "chunks": chunks}
            pages.append(page_entry)
            if executor and chunks:
                futures[executor.submit(extract_text_chunk_metadata, chunks, doc_title, openai_client,
                                        "gpt-4o-mini", sector)] = page_entry
        doc_fitz.close()

        if executor:
            from concurrent.futures import as_completed
            for future in as_completed(futures):
                page_entry = futures[future]
                metas = future.result()
                for c, meta in zip(page_entry["chunks"], metas):
                    c["structured_metadata"] = meta
    finally:
        if executor:
            executor.shutdown(wait=True)

    n_hard = sum(1 for p in pages if p["difficulty"] == "hard")
    return {
        "pdf": str(pdf_path), "n_pages": len(pages),
        "route_to_mineru": n_hard > 0,
        "hard_page_numbers": [p["page"] for p in pages if p["difficulty"] == "hard"],
        "pages": pages,
    }


def process_pdf_streaming(pdf_path, model, doc_title: str = None, strip_headers: bool = True,
                           chunk_backend: str = "rulebased", remove_boilerplate: bool = True,
                           embed_model=None, page_boxes: dict = None):
    """[32] 사용자 요청("PDF 업로드 시 5초 내외로 청킹까지 끝나 벡터DB에 들어가야") 지원 —
    `process_pdf()`처럼 문서 전체를 다 처리한 뒤 한 번에 반환하는 대신, **페이지가 끝나는 대로
    하나씩 yield**하는 제너레이터. 대형 문서(K-Wave류 70+페이지, `process_pdf()`로는 전체 처리에
    5초를 넘김, [32] 참고)에서 "문서 전체가 끝나야 검색 가능"이 아니라 "앞 페이지부터 순서대로
    바로 검색 가능"하게 만들고 싶을 때 이 함수를 쓴다 — 호출자가 각 페이지를 받는 즉시
    임베딩+벡터DB insert를 시작하면, 첫 페이지는 몇백 ms~1초 안에 이미 검색 가능해진다(전체
    문서 완료를 기다릴 필요 없음).

    구조화 출력(structured_metadata)은 이 스트리밍 경로에 없음 — 설계 원칙(표/이미지 메타데이터와
    마찬가지로 텍스트도 구조화 출력은 별도의 느린 백그라운드 잡)에 따라, 필요하면 호출자가 각
    페이지의 chunks를 받은 뒤 `structured_output.extract_text_chunk_metadata()`를 나중에 별도
    호출해서 붙이면 됨(`실험_생성단계_아키텍처설계.md` 3번 아키텍처의 "느린 레인"에 해당).

    파라미터는 `process_pdf()`와 동일(add_structured_metadata류 제외). 각 yield 값은 `process_pdf()`
    의 `pages[i]`와 같은 구조(`page`/`text`/`difficulty`/`needs_external_reader`/`chunks` 등) —
    호출자는 `needs_external_reader`로 hard 페이지를, 전체 순회 후 마지막 페이지 번호로 완료 여부를
    직접 추적하면 됨(별도 요약 객체를 안 두는 이유: yield 타입을 페이지 하나로 통일해 스트림
    소비 코드를 단순하게 유지하기 위함)."""
    if remove_boilerplate and embed_model is None:
        from embedding import get_embedding_model
        embed_model = get_embedding_model()

    from contextual_chunker import chunk_contextual_production

    doc_fitz = fitz.open(str(pdf_path))
    try:
        header_footer_templates = detect_headers_footers(doc_fitz) if strip_headers else None
        n_pages = doc_fitz.page_count

        for i in range(n_pages):
            page = doc_fitz[i]
            cached_boxes = (page_boxes.get(i + 1) if page_boxes else None)
            if cached_boxes is None:
                cached_boxes = run_yolo_layout(model, page, i)

            difficulty_result = assess_page_difficulty(model, doc_fitz, i, cached_boxes=cached_boxes)
            difficulty = difficulty_result.difficulty
            page_text = extract_page_text(doc_fitz, i, difficulty=difficulty,
                                           header_footer_templates=header_footer_templates)

            chunks = []
            if difficulty == "easy":
                chunks = chunk_contextual_production(model, doc_fitz, i, doc_title=doc_title,
                                                      backend=chunk_backend, cached_boxes=cached_boxes)
                near_last_pages = i + 1 > n_pages - max(2, n_pages // 3)
                if remove_boilerplate and chunks and near_last_pages:
                    raw_texts = [c["raw_chunk"] for c in chunks]
                    scored = detect_boilerplate_paragraphs_fast(raw_texts, embed_model)
                    chunks = [c for c, s in zip(chunks, scored) if not s["is_boilerplate"]]

            yield {**page_text, "difficulty_score": difficulty_result.difficulty_score, "chunks": chunks}
    finally:
        doc_fitz.close()


if __name__ == "__main__":
    import json

    ROOT = Path(__file__).resolve().parent.parent.parent
    for label, rel in [
        ("LGCNS", "pdf_pipeline/reference/LGCNS/20260721_company_279243000.pdf"),
        ("Construct", "pdf_pipeline/reference/Construct/20260721_industry_362851000.pdf"),
    ]:
        result = extract_pdf_text(ROOT / rel)
        print(f"=== {label} ===")
        print(f"  route_to_mineru={result['route_to_mineru']}, "
              f"PUA 아티팩트 있던 페이지={result['n_pages_with_pua_artifact']}개")
        for p in result["pages"]:
            if p["had_pua_artifact"] or p["needs_external_reader"]:
                print(f"  p{p['page']}: pua={p['had_pua_artifact']} "
                      f"needs_external_reader={p['needs_external_reader']}")

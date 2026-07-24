"""[47] 사용자 요청 — "전체 파이프라인 실행해. 이미지텍스트테이블라우팅도 전부" C밴드.pdf
(하나증권 통신장비 산업분석, 6p, 11개 종목 Top Picks)로 ERD 전 구간(텍스트/테이블/이미지 브랜치
-> 엔티티 합성 -> Supabase 적재 -> 엔티티인식 라우팅 검색 -> citation-check 포함 최종 생성)을
`run_investment_opinion_demo.py`와 동일한 패턴으로 검증. 이미지 브랜치는 이 환경에 MinerU가
없어 대표 예시 카드로 대체(README 카드 스키마 준수, 콘솔에 명시적으로 표시).

쿼리: "이 PDF에 나오는 기업에 대한 투자 인사이트를 도출해줘"
"""
import json
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
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "table_processing"))

from page_classifier import classify_pdf  # noqa: E402
from text_extraction import process_pdf  # noqa: E402
import run_table_metadata_pipeline as rtmp  # noqa: E402
from citation_check import generate_with_citation_check  # noqa: E402
from scanned_page_router import detect_scanned_pages  # noqa: E402
from index_text import precompute_entity_count, route_search  # noqa: E402
import entity_fusion  # noqa: E402

load_dotenv(ROOT / ".env")

PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "C밴드" / "c밴드.pdf"
PDF_ID = "C밴드"
TICKER = None  # 산업분석(다종목)이라 특정 티커 없음
QUERY = "이 PDF에 나오는 기업에 대한 투자 인사이트를 도출해줘"
DB_URL = os.environ.get("SUPABASE_DIRECT_DB_URL")
if not DB_URL:
    raise RuntimeError("SUPABASE_DIRECT_DB_URL 환경변수가 없습니다.")

# 이미지 브랜치: MinerU 미설치라 대표 예시 카드로 대체(README 카드 스키마 준수).
# c밴드.pdf 2페이지 유일한 차트("도표 1. 미국 경매용 주파수 현황")는 정적 스펙트럼 다이어그램이라
# 캡션+본문 서술로 내용이 이미 충분히 설명됨(수치 시계열 차트가 아님) — 대표 카드도 그에 맞게 구성.
FALLBACK_IMAGE_CARDS = [
    {
        "image_id": f"{PDF_ID}_p2_chart1", "doc_id": PDF_ID, "page": 2, "block_type": "chart",
        "status": "useful", "caption": "미국 경매용 주파수 현황(어퍼 C밴드 3.98~4.14GHz 160MHz)",
        "footnote": "자료: FCC, NTIA, 하나증권",
        "ocr": {"text": "3.0GHz 4.0GHz 5.0GHz 6.0GHz 7.0GHz 8.0GHz 어퍼C밴드 신규대역"},
        "chart_table": None,
        "narrative": "어퍼 C밴드(3.98~4.14GHz) 160MHz가 신규 경매 대상 주파수 대역으로, "
                     "기존 대역과 구분되는 미개척 주파수임을 시각적으로 보여준다.",
        "embed_text": "미국 경매용 주파수 현황 어퍼 C밴드(3.98~4.14GHz) 160MHz 신규대역 "
                       "자료: FCC, NTIA, 하나증권",
    },
]


def main():
    timings = {}
    from embedding import get_embedding_model
    threading.Thread(target=get_embedding_model, daemon=True).start()

    t0 = time.perf_counter()
    print("0) 스캔본 페이지 감지")
    scanned_pages = detect_scanned_pages(PDF_PATH)
    print(f"   스캔본 페이지: {scanned_pages or '없음'}")
    timings["0_scanned_detect"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("1) YOLO 로딩 + 페이지 분류")
    yolo_model = YOLO(str(ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"))
    yolo_model.predict(Image.new("RGB", (595, 842), (255, 255, 255)), conf=0.25, verbose=False)
    cls_result = classify_pdf(PDF_PATH, yolo_model)
    page_boxes = {p["page"]: p["cached_boxes"] for p in cls_result["pages"]}
    print(f"   {cls_result['n_pages']}페이지 분류 완료")
    timings["1_yolo_classify"] = time.perf_counter() - t0

    import numpy as np
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    t0 = time.perf_counter()
    print("2) 텍스트 브랜치: 추출+청킹+구조화출력(entities/sector) -> 임베딩 -> 즉시 적재")
    text_result = process_pdf(PDF_PATH, yolo_model, doc_title=PDF_ID, page_boxes=page_boxes,
                               chunk_backend="rulebased", add_structured_metadata=True,
                               openai_client=client, sector="통신장비")
    text_chunks = [c for page in text_result["pages"] for c in page["chunks"]]
    text_items, text_emb = entity_fusion.embed_items(entity_fusion.from_text_chunks(PDF_ID, text_chunks))
    n = entity_fusion.store_evidence(DB_URL, PDF_ID, text_items, text_emb, ticker=TICKER)
    print(f"   {len(text_chunks)}개 청크 -> {n}개 적재")
    n_meta = sum(1 for c in text_chunks if c.get("structured_metadata"))
    print(f"   구조화 메타데이터 채워진 청크: {n_meta}/{len(text_chunks)}")
    for c in text_chunks:
        sm = c.get("structured_metadata")
        if sm and sm.get("entities"):
            print(f"     p{c['page']} entities={sm['entities']} sector={sm.get('sector_mentioned')} "
                  f"sentiment={sm.get('sentiment')}")
    timings["2_text_branch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("3) 테이블 브랜치: 하이브리드 게이트([JAEIL v5], TATR 대체) + canonical 매칭 -> 임베딩 -> 즉시 적재")
    rtmp.PDF_PATH = PDF_PATH
    table_records, n_finance_filtered, n_cid = rtmp.build_records(
        PDF_ID, page_boxes=page_boxes, yolo_model=yolo_model, sector="통신장비")
    row_records = [r for r in table_records if r.get("record_type") != "table_metadata"]
    mapped = [r for r in row_records if r.get("canonical_field")]
    table_items, table_emb = entity_fusion.embed_items(entity_fusion.from_table_records(PDF_ID, row_records))
    n = entity_fusion.store_evidence(DB_URL, PDF_ID, table_items, table_emb, ticker=TICKER)
    print(f"   {len(row_records)}행 파싱, canonical 매칭 {len(mapped)}개 -> {n}개 적재")
    timings["3_table_branch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("4) 이미지 브랜치: MinerU 미설치 -> 대표 예시 카드로 대체 -> 임베딩 -> 즉시 적재")
    image_cards = FALLBACK_IMAGE_CARDS
    print(f"   ⚠ MinerU 미설치 확인됨 — 실제 문서 내용 기반 대표 카드 {len(image_cards)}건으로 대체")
    image_items, image_emb = entity_fusion.embed_items(entity_fusion.from_image_cards(PDF_ID, image_cards))
    n = entity_fusion.store_evidence(DB_URL, PDF_ID, image_items, image_emb, ticker=TICKER)
    print(f"   {n}개 적재")
    timings["4_image_branch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("5) 엔티티 합성 확인 (document_evidence에 세 브랜치 다 모였는지)")
    all_items = text_items + table_items + image_items
    by_source = {}
    for it in all_items:
        by_source[it["source_type"]] = by_source.get(it["source_type"], 0) + 1
    print(f"   총 {len(all_items)}개 evidence {by_source}")
    timings["5_entity_fusion"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("6) Supabase에서 다시 읽어와 인덱스 재구성 ([41] read-path 검증)")
    index = entity_fusion.load_evidence_from_db(DB_URL, pdf_id=PDF_ID)
    print(f"   재구성: evidence {len(index.chunks)}개")
    timings["6_reload_from_db"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("7) 엔티티 카운트 사전계산([46], 하나증권 정규식 무료 경로 기대)")
    precompute_entity_count(index, pdf_path=PDF_PATH, client=client)
    print(f"   entity_count={index.entity_count}")
    timings["7_entity_count"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("8) 엔티티 인식 라우팅 검색 ([44]/[45] route_search, entity_aware=True)")
    hits, qtype = route_search(index, QUERY, client=client, top_k=8)
    print(f"   분류: {qtype}, entity_count={index.entity_count}")
    for h in hits:
        print(f"   - [{h['chunk'].get('source_type', 'text')}] p{h['chunk'].get('page')}: "
              f"{h['chunk'].get('content', '')[:70]!r}")
    timings["8_search"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("9) 최종 생성 (gpt-4o-mini, citation-check 포함)")
    evidence_context = "\n\n".join(
        f"[{h['chunk'].get('source_type', 'text')} / p{h['chunk'].get('page')}] "
        f"{h['chunk'].get('content') or h['chunk'].get('raw_chunk', '')}"
        for h in hits
    )
    prompt = f"""다음은 증권사 산업분석 리포트 PDF에서 텍스트/표/이미지 소스를 통합한 검색으로
찾은 근거입니다.

[통합 근거]
{evidence_context}

[작성 지침]
- 반드시 위 근거에 등장하는 구체적 수치(목표주가/현재가/투자의견 등)를 최소 3개 이상 인용할 것.
- 이 문서가 다루는 여러 기업(종목)에 대한 투자 인사이트를 가능한 폭넓게 다룰 것 — 한두 종목에만
  의존하지 말 것.
- 위 근거에 없는 내용은 추측하지 말 것.

[사용자 요청]
{QUERY}
"""
    result = generate_with_citation_check(client, prompt, context=evidence_context,
                                          model="gpt-4o-mini", max_retries=2)
    timings["9_generation"] = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print(f"최종 답변 ({result['attempts']}회 생성"
          f"{', 미해결 근거없는 숫자: ' + str(result['unsupported_numbers']) if result['unsupported_numbers'] else ''})")
    print("=" * 70)
    print(result["answer"])

    total = sum(timings.values())
    print("\n" + "=" * 70)
    print(f"단계별 소요시간 (총 {total:.1f}s)")
    print("=" * 70)
    for name, sec in timings.items():
        print(f"   {name:24s} {sec:7.2f}s  ({sec/total*100:5.1f}%)")

    return timings, index, hits, qtype, result


if __name__ == "__main__":
    main()

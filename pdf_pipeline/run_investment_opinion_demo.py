"""LGCNS PDF 한 건으로 ERD 전체 흐름을 실행: 스캔본 페이지 감지(MinerU 전용 처리) -> 텍스트/
테이블/이미지 세 브랜치 -> 엔티티 합성(가중치 정제) -> 통합 Supabase 테이블 적재 -> 하이브리드
검색(BM25+BGE-m3-ko, 소스 가중치 반영) -> citation-check 포함 LLM 투자의견 생성까지 한 번에 돈다.

범위 제한:
    - 리랭킹: 의도적으로 보류(사용자 확인 완료 — 오버엔지니어링 방지, index_text.py 자체 설계
      원칙과도 일치)
    - 이미지 브랜치: 이 실행 환경에 MinerU가 설치돼 있지 않아(무거운 의존성, 별도 설치 필요)
      pdf_pipeline/image_processing/s2_onestop_mineru.py를 직접 실행할 수 없다. 실제
      onestop_cards.jsonl이 있으면 그걸 쓰고, 없으면(이 실행처럼) README의 카드 스키마를 그대로
      따르는 대표 예시 카드로 대체해 파이프라인 나머지 단계(합성/저장/검색/생성)를 끝까지
      검증한다 — 콘솔에 명시적으로 표시됨. 실제 MinerU 카드로 교체하려면 ONESTOP_CARDS_PATH만
      바꾸면 된다.
    - 스캔본 페이지가 감지되면 scanned_page_router.extract_text_via_mineru()로 MinerU 파싱
      결과를 가져와야 하는데, 이 역시 MinerU 미설치로 실제 호출은 못 한다 — 감지 자체(순수
      PyMuPDF)는 실행하고, 감지되면 "MinerU 미설치라 대체 불가" 경고만 명시적으로 낸다.

Usage:
    python pdf_pipeline/run_investment_opinion_demo.py
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
import entity_fusion  # noqa: E402

load_dotenv(ROOT / ".env")

PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "LGCNS" / "20260721_company_279243000.pdf"
PDF_ID = "LGCNS"
TICKER = "064400.KS"
QUERY = "이 PDF 내용을 바탕으로 이 회사에 대한 투자 의견을 제공해줘"
DB_URL = os.environ.get(
    "SUPABASE_DIRECT_DB_URL",
    "postgresql://postgres.itkxhdutnxircvbzwpon:SuperTeam24ever@aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres",
)
ONESTOP_CARDS_PATH = ROOT / "data" / "onestop" / PDF_ID / "onestop_cards.jsonl"

# MinerU 미설치 환경에서 이미지 브랜치 나머지 단계(합성/저장/검색/생성)를 검증하기 위한 대표
# 예시 카드 — onestop_cards.jsonl(README §5 카드 스키마)과 동일한 필드 구조. 실제 문서 내용이
# 아니라 이번 LGCNS PDF의 실제 수치(page1_4 텍스트 청크에서 확인된 클라우드&AI 매출 추이)를
# 반영해 만든 대표 사례.
FALLBACK_IMAGE_CARDS = [
    {
        "image_id": f"{PDF_ID}_p2_chart1", "doc_id": PDF_ID, "page": 2, "block_type": "chart",
        "status": "useful", "caption": "LG CNS 클라우드&AI 부문 분기별 매출 추이",
        "footnote": "단위: 십억원", "ocr": {"text": "717 872 880 1,118 765 921"},
        "chart_table": "| 분기 | 클라우드&AI 매출 |\n|---|---|\n| 1Q25 | 717 |\n| 2Q25 | 872 |\n"
                        "| 3Q25 | 880 |\n| 4Q25 | 1,118 |\n| 1Q26 | 765 |\n| 2Q26F | 921 |",
        "narrative": "클라우드&AI 부문 매출은 분기별로 우상향하며, 2Q26F 921십억원까지 성장할 전망이다.",
        "embed_text": "LG CNS 클라우드&AI 부문 분기별 매출 추이 단위: 십억원 717 872 880 1,118 765 921 "
                       "클라우드&AI 부문 매출은 분기별로 우상향하며, 2Q26F 921십억원까지 성장할 전망이다.",
    },
]


def main():
    timings = {}

    from embedding import get_embedding_model
    threading.Thread(target=get_embedding_model, daemon=True).start()

    t0 = time.perf_counter()
    print("0) 스캔본 페이지 감지 (텍스트 레이어 없음 + 이미지가 페이지 대부분을 덮는 페이지)")
    scanned_pages = detect_scanned_pages(PDF_PATH)
    if scanned_pages:
        print(f"   ⚠ {len(scanned_pages)}개 페이지({scanned_pages})가 어려운(스캔) PDF 페이지로 판정됨 "
              f"— only MinerU 처리 대상")
        print("   ⚠ 이 환경엔 MinerU가 설치돼 있지 않아 실제 MinerU 대체 추출은 수행하지 못함 "
              "(scanned_page_router.extract_text_via_mineru() 참고, MinerU 설치 후 재실행 필요)")
    else:
        print("   스캔본 페이지 없음 — 전 페이지 자체 파이프라인으로 처리")
    timings["0_scanned_page_detect"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("1) YOLO 로딩 + 페이지 분류")
    yolo_model = YOLO(str(ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"))
    yolo_model.predict(Image.new("RGB", (595, 842), (255, 255, 255)), conf=0.25, verbose=False)

    cls_result = classify_pdf(PDF_PATH, yolo_model)
    page_boxes = {p["page"]: p["cached_boxes"] for p in cls_result["pages"]}
    print(f"   {cls_result['n_pages']}페이지 분류 완료")
    timings["1_yolo_load_and_classify"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("2) 텍스트 브랜치: 추출 + 청킹")
    text_result = process_pdf(PDF_PATH, yolo_model, page_boxes=page_boxes,
                               chunk_backend="rulebased", remove_boilerplate=True,
                               add_structured_metadata=False)
    text_chunks = [c for page in text_result["pages"] for c in page["chunks"]]
    print(f"   {len(text_chunks)}개 청크 생성")
    timings["2_text_branch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("3) 테이블 브랜치: TATR + canonical 매칭")
    rtmp.PDF_PATH = PDF_PATH
    table_records, n_finance_filtered, n_cid = rtmp.build_records(
        PDF_ID, page_boxes=page_boxes, yolo_model=yolo_model)
    row_records = [r for r in table_records if r.get("record_type") != "table_metadata"]
    mapped = [r for r in row_records if r.get("canonical_field")]
    print(f"   {len(row_records)}행 파싱, canonical 매칭 {len(mapped)}개")
    timings["3_table_branch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("4) 이미지 브랜치: image_processing 카드 로드")
    if ONESTOP_CARDS_PATH.exists():
        image_cards = [json.loads(line) for line in ONESTOP_CARDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"   실제 onestop_cards.jsonl {len(image_cards)}건 로드: {ONESTOP_CARDS_PATH}")
    else:
        image_cards = FALLBACK_IMAGE_CARDS
        print(f"   ⚠ {ONESTOP_CARDS_PATH} 없음(MinerU 미실행) — 대표 예시 카드 {len(image_cards)}건으로 대체")
    timings["4_image_branch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("5) 엔티티 합성 (텍스트+테이블+이미지 -> 통합 evidence, 소스 가중치 부여)")
    evidence = entity_fusion.fuse(PDF_ID, text_chunks=text_chunks, table_records=row_records,
                                   image_cards=image_cards)
    by_source = {}
    for it in evidence:
        by_source[it["source_type"]] = by_source.get(it["source_type"], 0) + 1
    print(f"   총 {len(evidence)}개 evidence ({by_source})")
    timings["5_entity_fusion"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("6) 통합 하이브리드 인덱스 구축 (BGE-m3-ko dense + BM25)")
    index = entity_fusion.build_fused_index(PDF_ID, evidence)
    dim = index.embeddings.shape[1]
    print(f"   임베딩 차원: {dim}, evidence 수: {len(index.chunks)}")
    timings["6_build_fused_index"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("7) 통합 Supabase 테이블(document_evidence) 적재")
    n_stored = entity_fusion.store_evidence(DB_URL, PDF_ID, index, ticker=TICKER)
    print(f"   {n_stored}개 evidence -> document_evidence 적재 완료 (ticker={TICKER})")
    timings["7_supabase_store"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("8) 가중 하이브리드 검색 (BM25 + BGE-m3-ko + 소스 가중치)")
    hits = entity_fusion.weighted_hybrid_search(index, QUERY, top_k=8)
    for h in hits:
        print(f"   - [{h['source_type']}] page{h['chunk'].get('page')} score={h['score']:.3f} "
              f"(dense={h['dense_score']:.3f}, bm25={h['bm25_score']:.3f})")
    timings["8_hybrid_search"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("9) LLM 투자의견 생성 (gpt-4o-mini, citation-check 포함)")
    evidence_context = "\n\n".join(
        f"[{h['source_type']} / p{h['chunk'].get('page')}] {h['chunk']['content']}" for h in hits
    )

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = f"""다음은 한 기업 리포트 PDF에서 텍스트/표/이미지(차트) 세 소스를 통합한 하이브리드
검색(BM25+BGE-m3-ko, 소스별 가중치 반영)으로 찾은 근거입니다. 각 항목 앞의 [text/table/image]는
어느 브랜치에서 나온 근거인지를 나타냅니다.

[통합 근거]
{evidence_context}

[작성 지침]
- 반드시 위 근거에 등장하는 구체적 수치를 최소 3개 이상 인용할 것. 수치 없는 뭉뚱그린 서술만으로
  결론짓지 말 것.
- 가능하면 text/table/image 여러 소스의 근거를 섞어서 활용할 것(한 소스에만 의존하지 말 것).
- 긍정적 근거와 부정적/유의할 근거를 모두 찾아 균형 있게 제시할 것.
- 위 근거에 없는 내용은 추측하지 말 것.

[사용자 요청]
{QUERY}
"""
    result = generate_with_citation_check(
        client, prompt, context=evidence_context, model="gpt-4o-mini", max_retries=2)
    answer = result["answer"]
    timings["9_llm_generation"] = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"LLM 투자의견 출력 ({result['attempts']}회 생성"
          f"{', 미해결 근거없는 숫자: ' + str(result['unsupported_numbers']) if result['unsupported_numbers'] else ''})")
    print("=" * 60)
    print(answer)

    total = sum(timings.values())
    print("\n" + "=" * 60)
    print(f"단계별 소요시간 (총 {total:.1f}s)")
    print("=" * 60)
    for name, sec in timings.items():
        print(f"   {name:30s} {sec:7.2f}s  ({sec / total * 100:5.1f}%)")


if __name__ == "__main__":
    main()

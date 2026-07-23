"""LGCNS PDF 한 건으로 ERD 전체 흐름 시연: 텍스트+테이블 브랜치 실행 -> 임시 Supabase 테이블에
텍스트 청크 적재 -> 하이브리드 검색(BM25+BGE-m3-ko) -> LLM 투자의견 생성까지 한 번에 돌린다.

범위 제한(현재 코드베이스에 없는 부분은 스킵):
    - 이미지/차트/VLM 브랜치: MinerU/DocumentFigureClassifier 자체가 코드에 없어 제외
    - 정식 엔티티 합성(가중치 기반 병합): 아직 미구현이라, 텍스트 하이브리드 검색 결과 + 테이블
      canonical 매칭 결과를 단순 결합해서 LLM 컨텍스트로 사용
    - 임시 테이블에 대한 Postgres 네이티브 하이브리드 검색: 아직 없어서, 이미 검증된 인메모리
      build_index()/hybrid_search()(text_processing/index_text.py)로 실제 검색은 수행하고,
      결과 청크는 영속화를 위해 임시 테이블에도 적재.

Usage:
    python pdf_pipeline/run_investment_opinion_demo.py
"""

import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
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
from index_text import build_index, hybrid_search  # noqa: E402

load_dotenv(ROOT / ".env")

PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "LGCNS" / "20260721_company_279243000.pdf"
PDF_ID = "LGCNS"
QUERY = "이 PDF 내용을 바탕으로 이 회사에 대한 투자 의견을 제공해줘"
DB_URL = os.environ.get(
    "SUPABASE_DIRECT_DB_URL",
    "postgresql://postgres.itkxhdutnxircvbzwpon:SuperTeam24ever@aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres",
)


def main():
    print("1) YOLO 로딩 + 페이지 분류")
    yolo_model = YOLO(str(ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"))
    yolo_model.predict(Image.new("RGB", (595, 842), (255, 255, 255)), conf=0.25, verbose=False)

    cls_result = classify_pdf(PDF_PATH, yolo_model)
    page_boxes = {p["page"]: p["cached_boxes"] for p in cls_result["pages"]}
    print(f"   {cls_result['n_pages']}페이지 분류 완료")

    print("2) 텍스트 브랜치: 추출 + 청킹")
    text_result = process_pdf(PDF_PATH, yolo_model, page_boxes=page_boxes,
                               chunk_backend="rulebased", remove_boilerplate=True,
                               add_structured_metadata=False)
    n_chunks = sum(len(p["chunks"]) for p in text_result["pages"])
    print(f"   {n_chunks}개 청크 생성")

    print("3) 테이블 브랜치: TATR + canonical 매칭")
    rtmp.PDF_PATH = PDF_PATH
    table_records, n_finance_filtered, n_cid = rtmp.build_records(
        PDF_ID, page_boxes=page_boxes, yolo_model=yolo_model)
    row_records = [r for r in table_records if r.get("record_type") != "table_metadata"]
    mapped = [r for r in row_records if r.get("canonical_field")]
    print(f"   {len(row_records)}행 파싱, canonical 매칭 {len(mapped)}개")

    def _fmt_row(r):
        cells = r.get("cells") or r.get("numeric_values") or []
        cf = f" (canonical={r['canonical_field']})" if r.get("canonical_field") else ""
        return f"- p{r['page']} {r['raw_label']}: {cells}{cf}"

    # canonical 매칭 여부와 무관하게 "순수 재무제표 실측치"(financial_chunks에 이미 있다고 보고
    # 필터된 것) 외의 모든 표 행을 근거로 사용 — 세그먼트별 매출(예: 클라우드&AI)처럼 canonical
    # field가 아직 없는 항목도 실제 수치가 있으면 투자의견의 중요한 근거이므로 빠뜨리지 않는다.
    table_context = "\n".join(_fmt_row(r) for r in row_records) or "(표에서 추출된 항목 없음)"

    print("4) 텍스트 청크 하이브리드 인덱스(BGE-m3-ko dense + BM25) 구축")
    index = build_index(PDF_ID, text_result)
    dim = index.embeddings.shape[1]
    print(f"   임베딩 차원: {dim}, 청크 수: {len(index.chunks)}")

    print("5) 임시 Supabase 테이블 생성 + 텍스트 청크 적재")
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("drop table if exists pdf_chunks_temp")
        cur.execute(f"""
            create table pdf_chunks_temp (
                id text primary key,
                pdf_id text not null,
                page int not null,
                content text not null,
                embedding vector({dim}) not null,
                created_at timestamptz not null default now()
            )
        """)
        rows = [
            (cid, PDF_ID, chunk["page"], chunk["text"], emb.tolist())
            for cid, chunk, emb in zip(index.chunk_ids, index.chunks, index.embeddings)
        ]
        execute_values(
            cur, "insert into pdf_chunks_temp (id, pdf_id, page, content, embedding) values %s", rows)
    conn.close()
    print(f"   {len(rows)}개 청크 -> pdf_chunks_temp 적재 완료")

    print("6) 하이브리드 검색 (BM25 + BGE-m3-ko, weighted-sum fusion)")
    hits = hybrid_search(index, QUERY, top_k=5)
    for h in hits:
        print(f"   - page{h['chunk']['page']} score={h['score']:.3f} "
              f"(dense={h['dense_score']:.3f}, bm25={h['bm25_score']:.3f})")

    print("7) LLM 투자의견 생성 (gpt-4o-mini)")
    text_context = "\n\n".join(f"[p{h['chunk']['page']}] {h['chunk']['text']}" for h in hits)

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = f"""다음은 한 기업 리포트 PDF에서 하이브리드 검색(BM25+BGE-m3-ko)으로 찾은 관련 텍스트와,
표에서 추출한 지표(밸류에이션 배수, 세그먼트별 매출 등 — 원본 재무제표 실측치는 별도 DB에 이미
있다고 보고 여기서 제외됨)입니다.

[관련 텍스트]
{text_context}

[표에서 추출한 지표]
{table_context}

[작성 지침]
- 반드시 위 텍스트/표에 등장하는 구체적 수치(매출/영업이익 증감률, 세그먼트별 매출, PER/PBR/ROE 등
  밸류에이션 배수, 계약금액 등)를 최소 3개 이상 인용해서 근거로 제시할 것. 수치 없는 뭉뚱그린
  서술("긍정적", "안정적")만으로 결론짓지 말 것.
- 긍정적 근거와 부정적/유의할 근거를 모두 찾아 균형 있게 제시할 것 — 텍스트에 컨센서스 대비
  하회/부진 등 부정적 뉘앙스가 있으면 반드시 포함하고 빠뜨리지 말 것.
- 위 컨텍스트에 없는 내용은 추측하지 말 것.

[사용자 요청]
{QUERY}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = resp.choices[0].message.content

    print("\n" + "=" * 60)
    print("LLM 투자의견 출력")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()

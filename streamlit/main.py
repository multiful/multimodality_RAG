"""KOSPI200 투자 인사이트 대시보드 (Streamlit).

홈 화면(관심 종목 검색/그리드)과 종목 상세 화면(가격 차트 + AI 요약 + 실시간 질의응답)으로
구성된, 모바일 주식 앱 스타일의 데스크톱 웹 UI.

파이프라인(README/PRD의 "PDF -> 페이지 분류 -> 텍스트/테이블/이미지 -> 엔티티 합성 ->
Supabase -> 하이브리드 검색(BM25+BGE-m3-ko) -> LLM 투자 insight")을 그대로 재사용한다:
- 상시 사용 가능한 기본 근거: financial_chunks/company_profile_chunks 밀집 검색(1536차원,
  text-embedding-3-small) — 199개 KOSPI200 종목 전체에 미리 적재돼 있음.
- 애널리스트 PDF가 실제로 적재된 종목(document_evidence에 행이 있는 경우, 예: 064400.KS)은
  entity_fusion.load_evidence_from_db() + weighted_hybrid_search()로 BM25+BGE-m3-ko 가중
  하이브리드 검색까지 근거에 더하고, citation_check로 숫자 근거를 검증한 뒤 답변을 생성한다.
  SUPABASE_DIRECT_DB_URL이 없거나 psycopg2가 없으면 이 단계는 조용히 건너뛰고 밀집 검색
  결과만으로 답변한다(항상 동작은 하되, 가능할 때 더 정교해지는 구조).

"PDF 업로드" 화면에서는 애널리스트 PDF를 직접 올려 같은 파이프라인(스캔본 감지 -> YOLO 페이지
분류 -> 텍스트/표 브랜치 -> 임베딩 -> document_evidence 즉시 적재)을 그 자리에서 실행한다. 이미지/
차트 브랜치(MinerU)는 이 데모 환경에 설치돼 있지 않아 생략하며, 화면에 명시적으로 알린다.

Usage:
    streamlit run streamlit/main.py
"""

import os
import re
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pdf_pipeline"))
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "page_classification"))
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "text_processing"))
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "table_processing"))

load_dotenv(ROOT / ".env")

from supabase import create_client  # noqa: E402

from embeddings.gpt_embedder import GPTEmbedder  # noqa: E402
from generation import GPTGenerator  # noqa: E402
from supabase_store import SupabaseVectorStore  # noqa: E402

PROFILE_DIR = ROOT / "KOSPI200_output" / "kospi200_profiles"
TITLE_RE = re.compile(r"^# (.+?) \((.+?)\) 기업 프로필")

st.set_page_config(page_title="포트폴리오", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# 캐시된 리소스 / 데이터
# ---------------------------------------------------------------------------

@st.cache_resource
def get_supabase_client():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


@st.cache_resource
def get_embedder():
    return GPTEmbedder()


@st.cache_resource
def get_generator():
    return GPTGenerator()


@st.cache_resource
def get_financial_store():
    return SupabaseVectorStore(get_embedder(), table="financial_chunks")


@st.cache_resource
def get_profile_store():
    return SupabaseVectorStore(get_embedder(), table="company_profile_chunks")


@st.cache_resource
def get_yolo_model():
    """PDF 업로드 파이프라인의 페이지 분류/표 크롭에 쓰는 YOLOv11n 문서 레이아웃 모델.
    최초 1회만 로드하고, 첫 추론(콜드스타트 지연)도 여기서 미리 해둔다."""
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"))
    model.predict(Image.new("RGB", (595, 842), (255, 255, 255)), conf=0.25, verbose=False)
    return model


@st.cache_resource
def get_bge_model():
    """entity_fusion(텍스트/표 브랜치 임베딩)이 쓰는 BGE-m3-ko 모델. embedding.py의
    get_embedding_model()은 모듈 전역 싱글턴이라 YOLO와 달리 st.cache_resource 대상이
    아니었고, 그래서 첫 PDF 업로드 때 2GB+ 모델을 디스크에서 올리는 콜드로드 비용이
    업로드 대기 시간에 그대로 드러났다(실측: 5분 중 대부분). 여기서 감싸서 YOLO와 동일하게
    앱 시작 시 미리 워밍업해둔다."""
    from embedding import get_embedding_model
    return get_embedding_model()


_WARMUP_STARTED = False


def _warmup_heavy_models():
    """PDF 업로드 파이프라인이 쓰는 무거운 모델(YOLO/BGE-m3-ko)을 앱 프로세스 시작 시
    백그라운드 스레드로 미리 로드해, 실제 업로드 클릭 시점엔 이미 캐시돼 있게 한다
    (run_investment_opinion_demo.py가 쓰던 것과 동일한 패턴 — 콜드로드를 사용자 대기 시간
    밖으로 빼낸다). 프로세스당 한 번만 스레드를 띄우면 되므로(이후 세션은 st.cache_resource
    캐시를 그대로 재사용) 모듈 전역 플래그로 중복 실행을 막는다."""
    global _WARMUP_STARTED
    if _WARMUP_STARTED:
        return
    _WARMUP_STARTED = True
    import threading
    threading.Thread(target=get_yolo_model, daemon=True).start()
    threading.Thread(target=get_bge_model, daemon=True).start()


@st.cache_data(ttl=3600)
def load_ticker_universe() -> list[dict]:
    """로컬 프로필 md 파일(KOSPI200_output/kospi200_profiles/*_profile.md)의 제목 줄에서
    {name, ticker} 목록을 만든다 — KRX 로그인 없이 즉시 로드."""
    universe = []
    for path in sorted(PROFILE_DIR.glob("*_profile.md")):
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        m = TITLE_RE.match(first_line)
        if m:
            universe.append({"name": m.group(1), "ticker": m.group(2)})
    return universe


@st.cache_data(ttl=600)
def load_summaries(ticker: str) -> dict:
    client = get_supabase_client()
    profile = (
        client.table("company_profile_chunks").select("summary").eq("id", ticker).limit(1).execute()
    )
    financial = (
        client.table("financial_summaries").select("summary").eq("ticker", ticker).limit(1).execute()
    )
    return {
        "profile_summary": profile.data[0]["summary"] if profile.data else None,
        "financial_summary": financial.data[0]["summary"] if financial.data else None,
    }


@st.cache_data(ttl=600)
def has_document_evidence(ticker: str) -> bool:
    client = get_supabase_client()
    resp = (
        client.table("document_evidence")
        .select("id", count="exact")
        .eq("ticker", ticker)
        .limit(1)
        .execute()
    )
    return (resp.count or 0) > 0


@st.cache_data(ttl=300)
def load_price_history(ticker: str):
    return yf.Ticker(ticker).history(period="6mo")


# ---------------------------------------------------------------------------
# RAG 질의응답
# ---------------------------------------------------------------------------

def answer_question(ticker: str, query: str) -> dict:
    """financial_chunks/company_profile_chunks 밀집 검색을 기본 근거로 쓰고, 이 종목에 PDF
    리포트가 적재돼 있으면(document_evidence) 하이브리드(BM25+BGE-m3-ko) 검색 근거까지 더해
    GPT로 투자 인사이트를 생성한다."""
    evidence_lines = []

    for hit in get_financial_store().query(query, top_k=3, ticker=ticker) or []:
        evidence_lines.append(f"[financial_chunks] {hit['content']}")

    for hit in get_profile_store().query(query, top_k=2, ticker=ticker) or []:
        evidence_lines.append(f"[company_profile] {hit['content']}")

    used_hybrid = False
    db_url = os.environ.get("SUPABASE_DIRECT_DB_URL")
    if db_url and has_document_evidence(ticker):
        try:
            import entity_fusion

            index = entity_fusion.load_evidence_from_db(db_url, ticker=ticker)
            for hit in entity_fusion.weighted_hybrid_search(index, query, top_k=5):
                evidence_lines.append(
                    f"[{hit['source_type']}(PDF) score={hit['score']:.2f}] {hit['chunk']['content']}"
                )
            used_hybrid = True
        except Exception as e:
            st.caption(f"⚠ PDF 하이브리드 검색을 건너뜁니다: {e}")

    context = "\n\n".join(evidence_lines) if evidence_lines else "관련 근거를 찾지 못했습니다."

    if used_hybrid:
        import citation_check
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        prompt = (
            "다음은 이 기업에 대해 하이브리드 검색(밀집 임베딩+BM25, 소스별 가중치 반영)으로 찾은 "
            "근거입니다. 각 항목 앞 대괄호는 근거 출처를 나타냅니다.\n\n"
            f"[근거]\n{context}\n\n"
            "[작성 지침]\n"
            "- 근거에 등장하는 구체적 수치를 최소 2개 이상 인용할 것\n"
            "- 근거에 없는 내용은 추측하지 말 것\n"
            "- 한국어로, 투자자에게 도움이 되는 간결한 의견으로 답할 것\n\n"
            f"[사용자 질문]\n{query}"
        )
        result = citation_check.generate_with_citation_check(client, prompt, context=context, verbose=False)
        answer = result["answer"]
    else:
        answer = get_generator().generate(query, context)

    return {"answer": answer, "context": context, "used_hybrid": used_hybrid}


# ---------------------------------------------------------------------------
# PDF 업로드 -> 파이프라인 적재
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: Path, pdf_id: str, ticker: str, status) -> dict:
    """업로드된 PDF 한 건을 pdf_pipeline의 텍스트/표 브랜치로 처리해 document_evidence(Supabase)에
    즉시 적재한다. run_investment_opinion_demo.py와 동일한 흐름(스캔본 감지 -> YOLO 페이지 분류 ->
    텍스트/표 브랜치 -> 임베딩 -> 즉시 적재)을 재사용한다. 이미지/차트 브랜치(MinerU)는 이 데모
    환경에 설치돼 있지 않아 생략한다."""
    db_url = os.environ.get("SUPABASE_DIRECT_DB_URL")
    if not db_url:
        raise RuntimeError(
            "SUPABASE_DIRECT_DB_URL 환경변수가 설정되어 있지 않아 PDF를 적재할 수 없습니다."
        )

    from page_classifier import classify_pdf
    from text_extraction import process_pdf
    from scanned_page_router import detect_scanned_pages
    import entity_fusion
    import run_table_metadata_pipeline as rtmp

    counts: dict = {}

    status.write("스캔본 페이지 감지 중...")
    scanned_pages = detect_scanned_pages(pdf_path)
    if scanned_pages:
        counts["scanned_pages"] = scanned_pages

    status.write("페이지 레이아웃 분류 중 (YOLO)...")
    yolo_model = get_yolo_model()
    cls_result = classify_pdf(pdf_path, yolo_model)
    page_boxes = {p["page"]: p["cached_boxes"] for p in cls_result["pages"]}
    counts["n_pages"] = cls_result["n_pages"]

    status.write("텍스트 브랜치 처리 중 (추출 + 청킹 + 임베딩)...")
    text_result = process_pdf(pdf_path, yolo_model, page_boxes=page_boxes,
                               chunk_backend="rulebased", remove_boilerplate=True,
                               add_structured_metadata=False)
    text_chunks = [c for page in text_result["pages"] for c in page["chunks"]]
    text_items, text_emb = entity_fusion.embed_items(entity_fusion.from_text_chunks(pdf_id, text_chunks))
    counts["text_chunks"] = len(text_chunks)
    counts["text_stored"] = entity_fusion.store_evidence(db_url, pdf_id, text_items, text_emb, ticker=ticker)

    status.write("표 브랜치 처리 중 (하이브리드 표 파서 + canonical 매칭)...")
    # [수정] TATR(Table Transformer)은 pdf_pipeline/final/실험_4축_비교_스마트폰.md §14-17에서
    # 15문서 A/B 검증을 거쳐 row_parser.parse_table_hybrid()(pdfplumber text-strategy +
    # word-clustering 게이트)로 대체됐다 — build_records()가 더는 tatr_model/tatr_processor를
    # 받지 않아 get_tatr_model() 호출도 함께 제거(더 빠르고, TATR 체크포인트 로드 자체가 없음).
    rtmp.PDF_PATH = pdf_path
    table_records, _n_finance_filtered, _n_cid = rtmp.build_records(
        pdf_id, page_boxes=page_boxes, yolo_model=yolo_model,
    )
    row_records = [r for r in table_records if r.get("record_type") != "table_metadata"]
    table_items, table_emb = entity_fusion.embed_items(entity_fusion.from_table_records(pdf_id, row_records))
    counts["table_rows"] = len(row_records)
    counts["table_stored"] = entity_fusion.store_evidence(db_url, pdf_id, table_items, table_emb, ticker=ticker)

    status.write("이미지/차트 브랜치는 MinerU 미설치로 생략 (텍스트/표 근거만 적재)")
    counts["image_skipped"] = True

    return counts


# ---------------------------------------------------------------------------
# 화면: 홈
# ---------------------------------------------------------------------------

def render_home():
    title_col, upload_col = st.columns([5, 1])
    with title_col:
        st.title("홈")
    with upload_col:
        st.write("")
        if st.button("+ PDF 업로드", use_container_width=True):
            st.session_state["page"] = "upload"
            st.rerun()
    st.caption("KOSPI200 종목별 재무제표·기업 프로필을 바탕으로 AI가 투자 인사이트를 요약해드립니다.")

    query = st.text_input("종목명 또는 티커 검색", placeholder="예: Samsung, 005930.KS")

    universe = load_ticker_universe()
    if query:
        q = query.strip().lower()
        universe = [u for u in universe if q in u["name"].lower() or q in u["ticker"].lower()]

    st.subheader(f"관심 주식 ({len(universe)}개)")

    cols_per_row = 4
    for i in range(0, len(universe), cols_per_row):
        row = universe[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, row):
            with col, st.container(border=True):
                st.markdown(f"**{item['name']}**")
                st.caption(item["ticker"])
                if st.button("상세보기", key=f"open_{item['ticker']}", use_container_width=True):
                    st.session_state["page"] = "detail"
                    st.session_state["ticker"] = item["ticker"]
                    st.session_state["ticker_name"] = item["name"]
                    st.rerun()


# ---------------------------------------------------------------------------
# 화면: PDF 업로드
# ---------------------------------------------------------------------------

def render_upload():
    if st.button("← 홈으로"):
        st.session_state.pop("upload_result", None)
        st.session_state["page"] = "home"
        st.rerun()

    st.title("PDF 리포트 업로드")
    st.caption(
        "애널리스트 리포트 PDF를 올리면 텍스트/표 근거를 추출해 해당 종목의 문서 근거"
        "(document_evidence)에 즉시 적재합니다. 적재 후에는 종목 상세 화면에서 하이브리드 검색"
        "(BM25+BGE-m3-ko) 기반 질의응답에 이 PDF 근거가 함께 사용됩니다."
    )
    st.info("이미지/차트 근거 추출(MinerU)은 이 데모 환경에 설치되어 있지 않아 생략됩니다 — 텍스트·표 근거만 적재됩니다.")

    uploaded = st.file_uploader("PDF 파일 선택", type=["pdf"])

    universe = load_ticker_universe()
    ticker_options = ["-- 직접 입력 --"] + [f"{u['name']} ({u['ticker']})" for u in universe]
    picked = st.selectbox("연결할 종목", ticker_options)

    if picked == "-- 직접 입력 --":
        col1, col2 = st.columns(2)
        ticker = col1.text_input("티커", placeholder="예: 005930.KS")
        name = col2.text_input("종목명 (선택)", placeholder="예: Samsung")
    else:
        item = universe[ticker_options.index(picked) - 1]
        ticker, name = item["ticker"], item["name"]

    disabled = uploaded is None or not ticker
    if st.button("업로드 및 분석 시작", type="primary", disabled=disabled):
        pdf_id = f"upload_{uuid.uuid4().hex[:8]}"
        tmp_path = Path(tempfile.gettempdir()) / f"{pdf_id}.pdf"
        tmp_path.write_bytes(uploaded.getvalue())

        try:
            with st.status("PDF 분석 중...", expanded=True) as status:
                counts = ingest_pdf(tmp_path, pdf_id, ticker, status)
                status.update(label="분석 완료", state="complete")
        except Exception as e:
            st.error(f"처리 중 오류가 발생했습니다: {e}")
            st.session_state.pop("upload_result", None)
            return
        finally:
            tmp_path.unlink(missing_ok=True)

        has_document_evidence.clear()  # 방금 적재한 근거를 바로 조회할 수 있도록 캐시 무효화
        # st.button()은 클릭이 일어난 바로 그 rerun에서만 True를 반환하므로, 이 결과와 아래
        # "이 종목 질문하러 가기" 버튼을 이 if 블록 안에 그대로 두면 그 버튼을 누르는 순간(=새
        # rerun) 바깥 if가 다시 False가 되어 버튼 자체가 사라져 클릭이 무시된다(실측 확인).
        # session_state에 저장해 다음 rerun에서도 이 블록 밖에서 렌더링되게 한다.
        st.session_state["upload_result"] = {"counts": counts, "ticker": ticker, "name": name}

    result = st.session_state.get("upload_result")
    if result:
        counts = result["counts"]
        st.success(
            f"{counts['n_pages']}페이지 중 텍스트 {counts['text_stored']}건, "
            f"표 {counts['table_stored']}건을 '{result['ticker']}' 근거로 적재했습니다."
        )
        if counts.get("scanned_pages"):
            st.warning(f"스캔본으로 판정된 페이지 {counts['scanned_pages']}는 MinerU 미설치로 텍스트 대체를 건너뛰었습니다.")

        if st.button("이 종목 질문하러 가기 →"):
            st.session_state["page"] = "detail"
            st.session_state["ticker"] = result["ticker"]
            st.session_state["ticker_name"] = result["name"] or result["ticker"]
            del st.session_state["upload_result"]
            st.rerun()


# ---------------------------------------------------------------------------
# 화면: 종목 상세
# ---------------------------------------------------------------------------

def render_detail():
    ticker = st.session_state["ticker"]
    name = st.session_state.get("ticker_name", ticker)

    if st.button("← 홈으로"):
        st.session_state["page"] = "home"
        st.rerun()

    st.title(f"{name} ({ticker})")

    try:
        hist = load_price_history(ticker)
    except Exception:
        hist = None

    if hist is not None and not hist.empty:
        last = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2] if len(hist) > 1 else last
        change = last - prev
        pct = (change / prev * 100) if prev else 0
        st.metric(f"{ticker} 현재가", f"{last:,.2f}", f"{change:+,.2f} ({pct:+.2f}%)")
        st.line_chart(hist["Close"])
    else:
        st.info("가격 데이터를 불러오지 못했습니다.")

    st.divider()
    st.subheader("AI 투자 인사이트 요약")

    with st.spinner("요약 불러오는 중..."):
        summaries = load_summaries(ticker)

    tab_financial, tab_profile = st.tabs(["재무제표 요약", "기업 프로필 요약"])
    with tab_financial:
        if summaries["financial_summary"]:
            st.markdown(summaries["financial_summary"])
        else:
            st.info("이 종목의 재무제표 요약이 아직 없습니다.")
    with tab_profile:
        if summaries["profile_summary"]:
            st.markdown(summaries["profile_summary"])
        else:
            st.info("이 종목의 프로필 요약이 아직 없습니다.")

    st.divider()
    st.subheader("AI에게 질문하기")
    if has_document_evidence(ticker):
        st.caption(
            "이 종목은 애널리스트 리포트(PDF)가 적재되어 있어 텍스트/표/이미지 통합 "
            "하이브리드 검색(BM25+BGE-m3-ko)까지 함께 사용됩니다."
        )

    question = st.text_input("질문", placeholder="예: 이 회사 투자해도 될까?", key="qa_input")
    if st.button("질문하기", type="primary") and question:
        with st.spinner("근거를 찾고 답변을 생성하는 중..."):
            result = answer_question(ticker, question)
        st.markdown(result["answer"])
        with st.expander("근거 보기"):
            st.text(result["context"])


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main():
    st.session_state.setdefault("page", "home")
    _warmup_heavy_models()

    if st.session_state["page"] == "detail" and "ticker" in st.session_state:
        render_detail()
    elif st.session_state["page"] == "upload":
        render_upload()
    else:
        render_home()


if __name__ == "__main__":
    main()

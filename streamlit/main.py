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

Usage:
    streamlit run streamlit/main.py
"""

import os
import re
import sys
from pathlib import Path

import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pdf_pipeline"))

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
# 화면: 홈
# ---------------------------------------------------------------------------

def render_home():
    st.title("홈")
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

    if st.session_state["page"] == "detail" and "ticker" in st.session_state:
        render_detail()
    else:
        render_home()


if __name__ == "__main__":
    main()

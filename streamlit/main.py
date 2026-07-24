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

"PDF + 질문" 화면에서는 애널리스트 PDF와 질문을 함께 입력받아
`pdf_pipeline/run_investment_opinion_demo.main(pdf_path=, pdf_id=, ticker=, query=, sector=)`
(파이프라인_최종정리_핸드오프.md 기준 최신 구현 — 스캔본 감지 -> YOLO 페이지 분류 -> 텍스트/
표/이미지 3브랜치 동시실행 -> 엔티티 합성/Supabase 적재 -> 질의 분해 라우팅 검색
(BM25+BGE-m3-ko/HyDE/MQE) + KOSPI200 기업 DB 매칭(company_entity_linking) 병렬 실행 ->
citation-check 포함 gpt-4.1 생성)에 그대로 위임해, 업로드 즉시 사용자 질문에 맞춘 답변을
보여준다. 이 "공식" 파이프라인 함수를 그대로 재사용하고 로직을 이 파일에 복제하지 않는다
(협업 중 다른 파일 수정과의 충돌/드리프트 방지).

이미지/차트 브랜치는 기본적으로 이 pdf_id에 대해 사전 처리된 MinerU 카드가 없으면 예시 데이터로
대체된다(run_investment_opinion_demo.main() 자체 동작). "실제 이미지/차트 근거 추출" 체크박스를
켜면 image_processing.s2_onestop_mineru(MinerU CLI 원스톱 파서)를 애드훅 업로드용으로 먼저
호출해 실제 카드를 만들어둔 뒤 같은 함수를 호출한다 — 문서 페이지 수에 따라 수 분 걸릴 수 있어
기본은 off, 실패해도 예시 카드로 자연스럽게 계속 진행한다.

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
sys.path.insert(0, str(ROOT / "pdf_pipeline" / "image_processing"))

load_dotenv(ROOT / ".env")

from supabase import create_client  # noqa: E402

from embeddings.gpt_embedder import GPTEmbedder  # noqa: E402
from generation import GPTGenerator  # noqa: E402
from supabase_store import SupabaseVectorStore  # noqa: E402

PROFILE_DIR = ROOT / "KOSPI200_output" / "kospi200_profiles"
TITLE_RE = re.compile(r"^# (.+?) \((.+?)\) 기업 프로필")

st.set_page_config(page_title="포트폴리오", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# 테마 (supaste.com 참고 — 오프화이트 배경 + 인디고 포인트 + 세리프/모노 폰트 조합).
# 순수 CSS 오버레이라 기능 로직에는 영향 없음: .streamlit/config.toml의 라이트 테마 색상 위에
# 폰트·라운드 코너·pill 버튼·카드 그림자만 얹는다.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Fragment+Mono&family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
    /* 한글은 전용 서체가 없는 Bebas Neue/Inter/Fragment Mono 대신 Pretendard로 폴백된다 —
       각 font-family 목록에 'Pretendard'를 2순위로 넣어 영문/숫자는 지정 폰트, 한글은
       Pretendard로 자동 분기(브라우저가 글리프 단위로 폴백 처리). */

    :root {
        /* supaste.com 실측(원본 HTML) 기준 — 페이지는 흰 배경, 카드는 오프화이트 flat fill
           (반대로 페이지를 회색+카드를 흰색으로 하면 흔한 대시보드 톤이 되어버려 supaste
           특유의 "낮은 대비 flat 색면" 느낌이 안 남). */
        --bg: #ffffff;
        --surface: #f7f7f7;
        --surface-2: #ffffff;
        --text: #0d0d0d;
        --muted: #6b7280;
        --border: #e3e3e3;
        --primary: #5f61ed;
        --primary-hover: #4d4fd1;
        --primary-soft: #eef0ff;
        --radius-lg: 28px;
        --radius-md: 14px;
        --radius-pill: 999px;
        --shadow-sm: 0 1px 2px rgba(13,13,13,.04), 0 1px 1px rgba(13,13,13,.03);
        --shadow-md: 0 8px 24px rgba(13,13,13,.06), 0 2px 6px rgba(13,13,13,.04);
    }

    html, body, .stApp { background: var(--bg) !important; font-family: 'Inter', 'Pretendard', -apple-system, sans-serif; }

    h1 { font-family: 'Bebas Neue', 'Pretendard', sans-serif !important; font-weight: 400 !important;
         letter-spacing: 0.02em; color: var(--text) !important; }
    h2, h3 { font-family: 'Inter', 'Pretendard', sans-serif !important; font-weight: 600 !important; color: var(--text) !important; }

    [data-testid="stCaptionContainer"], .stCaption, small {
        font-family: 'Fragment Mono', 'Pretendard', monospace !important;
        color: var(--muted) !important;
        letter-spacing: 0.01em;
    }

    /* 버튼: supaste의 flat 회색 pill(보조) / 검정 pill(주요) 톤을 인디고 기준으로 재현 —
       외곽선 대신 색면 채움 위주(테두리는 hover에서만 강조로 잠깐 등장). */
    .stButton > button, .stDownloadButton > button {
        border-radius: var(--radius-pill) !important;
        border: 1px solid transparent !important;
        background: var(--surface) !important;
        color: var(--text) !important;
        font-weight: 500 !important;
        transition: border-color .15s ease, color .15s ease, background .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--primary) !important;
        color: var(--primary) !important;
    }
    button[kind="primary"] {
        background: var(--primary) !important;
        border-color: var(--primary) !important;
        color: #fff !important;
    }
    button[kind="primary"]:hover {
        background: var(--primary-hover) !important;
        border-color: var(--primary-hover) !important;
        color: #fff !important;
    }

    .stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {
        border-radius: var(--radius-md) !important;
        border-color: var(--border) !important;
        background: var(--surface-2) !important;
    }
    .stTextInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px var(--primary-soft) !important;
    }

    /* 카드: supaste의 카드는 테두리·그림자 없이 flat 색면 + 큰 radius만으로 구분됨 —
       hover에서만 아주 옅은 그림자로 살짝 뜨는 느낌만 추가. */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: var(--radius-lg) !important;
        border: 4px solid var(--border) !important;
        background: var(--surface) !important;
        box-shadow: none;
        transition: box-shadow .15s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover { box-shadow: var(--shadow-sm); }

    [data-testid="stAlert"], [data-testid="stExpander"], [data-testid="stStatusWidget"],
    [data-testid="stFileUploaderDropzone"] {
        border-radius: var(--radius-md) !important;
        border-color: var(--border) !important;
    }

    [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border) !important; }
    [data-baseweb="tab"] { font-family: 'Inter', 'Pretendard', sans-serif !important; font-weight: 500 !important; color: var(--muted) !important; }
    [aria-selected="true"][data-baseweb="tab"] { color: var(--primary) !important; }
    [data-baseweb="tab-highlight"] { background-color: var(--primary) !important; }

    [data-testid="stMetricValue"] { font-family: 'Bebas Neue', 'Pretendard', sans-serif !important; font-size: 2.2rem !important; letter-spacing: 0.02em; }
    [data-testid="stMetricLabel"] { font-family: 'Fragment Mono', 'Pretendard', monospace !important; color: var(--muted) !important; }

    hr { border-color: var(--border) !important; }

    /* ---- supaste 스타일 컴포넌트: 히어로 배지 / 2줄 헤드라인 / 배지 행 / 카드 아바타·태그 / 스텝 라벨 ---- */

    .pp-eyebrow {
        display: inline-block; font-family: 'Fragment Mono', 'Pretendard', monospace; font-size: .7rem;
        letter-spacing: .03em; color: var(--primary); background: var(--primary-soft);
        padding: 5px 12px; border-radius: var(--radius-pill); margin-bottom: 10px;
    }

    /* supaste 히어로/푸터의 시그니처 모티프: 굵은 산세리프 한 줄 + 이탤릭 세리프 한 줄. */
    .pp-hero2 { line-height: 1.08; margin: 2px 0 14px; }
    .pp-hero2-bold {
        display: block; font-family: 'Inter', 'Pretendard', sans-serif; font-weight: 700;
        font-size: 2.3rem; letter-spacing: -0.03em; color: var(--text);
    }
    .pp-hero2-italic {
        display: block; font-family: 'Bebas Neue', 'Pretendard', sans-serif;
        font-weight: 400; font-size: 2.3rem; letter-spacing: 0.02em; color: var(--text);
    }

    .pp-badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 18px; }
    .pp-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--surface); padding: 8px 14px; border-radius: var(--radius-pill);
        font-family: 'Inter', 'Pretendard', sans-serif; font-size: .82rem; font-weight: 500; color: var(--text);
    }
    .pp-badge svg { width: 14px; height: 14px; stroke: var(--primary); flex-shrink: 0; }

    .pp-card-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
    .pp-avatar {
        width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0;
        background: var(--primary-soft); color: var(--primary);
        display: flex; align-items: center; justify-content: center;
        font-family: 'Bebas Neue', 'Pretendard', sans-serif; font-size: 1.2rem; font-weight: 400;
    }
    .pp-ticker-pill {
        font-family: 'Fragment Mono', 'Pretendard', monospace; font-size: .68rem; color: var(--muted);
        background: var(--surface-2); border: 1px solid var(--border);
        padding: 3px 9px; border-radius: var(--radius-pill); white-space: nowrap;
    }
    .pp-ticker-pill-lg { font-size: .85rem; padding: 5px 14px; margin-left: 10px; vertical-align: middle; }
    .pp-card-name {
        /* 압축(말줄임) 대신, KOSPI200 중 가장 긴 종목명(영문 정식 법인명, 최대 3줄 분량)을
           기준으로 min-height를 잡아 모든 카드 높이를 맞춘다 — 어떤 이름도 잘리지 않는다. */
        font-family: 'Inter', 'Pretendard', sans-serif; font-weight: 600; font-size: 1.02rem; color: var(--text);
        white-space: normal; overflow-wrap: anywhere; line-height: 1.3; min-height: 3.9em;
    }

    .pp-hero-title { font-family: 'Bebas Neue', 'Pretendard', sans-serif !important; font-weight: 400 !important;
        letter-spacing: 0.02em; color: var(--text) !important; }

    .pp-step {
        display: flex; align-items: center; gap: 10px; margin: 22px 0 8px;
        font-family: 'Fragment Mono', 'Pretendard', monospace; font-size: .78rem; color: var(--muted); letter-spacing: .02em;
    }
    .pp-step-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 22px; height: 22px; border-radius: 50%; background: var(--primary); color: #fff;
        font-size: .68rem; font-weight: 600; flex-shrink: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _avatar_html(label: str) -> str:
    initial = (label or "?").strip()[0] if (label or "").strip() else "?"
    return f'<div class="pp-avatar">{initial}</div>'


def _step_label(n: int, text: str) -> None:
    st.markdown(
        f'<div class="pp-step"><span class="pp-step-num">{n:02d}</span>{text}</div>',
        unsafe_allow_html=True,
    )


def _hero_headline(bold_line: str, italic_line: str) -> None:
    """supaste.com 히어로/푸터의 시그니처 타이포 모티프(굵은 산세리프 한 줄 + 이탤릭 세리프
    한 줄, 예: "Copy once." / "Reuse anytime.")를 재현한 2줄 헤드라인."""
    st.markdown(
        f'''
        <div class="pp-hero2">
            <span class="pp-hero2-bold">{bold_line}</span>
            <span class="pp-hero2-italic">{italic_line}</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )


_PP_ICONS = {
    "chart": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
             'stroke-linecap="round" stroke-linejoin="round"><path d="M4 19V5M4 19h16M8 15l3-4 3 2 4-6"/></svg>',
    "doc": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
           'stroke-linecap="round" stroke-linejoin="round"><path d="M7 3h7l5 5v13H7z"/>'
           '<path d="M14 3v5h5M9 13h6M9 17h6"/></svg>',
    "sparkle": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
               'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.6 4.9L18.5 9.5l-4.9 1.6L12 16'
               'l-1.6-4.9L5.5 9.5l4.9-1.6z"/></svg>',
    "search": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
              'stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/>'
              '<path d="M21 21l-4.3-4.3"/></svg>',
}


def _badge_row(items: list[tuple[str, str]]) -> None:
    """supaste.com의 pill 배지 행("Local first" 등)을 참고한 아이콘+라벨 배지 나열."""
    pills = "".join(
        f'<span class="pp-badge">{_PP_ICONS.get(icon, "")}{label}</span>' for icon, label in items
    )
    st.markdown(f'<div class="pp-badge-row">{pills}</div>', unsafe_allow_html=True)


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
# PDF + 질문 -> 파이프라인 실행
# ---------------------------------------------------------------------------

def has_real_image_cards(pdf_id: str) -> bool:
    """run_investment_opinion_demo.main()의 이미지 브랜치는 이 pdf_id로 이미 만들어진
    실제 onestop_cards.jsonl(MinerU 산출물)이 있으면 그걸 쓰고, 없으면 대표 예시 카드로
    대체한다(그 함수 자체 문서화된 동작 — 이 파일은 그 로직을 건드리지 않는다)."""
    return (ROOT / "pdf_pipeline" / "data" / "onestop" / pdf_id / "onestop_cards.jsonl").exists()


def extract_real_image_cards(pdf_path: Path, pdf_id: str) -> bool:
    """image_processing.s2_onestop_mineru(MinerU CLI 원스톱 파서)를 애드훅 업로드용으로
    doc_id 사전등록(metadata.csv) 없이 직접 호출해, 이 PDF의 실제 이미지/차트 카드를 만든다.
    성공하면 run_investment_opinion_demo.main()이 그 자리에서 찾는 경로
    (pdf_pipeline/data/onestop/{pdf_id}/onestop_cards.jsonl)에 실제 카드가 생겨, 예시 카드
    대신 이를 근거로 쓴다. 차트를 서술형으로 해석하는 VLM+LLM 단계(with_chart_analysis)는
    CPU 데모 환경에서 문서당 지연이 커 기본 off(캡션·OCR 텍스트만 근거로 적재). 문서 페이지
    수에 따라 수 분 걸릴 수 있어 호출 여부는 UI 체크박스로 사용자가 선택한다. 실패하면
    예외를 그대로 올리지 않고 False만 반환 — 호출측이 예시 카드로 자연스럽게 계속 진행한다."""
    import argparse

    import s2_onestop_mineru

    args = argparse.Namespace(
        doc=pdf_id, pdf_abs=str(pdf_path), lang="korean", timeout_sec=1800,
        with_classifier=True, with_chart_analysis=False,
        chart_max_new_tokens=s2_onestop_mineru.CHART_MAX_NEW_TOKENS,
        narrative_model=s2_onestop_mineru.CFG["LLM_MODEL"],
        with_structured_output=False, force=False,
    )
    s2_onestop_mineru.process(args)
    return has_real_image_cards(pdf_id)


def run_pdf_query(pdf_path: Path, pdf_id: str, ticker: str | None, query: str,
                   sector: str | None = None) -> dict:
    """업로드된 PDF + 사용자 질문을 pdf_pipeline/run_investment_opinion_demo.main()에 그대로
    위임한다 — 스캔본 감지/YOLO 페이지 분류/텍스트·표·이미지 3브랜치 동시실행/엔티티 합성/
    Supabase 적재/질의 분해 라우팅 검색/기업명 정확 매칭(company_entity_linking)/citation-check
    포함 LLM 생성까지 이 한 함수가 전부 처리한다(파이프라인_최종정리_핸드오프.md 기준 최신
    구현). 이 프로젝트의 "공식" 파이프라인을 그대로 재사용해 로직을 여기 복제하지 않는다
    (협업 중 다른 파일이 바뀌어도 이 파일이 계속 최신 구현을 타게 하기 위함)."""
    import run_investment_opinion_demo as demo

    return demo.main(pdf_path=pdf_path, pdf_id=pdf_id, ticker=ticker, query=query, sector=sector)


# ---------------------------------------------------------------------------
# 화면: 홈
# ---------------------------------------------------------------------------

def render_home():
    st.markdown('<span class="pp-eyebrow">AI 리서치 대시보드</span>', unsafe_allow_html=True)
    title_col, upload_col = st.columns([5, 1])
    with title_col:
        _hero_headline("관심 종목,", "인사이트로 완성됩니다.")
    with upload_col:
        st.write("")
        if st.button("+ PDF로 질문하기", use_container_width=True):
            st.session_state["page"] = "upload"
            st.rerun()
    _badge_row([
        ("chart", "실시간 시세"),
        ("doc", "재무제표 요약"),
        ("sparkle", "AI 투자 인사이트"),
        ("search", "PDF 근거 검색"),
    ])
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
                st.markdown(
                    f'''
                    <div class="pp-card-head">
                        {_avatar_html(item["name"])}
                        <span class="pp-ticker-pill">{item["ticker"]}</span>
                    </div>
                    <div class="pp-card-name">{item["name"]}</div>
                    ''',
                    unsafe_allow_html=True,
                )
                if st.button("상세보기", key=f"open_{item['ticker']}", use_container_width=True):
                    st.session_state["page"] = "detail"
                    st.session_state["ticker"] = item["ticker"]
                    st.session_state["ticker_name"] = item["name"]
                    st.rerun()


# ---------------------------------------------------------------------------
# 화면: PDF + 질문
# ---------------------------------------------------------------------------

def render_upload():
    if st.button("← 홈으로"):
        st.session_state.pop("pdf_query_result", None)
        st.session_state["page"] = "home"
        st.rerun()

    st.markdown('<span class="pp-eyebrow">PDF 분석 · 하이브리드 검색</span>', unsafe_allow_html=True)
    _hero_headline("PDF를 올리고,", "질문으로 답을 찾아보세요.")
    st.caption(
        "애널리스트 리포트 PDF와 질문을 함께 올리면, 텍스트/표/이미지 3브랜치를 동시에 처리해 "
        "근거를 적재하고 질의 분해 검색(BM25+BGE-m3-ko/HyDE/MQE) + KOSPI200 기업 DB 매칭까지 "
        "종합한 답변을 바로 보여줍니다."
    )

    with st.container(border=True):
        _step_label(1, "리포트 업로드")
        uploaded = st.file_uploader("PDF 파일 선택", type=["pdf"])

        _step_label(2, "종목 연결 (선택)")
        universe = load_ticker_universe()
        ticker_options = ["-- 직접 입력 --", "지정 안 함 (여러 기업/섹터 리포트)"] + [
            f"{u['name']} ({u['ticker']})" for u in universe
        ]
        picked = st.selectbox("연결할 종목 (선택)", ticker_options)

        col1, col2 = st.columns(2)
        if picked == "-- 직접 입력 --":
            ticker = col1.text_input("티커", placeholder="예: 005930.KS") or None
            name = col2.text_input("종목명 (선택)", placeholder="예: Samsung")
        elif picked == "지정 안 함 (여러 기업/섹터 리포트)":
            ticker, name = None, None
            col1.caption("여러 기업을 다루는 산업 섹터 리포트 등 — 특정 종목에 묶지 않습니다.")
        else:
            item = universe[ticker_options.index(picked) - 2]
            ticker, name = item["ticker"], item["name"]

        _step_label(3, "질문")
        sector = st.text_input("업종/섹터 (선택)", placeholder="예: 건설, 반도체")
        query = st.text_area("질문", placeholder="예: 이 PDF 내용을 바탕으로 투자 의견을 알려줘", height=80)

        _step_label(4, "옵션")
        use_real_images = st.checkbox(
            "실제 이미지/차트 근거 추출 (MinerU) — 문서 페이지 수에 따라 수 분 소요될 수 있습니다"
        )
        if not use_real_images:
            st.info(
                "이미지/차트 근거는 이 PDF에 대해 사전 처리된 MinerU 카드가 없으면 예시 데이터로 "
                "대체됩니다 — 답변에서 이미지 출처가 인용되면 실제 업로드 문서 내용이 아닐 수 있으니 "
                "참고만 하세요(아래 결과에 표시됩니다). 실제 차트를 근거로 쓰려면 위 체크박스를 켜세요."
            )

        disabled = uploaded is None or not query.strip()
        if st.button("분석 시작", type="primary", disabled=disabled):
            pdf_id = f"upload_{uuid.uuid4().hex[:8]}"
            tmp_path = Path(tempfile.gettempdir()) / f"{pdf_id}.pdf"
            tmp_path.write_bytes(uploaded.getvalue())

            try:
                if use_real_images:
                    with st.spinner("MinerU로 이미지/차트 추출 중 (수 분 소요될 수 있습니다)..."):
                        try:
                            ok = extract_real_image_cards(tmp_path, pdf_id)
                        except Exception as e:
                            ok = False
                            st.warning(f"MinerU 이미지 추출에 실패해 예시 데이터로 대체합니다: {e}")
                        if not ok:
                            st.warning("MinerU 카드 생성에 실패해 이미지 근거는 예시 데이터로 대체됩니다.")

                with st.spinner("PDF + 질문 분석 중 (최대 1~2분 소요될 수 있습니다)..."):
                    result = run_pdf_query(tmp_path, pdf_id, ticker, query.strip(), sector=sector or None)
            except Exception as e:
                st.error(f"처리 중 오류가 발생했습니다: {e}")
                st.session_state.pop("pdf_query_result", None)
                return
            finally:
                tmp_path.unlink(missing_ok=True)

            has_document_evidence.clear()  # 방금 적재한 근거를 바로 조회할 수 있도록 캐시 무효화
            # st.button()은 클릭이 일어난 바로 그 rerun에서만 True를 반환하므로, 결과와 아래
            # 후속 버튼을 이 if 블록 안에 그대로 두면 그 버튼을 누르는 순간(=새 rerun) 바깥 if가
            # 다시 False가 되어 버튼 자체가 사라져 클릭이 무시된다(실측 확인). session_state에
            # 저장해 다음 rerun에서도 이 블록 밖에서 렌더링되게 한다.
            st.session_state["pdf_query_result"] = {
                "result": result, "pdf_id": pdf_id, "ticker": ticker, "name": name,
                "used_fallback_images": not has_real_image_cards(pdf_id),
            }

    saved = st.session_state.get("pdf_query_result")
    if saved:
        result = saved["result"]
        st.divider()
        st.subheader("답변")
        st.markdown(result["answer"])

        n_by_source = result.get("n_evidence_by_source", {})
        st.caption(
            f"근거 {sum(n_by_source.values())}건 "
            f"(텍스트 {n_by_source.get('text', 0)} / 표 {n_by_source.get('table', 0)} / "
            f"이미지 {n_by_source.get('image', 0)}) · 문서 내 기업 수 추정 "
            f"{result.get('entity_count', '?')} · 총 처리시간 {result.get('total_time_s', 0):.1f}s"
        )

        if saved["used_fallback_images"]:
            st.warning(
                "이 문서는 실제 이미지/차트 카드가 없어, 이미지 근거는 예시 데이터로 "
                "대체됐습니다 — 아래 답변/근거 중 [image] 출처는 실제 업로드 문서 내용이 "
                "아닐 수 있습니다."
            )

        matched = result.get("matched_companies") or []
        if matched:
            st.success(f"KOSPI200 DB와 매칭된 기업 {len(matched)}건: " + ", ".join(m["name"] for m in matched))

        citation = result.get("citation_result") or {}
        if citation.get("unsupported_numbers"):
            st.warning(f"근거로 확인되지 않은 숫자가 답변에 남아있습니다: {citation['unsupported_numbers']}")

        with st.expander("근거 보기"):
            for h in result.get("hits", []):
                chunk = h["chunk"]
                st.markdown(f"**[{chunk.get('source_type')}] p{chunk.get('page')}** (score={h['score']:.3f})")
                st.text((chunk.get("content") or "")[:500])

        with st.expander("단계별 처리 시간"):
            for step, sec in result.get("timings", {}).items():
                st.text(f"{step:35s} {sec:6.2f}s")

        if saved["ticker"] and st.button("이 종목 상세 화면으로 이동 →"):
            st.session_state["page"] = "detail"
            st.session_state["ticker"] = saved["ticker"]
            st.session_state["ticker_name"] = saved["name"] or saved["ticker"]
            del st.session_state["pdf_query_result"]
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

    st.markdown('<span class="pp-eyebrow">종목 상세</span>', unsafe_allow_html=True)
    st.markdown(
        f'<h1 class="pp-hero-title">{name}<span class="pp-ticker-pill pp-ticker-pill-lg">{ticker}</span></h1>',
        unsafe_allow_html=True,
    )

    try:
        hist = load_price_history(ticker)
    except Exception:
        hist = None

    with st.container(border=True):
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

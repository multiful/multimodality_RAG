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

"PDF + 질문" 화면은 2단계 인덱싱으로 동작한다(사용자 요청: "PDF 올리면 즉시 인덱싱, 질문
버튼 누를 때 인덱싱 시작하는 게 아니고" + "차트 VLM은 느리니 지연을 감당 못 함"):
- **1단계(빠름, 업로드 즉시 블로킹)**: 스캔본 감지 -> YOLO 페이지 분류 -> 텍스트/표 브랜치만
  동시 실행 -> document_evidence 즉시 적재. 사용자는 업로드 직후(질문 버튼 클릭 없이) 바로
  질문할 수 있다.
- **2단계(백그라운드 스레드)**: image_processing.s2_onestop_mineru로 이미지/차트 카드(OCR+분류)
  를 만들어 document_evidence에 추가 적재한다. 1단계를 막지 않는다 — 사용자가 1단계 완료 직후
  질문해도 되고, 2단계가 끝난 뒤 다시 질문하면 이미지 근거까지 반영된 답을 받는다. 차트->표
  VLM(4a/4b)은 켜지 않는다 — 15초+/장로 느리고 4b(서술형 해석)는 이 환경에 없는 Ollama가
  필요해 매번 연결 실패만 반복된다(실측). 대신 run_investment_opinion_demo._normalize_chart_
  card_signs()(정규식/산수 기반, 축 눈금을 시계열로 오독하지 않게 카드에 경고를 자동 삽입)를
  재사용해 VLM 없이도 같은 효과를 공짜로 얻는다.
- 질의 시점(search_and_generate)마다 entity_fusion.load_evidence_from_db()로 그 시점까지
  DB에 있는 근거를 매번 새로 읽으므로, 세션 메모리에 인덱스를 들고 있을 필요 없이 2단계
  완료 여부가 자동으로 다음 질문에 반영된다.

run_investment_opinion_demo.py(공식 파이프라인 데모)는 인덱싱+검색+생성을 한 호출로 묶어놔서
2단계 분리가 불가능해, 이 파일에서 같은 저수준 모듈(page_classifier/text_extraction/
run_table_metadata_pipeline/entity_fusion/index_text/company_entity_linking/citation_check)을
직접 호출해 두 단계로 나눈다 — 로직은 그 데모와 동일하되 이미지 브랜치만 분리했다.

Usage:
    streamlit run streamlit/main.py
"""

import hashlib
import json
import os
import re
import sys
import tempfile
import threading
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
LOGO_PATH = ROOT / "streamlit" / "static" / "logo.png"
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
        font-family: 'Inter', 'Pretendard', sans-serif; font-weight: 700;
        font-size: 1.6rem; letter-spacing: -0.03em; color: var(--text);
    }
    .pp-hero2-italic {
        font-family: 'Bebas Neue', 'Pretendard', sans-serif;
        font-weight: 400; font-size: 1.6rem; letter-spacing: 0.02em; color: var(--text);
    }

    .pp-badge-row { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin: 4px 0 18px; }
    .pp-center { text-align: center; }
    .pp-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--surface); padding: 8px 14px; border-radius: var(--radius-pill);
        font-family: 'Inter', 'Pretendard', sans-serif; font-size: .82rem; font-weight: 500; color: var(--text);
    }
    .pp-badge svg { width: 14px; height: 14px; stroke: var(--primary); flex-shrink: 0; }

    .pp-card-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
    .pp-price-badge {
        min-width: 36px; height: 30px; padding: 0 10px;
        flex-shrink: 0; color: var(--text);
        display: flex; align-items: center; justify-content: center; white-space: nowrap;
        font-family: 'Fragment Mono', 'Pretendard', monospace; font-size: .82rem; font-weight: 600;
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
        text-align: center;
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

    .pp-logo-wrap { text-align: center; }
    .pp-logo { display: inline-block; height: 52px; width: auto; margin: 4px 0 16px; }

    .pp-subtitle {
        font-family: 'Inter', 'Pretendard', sans-serif; font-weight: 500;
        font-size: 1.15rem; color: var(--muted); line-height: 1.5; margin: 2px 0 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def _load_logo_data_uri() -> str | None:
    """streamlit/static/logo.png를 base64 data URI로 읽어온다 — Streamlit의 정적 파일
    서빙(enableStaticServing) 설정 없이도 어디서든 <img>로 바로 쓸 수 있다."""
    if not LOGO_PATH.exists():
        return None
    import base64
    return "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode()


def _render_logo() -> None:
    uri = _load_logo_data_uri()
    if uri:
        st.markdown(
            f'<div class="pp-logo-wrap"><img src="{uri}" class="pp-logo" alt="logo"></div>',
            unsafe_allow_html=True,
        )


def _price_badge_html(price: float | None) -> str:
    text = f"₩{price:,.0f}" if price is not None else "-"
    return f'<div class="pp-price-badge">{text}</div>'


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
            <span class="pp-hero2-bold">{bold_line}</span> <span class="pp-hero2-italic">{italic_line}</span>
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


def _badge_pills_html(items: list[tuple[str, str]]) -> str:
    return "".join(
        f'<span class="pp-badge">{_PP_ICONS.get(icon, "")}{label}</span>' for icon, label in items
    )


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


@st.cache_resource
def warm_qwen_ollama() -> bool:
    """image_processing.s2_onestop_mineru의 4b(서술형 해석)가 쓰는 qwen3:8b(Ollama, localhost
    :11434)를 더미 요청으로 한 번 미리 불러둔다. Ollama는 안 쓰면 몇 분 뒤 모델을 메모리에서
    내리므로, 첫 실제 이미지 백그라운드 처리 때 로딩 지연이 그대로 드러나는 걸 막는다(YOLO/
    BGE-m3-ko와 동일한 이유). 실패해도(Ollama 데몬이 아직 안 떠 있는 등) 예외를 삼키고
    False만 반환 — 워밍업은 최적화일 뿐 필수 경로가 아니라 앱을 막아선 안 된다."""
    try:
        import requests

        requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "qwen3:8b", "messages": [{"role": "user", "content": "ping"}], "stream": False},
            timeout=60,
        )
        return True
    except Exception:
        return False


_WARMUP_STARTED = False


def _warmup_heavy_models():
    """PDF 업로드 파이프라인이 쓰는 무거운 모델(YOLO/BGE-m3-ko/qwen3:8b)을 앱 프로세스 시작 시
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
    threading.Thread(target=warm_qwen_ollama, daemon=True).start()
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


@st.cache_data(ttl=300)
def load_current_prices(tickers: tuple[str, ...]) -> dict[str, float | None]:
    """관심 주식 카드의 가격 배지에 쓸 현재가 일괄 조회. 카드 개수(최대 199개)만큼 yfinance를
    종목별로 따로 호출하면 홈 화면이 수십 초씩 멈춘 것처럼 보이므로, yf.download()로 한 번에
    배치 조회한다(5분 캐시)."""
    if not tickers:
        return {}
    import pandas as pd

    data = yf.download(list(tickers), period="1d", progress=False, group_by="ticker", threads=True)
    is_multi = isinstance(data.columns, pd.MultiIndex)
    prices: dict[str, float | None] = {}
    for t in tickers:
        try:
            close = (data[t]["Close"] if is_multi else data["Close"]).dropna()
            prices[t] = float(close.iloc[-1]) if not close.empty else None
        except Exception:
            prices[t] = None
    return prices


# ---------------------------------------------------------------------------
# RAG 질의응답
# ---------------------------------------------------------------------------

def answer_question(ticker: str, query: str) -> dict:
    """financial_chunks/company_profile_chunks 밀집 검색을 기본 근거로 쓰고, 이 종목에 PDF
    리포트가 적재돼 있으면(document_evidence) 하이브리드(BM25+BGE-m3-ko) 검색 근거까지 더해
    GPT로 투자 인사이트를 생성한다.

    [핸드오프 남은과제 8] 세 검색이 완전 순차라 지연이 sum이던 것을 ThreadPoolExecutor로
    병렬화(max) — 인제스트 쪽 index_pdf_fast()의 텍스트/표 동시 실행과 같은 패턴. 세 검색은
    서로의 결과를 읽지 않아 독립적이다. 주의: Streamlit API(st.cache_resource 스토어 획득,
    st.caption)는 워커 스레드에서 부르면 ScriptRunContext 경고/오류가 나므로 스토어 핸들과
    has_document_evidence() 판정은 메인 스레드에서 먼저 끝내고, 워커는 순수 검색 호출만 한다."""
    from concurrent.futures import ThreadPoolExecutor

    financial_store = get_financial_store()
    profile_store = get_profile_store()
    supabase_client = get_supabase_client()
    db_url = os.environ.get("SUPABASE_DIRECT_DB_URL")
    run_hybrid = bool(db_url) and has_document_evidence(ticker)

    def _hybrid_hits():
        import entity_fusion
        index = entity_fusion.load_evidence_from_db(db_url, ticker=ticker)
        return entity_fusion.weighted_hybrid_search(index, query, top_k=5)

    def _dividend_lines():
        # [배당 스코어링 배선] dividend_scores(적재·요약문 완비)를 소비하는 코드가 없어 LLM에
        # 도달하지 못하던 것을 연결. v2(고도화 산식) 최신 사업연도 요약문 1건 — 티커 정확
        # 조회라 검색 오류 여지 없음. 실패/데이터 없음이면 빈 리스트(보조 신호).
        try:
            resp = (
                supabase_client.table("dividend_scores")
                .select("content")
                .eq("ticker", ticker)
                .eq("score_version", "v2")
                .order("fiscal_year", desc=True)
                .limit(1)
                .execute()
            )
            return [r["content"] for r in (resp.data or []) if r.get("content")]
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=4) as ex:
        f_fin = ex.submit(lambda: financial_store.query(query, top_k=3, ticker=ticker) or [])
        f_prof = ex.submit(lambda: profile_store.query(query, top_k=2, ticker=ticker) or [])
        f_div = ex.submit(_dividend_lines)
        f_hyb = ex.submit(_hybrid_hits) if run_hybrid else None

        evidence_lines = []
        for hit in f_fin.result():
            evidence_lines.append(f"[financial_chunks] {hit['content']}")
        for hit in f_prof.result():
            evidence_lines.append(f"[company_profile] {hit['content']}")
        for line in f_div.result():
            evidence_lines.append(f"[dividend_scores] {line}")

        used_hybrid = False
        if f_hyb is not None:
            try:
                for hit in f_hyb.result():
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

def index_pdf_fast(pdf_path: Path, pdf_id: str, ticker: str | None, sector: str | None, status) -> dict:
    """1단계(빠름, 업로드 즉시 블로킹) — 스캔본 감지 -> YOLO 페이지 분류 -> 텍스트/표 브랜치만
    동시 실행해 document_evidence에 즉시 적재한다. 이미지/차트 VLM 분석(15초+/장)은 여기 포함
    하지 않는다 — run_investment_opinion_demo.main()처럼 3브랜치를 한꺼번에 동시실행하면 가장
    느린 이미지 브랜치가 끝날 때까지 텍스트/표도 발이 묶여, "업로드 즉시 질문 가능"이 안 된다.
    이미지는 index_images_background()가 별도 스레드로 뒤이어 처리한다."""
    db_url = os.environ.get("SUPABASE_DIRECT_DB_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DIRECT_DB_URL 환경변수가 설정되어 있지 않습니다.")

    from concurrent.futures import ThreadPoolExecutor

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

    def _text_branch():
        text_result = process_pdf(pdf_path, yolo_model, page_boxes=page_boxes,
                                   chunk_backend="rulebased", remove_boilerplate=True, sector=sector)
        text_chunks = [c for page in text_result["pages"] for c in page["chunks"]]
        text_items, text_emb = entity_fusion.embed_items(entity_fusion.from_text_chunks(pdf_id, text_chunks))
        n = entity_fusion.store_evidence(db_url, pdf_id, text_items, text_emb, ticker=ticker)
        return len(text_chunks), n

    def _table_branch():
        rtmp.PDF_PATH = pdf_path
        table_records, _n_finance_filtered, _n_cid = rtmp.build_records(
            pdf_id, page_boxes=page_boxes, yolo_model=yolo_model, sector=sector)
        row_records = [r for r in table_records if r.get("record_type") != "table_metadata"]
        table_items, table_emb = entity_fusion.embed_items(entity_fusion.from_table_records(pdf_id, row_records))
        n = entity_fusion.store_evidence(db_url, pdf_id, table_items, table_emb, ticker=ticker)
        return len(row_records), n

    status.write("텍스트 + 표 브랜치 동시 처리 중...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_text = ex.submit(_text_branch)
        f_table = ex.submit(_table_branch)
        counts["text_chunks"], counts["text_stored"] = f_text.result()
        counts["table_rows"], counts["table_stored"] = f_table.result()

    status.write(f"완료 — 텍스트 {counts['text_stored']}건 · 표 {counts['table_stored']}건 즉시 적재")
    return counts


_IMAGE_STAGE_LOCK = threading.Lock()


def _set_image_stage(pdf_id: str, state: str, message: str) -> None:
    with _IMAGE_STAGE_LOCK:
        st.session_state.setdefault("image_stage", {})[pdf_id] = {"state": state, "message": message}


def index_images_background(pdf_path: Path, pdf_id: str, ticker: str | None, sector: str | None) -> None:
    """2단계(백그라운드 스레드) — image_processing.s2_onestop_mineru로 이미지/차트 카드를
    만들고(OCR + 분류기), document_evidence에 이미지 근거를 추가 적재한다. 1단계(텍스트/표)
    완료 후 별도 스레드에서 돌려 사용자 질문을 막지 않는다.

    [수정] with_chart_analysis=True — Ollama(qwen3:8b)가 이 머신에 설치·확인됐으므로(로컬
    HTTP API, localhost:11434) 4a(MinerU2.5-Pro VLM, 차트→표) + 4b(qwen3:8b, 서술형 해석)를
    실제로 켠다. 4a가 성공해도 build_embed_text()는 narrative만 담고 chart_table 자체는 안
    담기 때문에(s2_onestop_mineru.py 확인) 4b(Ollama)가 없으면 4a 비용(15초+/장)이 그냥
    버려진다 — 그래서 Ollama 없이는 이 옵션을 켜봐야 소용없었다. 그와 별개로
    run_investment_opinion_demo._normalize_chart_card_signs()(팀 최신 수정 — 부호/OCR손상
    정규화 + "축 눈금을 시계열로 오독하지 말 것" 경고, chart_table/narrative가 이미 있으면
    경고를 안 붙임)는 VLM 성패와 무관하게 항상 적용해 이중 안전장치로 둔다.
    완료되면 entity_fusion.invalidate_evidence_cache()로 캐시를 지워, 다음 질문부터 즉시
    반영되게 한다. 실패해도 예외를 삼키고 상태만 기록 — 백그라운드 스레드의 예외는 Streamlit
    UI로 안 올라간다."""
    _set_image_stage(pdf_id, "running", "이미지/차트 근거(MinerU VLM + qwen3:8b) 백그라운드 처리 중...")
    try:
        import argparse

        import s2_onestop_mineru
        import entity_fusion
        from run_investment_opinion_demo import _normalize_chart_card_signs

        args = argparse.Namespace(
            doc=pdf_id, pdf_abs=str(pdf_path), lang="korean", timeout_sec=1800,
            with_classifier=True, with_chart_analysis=True,
            chart_max_new_tokens=s2_onestop_mineru.CHART_MAX_NEW_TOKENS,
            narrative_model=s2_onestop_mineru.CFG["LLM_MODEL"],
            with_structured_output=False, force=False,
        )
        s2_onestop_mineru.process(args)

        cards_path = ROOT / "pdf_pipeline" / "data" / "onestop" / pdf_id / "onestop_cards.jsonl"
        if not cards_path.exists():
            _set_image_stage(pdf_id, "failed", "MinerU 카드 생성 실패 — 텍스트/표 근거만 유지됩니다.")
            return

        cards = [json.loads(line) for line in cards_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        cards = _normalize_chart_card_signs(cards, pdf_path=pdf_path)
        image_items, image_emb = entity_fusion.embed_items(entity_fusion.from_image_cards(pdf_id, cards))
        db_url = os.environ["SUPABASE_DIRECT_DB_URL"]
        n = entity_fusion.store_evidence(db_url, pdf_id, image_items, image_emb, ticker=ticker)
        entity_fusion.invalidate_evidence_cache(pdf_id=pdf_id)
        _set_image_stage(pdf_id, "done", f"이미지/차트 근거 {n}건 추가 적재 완료 (MinerU VLM + qwen3:8b 서술형 해석)")
    except Exception as e:
        _set_image_stage(pdf_id, "failed", f"이미지/차트 근거 처리 실패: {e}")


def search_and_generate(pdf_path: Path, pdf_id: str, ticker: str | None, query: str) -> dict:
    """질의 시점에 document_evidence를 다시 읽어(entity_fusion.load_evidence_from_db) 그 순간
    까지 적재된 근거(텍스트/표는 항상 있음, 이미지는 2단계가 끝났으면 포함)로 질의 분해 라우팅
    검색 + 기업 DB 매칭(병렬) + citation-check 포함 LLM 생성을 수행한다. 세션에 인덱스를 들고
    있지 않고 매번 DB에서 새로 읽으므로, 2단계(이미지) 완료 여부가 재인덱싱 없이 자동 반영된다
    (run_investment_opinion_demo.main()의 6b~8단계와 동일 로직)."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI
    import entity_fusion
    from index_text import decompose_and_route_search, precompute_entity_count
    import company_entity_linking
    import citation_check

    db_url = os.environ["SUPABASE_DIRECT_DB_URL"]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    timings = {}

    t0 = time.perf_counter()
    index = entity_fusion.load_evidence_from_db(db_url, pdf_id=pdf_id, ticker=ticker)
    precompute_entity_count(index, pdf_path=pdf_path, client=client)
    timings["load_index_and_entity_count"] = time.perf_counter() - t0

    def _search():
        return decompose_and_route_search(index, query, client=client, top_k=8)

    def _link():
        import fitz
        doc = fitz.open(str(pdf_path))
        full_text = "\n".join(doc[i].get_text() for i in range(doc.page_count))
        doc.close()
        matched = company_entity_linking.find_mentioned_companies(full_text)
        db_context = company_entity_linking.fetch_company_db_context(db_url, matched)
        return matched, db_context

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_search = ex.submit(_search)
        f_link = ex.submit(_link)
        hits, subqueries = f_search.result()
        matched_companies, company_db_context = f_link.result()
    timings["search_and_entity_link"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    evidence_context = "\n\n".join(
        f"[{h['chunk'].get('source_type')} / p{h['chunk'].get('page')}] {h['chunk']['content']}" for h in hits
    )
    full_context = evidence_context
    if company_db_context:
        full_context += "\n\n=== 기업 DB 참고 정보(PDF에 언급된 기업을 KOSPI200 DB와 매칭) ===\n\n" + company_db_context

    prompt = f"""다음은 한 기업 리포트 PDF에서 텍스트/표/이미지(차트) 세 소스를 통합해 찾은 근거와,
그 PDF에 언급된 기업을 KOSPI200 DB(재무제표/기업프로필 요약)와 매칭해 가져온 보충 정보입니다.
각 항목 앞의 [text/table/image]는 어느 브랜치에서 나온 근거인지, "기업 DB 참고 정보" 구간은
DB에서 직접 조회한 정보임을 나타냅니다.

[통합 근거]
{full_context}

[작성 지침]
- 반드시 위 근거에 등장하는 구체적 수치를 최소 3개 이상 인용할 것. 수치 없는 뭉뚱그린 서술만으로
  결론짓지 말 것.
- 가능하면 text/table/image 여러 소스의 근거를 섞어서 활용할 것(한 소스에만 의존하지 말 것).
- "기업 DB 참고 정보"가 있으면 PDF 근거와 종합해서 활용하되, PDF에 없는 DB만의 수치를 인용할
  땐 출처가 DB임을 명시할 것.
- 긍정적 근거와 부정적/유의할 근거를 모두 찾아 균형 있게 제시할 것.
- 위 근거에 없는 내용은 추측하지 말 것.
- [image] 소스는 차트를 OCR로 읽은 원문이라 수치 앞뒤에 부호가 명시적으로 안 붙어 있을 수 있다.
  한국 증권 리포트 관례상 "값(N)"처럼 괄호로 감싼 숫자는 음수(하락/손실), 괄호 없는 숫자는
  양수(상승/이익)를 뜻한다. 축 눈금(예: "(38)(34)(30)...")은 데이터가 아니라 눈금선이므로 특정
  대상(기업명 등) 없이 나열된 괄호 숫자는 값으로 쓰지 말 것 — 차트의 Y축 눈금 목록과 X축 날짜
  목록이 개수가 같다고 순서대로 1:1로 짝지어 시계열 값으로 쓰지 말 것(둘 다 축 눈금일 뿐 실제
  데이터가 아니다). 근거에 날짜별 정확한 표(예: "날짜 | 목표주가" 행)가 있으면 그 표를 우선
  하고, 차트만 있고 정확한 값을 알 수 없으면 그렇다고 명시할 것.
- 사용자 질문이 인사이트/투자의견/투자 판단을 묻는 경우, 답변 마지막에 반드시 아래 형식으로
  한 줄 요약 추천을 덧붙일 것:
  "종합해봤을 때 [N]%로 [BUY/HOLD/SELL]을 추천합니다."
  N(0~100 정수)은 위에서 제시한 긍정/부정 근거의 비중을 스스로 판단한 확신도이며, 앞선 서술과
  논리적으로 일치해야 한다. 질문이 특정 수치·날짜를 묻는 단순 사실 확인이라면 이 요약 문장은
  생략할 것.

[사용자 요청]
{query}
"""
    result = citation_check.generate_with_citation_check(
        client, prompt, context=full_context, model="gpt-4.1", max_retries=1)
    timings["llm_generation"] = time.perf_counter() - t0

    by_source: dict = {}
    for c in index.chunks:
        by_source[c.get("source_type")] = by_source.get(c.get("source_type"), 0) + 1

    return {
        "answer": result["answer"], "hits": hits, "subqueries": subqueries,
        "matched_companies": matched_companies, "company_db_context": company_db_context,
        "entity_count": index.entity_count, "n_evidence_by_source": by_source,
        "citation_result": result, "timings": timings,
        "total_time_s": sum(timings.values()),
    }


# ---------------------------------------------------------------------------
# 화면: 홈
# ---------------------------------------------------------------------------

def render_home():
    badges = _badge_pills_html([
        ("chart", "실시간 시세"),
        ("doc", "재무제표 요약"),
        ("sparkle", "AI 투자 인사이트"),
        ("search", "PDF 근거 검색"),
    ])
    # 로고(_render_logo, main()에서 렌더)부터 이 배지 행까지가 사용자가 지정한 "가운데 정렬"
    # 대상 — 한 번의 st.markdown 안에 묶어야 실제로 같은 부모에 중첩돼 text-align이 먹는다
    # (별개의 st.markdown 호출로 나누면 각자 독립된 컨테이너라 열고 닫는 태그가 안 이어짐).
    st.markdown(
        f'''
        <div class="pp-center">
            <div class="pp-subtitle">관심 종목, 인사이트로 완성됩니다.</div>
            <div class="pp-badge-row">{badges}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    if st.button("+ PDF로 질문하기", use_container_width=True):
        st.session_state["page"] = "upload"
        st.rerun()
    st.markdown(
        '<div class="pp-center"><small>KOSPI200 종목별 재무제표·기업 프로필을 바탕으로 '
        'AI가 투자 인사이트를 요약해드립니다.</small></div>',
        unsafe_allow_html=True,
    )

    query = st.text_input("종목명 또는 티커 검색", placeholder="예: Samsung, 005930.KS")

    universe = load_ticker_universe()
    if query:
        q = query.strip().lower()
        universe = [u for u in universe if q in u["name"].lower() or q in u["ticker"].lower()]

    st.subheader(f"관심 주식 ({len(universe)}개)")

    with st.spinner("현재가 불러오는 중..."):
        prices = load_current_prices(tuple(u["ticker"] for u in universe))

    cols_per_row = 4
    for i in range(0, len(universe), cols_per_row):
        row = universe[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, row):
            with col, st.container(border=True):
                st.markdown(
                    f'''
                    <div class="pp-card-head">
                        {_price_badge_html(prices.get(item["ticker"]))}
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
    st.markdown('<div class="pp-center"><span class="pp-eyebrow">PDF 분석 · 하이브리드 검색</span></div>', unsafe_allow_html=True)

    if st.button("← 홈으로"):
        old = st.session_state.get("indexed_doc")
        if old and old.get("pdf_path"):
            Path(old["pdf_path"]).unlink(missing_ok=True)
        st.session_state.pop("indexed_doc", None)
        st.session_state.pop("qa_result", None)
        st.session_state["page"] = "home"
        st.rerun()

    _hero_headline("PDF를 올리면,", "바로 인덱싱을 시작합니다.")
    st.caption(
        "PDF를 올리는 즉시 텍스트/표 근거를 인덱싱해 바로 질문할 수 있습니다. 이미지/차트 근거는 "
        "백그라운드에서 따로 처리되고, 완료되면 다음 질문부터 자동으로 반영됩니다."
    )

    with st.container(border=True):
        _step_label(1, "종목 연결 (선택)")
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

        sector = st.text_input("업종/섹터 (선택)", placeholder="예: 건설, 반도체")
        use_real_images = st.checkbox(
            "이미지/차트 근거도 백그라운드에서 처리 (MinerU VLM + qwen3:8b — 텍스트/표 질문은 막지 않음)",
            value=True,
        )

        _step_label(2, "리포트 업로드 (올리는 즉시 텍스트·표 인덱싱 시작)")
        uploaded = st.file_uploader("PDF 파일 선택", type=["pdf"])

        if uploaded is not None:
            doc_key = hashlib.md5(uploaded.getvalue()).hexdigest()
            indexed = st.session_state.get("indexed_doc")
            if not indexed or indexed.get("doc_key") != doc_key:
                if indexed and indexed.get("pdf_path"):
                    Path(indexed["pdf_path"]).unlink(missing_ok=True)  # 이전 업로드 임시파일 정리

                pdf_id = f"upload_{uuid.uuid4().hex[:8]}"
                tmp_path = Path(tempfile.gettempdir()) / f"{pdf_id}.pdf"
                tmp_path.write_bytes(uploaded.getvalue())

                try:
                    with st.status(f"'{uploaded.name}' 인덱싱 중 (텍스트·표)...", expanded=True) as status:
                        counts = index_pdf_fast(tmp_path, pdf_id, ticker, sector or None, status)
                        status.update(label="텍스트·표 인덱싱 완료 — 바로 질문 가능", state="complete")
                except Exception as e:
                    st.error(f"인덱싱 중 오류가 발생했습니다: {e}")
                    return

                has_document_evidence.clear()
                st.session_state["indexed_doc"] = {
                    "doc_key": doc_key, "pdf_id": pdf_id, "pdf_path": str(tmp_path),
                    "ticker": ticker, "name": name, "counts": counts,
                }
                st.session_state.pop("qa_result", None)

                if use_real_images:
                    threading.Thread(
                        target=index_images_background,
                        args=(tmp_path, pdf_id, ticker, sector or None),
                        daemon=True,
                    ).start()

    indexed = st.session_state.get("indexed_doc")
    if indexed:
        counts = indexed["counts"]
        st.success(
            f"텍스트 {counts['text_stored']}건, 표 {counts['table_stored']}건 즉시 적재 완료 "
            f"({counts['n_pages']}페이지) — 아래에서 바로 질문하세요."
        )
        if counts.get("scanned_pages"):
            st.warning(f"스캔본으로 판정된 페이지 {counts['scanned_pages']}는 텍스트 품질이 낮을 수 있습니다.")

        img_status = st.session_state.get("image_stage", {}).get(indexed["pdf_id"])
        if img_status:
            if img_status["state"] == "running":
                st.info(f"⏳ {img_status['message']} (다른 입력을 하면 진행 상황이 갱신됩니다)")
            elif img_status["state"] == "done":
                st.success(f"✅ {img_status['message']}")
            elif img_status["state"] == "failed":
                st.warning(f"⚠ {img_status['message']}")

        st.divider()
        query = st.text_area("질문", placeholder="예: 이 PDF 내용을 바탕으로 투자 의견을 알려줘", height=80)
        if st.button("질문하기", type="primary", disabled=not query.strip()):
            with st.spinner("검색 + 답변 생성 중..."):
                try:
                    result = search_and_generate(
                        Path(indexed["pdf_path"]), indexed["pdf_id"], indexed["ticker"], query.strip())
                except Exception as e:
                    st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
                    result = None
            if result:
                st.session_state["qa_result"] = {
                    "result": result, "ticker": indexed["ticker"], "name": indexed["name"],
                }

    qa = st.session_state.get("qa_result")
    if qa:
        result = qa["result"]
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

        if qa["ticker"] and st.button("이 종목 상세 화면으로 이동 →"):
            st.session_state["page"] = "detail"
            st.session_state["ticker"] = qa["ticker"]
            st.session_state["ticker_name"] = qa["name"] or qa["ticker"]
            del st.session_state["qa_result"]
            st.rerun()


# ---------------------------------------------------------------------------
# 화면: 종목 상세
# ---------------------------------------------------------------------------

def render_detail():
    ticker = st.session_state["ticker"]
    name = st.session_state.get("ticker_name", ticker)

    st.markdown('<div class="pp-center"><span class="pp-eyebrow">종목 상세</span></div>', unsafe_allow_html=True)

    if st.button("← 홈으로"):
        st.session_state["page"] = "home"
        st.rerun()
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
            st.metric(f"{ticker} 현재가", f"₩{last:,.2f}", f"{change:+,.2f} ({pct:+.2f}%)")
            st.line_chart(hist["Close"])
        else:
            st.info("가격 데이터를 불러오지 못했습니다.")

    st.divider()
    st.subheader("AI 투자 인사이트 요약")

    with st.spinner("요약 불러오는 중..."):
        summaries = load_summaries(ticker)

    st.markdown("##### 재무제표 요약")
    if summaries["financial_summary"]:
        st.markdown(summaries["financial_summary"])
    else:
        st.info("이 종목의 재무제표 요약이 아직 없습니다.")

    st.markdown("##### 기업 프로필 요약")
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
    _render_logo()

    if st.session_state["page"] == "detail" and "ticker" in st.session_state:
        render_detail()
    elif st.session_state["page"] == "upload":
        render_upload()
    else:
        render_home()


if __name__ == "__main__":
    main()

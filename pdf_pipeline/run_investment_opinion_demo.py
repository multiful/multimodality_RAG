"""LGCNS PDF 한 건으로 ERD 전체 흐름을 실행: 스캔본 페이지 감지(MinerU 전용 처리) -> 텍스트/
테이블/이미지 세 브랜치 -> 엔티티 합성(가중치 정제) -> 통합 Supabase 테이블 적재 -> (문서 근거
검색 + 기업 DB 매칭)을 병렬로 -> citation-check 포함 LLM 투자의견 생성까지 한 번에 돈다.

[수정 이력]
1) 검색 단계를 entity_fusion.weighted_hybrid_search()(항상 고정 dense+BM25 가중합)에서
   index_text.route_search()로 교체 — 4문서 A/B(파이프라인_최종정리_핸드오프.md §5)에서
   route_search(entity_aware=True)가 ndcg 0.848(4문서 평균)로 dense-only(0.554)/고정 hybrid
   (0.428~0.470)/구 라우팅(0.754) 전부보다 높게 나온, 검증상 최고 성능 경로.
2) 그런데 route_search 단일 분류로는 복합 질의("인사이트 도출해주고(추상) + 건설업종 종목
   주간 수익률 최고 기업 추출(키워드형)")를 통째로 abstract로 오분류해 BM25를 다 버리고, 정밀
   키워드 매칭이 필요한 절의 정답 근거가 밀려나는 걸 실측으로 확인(도표3 건설업종 vs 도표4
   건자재업종 역전). index_text.decompose_and_route_search()로 재교체 — 질의를 성격별
   하위질의로 먼저 쪼갠 뒤 각자 맞는 전략(keyword_specific=hybrid RRF, abstract=HyDE/MQE)으로
   따로 검색해 라운드로빈으로 합친다(단일 성격 질의는 route_search와 동일하게 동작).
3) "기업명 및 티커"(ERD) — 사용자 지적: 사용자가 티커를 먼저 고르는 게 아니라, PDF 근거에
   DB(KOSPI200)가 아는 기업이 언급됐는지 파이프라인이 스스로 찾아 연결해야 함. dense(임베딩)
   매칭은 부정확해서(company_profile_chunks가 영문 위주라 한글 질의와 안 붙음, 4건 중 1건만
   히트) `company_entity_linking.py`로 교체 — pykrx 한글명 199개와 PDF 근거 텍스트를 정확
   문자열 매칭(이름→티커는 조회 문제지 의미 유사도 문제가 아님). 이 매칭+DB 조회(financial_
   summaries/company_profile_chunks)를 문서 근거 검색과 **병렬로**(ThreadPoolExecutor) 실행해
   둘 다 최종 프롬프트에 합쳐 넣는다.
4) 최종 생성 모델을 gpt-4o-mini에서 gpt-4.1(컨텍스트 ~1M, 더 강한 추론)로 교체.

[40] 사용자 지적("Entity Fusion sync barrier" — 세 브랜치를 다 모은 뒤 한 번에 DB 적재하면, 가장
느린 브랜치(이미지/VLM, 문서당 최대 152초+)가 끝날 때까지 몇 초면 끝나는 텍스트/테이블 결과까지
적재가 막힘) 반영 — 브랜치가 끝나는 즉시 그 브랜치분만 임베딩+Supabase 적재하도록 분리했다.
"엔티티 합성"은 이제 사전에 메모리에서 셋을 합친 뒤 쓰는 게 아니라, 같은 pdf_id/ticker로 태그된
채 같은 document_evidence 테이블에 각자 도착하는 것 자체가 합성 지점이다(entity_fusion.py 참고).
통합 하이브리드 인덱스는 각 브랜치에서 이미 계산한 임베딩을 재사용해서 만들어 재임베딩 비용도 없앴다.

[수정] [40]/[51]에서 검증된 "브랜치는 서로 독립적" 전제([49] RQ 큐, [51] 인덱싱/질의 준비
오버랩과 같은 계열의 발견 — test_concurrent_upload_query.py 등에서 실측)가 이 데모 자체에는
안 배선돼 있었음: 텍스트/테이블/이미지 3브랜치가 완전 순차 실행이라 총 대기시간이
max(text,table,image)가 아니라 sum(...)이었다(이미지 브랜치 최대 152초+가 그대로 더해짐).
세 브랜치 모두 자기 pdf_id/ticker로 독립적으로 적재하고(각자 store_evidence 호출), 서로의
결과를 읽지 않으며, YOLO는 1)에서 이미 전 페이지 caching이 끝난 `page_boxes`를 쓰므로(각
브랜치 내부에서 재추론 없음) 동시 실행해도 안전 — ThreadPoolExecutor로 실제 동시 실행하도록
배선한다. 표 브랜치는 `row_parser.parse_table_hybrid()`([JAEIL v5, 15문서 A/B 채택])로 TATR을
대체해 표 단계 자체도 더 빨라졌다.

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
    (다른 PDF/쿼리로: main(pdf_path=..., pdf_id=..., ticker=..., query=...) 직접 호출 —
    [수정] 팀원 인계를 위해 LGCNS 하드코딩을 걷어내고 main()이 인자를 받도록 파라미터화.
    인자를 안 주면 기존과 동일하게 LGCNS 기본값으로 동작(하위호환). ticker=None이면(예: 여러
    기업을 다루는 산업 섹터 리포트) document_evidence에 ticker 없이 pdf_id로만 태그된다.)
"""

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
from index_text import decompose_and_route_search, precompute_entity_count  # noqa: E402
import entity_fusion  # noqa: E402
import company_entity_linking  # noqa: E402

load_dotenv(ROOT / ".env")

DEFAULT_PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "LGCNS" / "20260721_company_279243000.pdf"
DEFAULT_PDF_ID = "LGCNS"
DEFAULT_TICKER = "064400.KS"
DEFAULT_QUERY = "이 PDF 내용을 바탕으로 이 회사에 대한 투자 의견을 제공해줘"
DB_URL = os.environ.get("SUPABASE_DIRECT_DB_URL")
if not DB_URL:
    raise RuntimeError(
        "SUPABASE_DIRECT_DB_URL 환경변수가 없습니다. .env에 설정하세요 "
        "(비밀번호를 코드에 하드코딩하지 않음 — 과거 하드코딩된 값은 유출된 것으로 간주하고 "
        "Supabase 대시보드에서 반드시 회전(rotate)할 것)."
    )

# MinerU 미실행/onestop_cards.jsonl 없음 환경에서 이미지 브랜치 나머지 단계(합성/저장/검색/생성)를
# 그래도 끝까지 검증하기 위한 대표 예시 카드(LGCNS 실측 수치 기반) — 다른 pdf_id로 돌릴 때도
# image_id/doc_id는 그 pdf_id로 자동 치환되지만 내용 자체는 LGCNS 사례임에 유의(진짜 카드가
# 있으면 이 폴백은 안 쓰인다).
def _fallback_image_cards(pdf_id: str) -> list:
    return [
        {
            "image_id": f"{pdf_id}_p2_chart1", "doc_id": pdf_id, "page": 2, "block_type": "chart",
            "status": "useful", "caption": "LG CNS 클라우드&AI 부문 분기별 매출 추이",
            "footnote": "단위: 십억원", "ocr": {"text": "717 872 880 1,118 765 921"},
            "chart_table": "| 분기 | 클라우드&AI 매출 |\n|---|---|\n| 1Q25 | 717 |\n| 2Q25 | 872 |\n"
                            "| 3Q25 | 880 |\n| 4Q25 | 1,118 |\n| 1Q26 | 765 |\n| 2Q26F | 921 |",
            "narrative": "클라우드&AI 부문 매출은 분기별로 우상향하며, 2Q26F 921십억원까지 성장할 전망이다.",
            "embed_text": "LG CNS 클라우드&AI 부문 분기별 매출 추이 단위: 십억원 717 872 880 1,118 765 921 "
                           "클라우드&AI 부문 매출은 분기별로 우상향하며, 2Q26F 921십억원까지 성장할 전망이다.",
        },
    ]


# [수정] 사용자 지적("건설업종 종목 주간 수익률이 가장 높은 기업은?") 검증 중 발견 — 차트를
# OCR로 읽은 텍스트("삼성E&A (7.4) GS건설 (14.3) ... IPARK현대산업개발 1.3")를 그대로 LLM에
# 주면(한국 증권 리포트 관례상 괄호=음수/하락) gpt-4o-mini가 괄호 숫자를 그냥 "큰 숫자"로
# 읽어 GS건설(-14.3%, 실제로는 가장 큰 하락)을 "가장 높은 수익률"이라고 답하는 오류를 실측으로
# 확인했다. 프롬프트에 "괄호=음수" 지침을 명시해도(아래 프롬프트 참고) gpt-4o-mini가 안정적으로
# 못 지킴 — LLM의 즉석 해석에 맡기는 대신, 데이터 자체를 정규화해 애매함을 원천 차단한다(citation
# -check의 extract_numbers()도 부호를 안 보므로 이 오류를 못 잡는다 — 인용 자체는 "그 숫자가
# 어딘가 있다"만 확인해서 통과함, citation_check.py 자체 한계로 문서화돼 있음). block_type=chart
# 카드의 embed_text에서만(라벨 있는 표/텍스트 청크의 괄호는 다른 의미일 수 있어 범위를 좁힘)
# "라벨 (N)" 패턴을 "라벨 -N"으로 치환해 부호를 명시적으로 만든다.
_CHART_PAREN_NEGATIVE_RE = re.compile(r"\((\d+(?:\.\d+)?)\)")
_CHART_TRAIL_PAREN_RE = re.compile(r"(?<![\d(])(\d+(?:\.\d+)?)\)")   # 여는괄호 없이 'N)'만 (겹친 텍스트 OCR 손상)
# [수정 — 오탐 해결, 재일] 숫자 토큰 정규식을 쉼표까지 포함하도록 바꿈. 기존 `\d+(?:\.\d+)?`은
# 쉼표를 몰라서, PP-OCR이 마침표를 쉼표로 읽은 순간("24.07" -> "24,07") 토큰이 "24"+"07"로 쪼개졌다.
# 반면 텍스트레이어 쪽은 같은 값을 "24.07" 한 토큰으로 잡으므로 조각은 **절대 매칭될 수 없고**,
# 결과적으로 멀쩡한 축 라벨이 통째로 [OCR손상]으로 지워졌다(실측 c밴드 p3: '07'이 텍스트레이어
# 토큰 집합에 없음 -> "24,[OCR손상]", "[OCR손상],[OCR손상]"). 이제 양쪽 모두 쉼표/마침표를 품은
# 하나의 토큰으로 뽑고, 구분자 표기 차이를 흡수한 변형 집합으로 비교한다.
_CHART_NUM_RE = re.compile(r"\d[\d,.]*\d|\d")


def _number_variants(token: str) -> set:
    """구분자 표기 차이를 흡수한 비교용 변형들 — 천단위 쉼표 제거형, 쉼표<->마침표 교환형.
    "24,07"과 "24.07", "35,000"과 "35000"이 서로 같은 값으로 취급되게 한다."""
    t = token.strip(".,")
    if not t:
        return set()
    return {v for v in {t, t.replace(",", ""), t.replace(",", "."), t.replace(".", ","),
                        t.replace(",", "").replace(".", "")} if v}


def _page_number_set(doc, page: int):
    """PDF 텍스트레이어(권위) 페이지 숫자 집합 — 이미지 OCR 손상값 대조용.
    차트 쪽과 **동일한 토큰화**를 쓰고, 각 토큰의 구분자 변형까지 모두 담아 표기 차이로 인한
    오탐을 없앤다."""
    try:
        out = set()
        for m in _CHART_NUM_RE.findall(doc[page - 1].get_text()):
            out |= _number_variants(m)
        return out
    except Exception:
        return None


# 한 카드의 숫자 중 이 비율 이상이 텍스트레이어에 없으면, 그 차트는 애초에 텍스트레이어가 덮지
# 않는 래스터 이미지로 보고 손상 판정을 통째로 건너뛴다(전멸 방지). 진짜 손상은 소수 토큰에서만
# 일어나므로 이 임계로 "일부 손상"과 "대조 불가"를 가른다.
_OCR_CHECK_MAX_MISS_RATIO = 0.5


_AXIS_LADDER_MIN = 5   # 등간격 숫자가 이만큼 이상이면 데이터가 아니라 축 눈금으로 본다
_AXIS_WARNING = ("\n[주의] 이 차트 OCR에는 **축 눈금 숫자만** 있고 실제 계열 값은 없다. 여기 숫자들을 "
                 "날짜와 순서대로 짝지어 시계열로 해석하지 말 것 — 값이 필요하면 같은 페이지의 표 근거를 쓸 것.")


def _looks_like_axis_ladder(text: str) -> bool:
    """[재일 — c밴드 사례] 차트 OCR 숫자들이 '등간격 사다리'면 데이터가 아니라 축 눈금으로 판정.

    배경: 스텝/라인 차트는 데이터 라벨이 안 찍혀 있어 OCR이 Y축 눈금(0, 10,000, ..., 80,000)과
    X축 날짜(24.07 ... 26.07)만 긁어온다. 두 목록의 개수가 우연히 같으면(실측 c밴드: 9개 vs 9개)
    생성 모델이 순서대로 1:1 짝지어 **존재하지 않는 목표주가 시계열**을 만들어낸다(실측: 4개 기업
    전부 실제와 정반대인 '하락 추세'로 답변, 실제는 전부 상향). 값 라벨이 붙은 막대차트(예:
    Construct 도표3 종목별 주간수익률)는 값이 등간격이 아니라 이 판정에 안 걸린다."""
    nums = sorted({float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or "")})
    if len(nums) < _AXIS_LADDER_MIN:
        return False
    from collections import Counter
    diffs = [round(b - a, 6) for a, b in zip(nums, nums[1:])]
    modal, n = Counter(diffs).most_common(1)[0]
    return modal > 0 and n >= _AXIS_LADDER_MIN - 1


def _find_onestop_cards(pdf_id: str, pdf_path=None):
    """[수정 — 재일] 이미지 브랜치 카드(onestop_cards.jsonl) 위치를 견고하게 찾는다.

    기존엔 `ROOT/pdf_pipeline/data/onestop/{pdf_id}/`로 **경로를 하드코딩**했는데, 이미지 모듈의
    데이터 루트는 `image_processing/common.py`의 CONFIG["DATA_DIR"]가 정하고 실제 산출물은 리포
    루트 `data/onestop/`에 있었다. 두 경로가 어긋나 있어 **실제 카드가 있어도 못 찾고** 예시 카드로
    폴백했다(그 상태에선 차트 VLM 결과도 당연히 안 실린다).

    또 s2는 디렉토리 이름으로 `doc_id`(예: industry_15)를 쓰고 데모는 `pdf_id`(예: Construct,
    upload_xxxx)를 쓰는 **명명 불일치**가 있어서, pdf_id로 못 찾으면 PDF 파일명(stem)으로도 찾고,
    그래도 없으면 각 후보 디렉토리의 카드에 기록된 doc_id/원본 경로로 역매칭한다."""
    from image_processing import common as _img_common
    roots = [Path(_img_common.CONFIG["DATA_DIR"]) / "onestop",
             ROOT / "data" / "onestop",
             ROOT / "pdf_pipeline" / "data" / "onestop"]
    names = [pdf_id]
    if pdf_path:
        names.append(Path(pdf_path).stem)
    for root in roots:
        for name in names:
            cand = root / name / "onestop_cards.jsonl"
            if cand.exists():
                return cand
    # 마지막 수단 — 카드 안에 기록된 원본 PDF 경로/파일명으로 역매칭
    if pdf_path:
        stem = Path(pdf_path).stem
        for root in roots:
            if not root.is_dir():
                continue
            for cand in root.glob("*/onestop_cards.jsonl"):
                try:
                    head = cand.read_text(encoding="utf-8").split("\n", 1)[0]
                except Exception:
                    continue
                if stem and stem in head:
                    return cand
    return roots[0] / pdf_id / "onestop_cards.jsonl"   # 없을 때 로그에 찍힐 기본 경로


def _normalize_chart_card_signs(cards: list, pdf_path=None) -> list:
    """block_type="chart" 카드의 embed_text 부호/OCR손상 정규화 — 데이터 단계에서 확정(LLM 즉석 해석 실수 방지).

    [겹친 텍스트 OCR 손상 대응, 재일]
      (1) "(N)" 완전괄호 -> "-N"(한국 증권 리포트 관례상 음수/하락). 기존.
      (2) "N)" 닫는괄호만(여는괄호 유실) -> "-N". 예: "금호건설(35.2)"이 겹친 텍스트로 "금호건5.2)"로 깨지며
          여는괄호+회사명 일부가 유실된 경우 최소한 부호(하락)라도 복원 → 손상값이 "최고"로 오답되는 걸 차단.
      (3) 텍스트레이어(권위) 대조: 이미지 OCR 숫자가 원문 페이지 어디에도 없으면 손상값으로 보고 "[OCR손상]"으로
          치환(수치 제거) → 예: 한샘 "1.7"이 겹친 텍스트로 "11.7"(축범위 +4 초과)로 깨진 경우, 원문엔 11.7이
          없으므로 제거되어 "가장 높은 수익률"의 근거로 못 쓰이게 됨(정답 KCC글라스 +2.5가 선택되도록).
      실측(Construct, N=4): 한샘 오답픽 1/4 -> 0/4. pdf_path 없으면 (1)(2)만 적용(무해).
      주의: (3)은 PP-OCR과 PyMuPDF의 반올림 표기가 다르면 정상값도 제거될 위험 — 이 문서군에선 깨끗했으나
      여러 문서로 오탐률 측정 필요(핸드오프 남은과제 참조)."""
    doc = None
    if pdf_path:
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
        except Exception:
            doc = None
    out = []
    for c in cards:
        if c.get("block_type") == "chart" and c.get("embed_text"):
            t = _CHART_PAREN_NEGATIVE_RE.sub(r"-\1", c["embed_text"])
            t = _CHART_TRAIL_PAREN_RE.sub(r"-\1", t)
            if doc is not None and c.get("page"):
                pn = _page_number_set(doc, c["page"])
                if pn:
                    def _damaged(tok: str) -> bool:
                        # 한 자리 숫자는 우연히 안 맞을 여지가 커서 대상에서 제외(기존 규칙 유지)
                        if len(re.sub(r"[^\d]", "", tok)) < 2:
                            return False
                        return not (_number_variants(tok) & pn)

                    toks = _CHART_NUM_RE.findall(t)
                    miss = [x for x in toks if _damaged(x)]
                    # [수정] 전멸 방지 — 대부분이 안 맞으면 이 페이지 텍스트레이어가 이 차트를
                    # 아예 안 덮는 것(래스터 차트)이므로 손상 판정을 건너뛴다. 안 그러면 정상
                    # 차트가 통째로 [OCR손상] 범벅이 돼 근거로서의 가독성이 사라진다.
                    if toks and len(miss) / len(toks) <= _OCR_CHECK_MAX_MISS_RATIO:
                        t = _CHART_NUM_RE.sub(
                            lambda m: "[OCR손상]" if _damaged(m.group(0)) else m.group(0), t)
            # (4) [재일 — c밴드 사례] 축 눈금만 읽힌 차트에 경고 문구를 붙인다. 차트분석(4a, MinerU
            # VLM)이 켜져 있어 실제 계열 값(chart_table/narrative)이 있으면 붙이지 않는다.
            if not (c.get("chart_table") or c.get("narrative")) and _looks_like_axis_ladder(t):
                t = t + _AXIS_WARNING
            c = {**c, "embed_text": t}
        out.append(c)
    if doc is not None:
        doc.close()
    return out


def main(pdf_path=None, pdf_id: str = None, ticker: str = None, query: str = None,
         add_structured_metadata: bool = False, sector: str = None, verbose: bool = True,
         gen_model: str = "gpt-4.1"):
    """[수정] 팀원 인계용 파라미터화 — 인자를 안 주면 기존 LGCNS 기본값 그대로 동작.
    ticker=None이면(예: 여러 기업을 다루는 산업 섹터 리포트) document_evidence에 ticker 컬럼
    없이 pdf_id로만 태그된다(entity_fusion.store_evidence의 기존 동작 그대로 재사용).
    add_structured_metadata=True면 텍스트/테이블 라우팅 끝에 OpenAI Structured Output까지
    돌려 각 청크/행에 정성적 메타데이터(엔티티/논조 등)를 채운다(유료 API 호출 추가).
    반환: 호출측이 검증/보고서 작성에 쓸 수 있도록 결과 dict를 그대로 돌려준다."""
    pdf_path = Path(pdf_path) if pdf_path else DEFAULT_PDF_PATH
    pdf_id = pdf_id or DEFAULT_PDF_ID
    query = query or DEFAULT_QUERY
    onestop_cards_path = _find_onestop_cards(pdf_id, pdf_path)

    def _p(*a):
        if verbose:
            print(*a)

    timings = {}

    from embedding import get_embedding_model
    threading.Thread(target=get_embedding_model, daemon=True).start()

    t0 = time.perf_counter()
    _p("0) 스캔본 페이지 감지 (텍스트 레이어 없음 + 이미지가 페이지 대부분을 덮는 페이지)")
    scanned_pages = detect_scanned_pages(pdf_path)
    if scanned_pages:
        _p(f"   ⚠ {len(scanned_pages)}개 페이지({scanned_pages})가 어려운(스캔) PDF 페이지로 판정됨 "
           f"— only MinerU 처리 대상")
    else:
        _p("   스캔본 페이지 없음 — 전 페이지 자체 파이프라인으로 처리")
    timings["0_scanned_page_detect"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _p("1) YOLO 로딩 + 페이지 분류")
    yolo_model = YOLO(str(ROOT / "pdf_pipeline" / "page_classification" / "models" / "yolo11n_doc_layout.pt"))
    yolo_model.predict(Image.new("RGB", (595, 842), (255, 255, 255)), conf=0.25, verbose=False)

    cls_result = classify_pdf(pdf_path, yolo_model)
    page_boxes = {p["page"]: p["cached_boxes"] for p in cls_result["pages"]}
    _p(f"   {cls_result['n_pages']}페이지 분류 완료")
    timings["1_yolo_load_and_classify"] = time.perf_counter() - t0

    import numpy as np

    def _run_text_branch():
        tb0 = time.perf_counter()
        text_result = process_pdf(pdf_path, yolo_model, page_boxes=page_boxes,
                                   chunk_backend="rulebased", remove_boilerplate=True,
                                   add_structured_metadata=add_structured_metadata, sector=sector)
        text_chunks = [c for page in text_result["pages"] for c in page["chunks"]]
        text_items, text_emb = entity_fusion.embed_items(entity_fusion.from_text_chunks(pdf_id, text_chunks))
        n = entity_fusion.store_evidence(DB_URL, pdf_id, text_items, text_emb, ticker=ticker)
        _p(f"   [텍스트] {len(text_chunks)}개 청크 생성 -> {n}개 즉시 적재")
        return text_items, text_emb, text_chunks, time.perf_counter() - tb0

    def _run_table_branch():
        tb0 = time.perf_counter()
        rtmp.PDF_PATH = pdf_path
        table_records, n_finance_filtered, n_cid = rtmp.build_records(
            pdf_id, page_boxes=page_boxes, yolo_model=yolo_model,
            add_structured_metadata=add_structured_metadata, sector=sector)
        # [재일 §8.4 갭 수정] 표 브랜치도 텍스트/이미지처럼 structured_metadata를 evidence에 부착.
        #  (1) 표 단위 table_metadata(record_type="table_metadata": table_title/type/notable/entities_mentioned)를
        #      table_idx로 매핑해 같은 표의 행에 붙임. (2) 행 content(라벨+셀)에서 KOSPI200 기업명을 결정적
        #      매칭해 entities로 채움 — LLM extract_table_metadata가 entities_mentioned를 자주 빈값으로 주던
        #      문제를 결정적 매칭으로 보완(§8.4: 표 entities 0건 → 채워짐).
        tmeta = {r.get("table_idx"): r for r in table_records if r.get("record_type") == "table_metadata"}
        row_records = [r for r in table_records if r.get("record_type") != "table_metadata"]
        mapped = [r for r in row_records if r.get("canonical_field")]
        _name_map = company_entity_linking.get_korean_name_map()
        for r in row_records:
            tm = tmeta.get(r.get("table_idx")) or {}
            row_text = f"{r.get('raw_label','')} " + " ".join(str(x) for x in (r.get("cells") or []))
            row_ents = [m["name"] for m in company_entity_linking.find_mentioned_companies(row_text, _name_map)]
            ents = sorted(set((tm.get("entities_mentioned") or []) + row_ents))
            sm = {k: tm[k] for k in ("table_title", "table_type_refined", "notable_finding") if tm.get(k)}
            if ents:
                sm["entities"] = ents
            if sm:
                r["structured_metadata"] = sm
        table_items, table_emb = entity_fusion.embed_items(entity_fusion.from_table_records(pdf_id, row_records))
        n = entity_fusion.store_evidence(DB_URL, pdf_id, table_items, table_emb, ticker=ticker)
        _p(f"   [테이블] {len(row_records)}행 파싱(하이브리드 게이트, [JAEIL v5]), "
           f"canonical 매칭 {len(mapped)}개 -> {n}개 즉시 적재")
        return table_items, table_emb, row_records, time.perf_counter() - tb0

    def _run_image_branch():
        tb0 = time.perf_counter()
        is_fallback = not onestop_cards_path.exists()
        if not is_fallback:
            image_cards = [json.loads(line) for line in onestop_cards_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            _p(f"   [이미지] 실제 onestop_cards.jsonl {len(image_cards)}건 로드: {onestop_cards_path}")
        else:
            image_cards = _fallback_image_cards(pdf_id)
            _p(f"   [이미지] ⚠ {onestop_cards_path} 없음(MinerU 미실행) — 대표 예시 카드 {len(image_cards)}건으로 대체")
        # [교차리뷰 수정] 폴백 예시 카드는 텍스트레이어 대조를 하지 않는다 — 카드 내용(LGCNS 사례)과
        # 대상 PDF가 다른 문서라, 대조하면 모든 숫자가 "원문에 없음"으로 찍혀 [OCR손상] 범벅이 된다
        # (실제 DB에서 pdf_id='Construct'에 이렇게 오염된 폴백 행이 발견돼 수정함).
        image_cards = _normalize_chart_card_signs(image_cards, pdf_path=None if is_fallback else pdf_path)
        if add_structured_metadata:
            import structured_output
            image_cards = structured_output.add_structured_metadata_to_cards(image_cards)
        image_items, image_emb = entity_fusion.embed_items(entity_fusion.from_image_cards(pdf_id, image_cards))
        if is_fallback:
            # [교차리뷰 수정] 폴백 카드는 파이프라인 후단(합성/검색/생성) 검증용 in-memory 전용 —
            # DB에 저장하면 실제 문서의 evidence로 둔갑해 이후 모든 재질의를 오염시킨다(실측:
            # pdf_id='Construct'에 LGCNS 예시 수치가 image evidence로 적재돼 있던 사고). 저장 안 함.
            n = 0
            _p(f"   [이미지] 폴백 카드 {len(image_items)}건은 DB에 저장하지 않음(오염 방지) — 이번 실행의 검색에만 사용")
        else:
            n = entity_fusion.store_evidence(DB_URL, pdf_id, image_items, image_emb, ticker=ticker)
        _p(f"   [이미지] {n}개 즉시 적재")
        return image_items, image_emb, image_cards, time.perf_counter() - tb0

    t0 = time.perf_counter()
    print("2-4) 텍스트/테이블/이미지 3브랜치 동시 실행 (서로 독립적 — [40]/[51]과 동일 전제, "
          "ThreadPoolExecutor로 실제 동시성 배선)")
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_text = ex.submit(_run_text_branch)
        f_table = ex.submit(_run_table_branch)
        f_image = ex.submit(_run_image_branch)
        text_items, text_emb, text_chunks, t_text = f_text.result()
        table_items, table_emb, row_records, t_table = f_table.result()
        image_items, image_emb, image_cards, t_image = f_image.result()
    # [수정] 개별 브랜치 시간(t_text/t_table/t_image)은 서로 겹쳐서 돌았으므로 timings에 그대로
    # 넣으면 "총 소요시간"이 실제 벽시계보다 부풀려짐(동시 실행분이 중복 합산) — 아래 "단계별
    # 소요시간" 총합/비율 계산에는 실제 벽시계(2-4_concurrent_wall)만 반영하고, 브랜치별 개별
    # 시간은 참고용으로 바로 위에서 따로 출력한다.
    timings["2-4_concurrent_wall(text/table/image)"] = time.perf_counter() - t0
    print(f"   [텍스트 {t_text:.2f}s / 테이블 {t_table:.2f}s / 이미지 {t_image:.2f}s] 개별 합계 "
          f"{t_text + t_table + t_image:.2f}s -> 동시 실행 벽시계 "
          f"{timings['2-4_concurrent_wall(text/table/image)']:.2f}s")

    t0 = time.perf_counter()
    print("5) 엔티티 합성 완료 확인 (세 브랜치가 각자 적재한 evidence 수 집계 — 추가 대기 없음)")
    all_items = text_items + table_items + image_items
    all_embeddings = np.concatenate([e for e in (text_emb, table_emb, image_emb) if e is not None])
    by_source = {}
    for it in all_items:
        by_source[it["source_type"]] = by_source.get(it["source_type"], 0) + 1
    print(f"   총 {len(all_items)}개 evidence ({by_source}) — 전부 document_evidence(ticker={ticker})에 있음")
    timings["5_entity_fusion"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("6) 통합 하이브리드 인덱스 구축 (이미 계산된 임베딩 재사용 — 재임베딩 없음)")
    index = entity_fusion.build_index_from_items(pdf_id, all_items, all_embeddings)
    dim = index.embeddings.shape[1]
    print(f"   임베딩 차원: {dim}, evidence 수: {len(index.chunks)}")
    timings["6_build_fused_index"] = time.perf_counter() - t0

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # [수정] 사용자 확인("4번 문제 — 최고성능인 신 라우터로 교체") — 4문서 A/B(핀터멘털/납기/
    # LGCNS/KWave, 파이프라인_최종정리_핸드오프.md §5)에서 route_search(entity_aware=True)가
    # ndcg 0.848(4문서 평균, 전 방법 중 최고 — 고정 weighted_hybrid_search 0.554~0.470보다 높음)로
    # 검증됐는데 이 데모는 그 검증된 경로를 안 쓰고 있었다. route_search()가 abstract 질의에서
    # entity_count로 HyDE/MQE를 가르므로, [46] 권장대로 인제스트 직후(질의 전에) 미리 계산해
    # 캐시해둔다 — 안 해두면 첫 질의가 그 계산 지연을 그대로 떠안는다.
    t0 = time.perf_counter()
    print("6b) 문서 엔티티 수 사전계산 (HyDE/MQE 분기 판단용)")
    precompute_entity_count(index, pdf_path=pdf_path, client=client)
    print(f"   entity_count={index.entity_count}")
    timings["6b_precompute_entity_count"] = time.perf_counter() - t0

    # [수정] "기업명 및 티커"(ERD) — PDF 근거 검색과 기업 DB 매칭을 병렬로 돌린다("두 개 DB를
    # 병렬로 검색"). 서로 독립적(하나는 document_evidence 인덱스, 하나는 financial_summaries/
    # company_profile_chunks + 원본 PDF 텍스트)이라 동시 실행해도 안전.
    def _run_document_search():
        hits, subqueries = decompose_and_route_search(index, query, client=client, top_k=8)
        return hits, subqueries

    def _run_entity_linking():
        import fitz
        doc = fitz.open(str(pdf_path))
        full_text = "\n".join(doc[i].get_text() for i in range(doc.page_count))
        doc.close()
        matched = company_entity_linking.find_mentioned_companies(full_text)
        db_context = company_entity_linking.fetch_company_db_context(DB_URL, matched)
        return matched, db_context

    t0 = time.perf_counter()
    print("7) 문서 근거 검색(질의 분해 라우팅) + 기업 DB 매칭 — 병렬 실행")
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_search = ex.submit(_run_document_search)
        f_link = ex.submit(_run_entity_linking)
        hits, subqueries = f_search.result()
        matched_companies, company_db_context = f_link.result()
    for sq in subqueries:
        print(f"   [하위질의] {sq['type']}->{sq['resolved_type']}: {sq['query']}")
    for h in hits:
        source_type = h["chunk"].get("source_type")
        print(f"   - [{source_type}] page{h['chunk'].get('page')} score={h['score']:.4f}")
    print(f"   [기업 DB 매칭] {len(matched_companies)}건: "
          f"{[m['name'] for m in matched_companies] if matched_companies else '없음'}")
    timings["7_search_and_entity_link"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print(f"8) LLM 투자의견 생성 ({gen_model}, citation-check 포함)")
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
- "기업 DB 참고 정보"의 원본 숫자(원단위 금액 등)를 인용할 땐 **단위 변환·반올림 없이 자릿수 그대로**
  옮길 것(예: "1,234,500,000원"을 "12.3억"으로 바꾸지 말 것). 표기를 바꾸면 근거 검증(citation-check)에서
  근거 없는 숫자로 오인돼 불필요한 재생성이 발생한다.
- [image] 소스는 차트를 OCR로 읽은 원문이라 수치 앞뒤에 부호가 명시적으로 안 붙어 있을 수 있다.
  한국 증권 리포트 관례상 "값(N)"처럼 **괄호로 감싼 숫자는 음수(하락/손실)**, 괄호 없는 숫자는
  양수(상승/이익)를 뜻한다 — "가장 높다/많다"류 질문에 답할 때 괄호 유무를 반드시 부호로
  해석해서 판단할 것(괄호 안 숫자의 절댓값이 크다고 그게 "가장 높은" 값이 아니다 — 오히려
  가장 큰 하락폭이다). 축 눈금(예: "(38)(34)(30)...")은 데이터가 아니라 눈금선이므로 특정
  대상(기업명 등) 없이 나열된 괄호 숫자는 값으로 쓰지 말 것.
- 근거 안의 "[OCR손상]" 표기는 원문 대조에서 신뢰 불가로 판정돼 제거된 값이다 — 그 항목의
  수치는 존재하지 않는 것으로 취급하고, 어떤 수치 판단(최고/최저/비교/합산)의 근거로도 절대
  쓰지 말 것. "[OCR손상]"이 섞인 항목의 기업/지표를 굳이 언급해야 하면 "값 판독 불가"로만 서술할 것.

[사용자 요청]
{query}
"""
    result = generate_with_citation_check(
        client, prompt, context=full_context, model=gen_model, max_retries=2)
    answer = result["answer"]
    timings["8_llm_generation"] = time.perf_counter() - t0

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

    return {
        "pdf_id": pdf_id, "ticker": ticker, "query": query, "subqueries": subqueries,
        "entity_count": index.entity_count,
        "matched_companies": matched_companies, "company_db_context": company_db_context,
        "text_chunks": text_chunks, "row_records": row_records, "image_cards": image_cards,
        "all_items": all_items, "n_evidence_by_source": by_source,
        "hits": hits, "answer": answer, "citation_result": result,
        "timings": timings, "total_time_s": total,
    }


if __name__ == "__main__":
    main()

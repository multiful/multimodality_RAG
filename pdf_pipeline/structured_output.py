"""[25] 각 라우팅(텍스트/표) 끝에 붙는 LLM 구조화 출력(Structured Output) — 사용자 요청
("스트럭처 아웃풋 각 라우팅 끝에 도입", "open api 사용할 것")에 따라 OpenAI Structured Outputs
(response_format=Pydantic 모델, `chat.completions.parse`)로 rule-based 추출이 못 잡는 정성적
메타데이터(엔티티/논조/특이사항 등)만 보완 추출한다.

설계 원칙:
- rule-based가 이미 안정적으로 뽑는 값(table_processing의 canonical_field/numeric_value/unit,
  text_processing의 section_path 등)은 여기서 다시 뽑지 않는다 — LLM은 자유 텍스트 이해가
  필요한 것(요약/논조/신규 엔티티/특이사항)만 담당.
- 아래 필드는 초안이다 — 실제 채택할 필드는 사용자가 직접 가감할 예정.
- API 키는 환경변수 OPENAI_API_KEY에서만 읽음(하드코딩 금지, contextual_chunker.generate_context_openai
  와 동일한 관례).
"""

import os
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

_DEFAULT_MODEL = "gpt-4o-mini"

sys.path.insert(0, str(Path(__file__).resolve().parent / "table_processing"))  # [37] sector_schema.yaml 재사용


def _get_client(client=None):
    if client is not None:
        return client
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _retryable_parse(client, **kwargs):
    """[37] 사용자 지적("페이지 늘어날수록 조정할 부분") 반영 — 표/청크 수가 많은 대형 문서는
    구조화 출력 API 호출이 수십~수백 건 나갈 수 있는데, 지금까지 재시도/백오프가 전혀 없어서
    레이트리밋(429)이나 일시적 5xx에 그대로 실패했다. `tenacity`로 지수 백오프 재시도 추가 —
    429(RateLimitError)/5xx(APIStatusError, internal_server_error 등)/연결 오류/타임아웃만 재시도
    (스키마 위반 같은 4xx 요청 자체 오류는 재시도해도 똑같이 실패하므로 즉시 실패 처리)."""
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    @retry(
        retry=retry_if_exception_type((RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call():
        return client.chat.completions.parse(**kwargs)

    return _call()


def _sector_hint(sector: str = None) -> str:
    """[37] 사용자 요청("문서 종류에 따라 그에 알맞는 구조화 출력") 반영 — 스키마 자체를 섹터별로
    N개 따로 만들지 않고(오버엔지니어링 방지), 이미 있는 `sector_schema.yaml`의 섹터별 큐레이션
    필드 별칭을 프롬프트 힌트로 주입 — 같은 스키마 형태를 유지하면서 "이 섹터에서 특히 중요한
    지표"에 LLM의 주의를 끌어 metric_mentions/notable_finding 등의 품질을 높인다."""
    if not sector:
        return ""
    try:
        from canonical_field_schema import SECTOR_TABLE_TYPES, FIELD_BY_KEY
        table_types = SECTOR_TABLE_TYPES.get(sector, {}).get("table_types", {})
        field_keys = {fk for fields in table_types.values() for fk in fields}
        aliases = []
        for fk in field_keys:
            aliases.extend(FIELD_BY_KEY[fk].aliases[:2] if fk in FIELD_BY_KEY else [])
        if not aliases:
            return ""
        sample = ", ".join(dict.fromkeys(aliases))[:300]
        return (f"\n이 문서는 '{sector}' 섹터 리포트입니다. 이 섹터에서 흔히 쓰이는 지표 예시(참고용): "
                f"{sample}. **주의: 이건 어떤 지표를 찾아야 할지 감을 잡는 참고 목록일 뿐입니다 — "
                f"위 chunk/표 원문에 실제로 등장하지 않는 지표는 절대 지어내지 말고, 원문에 실제로 "
                f"있는 내용만 뽑으세요.**\n")
    except Exception:
        return ""  # sector_schema 조회 실패해도 구조화 출력 자체는 계속 진행(핵심 기능 아님)


# ---------- 텍스트 라우팅 끝: 청크 단위 구조화 메타데이터 ----------

class TextChunkMetadata(BaseModel):
    chunk_index: int
    entities: list[str]
    sector_mentioned: Optional[str]
    topic: str
    metric_mentions: list[str]
    time_period: Optional[str]
    sentiment: Literal["positive", "neutral", "negative"]
    contains_forward_looking_statement: bool
    summary: str


class _TextChunkMetadataBatch(BaseModel):
    items: list[TextChunkMetadata]


_TEXT_METADATA_PROMPT = (
    "다음은 증권사 리포트의 문단(chunk) 목록입니다. 각 chunk마다 아래 항목을 뽑아 chunk_index로 "
    "표시된 순서에 맞춰 items 배열로 반환하세요(입력 chunk 개수와 items 개수가 반드시 같아야 함).\n\n"
    "- entities: 언급된 기업/기관명(없으면 빈 배열)\n"
    "- sector_mentioned: 언급된 산업/섹터명(없으면 null)\n"
    "- topic: 이 chunk의 핵심 주제 한 줄\n"
    "- metric_mentions: 언급된 재무/수치 지표명(라벨만, 값은 이미 별도 파이프라인에서 추출되므로 값은 적지 말 것)\n"
    "- time_period: 언급된 시점/기간(예: 2026E, 3Q25, 없으면 null)\n"
    "- sentiment: 투자 관점에서 이 chunk의 논조(positive/neutral/negative)\n"
    "- contains_forward_looking_statement: 전망/추정 문장 포함 여부\n"
    "- summary: 1문장 요약\n\n"
    "{doc_title_line}{chunks_block}"
)


def extract_text_chunk_metadata(chunks: list, doc_title: str = None, client=None,
                                 model_name: str = _DEFAULT_MODEL, sector: str = None) -> list:
    """텍스트 라우팅 끝에서 호출 — 한 페이지의 chunk들(raw_chunk + section_path)을 한 번의 API
    호출로 배치 처리(청크마다 호출하면 페이지당 API 호출 수가 너무 많아짐). 반환은 입력 chunks와
    같은 길이/순서의 dict 리스트(항목별 병합은 호출측이 수행 — chunk_index로 정렬해 매칭하므로
    LLM이 순서를 안 지켜도 안전). sector: [37] `sector_classifier.classify_pdf_sector()`로 이미
    판별한 섹터명을 넘기면 그 섹터에 특화된 지표 힌트를 프롬프트에 주입(선택, 없어도 동작)."""
    if not chunks:
        return []
    client = _get_client(client)
    chunks_block = "\n\n".join(
        f"[chunk_index={i}] (섹션: {' > '.join(c.get('section_path') or []) or '없음'})\n{c['raw_chunk']}"
        for i, c in enumerate(chunks)
    )
    doc_title_line = f"문서 제목: {doc_title}\n\n" if doc_title else ""
    prompt = _TEXT_METADATA_PROMPT.format(doc_title_line=doc_title_line, chunks_block=chunks_block)
    prompt += _sector_hint(sector)

    resp = _retryable_parse(client, model=model_name, messages=[{"role": "user", "content": prompt}],
                             response_format=_TextChunkMetadataBatch)
    parsed = resp.choices[0].message.parsed
    by_index = {item.chunk_index: item for item in parsed.items}
    return [by_index[i].model_dump() if i in by_index else None for i in range(len(chunks))]


# ---------- 표 라우팅 끝: 표 단위 구조화 메타데이터 ----------

class TableMetadata(BaseModel):
    table_title: Optional[str]
    entities_mentioned: list[str]
    time_periods_covered: list[str]
    table_type_refined: str
    unmapped_fields_summary: Optional[str]
    notable_finding: Optional[str]


_TABLE_METADATA_PROMPT = (
    "다음은 증권사 리포트에서 추출한 표 하나의 원문 텍스트와, 이미 규칙 기반으로 표준 필드에 매칭된 "
    "값 목록입니다. 규칙 기반으로 이미 뽑힌 값(canonical_field로 매칭된 것들)은 다시 반복하지 말고, "
    "아래 항목만 이 표 내용을 바탕으로 채우세요.\n\n"
    "- table_title: 표 제목/주제 한 줄(원문에 명시된 제목이 없으면 내용 기반으로 추론)\n"
    "- entities_mentioned: 표에 언급된 기업/기관명\n"
    "- time_periods_covered: 표가 다루는 시점 목록(예: [\"2024\", \"2025\", \"2026E\"])\n"
    "- table_type_refined: 이 표의 세부 유형(예: '실적 요약', '계약 공시', '투자지표', '세그먼트별 매출')\n"
    "- unmapped_fields_summary: '규칙 기반 매칭 안 된 행'이 어떤 정보인지 한 줄 요약(없으면 null)\n"
    "- notable_finding: 이 표에서 눈에 띄는 점(급증/급감/이례적 수치 등, 없으면 null)\n\n"
    "<표 원문>\n{table_text}\n</표 원문>\n\n"
    "<규칙 기반 매칭된 필드>\n{mapped_summary}\n</규칙 기반 매칭된 필드>\n\n"
    "<규칙 기반 매칭 안 된 행 라벨>\n{unmapped_labels}\n</규칙 기반 매칭 안 된 행 라벨>"
)


def extract_table_metadata(table_text: str, mapped_records: list, unmapped_labels: list,
                            client=None, model_name: str = _DEFAULT_MODEL, sector: str = None) -> dict:
    """표 라우팅(run_table_metadata_pipeline) 끝에서 표 하나마다 호출. mapped_records: 이미
    canonical_field가 매칭된 레코드들(raw_label 요약용), unmapped_labels: 매칭 안 된 행의 원본
    라벨 리스트. 빈 표(행 없음)에는 호출하지 않도록 호출측에서 가드할 것. sector: [37] 섹터
    특화 지표 힌트 주입용(선택)."""
    client = _get_client(client)
    mapped_summary = ", ".join(
        f"{r['canonical_field']}={r.get('raw_label') or r.get('cells')}" for r in mapped_records
    ) or "(없음)"
    unmapped_str = ", ".join(unmapped_labels) or "(없음)"
    prompt = _TABLE_METADATA_PROMPT.format(
        table_text=table_text, mapped_summary=mapped_summary, unmapped_labels=unmapped_str)
    prompt += _sector_hint(sector)

    resp = _retryable_parse(client, model=model_name, messages=[{"role": "user", "content": prompt}],
                             response_format=TableMetadata)
    return resp.choices[0].message.parsed.model_dump()


# ---------- 이미지 라우팅 끝: 이미지 단위 구조화 메타데이터 ----------
#
# [39] 팀원 이미지 모듈(pdf_pipeline/image_processing/, onestop_cards.jsonl 카드 스키마)이
# 실제로 merge된 뒤 배선. 카드는 이미 캡션/각주/OCR텍스트/(선택)차트표+서술형해석까지 채워진
# 상태로 들어오므로(README §5 카드 스키마), "이미지 자체(bytes)"가 아니라 이 텍스트 필드들을
# 입력으로 쓴다 — vision API 호출 불필요, 텍스트 전용 구조화 출력으로 충분.

class ImageMetadata(BaseModel):
    image_type: Literal["chart", "logo", "photo", "diagram", "other"]
    caption_or_title: Optional[str]
    entities_mentioned: list[str]
    described_content: str
    key_values_or_trend: Optional[str]
    time_period: Optional[str]


_IMAGE_METADATA_PROMPT = (
    "다음은 증권사 리포트에 포함된 이미지/차트 블록 하나에 대한 정보입니다(MinerU 탐지 타입, "
    "캡션/각주, OCR로 읽은 크롭 내부 텍스트, 있다면 차트에서 추출한 데이터표와 서술형 해석 포함). "
    "이 정보만 근거로 아래 항목을 채우세요 — 여기 없는 내용은 추측하지 마세요.\n\n"
    "- image_type: chart(차트/그래프) / logo(로고) / photo(사진) / diagram(도식) / other 중 하나\n"
    "- caption_or_title: 원문에 캡션/제목이 있으면 그대로, 없으면 null\n"
    "- entities_mentioned: 등장/언급된 기업/기관명(범례, OCR 텍스트, 캡션 등에서)\n"
    "- described_content: 이 이미지가 무엇을 보여주는지 1~2문장 설명\n"
    "- key_values_or_trend: 차트라면 읽을 수 있는 핵심 수치나 추세(급등/급락 등, 없으면 null)\n"
    "- time_period: 시계열 차트라면 다루는 기간(없으면 null)\n\n"
    "<MinerU 탐지 타입>\n{block_type}\n</MinerU 탐지 타입>\n\n"
    "<캡션>\n{caption}\n</캡션>\n\n"
    "<각주>\n{footnote}\n</각주>\n\n"
    "<OCR 텍스트>\n{ocr_text}\n</OCR 텍스트>\n\n"
    "<차트 추출표(MinerU VLM, 있는 경우만)>\n{chart_table}\n</차트 추출표>\n\n"
    "<서술형 해석(참고용 — §3.4에 따라 근거는 항상 위 차트표를 우선할 것)>\n{narrative}\n</서술형 해석>"
)


def extract_image_metadata(card: dict, client=None, model_name: str = _DEFAULT_MODEL) -> dict:
    """이미지/차트 카드(onestop_cards.jsonl의 행 하나) -> 구조화 메타데이터. status="useful"인
    카드에만 호출할 것(호출측에서 가드) — discarded/handoff/skipped 카드는 내용이 비어있거나
    표 파트 소관이라 이 함수 대상이 아니다."""
    client = _get_client(client)
    ocr_text = (card.get("ocr") or {}).get("text") or ""
    prompt = _IMAGE_METADATA_PROMPT.format(
        block_type=card.get("block_type") or "(알수없음)",
        caption=card.get("caption") or "(없음)",
        footnote=card.get("footnote") or "(없음)",
        ocr_text=ocr_text or "(없음)",
        chart_table=card.get("chart_table") or "(없음 — 차트분석 미실행)",
        narrative=card.get("narrative") or "(없음)",
    )
    resp = _retryable_parse(client, model=model_name, messages=[{"role": "user", "content": prompt}],
                             response_format=ImageMetadata)
    return resp.choices[0].message.parsed.model_dump()


def add_structured_metadata_to_cards(cards: list, openai_client=None, model_name: str = _DEFAULT_MODEL,
                                      workers: int = 8) -> list:
    """cards(onestop_cards.jsonl을 읽은 dict 리스트) 중 status="useful"인 것만 골라 구조화
    메타데이터를 뽑아 각 카드에 `structured_metadata` 필드로 채워 반환(원본 리스트는 그대로,
    새 리스트 반환). run_table_metadata_pipeline.py의 [29]와 동일하게 로컬 연산은 이미 끝난
    카드들이므로 API 호출만 ThreadPoolExecutor로 한꺼번에 병렬 디스패치."""
    if openai_client is None:
        import os
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    targets = [c for c in cards if c.get("status") == "useful"]
    if not targets:
        return cards

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(extract_image_metadata, card, openai_client, model_name): card["image_id"]
            for card in targets
        }
        for future in as_completed(future_to_id):
            image_id = future_to_id[future]
            try:
                results[image_id] = future.result()
            except Exception as e:  # noqa: BLE001 — 카드 하나 실패해도 나머지는 계속 진행
                results[image_id] = {"error": str(e)}

    return [
        {**c, "structured_metadata": results[c["image_id"]]} if c["image_id"] in results else c
        for c in cards
    ]

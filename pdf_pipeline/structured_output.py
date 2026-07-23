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
from typing import Literal, Optional

from pydantic import BaseModel

_DEFAULT_MODEL = "gpt-4o-mini"


def _get_client(client=None):
    if client is not None:
        return client
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


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
                                 model_name: str = _DEFAULT_MODEL) -> list:
    """텍스트 라우팅 끝에서 호출 — 한 페이지의 chunk들(raw_chunk + section_path)을 한 번의 API
    호출로 배치 처리(청크마다 호출하면 페이지당 API 호출 수가 너무 많아짐). 반환은 입력 chunks와
    같은 길이/순서의 dict 리스트(항목별 병합은 호출측이 수행 — chunk_index로 정렬해 매칭하므로
    LLM이 순서를 안 지켜도 안전)."""
    if not chunks:
        return []
    client = _get_client(client)
    chunks_block = "\n\n".join(
        f"[chunk_index={i}] (섹션: {' > '.join(c.get('section_path') or []) or '없음'})\n{c['raw_chunk']}"
        for i, c in enumerate(chunks)
    )
    doc_title_line = f"문서 제목: {doc_title}\n\n" if doc_title else ""
    prompt = _TEXT_METADATA_PROMPT.format(doc_title_line=doc_title_line, chunks_block=chunks_block)

    resp = client.chat.completions.parse(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format=_TextChunkMetadataBatch,
    )
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
                            client=None, model_name: str = _DEFAULT_MODEL) -> dict:
    """표 라우팅(run_table_metadata_pipeline) 끝에서 표 하나마다 호출. mapped_records: 이미
    canonical_field가 매칭된 레코드들(raw_label 요약용), unmapped_labels: 매칭 안 된 행의 원본
    라벨 리스트. 빈 표(행 없음)에는 호출하지 않도록 호출측에서 가드할 것."""
    client = _get_client(client)
    mapped_summary = ", ".join(
        f"{r['canonical_field']}={r.get('raw_label') or r.get('cells')}" for r in mapped_records
    ) or "(없음)"
    unmapped_str = ", ".join(unmapped_labels) or "(없음)"
    prompt = _TABLE_METADATA_PROMPT.format(
        table_text=table_text, mapped_summary=mapped_summary, unmapped_labels=unmapped_str)

    resp = client.chat.completions.parse(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format=TableMetadata,
    )
    return resp.choices[0].message.parsed.model_dump()


# ---------- 이미지 라우팅 끝: 이미지 단위 구조화 메타데이터(초안, 아직 미배선) ----------
#
# 팀원 이미지 모듈 통합 전에 스키마만 먼저 설계(사용자 요청: "테이블, 이미지, 텍스트 별로
# 설정하는게 좋지 않나"). 표/텍스트와 마찬가지로 이미지도 내용 유형이 아예 다르므로(차트/로고/
# 사진) 별도 스키마가 필요하다고 판단해 미리 틀만 잡아둔다. **주의**: 아래 함수는 "이미지에
# 대한 캡션/설명 텍스트가 이미 있다"는 가정으로 짠 텍스트 전용 프롬프트다 — 팀원 모듈이 실제로
# 캡션 문자열을 주는지, 아니면 이미지 자체(bytes)를 던지는지에 따라 프롬프트/입력 방식(vision
# 멀티모달 메시지로 교체 필요할 수 있음)을 다시 맞춰야 한다. 그 전까지는 호출하지 말 것.

class ImageMetadata(BaseModel):
    image_type: Literal["chart", "logo", "photo", "diagram", "other"]
    caption_or_title: Optional[str]
    entities_mentioned: list[str]
    described_content: str
    key_values_or_trend: Optional[str]
    time_period: Optional[str]


_IMAGE_METADATA_PROMPT = (
    "다음은 증권사 리포트에 포함된 이미지 하나에 대한 설명(캡션 또는 VLM이 생성한 이미지 설명)"
    "입니다. 아래 항목을 채우세요.\n\n"
    "- image_type: chart(차트/그래프) / logo(로고) / photo(사진) / diagram(도식) / other 중 하나\n"
    "- caption_or_title: 원문에 캡션/제목이 있으면 그대로, 없으면 null\n"
    "- entities_mentioned: 이미지에 등장/언급된 기업/기관명(범례, 라벨 등)\n"
    "- described_content: 이 이미지가 무엇을 보여주는지 1~2문장 설명\n"
    "- key_values_or_trend: 차트라면 읽을 수 있는 핵심 수치나 추세(급등/급락 등, 없으면 null)\n"
    "- time_period: 시계열 차트라면 다루는 기간(없으면 null)\n\n"
    "<이미지 설명>\n{image_description}\n</이미지 설명>"
)


def extract_image_metadata(image_description: str, client=None, model_name: str = _DEFAULT_MODEL) -> dict:
    """[미배선] 이미지 캡션/설명 텍스트 하나를 받아 구조화 메타데이터로 변환 — 팀원의 이미지
    추출 모듈이 캡션 텍스트를 주는 경우를 가정한 초안. 실제 입력 형태가 확정되면 프롬프트나
    호출 방식(vision API 등)을 다시 맞출 것."""
    client = _get_client(client)
    prompt = _IMAGE_METADATA_PROMPT.format(image_description=image_description)
    resp = client.chat.completions.parse(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format=ImageMetadata,
    )
    return resp.choices[0].message.parsed.model_dump()

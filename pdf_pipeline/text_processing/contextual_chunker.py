"""[3]/[5] 문맥적 청킹(Contextual Chunking) — Anthropic의 Contextual Retrieval 기법을 그대로 적용.
기본 분할(고정 길이/문단 단위)은 계층적 청킹만큼 정교하지 않지만, **각 청크 앞에 "이 청크가 전체
문서에서 어떤 맥락에 있는지" 설명을 덧붙인다**는 게 핵심 차별점. 계층적 청킹은 구조 경로를
메타데이터로만 남기고 본문은 그대로 두는 반면, 이 방식은 본문 자체에 맥락을 주입한다 — "동사",
"당사" 같은 대명사로 지칭된 회사명이 청크 안에 없어도(원문이 그렇게 쓰여 있으므로), 청크 하나만
독립적으로 봤을 때(실제 RAG 검색 상황) 엔티티를 놓치지 않게 하는 것이 목적.

[5]에서 컨텍스트 생성 백엔드를 3가지로 분리(사용자 피드백 — 로컬 VLM 호출이 매 청크마다 걸려
지연이 큼): "qwen"(로컬 Qwen2.5-VL-7B, 정확도는 가장 좋으나 느림), "openai"(OpenAI 경량 모델,
API 비용 발생하나 빠름), "rulebased"(LLM 호출 자체를 없애고 계층적 청킹의 section_path +
문서 메타데이터를 템플릿 문자열로 조립 — 비용/지연 거의 0이지만 표현력이 규칙 이상으로는 못 감).
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import split_sentences  # noqa: E402


def _naive_split(text: str, base_chars: int = 250) -> list:
    """기본 분할: 문장/줄 경계에서 base_chars만큼 그리디하게 누적.
    [3] 구현 노트(버그 발견·수정, 2건): (1) 처음엔 `\\n+` 기준으로 "문단"을 나눈 뒤 문장
    재분할했는데, 이 PDF들은 줄바꿈된 시각적 "줄"마다 개별 `\\n`이 들어가 있어(hierarchical_
    chunker에서 처음 발견한 것과 같은 문제) 문단이 아니라 줄 단위로 잘못 쪼개짐 — 문장 분리
    (semantic_chunker의 `split_sentences`)로 교체해 해결. (2) 그런데 뉴스 불릿처럼 마침표 없이
    줄바꿈만으로 항목이 끝나는 콘텐츠(construct_p5)에서, 여기서 `\\n`을 공백으로 지워버리면
    `split_sentences`가 줄 경계를 못 보고 여러 뉴스 항목이 통째로 한 "문장"으로 뭉쳐버림(recall
    20%까지 급락한 근본 원인, [4]에서 발견) — `split_sentences`가 이제 줄바꿈 자체도 경계로
    인정하도록 고쳐졌으므로, 여기서 줄바꿈을 미리 지우지 않고 그대로 전달해야 한다."""
    sentences = split_sentences(text)
    chunks, buf = [], ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > base_chars:
            chunks.append(buf.strip())
            buf = s
        else:
            buf = (buf + " " + s).strip()
    if buf:
        chunks.append(buf.strip())
    return chunks


_CONTEXT_PROMPT_TEMPLATE = (
    "다음은 증권사 리포트 전체 문서와 그 안의 특정 조각(chunk)입니다.\n\n"
    "<document>\n{full_doc_text}\n</document>\n\n"
    "<chunk>\n{chunk_text}\n</chunk>\n\n"
    "이 chunk가 전체 문서에서 어떤 맥락에 있는지(어느 회사/주제에 대한 내용인지 포함) "
    "검색 정확도를 높이기 위한 간결한 설명을 1~2문장으로만 작성하세요. 다른 설명 없이 "
    "그 문장만 출력하세요."
)


def generate_context_qwen(chunk_text: str, full_doc_text: str, model, processor, device,
                           max_new_tokens: int = 80) -> str:
    """백엔드 1: 로컬 Qwen2.5-VL-7B-Instruct — 이 프로젝트 기존 LLM 호출과 동일 모델 재사용,
    가장 정확하지만 청크마다 로컬 VLM 추론이 걸려 지연이 큼([3] 실측: 페이지당 6개 청크에
    ~87초)."""
    import torch
    prompt = _CONTEXT_PROMPT_TEMPLATE.format(full_doc_text=full_doc_text, chunk_text=chunk_text)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.3)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    result = processor.decode(trimmed, skip_special_tokens=True).strip()
    del inputs, out
    if device == "mps":
        torch.mps.empty_cache()
    return result


def generate_context_openai(chunk_text: str, full_doc_text: str, client=None,
                             model_name: str = "gpt-4o-mini", max_tokens: int = 80) -> str:
    """백엔드 2: OpenAI 경량 모델(API) — [5] 사용자 요청으로 추가, 지연/비용 트레이드오프 A/B용.
    API 키는 환경변수 OPENAI_API_KEY에서만 읽음(코드/저장소에 절대 하드코딩하지 않음 — 채팅에
    노출된 키는 대화 종료 후 즉시 폐기/재발급 권장)."""
    from openai import OpenAI
    client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = _CONTEXT_PROMPT_TEMPLATE.format(full_doc_text=full_doc_text, chunk_text=chunk_text)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


_NUM_HEADER_RE = re.compile(r"^\s*\d")


def generate_context_rulebased(chunk_text: str, section_path: list, doc_title: str = None) -> str:
    """백엔드 3: 규칙 기반(LLM 호출 없음, 비용/지연 거의 0) — 사용자 제안 반영. 계층적 청킹의
    section_path(문서 제목 -> 섹션 헤더 경로)를 그대로 문자열로 조립해 컨텍스트 접두어로 사용.
    LLM처럼 "이 청크의 요지"까지 요약하진 못하고 "어디에 속하는지" 구조 정보만 제공하는 한계는
    있음(표현력 상한 명확) — 그래도 대명사("동사"/"당사") 문제의 상당 부분은 "이 chunk는 [문서
    제목] 문서의 [섹션명] 부분입니다" 정도만으로도 완화됨(엔티티명이 섹션 경로 어딘가에 있는
    경우가 많아서)."""
    parts = [doc_title] if doc_title else []
    parts.extend(section_path or [])
    if not parts:
        return "이 chunk는 문서 본문의 일부입니다."
    path_str = " > ".join(dict.fromkeys(p for p in parts if p))
    return f"이 chunk는 다음 문맥에 속합니다: {path_str}"


def chunk_contextual(text: str, full_doc_text: str, page: int, base_chars: int = 250,
                      backend: str = "rulebased", **backend_kwargs) -> list:
    """반환: [{text(맥락 포함), raw_chunk(원문만), context_prefix, page}, ...]

    backend: "qwen"(model, processor, device 필요) | "openai"(client 또는 환경변수) |
    "rulebased"(section_path 필요, doc_title 옵션) — backend_kwargs로 각 백엔드별 인자 전달.

    [5] 기본값을 "rulebased"로 변경 — 사용자 지시("지연을 단축해야 함")에 따른 결정. A/B 실측
    (3문서 평균) 결과 openai(경량 API)는 rulebased 대비 entity_recall 이득이 전혀 없으면서
    (둘 다 59.3%) API 비용만 발생하고 fact_recall은 오히려 낮았음(51.7% vs 61.7%) — 유료 API로
    바꿀 이유가 없어 기각. 로컬 Qwen만 유일하게 더 높은 recall(89.3%)을 내지만 지연이 2.4배라
    "지연 단축" 지시와 상충 — 그래서 rulebased를 기본으로 채택. 단, 이 함수를 독립적으로 쓰면
    아래 `chunk_contextual_production`보다 문맥 품질이 떨어짐(placeholder 수준 section_path만
    쓰게 되므로) — 실제 배포는 `chunk_contextual_production`을 통해 쓸 것을 권장."""
    base_chunks = _naive_split(text, base_chars)
    results = []
    for raw_chunk in base_chunks:
        if backend == "qwen":
            context = generate_context_qwen(raw_chunk, full_doc_text,
                                             backend_kwargs["model"], backend_kwargs["processor"],
                                             backend_kwargs["device"])
        elif backend == "openai":
            context = generate_context_openai(raw_chunk, full_doc_text,
                                               client=backend_kwargs.get("client"),
                                               model_name=backend_kwargs.get("model_name", "gpt-4o-mini"))
        elif backend == "rulebased":
            context = generate_context_rulebased(raw_chunk, backend_kwargs.get("section_path"),
                                                  backend_kwargs.get("doc_title"))
        else:
            raise ValueError(f"unknown backend: {backend}")
        results.append({
            "text": f"{context}\n\n{raw_chunk}",
            "raw_chunk": raw_chunk,
            "context_prefix": context,
            "page": page,
        })
    return results


def chunk_contextual_production(model, doc_fitz, page_idx: int, doc_title: str = None,
                                 backend: str = "rulebased", full_doc_text: str = None,
                                 cached_boxes: list = None, **backend_kwargs) -> list:
    """[5]/[7] 프로덕션 파이프라인 — `chunk_contextual(backend="rulebased")`를 독립적으로 쓰면
    청크마다 같은 placeholder급 컨텍스트만 붙는 문제([5] A/B 테스트에서 실제로 그렇게
    구성했었음)를 해결: 계층적 청킹(`chunk_hierarchical`)으로 실제 문단 경계 + **청크별로 다른
    진짜 section_path**를 먼저 얻고, 그 각각에 컨텍스트를 주입한다.
    - 청킹 경계: YOLO 구조 기반(계층적 청킹의 강점 — [3]에서 청킹 자체 비용이 0.6~0.8s로
      가장 저렴했던 그 방식 그대로) — backend에 상관없이 항상 동일하게 사용
    - 컨텍스트 생성은 backend로 선택:
      - "rulebased"(기본, [6]에서 채택): section_path를 그대로 문자열로 조립, LLM 호출 없음
      - "openai"/"qwen": [5]에서 이미 검증된 컨텍스트 생성기를 재사용하되, [5]의 naive
        base_chars 분할이 아니라 **여기서 만든 진짜 문단 경계 청크**에 적용 — [7]에서
        사용자 제안으로 추가("SLM을 계층적 청킹에 붙이면 rulebased보다 낫지 않겠냐")한 A/B 대상.
        full_doc_text(전체 문서 텍스트, LLM 프롬프트용)를 넘겨야 함."""
    from hierarchical_chunker import chunk_hierarchical
    hier_chunks = chunk_hierarchical(model, doc_fitz, page_idx, cached_boxes=cached_boxes)
    results = []
    for c in hier_chunks:
        if backend == "rulebased":
            context = generate_context_rulebased(c["text"], c["section_path"], doc_title)
        elif backend == "openai":
            context = generate_context_openai(c["text"], full_doc_text,
                                               client=backend_kwargs.get("client"),
                                               model_name=backend_kwargs.get("model_name", "gpt-4o-mini"))
        elif backend == "qwen":
            context = generate_context_qwen(c["text"], full_doc_text,
                                             backend_kwargs["model"], backend_kwargs["processor"],
                                             backend_kwargs["device"])
        else:
            raise ValueError(f"unknown backend: {backend}")
        results.append({
            "text": f"{context}\n\n{c['text']}",
            "raw_chunk": c["text"],
            "context_prefix": context,
            "section_path": c["section_path"],
            "page": c["page"],
        })
    return results



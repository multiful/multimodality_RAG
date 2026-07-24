"""LLM 답변에 등장하는 숫자가 실제로 제공된 컨텍스트(근거)에 있는지 검증하고, 없으면 피드백과
함께 재생성을 요청한다. 근거 없는 숫자를 완전히 지어내는 케이스(예: 컨텍스트에 없는 주가/날짜를
답변이 만들어내는 경우)를 잡기 위한 최소 구현 — LangGraph 등 별도 프레임워크 없이 순수 함수로.

한계: "숫자가 컨텍스트 어딘가에 존재하는지"만 확인하므로, 단위 오귀속(예: 컨텍스트의 "365(십억원)"를
답변이 "365억원"으로 잘못 환산해 쓰는 경우)이나 틀린 연도에 숫자를 갖다붙이는 경우는 못 잡는다 —
숫자 자체는 컨텍스트에 실재하기 때문. 완전히 근거 없는 숫자를 지어내는 것만 걸러낸다.
"""

import re

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def _retryable_create(client, **kwargs):
    """[41] structured_output._retryable_parse()와 동일한 tenacity 재시도(429/5xx/연결오류/타임아웃만,
    스키마 오류 같은 4xx는 즉시 실패) — 사용자가 실제로 보는 최종 답변 생성 호출에는 이 보호가
    빠져 있어서 일시적 API 오류 한 번에 전체 요청이 그냥 실패하던 것을 수정."""
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    @retry(
        retry=retry_if_exception_type((RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call():
        return client.chat.completions.create(**kwargs)

    return _call()


def extract_numbers(text: str, min_digits: int = 3) -> set:
    """텍스트에서 숫자 토큰을 뽑는다(쉼표 제거). 자릿수가 너무 짧은 숫자(연도의 일부, 소제목
    번호 등)는 우연히 겹치는 오탐이 많아 기본적으로 제외."""
    numbers = set()
    for m in NUMBER_RE.findall(text):
        cleaned = m.replace(",", "")
        if len(cleaned.replace(".", "").lstrip("0")) >= min_digits:
            numbers.add(cleaned)
    return numbers


def find_unsupported_numbers(answer: str, context: str) -> list:
    """answer에 등장하는 숫자 중 context(원문 근거) 어디에도 없는 것들을 반환."""
    context_numbers = extract_numbers(context)
    answer_numbers = extract_numbers(answer)
    return sorted(n for n in answer_numbers if n not in context_numbers)


def generate_with_citation_check(client, prompt: str, context: str, model: str = "gpt-4o-mini",
                                  max_retries: int = 2, verbose: bool = True) -> dict:
    """LLM 답변을 생성하고, 근거 없는 숫자가 있으면 피드백과 함께 최대 max_retries회 재생성.

    반환: {"answer": str, "attempts": int, "unsupported_numbers": list(마지막 시도 기준)}"""
    messages = [{"role": "user", "content": prompt}]
    unsupported = []
    for attempt in range(max_retries + 1):
        # [수정] _retryable_create()가 정의만 되고 실제로는 안 쓰이고 있었음(docstring은 "수정
        # 완료"라 주장했지만 이 호출부는 계속 client.chat.completions.create()를 직접 불렀음) —
        # 사용자가 실제로 보는 최종 답변 생성 호출이라 재시도 보호가 특히 중요한 지점이었는데
        # 일시적 429/5xx 한 번에 데모 전체가 죽는 회귀 상태였음. 실제로 배선.
        resp = _retryable_create(client, model=model, messages=messages)
        answer = resp.choices[0].message.content
        unsupported = find_unsupported_numbers(answer, context)
        if not unsupported:
            return {"answer": answer, "attempts": attempt + 1, "unsupported_numbers": []}

        if verbose:
            print(f"   [검증] {attempt + 1}회차: 근거 없는 숫자 발견 {unsupported} -> 재생성 요청")
        if attempt < max_retries:
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content":
                f"방금 답변에 나온 숫자 중 다음 값들은 제공된 컨텍스트 어디에도 없습니다: "
                f"{unsupported}. 이 숫자들이 실제로 근거에 있는지 다시 확인하고, 근거가 없다면 "
                "해당 문장을 삭제하거나 컨텍스트에 실제로 있는 수치로 바꿔서 답변 전체를 다시 "
                "작성해주세요. 근거 없는 숫자를 만들어내지 마세요."})

    return {"answer": answer, "attempts": max_retries + 1, "unsupported_numbers": unsupported}

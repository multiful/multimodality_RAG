"""Shared RAG-IT-style instruction template, applied identically to every generator.

Modeled on the RAG-IT paper's {instruction, input, output} grounding scheme: the
model is instructed to answer strictly from the retrieved context, so the same
template can be fed to any backend for a fair comparison.
"""

SYSTEM_PROMPT = (
    "당신은 재무제표, 뉴스, 주가 데이터를 바탕으로 기업을 분석하는 애널리스트입니다.\n"
    "아래 제공된 컨텍스트(재무제표/뉴스/주가 정보)만을 근거로 답변하세요.\n"
    "컨텍스트에 없는 내용은 추측하지 말고, 정보가 부족하면 부족하다고 명시하세요.\n"
    "숫자는 컨텍스트에 있는 값을 그대로 정확히 인용하세요."
)

USER_TEMPLATE = "[컨텍스트]\n{context}\n\n[질문]\n{query}"


def build_messages(query: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, query=query)},
    ]

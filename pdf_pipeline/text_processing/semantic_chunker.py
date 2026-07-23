"""[3] 시멘틱 청킹(Semantic Chunking) — 문서 구조(제목/섹션)와 무관하게, 문장 임베딩(BGE-M3)의
연속 문장 간 코사인 유사도가 크게 떨어지는 지점("주제 전환")을 청크 경계로 삼는다. 계층적
청킹과 달리 Section-header 같은 명시적 구조 신호를 전혀 쓰지 않고, 순수하게 문장 의미
유사도만으로 경계를 정한다 — Greg Kamradt/LlamaIndex의 percentile 기반 semantic chunking과
동일한 방식(연속 문장 쌍의 거리(1-유사도) 분포에서 상위 percentile을 breakpoint로 채택).
"""

import re

import numpy as np

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?다요음])\s+(?=[가-힣A-Z(\[])")

# [4] 버그 발견·수정: 뉴스 불릿/리스트 항목("정부는...공식 의제로 올려", "다만...추진하겠다는
# 방침")은 마침표 없이 줄바꿈만으로 항목이 끝나는 경우가 많음 — 원래 코드는 마침표/종결어미
# 뒤 공백만 문장 경계로 봐서 이런 줄들이 전부 하나로 뭉쳐짐(construct_p5에서 5개 뉴스 항목 중
# 4개가 1017자짜리 "문장" 하나로 뭉쳐, semantic/contextual 청킹 모두 max_chars 상한이 무력화되고
# 뒷부분 엔티티가 LLM 추출 max_new_tokens 안에 못 들어가 recall이 20%까지 떨어짐). 줄바꿈도
# 항상 유효한 경계로 인정하도록 수정 — 문장 종결부호가 없어도 "한 줄 = 최소 하나의 의미 단위"로
# 취급(불릿/헤드라인 형태 콘텐츠에 안전, 줄바꿈된 일반 문단에도 무해 — 어차피 문장 중간에서
# 줄바꿈되는 경우는 없고 [1]에서 검증된 get_textbox()/whole_page 추출은 문단 끝에서만 줄바꿈됨).
_LINE_SPLIT_RE = re.compile(r"\n+")


def split_sentences(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    units = []
    for line in _LINE_SPLIT_RE.split(text):
        line = line.strip()
        if not line:
            continue
        units.extend(p.strip() for p in _SENT_SPLIT_RE.split(line) if p.strip())
    return units


def chunk_semantic(text: str, embed_model, page: int, breakpoint_percentile: int = 75,
                    min_sentences_per_chunk: int = 1, max_chars: int = 500) -> list:
    """반환: [{text, page, n_sentences}, ...]. embed_model은 SentenceTransformer 인스턴스.

    max_chars: 실측 발견 — 금융 애널리스트 리포트는 한 섹션 내 주제 일관성이 매우 높아서(계속
    "이 회사 실적/전망" 얘기), percentile 기반 유사도 breakpoint만으로는 1200자 넘는 거대 청크가
    나오는 경우 발견(문서 구조와 무관하게 순수 의미 유사도만 보므로, 주제가 안 바뀌면 안 쪼갬).
    실제 배포에서는 임베딩 모델의 컨텍스트 한도/검색 단위 세분성을 위해 상한이 필요해, breakpoint로
    나온 그룹이 max_chars를 넘으면 그 안에서 상대적으로 거리가 큰 지점부터 추가로 쪼갠다."""
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [{"text": text, "page": page, "n_sentences": len(sentences)}] if text.strip() else []

    embeddings = embed_model.encode(sentences, normalize_embeddings=True)
    sims = [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]
    distances = [1 - s for s in sims]

    threshold = np.percentile(distances, breakpoint_percentile)
    breakpoints = [i for i, d in enumerate(distances) if d >= threshold]

    groups, start = [], 0
    for bp in breakpoints:
        end = bp + 1  # sentences[start:end] 를 한 그룹으로
        if end - start >= min_sentences_per_chunk:
            groups.append((start, end))
            start = end
    if start < len(sentences):
        groups.append((start, len(sentences)))

    chunks = []
    for g_start, g_end in groups:
        group_sents = sentences[g_start:g_end]
        joined = " ".join(group_sents)
        if len(joined) <= max_chars:
            chunks.append({"text": joined, "page": page, "n_sentences": len(group_sents)})
            continue
        # 그룹이 max_chars를 넘으면 문장 단위로 그리디하게 재분할(단순하고 예측 가능한 방식 —
        # 거리 기반 재정렬은 문장 수가 적을 때 불안정해서 단순 누적 방식으로 교체)
        buf_sents, buf_len = [], 0
        for s in group_sents:
            if buf_sents and buf_len + len(s) + 1 > max_chars:
                chunks.append({"text": " ".join(buf_sents), "page": page, "n_sentences": len(buf_sents)})
                buf_sents, buf_len = [], 0
            buf_sents.append(s)
            buf_len += len(s) + 1
        if buf_sents:
            chunks.append({"text": " ".join(buf_sents), "page": page, "n_sentences": len(buf_sents)})
    return chunks

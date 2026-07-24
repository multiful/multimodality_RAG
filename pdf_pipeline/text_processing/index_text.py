"""[35] 텍스트 인덱싱 틀 — BM25 + BGE-m3-ko 하이브리드 검색 스켈레톤.

사용자 요청: "텍스트의 bm25 + bge-m3-ko로 인덱싱 과정 연결해놓게 틀 잡아놓자." Supabase
스키마가 아직 팀원 쪽에서 확정 전이라(실제 Supabase 프로젝트 확인 결과 스키마 미정, 대화 기록
참고), 이번엔 저장소를 **인메모리**로 두고 인터페이스(`build_index`/`hybrid_search`)만 먼저
잡는다 — 나중에 Supabase(pgvector + full-text search)로 교체할 때 이 두 함수의 내부 구현만
바꾸면 되고, 호출부(인덱싱 스크립트/쿼리 핸들러)는 그대로 재사용 가능하도록 설계.

다이어그램 반영: "핵심모델: DENSE(BGE-m3-ko), 보조모델: BM25" — dense_weight를 bm25_weight보다
높게 기본 설정. Rank Fusion 기본값은 min-max 정규화 후 가중합(fusion="weighted_sum")이고,
[42]에서 RRF(fusion="rrf")를 추가해 dense-only/dense+BM25(가중합)/dense+BM25(RRF) 세 방식을
`evaluate_hybrid_search.py`로 비교 검증했다.

[43] 검색 증강 기법 A/B 테스트 추가 — `classify_query_type()`(규칙 기반 쿼리 타입 분류),
`mqe_search()`(Multi-Query Expansion), `hyde_search()`(Hypothetical Document Embeddings),
`route_search()`(타입별 라우팅: summary형만 MQE로, 나머지는 hybrid RRF로). 전체 비교는
`evaluate_retrieval_ab.py`에서 NDCG/MAP/MRR/Recall/Precision/F1/지연으로 검증.

[44] 사용자 지적("분류를 잘 하는지 모르겠다", "2라우팅으로 충분해?") 반영 — (1)
`classify_query_type_llm()` 추가, 25개 라벨셋(`query_type_labeled_set.json`)으로 규칙기반(80%)
vs gpt-4o-mini(84%) vs gpt-4o(92%) 실측 비교 후 `route_search()` 기본 분류기를 LLM(gpt-4o)으로
변경(`evaluate_query_classifier.py`). (2) 라우팅은 fusion 전략 기준으로 이진(keyword_specific/
abstract)이면 충분함을 재확인 — route_search()가 그 결론 반영. (3) 사용자 참고 다이어그램의
"Parent-Child Retrieval" 반영해 `expand_to_parent_context()` 추가 — 재색인 없이 검색된 청크와
같은 페이지의 나머지 청크를 원문 순서로 붙여 요약형 질의에 필요한 맥락을 보강.

[51] 사용자 지적("업로드와 쿼리가 같이 주어지는데 9초 기다리고 받는 게 좋나?") 반영 —
`ingest_and_search()` 추가. 인덱싱(청킹+임베딩)과 질의 준비(분류+MQE 하위질의 생성, 엔티티
카운트)는 서로 의존관계가 없어(질의 준비는 질의 텍스트/원본 PDF 텍스트만 있으면 됨, 인덱스
불필요) 동시에 실행하면 벽시계 지연을 겹치는 만큼 숨길 수 있음을 실측(`test_concurrent_
upload_query.py`) — 다기업(MQE) 문서 48% 단축, 단일기업(HyDE) 문서는 14%만 단축(HyDE 가상
문단 생성이 엔티티 수를 알아야 결정돼 인덱싱과 못 겹쳤음). speculative_hyde=True로 그 갭도
마저 겹치되, 결과가 버려질 수 있는(다기업으로 판명나면) 토큰 낭비 트레이드오프는 코드에 명시.
`route_search(index, query)`는 인덱스가 이미 있는 후속 질의에 그대로 씀 — 대체 아님, 병행.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list:
    """BM25용 토크나이저 — 형태소 분석기 없이 한글 어절/영숫자 단위로만 쪼갬(정교한 한국어
    토크나이저는 이후 개선 과제로 남김 — 지금은 dense가 메인이라 BM25는 보조 신호 정도로 충분)."""
    return re.findall(r"[가-힣]+|[A-Za-z0-9]+", text.lower())


@dataclass
class TextIndex:
    pdf_id: str
    chunk_ids: list = field(default_factory=list)
    chunks: list = field(default_factory=list)   # process_pdf() chunk dict 그대로(text/raw_chunk/section_path/page/structured_metadata 등)
    embeddings: object = None                     # np.ndarray (N, dim), normalize_embeddings=True
    bm25: object = None                            # rank_bm25.BM25Okapi
    entity_count: int = None                       # [45] count_document_entities() 결과 캐시 —
                                                    # 문서(인덱스) 속성이라 질의마다 재계산하면 안 됨.
                                                    # route_search()가 None이면 1회 계산해 여기 채움.


def build_index(pdf_id: str, process_pdf_result: dict, embed_model=None) -> TextIndex:
    """`text_extraction.process_pdf()`(또는 `process_pdf_streaming()`을 다 모은 결과)의 pages에서
    모든 chunks를 모아 (1) BGE-m3-ko 임베딩, (2) BM25 인덱스를 만든다. 임베딩 대상은 `c["text"]`
    (컨텍스트 접두어 포함본, [5]/[9]에서 검증된 대로 이게 실제 검색 품질이 더 좋았음) — raw_chunk가
    아님에 주의."""
    if embed_model is None:
        from embedding import get_embedding_model
        embed_model = get_embedding_model()

    chunks = [c for page in process_pdf_result["pages"] for c in page["chunks"]]
    if not chunks:
        return TextIndex(pdf_id=pdf_id)

    chunk_ids = [f"{pdf_id}_p{c['page']}_{i}" for i, c in enumerate(chunks)]
    texts = [c["text"] for c in chunks]

    from embedding import embed_texts
    embeddings = embed_texts(texts)

    bm25 = BM25Okapi([_tokenize(t) for t in texts])

    return TextIndex(pdf_id=pdf_id, chunk_ids=chunk_ids, chunks=chunks, embeddings=embeddings, bm25=bm25)


def _rrf_scores(scores, k: int = 60):
    """[42] 사용자 요청("RRF를 먼저 도입해서 시험") 반영 — RRF(Reciprocal Rank Fusion, Cormack et
    al. 2009)는 점수 값 자체가 아니라 "순위"만 이용해 결합한다. dense(코사인 유사도, -1~1)와
    BM25(비정규화, 코퍼스마다 범위가 다름)처럼 스케일이 다른 두 신호를 min-max 정규화 없이
    결합할 수 있어서, 이상치 하나가 정규화 전체를 흔드는 문제에서 자유롭다. k=60은 원 논문
    기본값(대부분의 IR 구현체가 그대로 씀 — 코퍼스 규모별 재튜닝은 드묾)."""
    import numpy as np
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return 1.0 / (k + ranks)


def _apply_source_weights(index: TextIndex, fused):
    """[48] 사용자 지적("가중치가 실제로 적용이 안 되는데?") 반영 — entity_fusion.py가 evidence
    아이템마다 계산해둔 `weight`(예: canonical_field 매칭 안 된 표 행은 0.6으로 강등, [48])가
    entity_fusion.weighted_hybrid_search()에서만 곱해지고 있었고, route_search()/mqe_search()/
    hyde_search()/hybrid_search() 등 실제 라우팅에 쓰이는 경로에는 전혀 반영이 안 되고
    있었음(실측: C밴드.pdf에서 표 가중치를 0.6으로 낮췄는데도 route_search() 결과가 그대로였던
    이유). 모든 융합 함수가 마지막에 이 함수 하나만 거치면 weight가 어느 경로로 검색하든
    일관되게 반영됨(weight 필드가 없는 청크는 1.0 기본값이라 순수 텍스트 전용 인덱스에는
    영향 없음)."""
    import numpy as np
    weights = np.array([c.get("weight", 1.0) for c in index.chunks])
    return fused * weights


def hybrid_search(index: TextIndex, query: str, embed_model=None, top_k: int = 5,
                   dense_weight: float = 0.7, bm25_weight: float = 0.3,
                   fusion: str = "weighted_sum", rrf_k: int = 60) -> list:
    """Dense(코사인 유사도) + BM25 점수를 결합(Rank Fusion). fusion="weighted_sum"(기본, 기존 동작
    그대로)은 각 점수를 min-max 정규화 후 가중합. fusion="rrf"는 [42] 순위 기반 RRF로 결합 —
    dense_weight/bm25_weight는 이 모드에서 쓰이지 않고 rrf_k만 적용됨.
    반환: [{chunk_id, chunk, score, dense_score, bm25_score}, ...] score 내림차순 top_k."""
    if not index.chunks:
        return []
    if embed_model is None:
        from embedding import get_embedding_model
        embed_model = get_embedding_model()

    import numpy as np
    from embedding import embed_texts
    query_emb = embed_texts([query])[0]
    dense_scores = np.asarray(index.embeddings) @ query_emb
    bm25_scores = np.asarray(index.bm25.get_scores(_tokenize(query)))

    if fusion == "dense_only":
        # [43] 사용자 가설("키워드성 질의엔 BM25+dense, 추상적 질의엔 dense 위주") 검증용 —
        # 실측(LGCNS)으로 BM25가 abstract 질의에서 순수 노이즈일 뿐 아니라, 길이정규화 때문에
        # 짧고 무의미한 청크(제목/티커 한 줄)를 오히려 우대해 "적극적으로 해가 됨"을 확인했음
        # (실험.md [43] 참고) — 그런 경우 BM25를 아예 빼는 옵션.
        fused = dense_scores
    elif fusion == "rrf":
        fused = _rrf_scores(dense_scores, k=rrf_k) + _rrf_scores(bm25_scores, k=rrf_k)
    else:
        def _normalize(arr):
            span = arr.max() - arr.min()
            return (arr - arr.min()) / span if span > 0 else np.zeros_like(arr)
        fused = dense_weight * _normalize(dense_scores) + bm25_weight * _normalize(bm25_scores)

    fused = _apply_source_weights(index, fused)
    order = np.argsort(-fused)[:top_k]
    return [
        {"chunk_id": index.chunk_ids[i], "chunk": index.chunks[i], "score": float(fused[i]),
         "dense_score": float(dense_scores[i]), "bm25_score": float(bm25_scores[i])}
        for i in order
    ]


def _raw_scores(index: TextIndex, query: str, query_embedding=None):
    """dense_scores/bm25_scores 계산 공통 로직 — hybrid_search/mqe_search/hyde_search가 공유.
    query_embedding을 직접 주면(HyDE처럼 원 질의가 아닌 다른 텍스트의 임베딩을 dense 쪽에 쓰고
    싶을 때) 그걸 그대로 쓰고, BM25는 항상 원 질의(query)의 토큰으로 계산한다(BM25는 어휘
    매칭이라 가상 문서가 아니라 실제 사용자 질의 단어와 맞춰야 함)."""
    import numpy as np
    if query_embedding is None:
        from embedding import embed_texts
        query_embedding = embed_texts([query])[0]
    dense_scores = np.asarray(index.embeddings) @ np.asarray(query_embedding)
    bm25_scores = np.asarray(index.bm25.get_scores(_tokenize(query)))
    return dense_scores, bm25_scores


QUERY_TYPES = ("factoid", "list", "comparison", "summary")

_TYPE_KEYWORDS = {
    "summary": ["요약", "인사이트", "도출", "총평", "정리해", "분석해줘", "브리핑", "리뷰해"],
    # [43] "리스트"를 단독 키워드로 뒀다가 "애널리스트"(analyst)에 부분 문자열로 걸리는 오탐을
    # 실측으로 발견 — 한글 외래어 표기가 통째로 들어있는 단어와 충돌하기 쉬운 키워드라 제거하고
    # "~리스트를"/"~리스트가"처럼 조사가 붙는 실제 사용 패턴만 남김.
    "list": ["top pick", "탑픽", "추천 종목", "종목들", "리스트를", "리스트가", "뭐가 있", "어떤 것들", "나열"],
    "comparison": ["대비", "비교", "얼마나 늘었", "얼마나 줄었", "성장률", "추세", "변동", "전분기", "전년"],
}


def classify_query_type(query: str) -> str:
    """[43] 사용자 요청("쿼리타입분류는 너가 적당하게 분류기준 잡아서") — 정교한 분류기(임베딩
    기반/LLM 기반)까지는 오버엔지니어링이라 판단, 4가지 타입에 대한 키워드 휴리스틱으로 시작:

      - factoid    : 특정 수치/사실 하나를 묻는 질의(예: "매출액이 얼마야?") — 정답 청크가 보통
                     문서 안에 1개. 다른 타입 키워드에 안 걸리면 기본값으로 여기 배정.
      - list       : 여러 항목 나열을 묻는 질의(예: "TOP PICK 종목이 뭐야?") — 정답이 소수의
                     항목 집합.
      - comparison : 비교/추세를 묻는 질의(예: "전분기 대비 얼마나 늘었어?").
      - summary    : 여러 근거를 종합해야 하는 요약/분석 질의(예: "이 기업의 이벤트를 요약하고
                     투자 인사이트를 도출해줘") — 정답 청크가 문서 전반에 넓게 퍼져 있어서 커버리지
                     (recall)가 핵심이고, 단일 top-1 정확도보다 다양성이 중요.

    이 타입이 route_search()에서 검색 전략 선택에 쓰인다(summary만 MQE로 라우팅, 나머지는
    기본 하이브리드+RRF — MQE는 LLM 호출이 추가로 드는 만큼 커버리지가 실제로 중요한 질의에만
    씀)."""
    q = query.lower()
    for qtype in ("summary", "list", "comparison"):
        if any(kw in q for kw in _TYPE_KEYWORDS[qtype]):
            return qtype
    return "factoid"


_LLM_CLASSIFY_PROMPT = (
    "다음 사용자 질의가 증권사 리포트 검색에서 어느 쪽에 가까운지 판단하세요.\n\n"
    "- keyword_specific: 특정 수치/사실/고유명사(기업명, 티커, 금액, 날짜, 지표명 등)를 콕 집어 "
    "묻는 질의. 문서 안의 구체적인 어휘와 직접 매칭되는 게 보통(예: '매출액이 얼마야?', "
    "'목표주가는?', 'TOP PICK 종목은?', '전분기 대비 얼마나 늘었어?').\n"
    "- abstract: 여러 근거를 종합/요약/판단해야 답할 수 있는 질의. 문서 어휘와 직접 안 겹치는 "
    "메타/평가성 언어를 쓰는 경우가 많음(예: '이 회사 어때?', '투자할 만해?', '브리핑해줘', "
    "'전반적으로 설명해줘', '이벤트 요약과 인사이트를 도출해줘').\n\n"
    "질의: \"{query}\"\n\n"
    "keyword_specific 또는 abstract 중 하나만, 다른 텍스트 없이 그대로 출력하세요."
)


def classify_query_type_llm(query: str, client=None, model: str = "gpt-4o") -> str:
    """[44] 사용자 지적("규칙 기반 분류가 잘 되는지 모르겠다") 반영 — classify_query_type()의
    키워드 휴리스틱은 트리거 단어(요약/인사이트/도출 등)에 없는 표현("이 회사 어때?", "투자할
    만해?")은 놓치고 기본값 factoid(=keyword_specific)로 오분류한다 — 직접 만든 25개 라벨셋
    (query_type_labeled_set.json, 트리거 단어를 의도적으로 피한 abstract 표현 6개 포함)으로
    실측한 결과 규칙 기반 80%(5개 전부 이 블라인드스팟), gpt-4o-mini 84%, gpt-4o 92%
    (evaluate_query_classifier.py) — mini가 빠르고 쌀 거라 기본값으로 뒀었는데 실제로는 4o가
    유의미하게 더 정확해서(단순 이진 분류인데도) 기본값을 gpt-4o로 바꿈. 이 판정이 잘못되면
    "abstract 질의를 잘 처리하자"는 이번 개선 전체가 무력화되는 비대칭적 리스크가 있어 정확도를
    우선(N=25로 작아서 확정적이진 않음 — 더 큰 라벨셋으로 재검증 권장)."""
    if client is None:
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": _LLM_CLASSIFY_PROMPT.format(query=query)}],
    )
    answer = resp.choices[0].message.content.strip().lower()
    return "abstract" if "abstract" in answer else "keyword_specific"


_CLASSIFY_AND_EXPAND_PROMPT = (
    "다음은 증권사 리포트에 대한 사용자 질의입니다: \"{query}\"\n\n"
    "먼저 판단하세요:\n"
    "- 이 질의가 특정 수치/사실/고유명사(기업명, 티커, 금액, 날짜, 지표명 등)를 콕 집어 묻는 "
    "keyword_specific 질의라면, 다른 말 없이 정확히 KEYWORD_SPECIFIC 한 줄만 출력하세요.\n"
    "- 이 질의가 여러 근거를 종합/요약/판단해야 답할 수 있는 abstract 질의라면(문서 어휘와 직접 "
    "안 겹치는 메타/평가성 언어를 쓰는 경우가 많음 — 예: '이 회사 어때?', '브리핑해줘', '이벤트 "
    "요약과 인사이트를 도출해줘'), 이 질의에 제대로 답하기 위해 리포트에서 확인해야 할 구체적인 "
    "하위 질문 {n}개를 한 줄씩(번호/설명 없이 질문 텍스트만) 출력하세요. 각 하위 질문은 서로 "
    "다른 측면(실적/이벤트, 목표주가/투자의견, 리스크, 개별 대상별 현황 등)을 다루도록 하세요.\n\n"
    "KEYWORD_SPECIFIC 한 줄, 또는 하위 질문 {n}줄 중 하나만 출력하고 그 외 텍스트는 넣지 마세요."
)


def _classify_and_expand(query: str, client=None, n_subqueries: int = 4,
                          model: str = "gpt-4o") -> tuple:
    """[44] 사용자 지적("지금 병목 해결해줘") 반영 — route_search()가 기존엔 (1)
    classify_query_type_llm() 호출로 분류(~700ms) 후 abstract면 (2) mqe_search() 내부에서 또
    한 번 LLM을 불러 하위질의를 생성(~1.5s)해서, abstract 질의는 LLM 라운드트립을 순차로 2번
    거쳤다(합 ~2.2s+). 분류와 하위질의 생성을 프롬프트 하나로 합쳐 한 번의 호출로 끝낸다 —
    keyword_specific이면 그 자리에서 'KEYWORD_SPECIFIC' 한 줄만 받고, abstract면 그 한 번의
    응답 자체가 이미 하위질의 목록이라 별도 생성 호출이 필요 없다. abstract 질의의 LLM
    라운드트립을 2회->1회로 줄임(keyword_specific 질의는 원래도 1회 호출이라 변화 없음).
    반환: ("keyword_specific", None) 또는 ("abstract", [하위질의, ...])."""
    if client is None:
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user",
                   "content": _CLASSIFY_AND_EXPAND_PROMPT.format(query=query, n=n_subqueries)}],
    )
    lines = [l.strip("-•. \t") for l in resp.choices[0].message.content.splitlines() if l.strip()]
    if len(lines) == 1 and "keyword_specific" in lines[0].lower():
        return "keyword_specific", None
    return "abstract", lines[:n_subqueries]


_MQE_SUBQUERY_PROMPT = (
    "다음은 증권사 리포트에 대한 사용자 질의입니다: \"{query}\"\n"
    "이 질의에 제대로 답하려면 리포트에서 어떤 구체적인 정보들을 확인해야 하는지, "
    "검색에 쓸 구체적인 하위 질문 {n}개를 한 줄씩 생성하세요. "
    "각 하위 질문은 서로 다른 측면(예: 실적/이벤트, 목표주가/투자의견, 리스크, 개별 대상별 "
    "현황 등)을 다루도록 하고, 번호나 설명 없이 질문 텍스트만 줄바꿈으로 구분해 출력하세요."
)


def _fuse_multi_query(index: TextIndex, queries: list, top_k: int, rrf_k: int = 60,
                       use_bm25: bool = True) -> list:
    """[44] mqe_search()에서 분리 — 이미 생성된 하위질의 목록으로 RRF 융합 검색만 수행(LLM 호출
    없음). route_search()의 병목 수정([44] 분류+하위질의 생성 단일 호출화)이 이 부분을 재사용."""
    import numpy as np
    fused = np.zeros(len(index.chunks))
    for q in queries:
        dense_scores, bm25_scores = _raw_scores(index, q)
        fused += _rrf_scores(dense_scores, k=rrf_k)
        if use_bm25:
            fused += _rrf_scores(bm25_scores, k=rrf_k)

    fused = _apply_source_weights(index, fused)
    order = np.argsort(-fused)[:top_k]
    return [
        {"chunk_id": index.chunk_ids[i], "chunk": index.chunks[i], "score": float(fused[i])}
        for i in order
    ]


def mqe_search(index: TextIndex, query: str, client=None, top_k: int = 5, n_subqueries: int = 4,
               model: str = "gpt-4o-mini", rrf_k: int = 60, use_bm25: bool = True) -> list:
    """[43] MQE(Multi-Query Expansion) — 사용자 요청(A/B 테스트 대상) 반영. 넓은 질의 하나로만
    검색하면 상위 결과가 비슷한 톤의 청크로 쏠릴 위험이 있음(특히 summary형 질의처럼 문서 전반의
    다양한 근거를 모아야 하는 경우). LLM으로 원 질의를 구체적인 하위 질문 n_subqueries개로 쪼갠
    뒤, 각 하위 질의(+원 질의)로 따로 검색해서 결과를 RRF로 합쳐 커버리지를 넓힌다.

    use_bm25=False: [43] 사용자 가설("추상적 질의엔 BM25를 빼는 게 나을 수도") 검증용 — 실측으로
    abstract 질의는 BM25 토큰이 청크와 거의 안 겹치고(LGCNS 사례: 질의 토큰 10개 중 겹친 건
    조사 "이" 하나뿐), 그나마도 BM25 길이정규화가 그 우연한 매치를 가장 짧고 무의미한 청크에
    가장 후하게 쳐줘서 순위를 오염시킴(실험.md [43]) — 하위질의 각각에서 BM25 기여를 빼고
    dense 랭크만 RRF로 합치는 옵션.

    temperature=0: [43] A/B 비교 재현성을 위해 고정(기본 온도로는 하위질의 생성이 실행마다
    달라져서 use_bm25 True/False 비교가 LLM 샘플링 노이즈에 묻히는 걸 실측으로 확인).

    독립적으로 분류+하위질의 생성을 각각 호출하는 이 함수는 A/B 비교용으로 남겨두고, 실제
    route_search()는 [44]에서 두 호출을 하나로 합친 _classify_and_expand()를 쓴다(지연 단축)."""
    if client is None:
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": _MQE_SUBQUERY_PROMPT.format(query=query, n=n_subqueries)}])
    sub_queries = [line.strip("-•. \t") for line in resp.choices[0].message.content.splitlines() if line.strip()]
    all_queries = [query] + sub_queries[:n_subqueries]

    hits = _fuse_multi_query(index, all_queries, top_k=top_k, rrf_k=rrf_k, use_bm25=use_bm25)
    return hits, all_queries


def hyde_search(index: TextIndex, query: str, client=None, top_k: int = 5,
                model: str = "gpt-4o-mini", rrf_k: int = 60, use_bm25: bool = True) -> list:
    """[43] HyDE(Hypothetical Document Embeddings, Gao et al. 2022) — 사용자 요청(A/B 테스트 대상)
    반영. 원 질의(보통 짧은 문장)를 그대로 임베딩하는 대신, LLM에게 "이 질의에 대한 답이 될 법한
    가상의 리포트 문단"을 생성시켜 그 문단을 임베딩한다 — 가상의 상세 문단이 실제 관련 청크와
    임베딩 공간에서 더 가까운 경우가 많다는 아이디어. BM25는 여전히 원 질의 토큰으로 계산(가상
    문단은 실제 사실이 아니므로 어휘 매칭에는 원 질의가 더 안전)하고, RRF로 dense(가상문단)+
    BM25(원질의)를 결합한다. use_bm25=False면 mqe_search와 동일한 이유로 BM25를 아예 뺌."""
    if client is None:
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = (
        f"다음 질의에 대한 답이 될 법한 가상의 증권사 리포트 문단을 작성하세요(실제 사실은 모르니 "
        f"형식/톤/다룰 법한 항목만 그럴듯하게 채우고, 구체적 수치는 꾸며내되 사실 여부는 중요하지 "
        f"않습니다 — 이 문단은 검색용 임베딩에만 쓰입니다): \"{query}\""
    )
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                          temperature=0)
    hypothetical_passage = resp.choices[0].message.content

    import numpy as np
    from embedding import embed_texts
    hyde_embedding = embed_texts([hypothetical_passage])[0]
    dense_scores, bm25_scores = _raw_scores(index, query, query_embedding=hyde_embedding)
    fused = _rrf_scores(dense_scores, k=rrf_k)
    if use_bm25:
        fused += _rrf_scores(bm25_scores, k=rrf_k)

    fused = _apply_source_weights(index, fused)
    order = np.argsort(-fused)[:top_k]
    return [
        {"chunk_id": index.chunk_ids[i], "chunk": index.chunks[i], "score": float(fused[i])}
        for i in order
    ], hypothetical_passage


def route_search(index: TextIndex, query: str, client=None, top_k: int = 5,
                  classifier: str = "llm", entity_aware: bool = True) -> tuple:
    """[43]/[44] 쿼리 분류 라우팅 — 두 갈래(이진, [44]에서 4종 분류를 이 축으로 통일):

      - abstract(추상/종합형): mqe_search(use_bm25=False) — [43] 실측(LGCNS)으로 이런 질의는
        BM25 토큰이 청크와 사실상 안 겹치고("이벤트/요약/인사이트/도출" 같은 메타 어휘가 실제
        문서 어휘 "매출/영업이익/목표주가"와 겹칠 리 없음), 그나마 우연히 겹치는 조사(예: "이")
        마저 BM25 길이정규화가 가장 짧고 무의미한 청크(제목/티커 한 줄)에 가장 후하게 점수를
        줘서 순위를 오염시킴을 확인 — dense(+MQE로 다기업/다측면 커버리지 확보) 위주로 감.
      - keyword_specific(factoid/list/comparison 통합): hybrid_search(fusion="rrf") — 이런
        질의는 "매출액", "영업이익", 티커명처럼 청크에 그대로 등장하는 구체적 어휘를 담고 있어
        BM25가 정확한 어휘 매칭으로 dense를 보완(실측: 15개 factoid 위주 질의에서 하이브리드가
        dense-only 대비 recall@1 86.7%->93.3%로 개선, `result_hybrid_search_eval.json` 참고).
      두 갈래로 충분한지(3종 이상 세분화 필요한지)는 fusion 전략 기준으로는 이 실측들로 충분히
      뒷받침됨 — list/comparison은 top_k를 키우는 정도의 추가 튜닝 여지는 있지만 fusion 방식
      자체는 factoid와 동일하게 둬도 됨(실험.md [44] 참고).

    classifier="llm"(기본, [44]): gpt-4o 기반 분류 — 규칙 기반보다 12%p 더 정확(25개 라벨셋
    기준 80%->92%), 특히 트리거 단어 없는 abstract 표현(예: "이 회사 어때?")을 규칙 기반은
    100% 놓치는데 LLM은 다 잡음. [44] 병목 수정: 분류와 abstract일 때의 MQE 하위질의 생성을
    `_classify_and_expand()` 한 번의 LLM 호출로 합쳐서, abstract 질의가 LLM 라운드트립을
    순차로 2번(분류+생성, 합 ~2.2s+) 거치던 걸 1번으로 줄임(keyword_specific은 원래도 1회라
    변화 없음). classifier="rule"이면 무료/무지연 규칙 기반(classify_query_type()의 4종 분류를
    keyword_specific/abstract로 접어서 사용) — 지연이 민감한 상황에서 대안으로 남겨둠.

    entity_aware=True(기본, [45]): abstract로 분류된 질의를 문서의 entity_count로 다시 갈래를
    나눈다 — entity_count<=1(단일기업 문서)이면 hyde_search(use_bm25=False), 그 외(2개 이상,
    다기업 문서)면 MQE 경로. 실측(LGCNS=1기업/납기=3기업/KWave=10기업, `evaluate_retrieval_ab.py`)
    으로 이 경계가 뚜렷했음 — 단일기업만 HyDE가 확실히 우세(LGCNS ndcg 0.967 vs MQE 0.749)하고,
    기업이 2개만 돼도(납기) 이미 MQE가 우세(0.792 vs HyDE 0.403~0.526)해서 임계치를 1/2 사이로
    잡음. entity_aware=False면 [44]와 동일하게 abstract는 항상 MQE(구 동작, 비교용으로 남겨둠).

    [46] 사용자 지적("캐시 미스/지연 병목 해결") 반영 — index.entity_count는 여기서 계산하지
    않는 걸 강력 권장. 인제스트 시점에 `precompute_entity_count(index, pdf_path=...)`를 미리
    한 번 불러 채워두면(하나증권 포맷이면 정규식이라 LLM 호출 0회, 무료/즉시), route_search()는
    캐시만 읽어 추가 지연이 0이다. 혹시 안 채워져 있으면(index.entity_count is None) 이번
    호출에서 딱 1회 계산해 채우긴 하지만(pdf_path 없이 호출되므로 하나증권 정규식 경로를 못
    타 LLM 폴백만 가능 — [46]의 count_document_entities() 참고), 그 1회의 지연을 사용자가
    그대로 체감하게 되니 프로덕션에서는 반드시 인제스트 시점 precompute를 쓸 것.
    반환: (hits, query_type)."""
    if classifier == "llm":
        qtype, sub_queries = _classify_and_expand(query, client=client)
    else:
        qtype = "abstract" if classify_query_type(query) == "summary" else "keyword_specific"
        sub_queries = None

    if qtype != "abstract":
        return hybrid_search(index, query, top_k=top_k, fusion="rrf"), qtype

    if not entity_aware:
        if sub_queries is not None:
            hits = _fuse_multi_query(index, [query] + sub_queries, top_k=top_k, use_bm25=False)
        else:
            hits, _ = mqe_search(index, query, client=client, top_k=top_k, use_bm25=False)
        return hits, qtype

    if index.entity_count is None:
        index.entity_count = count_document_entities(index, client=client)

    if index.entity_count <= 1:
        hits, _ = hyde_search(index, query, client=client, top_k=top_k, use_bm25=False)
    elif sub_queries is not None:
        hits = _fuse_multi_query(index, [query] + sub_queries, top_k=top_k, use_bm25=False)
    else:
        hits, _ = mqe_search(index, query, client=client, top_k=top_k, use_bm25=False)
    return hits, qtype


# ---------- [52] 복합 질의 분해 라우팅 ----------
# 사용자 지적(2026-07-24, Construct PDF 실측) — route_search()는 질의 전체를 통째로 하나의
# 타입(keyword_specific/abstract)으로만 분류한다. "여기 나온 기업의 인사이트 도출해주고(추상,
# 종합 필요) + 건설업종 종목 주간 수익률이 가장 높은 기업은 어딘지 추출해(키워드/비교형, 정밀
# 매칭 필요)"처럼 성격이 다른 절이 한 질의에 섞이면, 분류기(rule/LLM 둘 다 "인사이트"/"도출"
# 트리거 단어 때문에)가 전체를 abstract로만 보고 abstract 경로의 use_bm25=False가 BM25를
# 통째로 버려서, 키워드 정밀 매칭이 필요한 뒷절의 정답 근거가 밀려나는 걸 실측으로 확인함
# (도표3 "건설업종" vs 도표4 "건자재업종" 차트가 근소하게 역전 — BM25 없이는 캡션의 정확한
# 어휘 매칭 이점을 못 씀). 이 함수는 질의를 먼저 성격별로 쪼갠 뒤 각 하위질의를 route_search와
# 동일한 갈래(keyword_specific→hybrid RRF, abstract→entity_count 기반 HyDE/MQE)로 따로 검색해
# RRF로 합친다 — route_search를 대체하는 상위 호환(단일 성격 질의는 분해 결과가 원 질의 그대로
# 1개라 route_search와 동일하게 동작).

_DECOMPOSE_PROMPT = (
    "다음 사용자 질의를 분석하세요: \"{query}\"\n\n"
    "이 질의 안에 성격이 다른 하위 요청이 여러 개 섞여 있는지 판단하세요:\n"
    "- keyword_specific 성격: 특정 수치/사실/고유명사(기업명, 티커, 금액, 날짜, 지표명 등)를 콕 "
    "집어 찾거나, 여러 대상 중 조건(최고/최저/비교)에 맞는 것을 추출하는 요청.\n"
    "- abstract 성격: 여러 근거를 종합/요약/평가해야 답할 수 있는 요청(예: 인사이트 도출, 브리핑, "
    "전반적 평가).\n\n"
    "규칙:\n"
    "1. 한 가지 성격만 있으면(복합 아님) 원본 질의를 그대로 한 줄만 출력하세요: "
    "\"SINGLE|<원본 질의 그대로>\"\n"
    "2. 성격이 다른 하위 요청이 여러 개 섞여 있으면, 각각을 독립적으로 검색 가능한 완전한 문장으로 "
    "나눠(원래 문맥/의미를 각자 온전히 보존, 대명사·생략된 주어 채우기) 한 줄씩 "
    "\"keyword_specific|<하위질의>\" 또는 \"abstract|<하위질의>\" 형식으로 출력하세요.\n"
    "다른 설명 없이 위 형식의 줄만 출력하세요."
)


def decompose_query(query: str, client=None, model: str = "gpt-4o") -> list:
    """질의를 성격별 하위질의로 분해. 단일 성격이면 [{"type": "single", "query": 원본}] 하나만
    반환(하위 라우팅에서 "single"은 route_search와 동일하게 LLM 재분류를 한 번 더 거쳐
    keyword_specific/abstract를 정함 — 분해 단계에서 이미 판단했지만 재사용 안 하는 이유는
    SINGLE 태그 자체가 "분해 불필요"라는 의미이지 타입까지 확정하는 게 아니기 때문). 파싱
    실패(형식 안 맞는 응답 등) 시 안전망으로 [{"type":"single","query":원본}] 반환."""
    if client is None:
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": _DECOMPOSE_PROMPT.format(query=query)}])
    lines = [l.strip() for l in resp.choices[0].message.content.splitlines() if l.strip()]
    parsed = []
    for line in lines:
        if "|" not in line:
            continue
        qtype, subq = line.split("|", 1)
        qtype, subq = qtype.strip().lower(), subq.strip().strip("-•. \t")
        if not subq:
            continue
        if qtype in ("single", "keyword_specific", "abstract"):
            parsed.append({"type": qtype, "query": subq})
    if not parsed:
        parsed = [{"type": "single", "query": query}]
    return parsed


def decompose_and_route_search(index: TextIndex, query: str, client=None, top_k: int = 8,
                                classifier: str = "llm", entity_aware: bool = True,
                                decompose_model: str = "gpt-4o") -> tuple:
    """[52] route_search()의 상위 호환 — 복합 질의를 먼저 성격별로 분해한 뒤 하위질의마다 각자
    맞는 전략으로 검색해 RRF로 합친다. 단일 성격 질의는 decompose_query()가 하위질의 1개(원본
    그대로)만 돌려주므로 route_search()와 동일하게 동작(추가 비용은 분해 판단 LLM 호출 1회뿐).

    하위질의 처리:
      - type="single": route_search()와 동일하게 재분류(keyword_specific/abstract)해 라우팅.
      - type="keyword_specific": hybrid_search(fusion="rrf")로 직접 검색(정밀 어휘 매칭 유지).
      - type="abstract": entity_aware=True면 index.entity_count로 HyDE(<=1)/MQE(>1) 분기,
        False면 항상 MQE — route_search()의 entity_aware 분기 로직 그대로 재사용.

    융합: 하위질의별 결과를 각자 top_k만큼 뽑은 뒤 **라운드로빈으로 인터리빙**한다(하위질의0의
    1위, 하위질의1의 1위, 하위질의0의 2위, ... 순서로 중복 제거하며 채움). [실측으로 발견]
    순위 기반 RRF 합산(모든 하위질의를 동등한 "투표"로 합산)을 먼저 시도했더니, "인사이트
    도출"처럼 포괄적인 abstract 하위질의가 코퍼스 전반에 폭넓게 점수를 흩뿌리면서 keyword_
    specific 하위질의가 정확히 1위로 짚어낸 근거(도표3 건설업종)가 abstract 쪽에서 낮은 순위인
    바람에 합산 후 다른 청크(도표4 건자재업종, 두 하위질의 모두에서 중상위)에 밀려나는 걸 확인함
    — 하위질의 수가 늘수록 "정밀 추출"형 하위질의의 1위가 희석되는 구조적 문제. 라운드로빈은
    각 하위질의의 1위를 최상위 `len(subqueries)`개 슬롯 안에 무조건 넣어 이 희석을 원천 차단한다.

    반환: (hits, subqueries) — subqueries: decompose_query()가 반환한 원본 리스트(디버깅/검증용,
    각 하위질의가 실제로 어느 전략을 탔는지는 호출측이 로그로 확인 가능하도록 "resolved_type"
    키를 추가해 채워 넣음)."""
    if client is None:
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    subqueries = decompose_query(query, client=client, model=decompose_model)

    per_subquery_hits = []
    for sq in subqueries:
        sub_q = sq["query"]
        sq_type = sq["type"]
        if sq_type == "single":
            hits, resolved_qtype = route_search(index, sub_q, client=client, top_k=top_k,
                                                 classifier=classifier, entity_aware=entity_aware)
            sq["resolved_type"] = resolved_qtype
        elif sq_type == "keyword_specific":
            hits = hybrid_search(index, sub_q, top_k=top_k, fusion="rrf")
            sq["resolved_type"] = "keyword_specific"
        else:  # abstract
            if entity_aware:
                if index.entity_count is None:
                    index.entity_count = count_document_entities(index, client=client)
                if index.entity_count <= 1:
                    hits, _ = hyde_search(index, sub_q, client=client, top_k=top_k, use_bm25=False)
                    sq["resolved_type"] = "abstract(HyDE)"
                else:
                    hits, _ = mqe_search(index, sub_q, client=client, top_k=top_k, use_bm25=False)
                    sq["resolved_type"] = "abstract(MQE)"
            else:
                hits, _ = mqe_search(index, sub_q, client=client, top_k=top_k, use_bm25=False)
                sq["resolved_type"] = "abstract(MQE)"
        per_subquery_hits.append(hits)

    seen_ids = set()
    result_hits = []
    max_len = max((len(h) for h in per_subquery_hits), default=0)
    for rank in range(max_len):
        for hits in per_subquery_hits:
            if rank >= len(hits):
                continue
            h = hits[rank]
            if h["chunk_id"] in seen_ids:
                continue
            seen_ids.add(h["chunk_id"])
            result_hits.append(h)
            if len(result_hits) >= top_k:
                return result_hits, subqueries
    return result_hits, subqueries


def ingest_and_search(pdf_path, yolo_model, query: str, doc_title: str = None,
                       page_boxes: dict = None, client=None, top_k: int = 5,
                       speculative_hyde: bool = True) -> tuple:
    """[51] 사용자 지적("업로드와 쿼리가 같이 주어지는데.. 9초 기다리고 쿼리를 받는 게 좋나?")
    반영 — PDF와 질의가 동시에 들어오는 실사용 시나리오 전용 진입점. `route_search(index, query)`
    는 인덱스가 이미 만들어져 있다고 전제하는 함수라 이 오버랩을 표현할 수 없어서 새로 만듦
    (기존 route_search는 "인덱스는 있고 새 질의만 온" 경우 — 예: 같은 문서에 대한 후속 질문 —
    에 그대로 씀).

    질의 분류(+MQE 하위질의 생성, `_classify_and_expand`)는 질의 텍스트만 있으면 되고 문서가
    전혀 필요 없다. 엔티티 카운트도 하나증권 포맷이면 원본 PDF 텍스트만 있으면 되고(정규식)
    청킹/임베딩과 무관하다. 즉 "인덱싱"과 "질의 준비"는 서로 의존관계가 없어 동시에 돌리면
    벽시계 지연을 겹치는 만큼 숨길 수 있다 — 실측(`test_concurrent_upload_query.py`)으로 다기업
    (MQE 경로) 문서는 총 지연이 48%(3.40s->1.76s), 단일기업(HyDE 경로)은 14%(6.35s->5.46s)
    줄어드는 걸 확인. 단일기업 쪽이 덜 줄어드는 이유: 분류 호출 자체가 MQE 하위질의까지는
    같이 만들어주지만 HyDE의 "가상 문단 생성"은 별도 호출이고, 그 경로로 갈지는 엔티티 수를
    알아야 정해지므로 인덱싱과 겹치게 미리 준비를 못 해뒀었기 때문.

    speculative_hyde=True(기본): 그 갭까지 마저 메우기 위해 HyDE 가상 문단 생성도 인덱싱과
    동시에 무조건 미리 만들어둔다(엔티티 수를 몰라도 질의만으로 생성 가능) — 나중에
    entity_count<=1이 아니라고 판명되면(다기업이라 MQE를 쓰게 되면) 이 생성 결과는 버려짐.
    **트레이드오프(사용자에게 명시)**: 버려지는 경우 그 호출의 토큰/비용이 그대로 낭비된다
    (지연은 인덱싱과 겹쳐서 손해가 없지만 비용은 남음) — 속도를 우선한 선택. speculative_hyde=
    False로 끄면 이 낭비 없이(단일기업 판명 후에만 HyDE 호출) 대신 그 경우 절감폭이 다시
    14%로 줄어듦.

    반환: (index, hits, qtype, entity_count)."""
    import fitz
    from concurrent.futures import ThreadPoolExecutor
    from text_extraction import process_pdf_streaming

    if client is None:
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _build_index():
        pages_chunks = [page["chunks"] for page in
                         process_pdf_streaming(pdf_path, yolo_model, doc_title=doc_title, page_boxes=page_boxes)]
        flat = [c for pc in pages_chunks for c in pc]
        from embedding import embed_texts
        texts = [c["text"] for c in flat]
        embeddings = embed_texts(texts)
        bm25 = BM25Okapi([_tokenize(t) for t in texts])
        chunk_ids = [f"{doc_title}_p{c['page']}_{i}" for i, c in enumerate(flat)]
        return TextIndex(pdf_id=doc_title, chunk_ids=chunk_ids, chunks=flat, embeddings=embeddings, bm25=bm25)

    def _entity_count():
        full_text = "\n".join(page.get_text() for page in fitz.open(str(pdf_path)))
        return count_document_entities_hana(full_text)  # None이면 나중에 index로 LLM 폴백

    n_workers = 4 if speculative_hyde else 3
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            "index": ex.submit(_build_index),
            "classify": ex.submit(_classify_and_expand, query, client),
            "entity": ex.submit(_entity_count),
        }
        if speculative_hyde:
            hyde_prompt = (
                f"다음 질의에 대한 답이 될 법한 가상의 증권사 리포트 문단을 작성하세요(실제 사실은 "
                f"모르니 형식/톤/다룰 법한 항목만 그럴듯하게 채우고, 구체적 수치는 꾸며내되 사실 "
                f"여부는 중요하지 않습니다 — 이 문단은 검색용 임베딩에만 쓰입니다): \"{query}\""
            )
            futures["hyde"] = ex.submit(
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0,
                    messages=[{"role": "user", "content": hyde_prompt}]
                ).choices[0].message.content)

        index = futures["index"].result()
        qtype, sub_queries = futures["classify"].result()
        entity_count = futures["entity"].result()
        if entity_count is None:  # 하나증권 포맷 아님 -> 인덱스 있으니 이제 LLM 폴백으로 확정
            entity_count = count_document_entities(index, client=client)
        index.entity_count = entity_count

        if qtype != "abstract":
            return index, hybrid_search(index, query, top_k=top_k, fusion="rrf"), qtype, entity_count

        if entity_count <= 1:
            if speculative_hyde:
                hypothetical_passage = futures["hyde"].result()
                from embedding import embed_texts
                hyde_embedding = embed_texts([hypothetical_passage])[0]
                dense_scores, _ = _raw_scores(index, query, query_embedding=hyde_embedding)
                fused = _apply_source_weights(index, _rrf_scores(dense_scores))
                import numpy as np
                order = np.argsort(-fused)[:top_k]
                hits = [{"chunk_id": index.chunk_ids[i], "chunk": index.chunks[i], "score": float(fused[i])}
                        for i in order]
            else:
                hits, _ = hyde_search(index, query, client=client, top_k=top_k, use_bm25=False)
        else:
            hits = _fuse_multi_query(index, [query] + sub_queries, top_k=top_k, use_bm25=False)
        return index, hits, qtype, entity_count


_HANA_PUBLISHER_MARKER = "하나증권"
_HANA_RATING_SECTION_RE = re.compile(r"투자의견 변동 내역[^\n]*\n\s*([^\n]+)")


def count_document_entities_hana(full_pdf_text: str):
    """[46] 사용자 지적("하나증권에 대해서만 수집... 양식은 거의 비슷하니까") 반영 — 하나증권
    Weekly류 리포트는 "투자의견 변동 내역 및 목표주가 괴리율" 문구 바로 다음 줄에 회사명이
    오는 구조가 공통(실측: 납기=2개, Construct=8개, SmartPhone=10개, 전부 정확히 일치, LLM
    호출 0회/토큰 0). 이 신호가 담긴 페이지가 리딩오더 hard 판정(컬럼 인터리빙)으로 텍스트
    청크에서 제외되는 경우가 있어서(납기 9페이지가 그 예) 이미 청킹/필터링된 index.chunks가
    아니라 원본 PDF 페이지 텍스트(청킹 전)에서 직접 뽑아야 한다 — 그래서 인자가 TextIndex가
    아니라 원문 텍스트. 하나증권 문서가 아니면 None을 반환해 호출측이 다른 방법으로 폴백하게 함."""
    if _HANA_PUBLISHER_MARKER not in full_pdf_text:
        return None
    companies = {c.strip() for c in _HANA_RATING_SECTION_RE.findall(full_pdf_text)}
    return len(companies) or None


def count_document_entities(index: TextIndex, full_pdf_text: str = None, client=None,
                             model: str = "gpt-4o-mini") -> int:
    """[45]/[46] 문서가 개별 투자의견(매수/중립/매도+목표주가)을 부여하는 서로 다른 기업이 몇
    개인지 센다. 우선순위(사용자 지적 "LLM 또 쓰는거 과소비 아니야?" 반영 — 비용 낮은 순):

      1) structured_metadata.entities 집계 — text/table/image 각 라우팅이 structured_output으로
         이미 뽑아둔 엔티티가 있으면(add_structured_metadata=True로 인제스트된 경우) 그걸 그대로
         모아서 씀. 새 LLM 호출 0회 — 이미 다른 목적으로 계산된 걸 재사용하는 것뿐.
      2) count_document_entities_hana(full_pdf_text) — [46] 하나증권 포맷이면 정규식으로 무료/
         즉시 계산. full_pdf_text를 안 주면 이 단계는 건너뜀(호출측이 인제스트 시점에 원본
         페이지 텍스트를 같이 넘겨야 이 무료 경로를 탐).
      3) LLM 폴백 — 위 둘 다 안 되면(하나증권이 아닌 문서, structured_metadata도 없음) 마지막
         수단. 토큰 절약을 위해 [45]의 "전체 문서 4만자" 대신 "투자의견/목표주가/매수/BUY 등
         키워드가 있는 청크만" 골라서 넘김(대부분의 청크는 무관 — 실측 KWave 기준 225청크 중
         후보만 추리면 수십 개 수준으로 줄어 토큰이 크게 줄어듦).

    문서당 1회만(질의마다 아님) — 인제스트 시점 비용. model 기본값 gpt-4o-mini로 충분(단순
    카운팅 작업이라 4o급 추론 불필요)."""
    # [재일 — 스키마 감사] 이 무료 경로가 실제로는 항상 죽어 있었다. entity_fusion이 만든 융합
    # 인덱스의 청크는 structured_metadata를 최상위가 아니라 `metadata.structured_metadata`로
    # 한 겹 중첩해 담는데(entity_fusion.py from_text_chunks/from_table_records/from_image_cards
    # 셋 다), 여기선 최상위만 봐서 항상 None -> 1)번 경로가 통째로 스킵되고 매 인제스트마다 3)번
    # LLM 폴백이 돌았다. 두 위치를 모두 보고, 브랜치마다 다른 키 이름(text=entities,
    # table/image=entities_mentioned)도 함께 읽는다.
    def _sm(c):
        return c.get("structured_metadata") or (c.get("metadata") or {}).get("structured_metadata") or {}

    entity_sets = []
    for c in index.chunks:
        sm = _sm(c)
        if sm:
            entity_sets.append(set(sm.get("entities") or sm.get("entities_mentioned") or []))
    if entity_sets:
        all_entities = set().union(*entity_sets)
        if all_entities:
            return len(all_entities)

    if full_pdf_text:
        hana_count = count_document_entities_hana(full_pdf_text)
        if hana_count is not None:
            return hana_count

    if client is None:
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    keywords = ("투자의견", "목표주가", "매수", "buy", "neutral", "중립", "비중")
    # [48] 사용자 지적으로 실측 발견 — entity_fusion.load_evidence_from_db()가 만든 融합 인덱스는
    # 청크가 "raw_chunk"가 아니라 "content" 키를 쓴다(text_extraction.process_pdf()의 순수 텍스트
    # 청크와 스키마가 다름). 둘 다 처리하도록 방어.
    texts = [c.get("raw_chunk") or c.get("content") or "" for c in index.chunks]
    candidates = [t for t in texts if any(k in t.lower() for k in keywords)]
    excerpt = "\n".join(candidates) if candidates else "\n".join(texts)
    prompt = (
        "다음은 증권사 리포트에서 투자의견 관련 문구가 있는 문단만 추린 것입니다. 이 문서가 "
        "개별 투자의견(매수/중립/매도 등급 + 목표주가)을 명시적으로 부여하는 서로 다른 기업이 "
        "몇 개인지 세어 숫자만 출력하세요(주가 동향 표에만 이름이 나열되고 개별 투자의견/"
        "목표주가 언급이 없는 기업은 세지 마세요).\n\n" + excerpt[:12000]
    )
    resp = client.chat.completions.create(model=model, temperature=0,
                                          messages=[{"role": "user", "content": prompt}])
    m = re.search(r"\d+", resp.choices[0].message.content)
    return int(m.group()) if m else 1


def precompute_entity_count(index: TextIndex, pdf_path=None, client=None) -> int:
    """[46] 사용자 지적("캐시 미스 해결해주고, 지연 병목 해결해주고") 반영 — route_search()가
    첫 질의에서 entity_count를 그때야 계산하면(이전 버전) LLM 호출 지연이 질의 응답 시간에
    그대로 노출된다. 인제스트 파이프라인이 TextIndex를 만든 직후 이 함수를 한 번 불러 index.
    entity_count를 미리 채워두면, 이후 모든 route_search() 호출은 지연 0인 캐시 읽기만 한다.
    pdf_path를 주면 count_document_entities_hana()의 무료 정규식 경로까지 탈 수 있음(원본
    PDF 페이지 텍스트가 필요해서 — hard 판정으로 청크에서 빠진 페이지의 신호까지 잡기 위함)."""
    full_pdf_text = None
    if pdf_path is not None:
        import fitz
        doc = fitz.open(str(pdf_path))
        full_pdf_text = "\n".join(doc[i].get_text() for i in range(doc.page_count))
        doc.close()
    index.entity_count = count_document_entities(index, full_pdf_text=full_pdf_text, client=client)
    return index.entity_count


def expand_to_parent_context(hits: list, index: TextIndex, parent_key: str = "page") -> list:
    """[44] Parent-Child Retrieval — 사용자 참고 다이어그램("Parent-Child Retrieval") 반영. 지금
    청킹(계층적 분할)이 문장/불릿 단위로 꽤 잘게 쪼개다 보니(예: "-음반/음원은 CORTIS 미니 2집이
    300만장 이상 판매되며..."), summary형 질의처럼 종합이 필요한 경우 검색된 개별 청크(child)만
    LLM에 주면 앞뒤 맥락(같은 페이지/섹션의 나머지 문장)이 빠진 파편적 근거가 된다.

    재색인 없이 기존 인덱스 그대로 활용하는 실용적 구현 — "부모"를 별도 계층으로 저장하는 대신,
    검색된 child 청크와 parent_key(기본 page, 필요시 section_path로 바꿀 수 있음)가 같은 index
    내 다른 청크들을 원문 순서대로 모아 parent_context로 붙여준다(재계산/재임베딩 없음, 청크
    표시 순서만 이용). 여러 hit이 같은 페이지에서 나오면 parent_context를 한 번만 계산해 공유.
    반환: 각 hit에 "parent_context"(해당 페이지 전체 원문, 순서대로 이어붙임) 키를 추가한 새 리스트."""
    parent_cache = {}
    expanded = []
    for h in hits:
        key = h["chunk"].get(parent_key)
        if key not in parent_cache:
            siblings = [c["raw_chunk"] for c in index.chunks if c.get(parent_key) == key]
            parent_cache[key] = "\n".join(siblings)
        expanded.append({**h, "parent_context": parent_cache[key]})
    return expanded


def save_index(index: TextIndex, path: Path) -> None:
    """[35] 임시 저장 — Supabase 스키마 확정 전까지 로컬 pickle로 인덱스를 보존해두는 용도.
    실제 서비스 저장소(Supabase pgvector 등)로 교체할 때 이 함수만 바꾸면 됨."""
    import pickle
    Path(path).write_bytes(pickle.dumps(index))


def load_index(path: Path) -> TextIndex:
    import pickle
    return pickle.loads(Path(path).read_bytes())

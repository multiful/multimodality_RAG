"""[4] Boilerplate(법적고지/Compliance Notice/Disclaimer) 제거 — Sentence Embedding 기반 채택.

[43] 사용자 지적("저정보량 청크가 필터링을 통과해 검색 결과를 오염시킨다") 반영 — 실측(KWave
73p)에서 두 가지 저정보량 패턴을 발견해 일반화된 필터로 추가:
  1) 인용/출처 전용 청크("자료: OO, DS투자증권 리서치센터"류) — 이하 `is_low_information_
     fragment()`. 임베딩 유사도 대신 구조 규칙(모든 줄이 "자료:"로 시작)만 보므로 임베딩 비용도
     없고, 브로커/문서마다 표현이 달라도(어느 증권사 리포트든 차트/표 밑에 "자료: 출처" 각주가
     붙는 건 공통 관례) 일반화된다. "주:"(각주/노트, 회계기준 설명 등 실제 분석 내용을 담을 수
     있음, 예: "주: K-IFRS 회계기준 개정으로...")는 의도적으로 제외 — "자료:"만 항상 순수 출처
     표기이고 "주:"는 그렇지 않다는 걸 실측으로 확인.
  2) Compliance Notice 섹션 전체 — 개별 문장 단위 semantic 유사도 검사(SEED_BOILERPLATE_
     SENTENCES)는 시드에 없는 새 표현(예: "당사 리서치센터 연구원은 ... 카카오톡 메신저 등으로
     개별 접촉하지 않습니다")이나 짧은 소제목("- Compliance Notice -", "[ 업종 투자의견 ]")은
     안 잡는다 — 소제목은 완전한 문장이 아니라서 시드 문장들과 코사인 유사도가 구조적으로 낮게
     나옴. 문장 단위 판정을 무한정 확장하는 대신, "이 섹션 표지가 한 번 등장하면 문서 끝까지
     전부 법적/공시 정형 문구뿐"이라는 실제 증권사 리포트 관례를 이용해 `is_boilerplate_
     section_marker()`로 섹션 진입을 감지하고, 그 뒤로는 개별 문장 유사도와 무관하게 전부
     제외한다(호출측인 text_extraction.py의 페이지 루프에서 플래그로 적용).

사용자가 제시한 3가지 옵션(SimHash/MinHash/Sentence Embedding) 중 Sentence Embedding을 채택한
이유: SimHash/MinHash는 "거의 동일한 문자열"(오탈자 수준 차이까지만 허용)이 대량 문서에 걸쳐
반복될 때 계산 효율이 뛰어나지만(해시 비교라 O(1)에 가까움), 지금 있는 참고 PDF 2건은 서로 다른
증권사(교보증권/하나증권)라 **문구 자체가 다름**("이 조사자료는 투자참고자료로만 활용하시기
바라며..." vs "본 조사자료는 고객의 투자에 정보를 제공할 목적으로 작성되었으며...") — 의미는
같지만 표현이 달라 SimHash/MinHash로는 안 잡힘. Sentence Embedding(이미 BGE-M3를 semantic_chunker
에서 쓰고 있어 재사용)은 패러프레이즈에도 강해 이 상황에 더 적합. 문서 수가 많아져 "완전히
동일한 문구"가 대량 반복되는 상황이 되면 SimHash/MinHash가 계산 비용 면에서 더 유리해질 수 있음
— 코퍼스 규모가 커지면 재검토 권장(정직하게 기록).
"""

# 실측 확인된 전형적인 법적고지/면책조항 문장 시드 — LG CNS(교보증권)/Construct(하나증권) 두
# 증권사의 Compliance Notice에서 실제로 확인한 문장을 일반화. 운영 시에는 새 문서에서 반복
# 발견되는 문장을 이 시드에 계속 추가해 나가는 식으로 확장 가능.
SEED_BOILERPLATE_SENTENCES = [
    "이 조사자료는 투자참고자료로만 활용하시기 바라며 고객의 증권투자 결과에 대한 법적 책임소재의 증빙자료로 사용될 수 없습니다.",
    "본 조사자료는 고객의 투자에 정보를 제공할 목적으로 작성되었으며 어떠한 경우에도 무단 복제 및 배포될 수 없습니다.",
    "이 자료에 게재된 내용들은 작성자의 의견을 정확하게 반영하고 있으며 외부의 부당한 압력이나 간섭 없이 작성되었음을 확인합니다.",
    "본 자료를 작성한 애널리스트는 자료의 작성과 관련하여 외부의 압력이나 부당한 간섭을 받지 않았으며 본인의 의견을 정확하게 반영하여 작성하였습니다.",
    "당사는 해당회사의 지분을 1% 이상 보유하고 있지 않습니다.",
    "본 자료는 기관투자가 등 제3자에게 사전 제공한 사실이 없습니다.",
    "투자의견의 유효기간은 추천일 이후 12개월을 기준으로 적용하며 목표주가 대비 상승여력에 따라 매수/중립/매도로 구분합니다.",
]


# [43] 섹션 표지 마커 — 실측(LGCNS/교보증권)으로 확인된, "이 뒤로는 문서 끝까지 법적/공시
# 정형 문구뿐"임을 알리는 소제목류. 새 증권사 문서에서 다른 마커가 반복 발견되면 계속 추가.
BOILERPLATE_SECTION_MARKERS = [
    "compliance notice", "투자의견 비율공시", "업종 투자의견", "투자등급관련사항",
]


def is_boilerplate_section_marker(text: str) -> bool:
    """[43] 법적고지/투자의견 공시 섹션의 시작을 알리는 제목류인지 판정. 이게 한 번 등장하면
    그 뒤로는 문서 끝까지 전부 정형화된 법적/공시 문구뿐이라는 게 실측으로 확인됨 — 개별 문장
    semantic 유사도로는 못 잡는 짧은 소제목까지 구조적으로 잡기 위한 보조 신호."""
    t = text.strip().lower()
    return any(marker in t for marker in BOILERPLATE_SECTION_MARKERS)


def is_low_information_fragment(text: str) -> bool:
    """[43] 인용/출처 전용 청크 판정 — 줄바꿈으로 나눈 모든 줄이 "자료:"(또는 "자료 :")로
    시작하면 순수 출처 표기로 간주(예: "자료: FnGuide, DS투자증권 리서치센터"). "주:"(노트/각주)는
    실제 분석 내용을 담을 수 있어(예: 회계기준 변경 설명) 제외 — 실측으로 "자료:"만 항상 순수
    인용이고 "주:"는 아니라는 걸 확인했음. 임베딩 없이 구조만 보는 값싼 필터라 전 페이지에 상시
    적용 가능(위치 사전필터 불필요)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True
    return all(l.startswith(("자료:", "자료 :")) for l in lines)


_seed_embedding_cache = {}


def _get_seed_embeddings(embed_model, seeds: list):
    """[32] 사용자 지적("병목 해결") 반영 — 프로파일링 결과 `detect_boilerplate_paragraphs_fast`가
    process_pdf()에서 페이지마다(K-Wave 기준 14회) 호출되는데, 매번 고정된 7개 시드 문장을
    다시 인코딩하고 있었다(실제 변하는 페이지 문단과 달리 시드는 상수인데도). 프로세스 내
    (embed_model, seeds) 조합별로 한 번만 인코딩하도록 캐싱 — 시드 재인코딩 비용을 사실상 0으로."""
    cache_key = (id(embed_model), tuple(seeds))
    if cache_key not in _seed_embedding_cache:
        _seed_embedding_cache[cache_key] = embed_model.encode(seeds, normalize_embeddings=True)
    return _seed_embedding_cache[cache_key]


def detect_boilerplate_paragraphs(paragraphs: list, embed_model, seed_sentences: list = None,
                                   similarity_threshold: float = 0.62) -> list:
    """각 문단이 알려진 boilerplate 문장들과 의미적으로 얼마나 비슷한지(코사인 유사도) 계산해
    threshold 이상이면 boilerplate로 판정. 반환: [{text, is_boilerplate, max_similarity}, ...]"""
    if not paragraphs:
        return []
    seeds = seed_sentences or SEED_BOILERPLATE_SENTENCES
    seed_embs = _get_seed_embeddings(embed_model, seeds)
    para_embs = embed_model.encode(paragraphs, normalize_embeddings=True)
    results = []
    for p, emb in zip(paragraphs, para_embs):
        sims = seed_embs @ emb
        max_sim = float(sims.max())
        results.append({"text": p, "is_boilerplate": max_sim >= similarity_threshold,
                         "max_similarity": round(max_sim, 4)})
    return results


def strip_boilerplate(paragraphs: list, embed_model, **kwargs) -> list:
    """boilerplate로 판정된 문단을 제외한 나머지만 반환."""
    scored = detect_boilerplate_paragraphs(paragraphs, embed_model, **kwargs)
    return [s["text"] for s in scored if not s["is_boilerplate"]]


# [5] 사용자 피드백 반영 — 임베딩 연산 비용 최적화. 두 단계로 사전 필터링해 실제 임베딩 호출
# (배치라 해도 인코딩 자체는 비용이 있음)이 필요한 문단 수를 줄인다:
# 1) 위치 필터: Compliance Notice는 실측 두 문서 모두 마지막 페이지(또는 문서 앞부분 하단)에만
#    등장 — 페이지 번호로 후보를 좁혀 애초에 검사 대상에서 제외.
# 2) Jaccard 사전 필터: 시드 문장과 단어 집합이 크게 겹치는(almost-exact) 문단은 임베딩 없이
#    바로 boilerplate로 확정 — 완전히 겹치지 않는 애매한 경우만 임베딩으로 넘김.
def _tokenize(s: str) -> set:
    return set(s.replace(".", " ").replace(",", " ").split())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def detect_boilerplate_paragraphs_fast(paragraphs: list, embed_model, seed_sentences: list = None,
                                        similarity_threshold: float = 0.62,
                                        jaccard_shortcut: float = 0.5,
                                        paragraph_pages: list = None,
                                        last_n_pages_only: int = None) -> list:
    """`detect_boilerplate_paragraphs`와 동일한 판정 기준이지만, 위치+Jaccard 사전 필터로
    실제 임베딩 인코딩이 필요한 문단 수를 줄인 버전. paragraph_pages(문단별 페이지 번호,
    paragraphs와 같은 길이)와 last_n_pages_only를 주면 그 범위 밖 문단은 임베딩 없이 즉시
    "content"로 확정(위치상 boilerplate일 수 없다고 가정)."""
    if not paragraphs:
        return []
    seeds = seed_sentences or SEED_BOILERPLATE_SENTENCES
    seed_token_sets = [_tokenize(s) for s in seeds]

    results = [None] * len(paragraphs)
    candidates = list(range(len(paragraphs)))
    if last_n_pages_only is not None and paragraph_pages is not None:
        max_page = max(paragraph_pages)
        cutoff = max_page - last_n_pages_only + 1
        for i in range(len(paragraphs)):
            if paragraph_pages[i] < cutoff:
                results[i] = {"text": paragraphs[i], "is_boilerplate": False,
                              "max_similarity": 0.0, "method": "skipped(위치필터)"}
        candidates = [i for i in candidates if results[i] is None]

    need_embedding = []
    for i in candidates:
        p_tokens = _tokenize(paragraphs[i])
        max_jaccard = max((_jaccard(p_tokens, st) for st in seed_token_sets), default=0.0)
        if max_jaccard >= jaccard_shortcut:
            results[i] = {"text": paragraphs[i], "is_boilerplate": True,
                          "max_similarity": round(max_jaccard, 4), "method": "jaccard"}
        else:
            need_embedding.append(i)

    if need_embedding:
        seed_embs = _get_seed_embeddings(embed_model, seeds)
        para_embs = embed_model.encode([paragraphs[i] for i in need_embedding], normalize_embeddings=True)
        for idx, emb in zip(need_embedding, para_embs):
            sims = seed_embs @ emb
            max_sim = float(sims.max())
            results[idx] = {"text": paragraphs[idx], "is_boilerplate": max_sim >= similarity_threshold,
                            "max_similarity": round(max_sim, 4), "method": "embedding"}
    return results

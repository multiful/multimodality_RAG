"""Layer3: 규칙 기반 뉴스 선정.

흐름: 하드 필터(기업 매칭·시간·중복) → 4요소 가중 랭킹(관련성/최신성/소스신뢰도/이벤트성) → 최종 top-N.

score_i = w_r * rel_i + w_t * exp(-λΔt_i) + w_s * src_i + w_e * event_i
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import numpy as np

from embeddings.bge_ko_embedder import BGEKoEmbedder
from src.finance.layer3_naver_news import search_news_paged
from src.finance.layer3_qwen3_llm import generate as qwen3_generate

# 매체 등급표 (제안 가중치: 통신사·경제지=1.0 / 일반지·방송사=0.7 / 기타=0.4)
SOURCE_TIERS: dict[str, float] = {
    # 통신사
    "yna.co.kr": 1.0, "newsis.com": 1.0, "news1.kr": 1.0, "newspim.com": 1.0,
    # 경제지
    "mk.co.kr": 1.0, "hankyung.com": 1.0, "sedaily.com": 1.0, "edaily.co.kr": 1.0,
    "fnnews.com": 1.0, "asiae.co.kr": 1.0, "mt.co.kr": 1.0, "moneys.mt.co.kr": 1.0,
    "biz.chosun.com": 1.0, "wowtv.co.kr": 1.0, "yonhapnews.co.kr": 1.0,
    "biz.heraldcorp.com": 1.0,
    # 일반지·방송사
    "chosun.com": 0.7, "joongang.co.kr": 0.7, "donga.com": 0.7, "hani.co.kr": 0.7,
    "khan.co.kr": 0.7, "ytn.co.kr": 0.7, "sbs.co.kr": 0.7, "kbs.co.kr": 0.7,
    "imbc.com": 0.7, "jtbc.co.kr": 0.7, "hankookilbo.com": 0.7, "seoul.co.kr": 0.7,
}
DEFAULT_SOURCE_TIER = 0.4

EVENT_KEYWORDS = [
    "공시", "잠정실적", "어닝서프라이즈", "어닝쇼크", "실적", "수주", "계약체결",
    "소송", "패소", "승소", "인수", "합병", "M&A", "유상증자", "무상증자",
    "자사주매입", "자사주소각", "배당", "상장폐지", "IPO", "특허", "승인",
    "FDA", "임상", "리콜", "감사의견", "회생절차", "지분매각", "투자유치",
]

WEIGHTS = {"rel": 0.40, "recency": 0.25, "src": 0.15, "event": 0.20}
RECENCY_LAMBDA = 0.1  # 반감기 ln(2)/0.1 ≈ 6.93일 (~7일)
DEDUP_SIM_THRESHOLD = 0.9
SEARCH_WINDOW_DAYS = 2
LLM_CANDIDATE_POOL = 12  # 규칙 기반 랭킹 상위 몇 건을 LLM 검증에 넘길지


@dataclass
class ScoredArticle:
    title: str
    description: str
    link: str
    originallink: str
    pub_date: datetime
    match_position: str  # "title" | "lead"
    rel: float
    recency_decay: float
    src: float
    src_tier_label: str
    event: int
    score: float
    reasoning: str = ""


def _company_match_position(article: dict, name_ko: str, aliases: list[str]) -> str | None:
    """기업명/별칭이 제목 또는 리드문(description)에 등장하는지 확인한다.

    본문 전체가 아닌 검색 API가 주는 title/description(리드문 스니펫)만 근거로 삼으므로,
    본문 깊숙이 1회만 언급된 기사(예: 시황 기사의 종목 나열)는 애초에 후보에서 걸러진다.
    """
    needles = [name_ko] + [a.strip() for a in aliases if a.strip()]
    title = article["title"]
    if any(n in title for n in needles):
        return "title"
    if any(n in article["description"] for n in needles):
        return "lead"
    return None


def _within_date_window(pub_date: datetime, now: datetime, window_days: int) -> bool:
    if pub_date.tzinfo is None:
        pub_date = pub_date.replace(tzinfo=timezone.utc)
    lower = now - timedelta(days=window_days)
    return lower <= pub_date <= now


def hard_filter(
    articles: list[dict],
    name_ko: str,
    aliases: list[str],
    now: datetime | None = None,
    window_days: int = SEARCH_WINDOW_DAYS,
) -> list[dict]:
    """1단계: 기업 매칭 + 날짜 윈도 하드 필터."""
    now = now or datetime.now(timezone.utc)
    kept = []
    for art in articles:
        pos = _company_match_position(art, name_ko, aliases)
        if pos is None:
            continue
        if not _within_date_window(art["pub_date"], now, window_days):
            continue
        art = {**art, "match_position": pos}
        kept.append(art)
    return kept


def dedup_by_title_similarity(
    articles: list[dict], embedder: BGEKoEmbedder, threshold: float = DEDUP_SIM_THRESHOLD
) -> list[dict]:
    """제목 임베딩 코사인 유사도 > threshold면 같은 사건의 받아쓰기 기사로 보고 최신 1건만 유지."""
    if not articles:
        return []
    ordered = sorted(articles, key=lambda a: a["pub_date"], reverse=True)  # 최신 우선
    titles = [a["title"] for a in ordered]
    vecs = np.array(embedder.embed(titles))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    unit = vecs / norms

    kept: list[dict] = []
    kept_vecs: list[np.ndarray] = []
    for art, vec in zip(ordered, unit):
        if kept_vecs:
            sims = np.array(kept_vecs) @ vec
            if sims.max() > threshold:
                continue  # 이미 유지된(더 최신) 유사 기사가 있음 -> 스킵
        kept.append(art)
        kept_vecs.append(vec)
    return kept


def _source_tier(url: str) -> tuple[float, str]:
    domain = urlparse(url).netloc.lower().removeprefix("www.")
    for known_domain, tier in SOURCE_TIERS.items():
        if domain == known_domain or domain.endswith("." + known_domain):
            label = "통신사·경제지" if tier == 1.0 else "일반지·방송사"
            return tier, label
    return DEFAULT_SOURCE_TIER, "기타"


def _has_event_keyword(text: str) -> bool:
    return any(kw in text for kw in EVENT_KEYWORDS)


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
    return a_norm @ b_norm.T


def rank_articles(
    articles: list[dict], topic: str, embedder: BGEKoEmbedder, now: datetime | None = None
) -> list[ScoredArticle]:
    """2단계: rel/최신성/소스신뢰도/이벤트성 4요소 가중 랭킹."""
    now = now or datetime.now(timezone.utc)
    if not articles:
        return []

    texts = [f"{a['title']} {a['description']}".strip() for a in articles]
    doc_vecs = np.array(embedder.embed(texts))
    topic_vec = np.array(embedder.embed([topic]))
    rel_scores = _cosine_sim_matrix(doc_vecs, topic_vec)[:, 0]

    scored: list[ScoredArticle] = []
    for art, rel in zip(articles, rel_scores):
        pub_date = art["pub_date"]
        if pub_date.tzinfo is None:
            pub_date = pub_date.replace(tzinfo=timezone.utc)
        delta_days = max((now - pub_date).total_seconds() / 86400, 0.0)
        decay = math.exp(-RECENCY_LAMBDA * delta_days)

        src, src_label = _source_tier(art["originallink"])
        event = 1 if _has_event_keyword(f"{art['title']} {art['description']}") else 0

        score = (
            WEIGHTS["rel"] * rel
            + WEIGHTS["recency"] * decay
            + WEIGHTS["src"] * src
            + WEIGHTS["event"] * event
        )

        reasoning = (
            f"관련성 {rel:.2f}(리포트 주제와의 제목·리드문 코사인 유사도) · "
            f"최신성 {delta_days:.1f}일 경과(감쇠 {decay:.2f}, 반감기 ~7일) · "
            f"소스 신뢰도 {src_label}({src:.1f}) · "
            f"이벤트성 {'하드 이벤트 키워드 포함(+1)' if event else '없음(0)'} "
            f"→ 종합점수 {score:.3f} (제목/리드문 매칭 위치: {art['match_position']})"
        )

        scored.append(
            ScoredArticle(
                title=art["title"],
                description=art["description"],
                link=art["link"],
                originallink=art["originallink"],
                pub_date=pub_date,
                match_position=art["match_position"],
                rel=float(rel),
                recency_decay=decay,
                src=src,
                src_tier_label=src_label,
                event=event,
                score=score,
                reasoning=reasoning,
            )
        )
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def _build_verification_prompt(
    topic: str, name_ko: str, candidates: list[ScoredArticle], top_n: int
) -> str:
    lines = [
        "너는 금융 리포트에 인용할 뉴스를 최종 검수하는 애널리스트다.",
        f"기업: {name_ko}",
        f"리포트 핵심 주제: {topic}",
        f"아래는 규칙 기반 랭킹 점수로 1차 선별된 뉴스 후보 {len(candidates)}건이다.",
        f"각 후보가 실제로 '{name_ko}'의 '{topic}'과 직접 관련된 유의미한 기사인지 검토하고,",
        f"가장 적합한 상위 {top_n}건을 선택하라.",
        "단순 시황 종목 나열, 광고성 기사, 주제와 무관한 기사는 규칙점수가 높아도 제외하라.",
        "",
    ]
    for i, c in enumerate(candidates):
        lines.append(f"[{i}] 제목: {c.title}\n    리드문: {c.description}\n    규칙점수: {c.score:.3f}")
    lines.append("")
    lines.append(
        f"다음 JSON 배열 형식으로만 답하라 (그 외 설명 문장 없이): "
        f'[{{"index": 후보번호(int), "reasoning": "선정 사유 1~2문장(한국어)"}}, ...] '
        f"정확히 {top_n}개 항목만 포함하라."
    )
    return "\n".join(lines)


def _parse_llm_selection(raw_text: str) -> list[dict]:
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON 배열을 찾지 못했습니다: {raw_text[:200]}")
    return json.loads(match.group(0))


def verify_with_llm(
    candidates: list[ScoredArticle], topic: str, name_ko: str, top_n: int = 5
) -> list[ScoredArticle]:
    """LLM reasoning 검증: Qwen3(기본 0.6B, src/finance/layer3_qwen3_llm.py)가 규칙 기반 상위 후보를 재검토해 최종 top_n을 선정한다.

    파싱 실패 등 LLM 호출이 실패하면 규칙 기반 상위 top_n으로 안전하게 대체한다.
    """
    if not candidates:
        return []
    pool = candidates[:LLM_CANDIDATE_POOL]
    prompt = _build_verification_prompt(topic, name_ko, pool, top_n)

    try:
        raw = qwen3_generate(prompt, max_new_tokens=800)
        parsed = _parse_llm_selection(raw)
    except Exception as exc:  # noqa: BLE001 - LLM 호출/파싱 실패 시 규칙 기반으로 안전하게 폴백
        print(f"[warn] LLM 검증 실패({exc}), 규칙 기반 상위 {top_n}건으로 대체")
        return candidates[:top_n]

    selected: list[ScoredArticle] = []
    for item in parsed[:top_n]:
        idx = item.get("index")
        if not isinstance(idx, int) or not (0 <= idx < len(pool)):
            continue
        art = pool[idx]
        llm_reasoning = item.get("reasoning")
        if llm_reasoning:
            art.reasoning = f"{llm_reasoning} [규칙 랭킹 근거: {art.reasoning}]"
        selected.append(art)

    return selected if selected else candidates[:top_n]


def select_news(
    name_ko: str,
    query: str,
    topic: str | None = None,
    aliases: list[str] | None = None,
    top_n: int = 5,
    embedder: BGEKoEmbedder | None = None,
    now: datetime | None = None,
    use_llm_verification: bool = True,
) -> list[ScoredArticle]:
    """네이버 검색 API 실시간 결과 → 하드 필터 → 4요소 가중 랭킹 → LLM(Qwen3) reasoning 검증 → 최종 top_n건."""
    aliases = aliases or []
    topic = topic or name_ko
    embedder = embedder or BGEKoEmbedder()
    now = now or datetime.now(timezone.utc)

    raw = search_news_paged(query, sort="date", max_results=300)
    filtered = hard_filter(raw, name_ko, aliases, now=now)
    deduped = dedup_by_title_similarity(filtered, embedder)
    ranked = rank_articles(deduped, topic, embedder, now=now)

    if use_llm_verification:
        return verify_with_llm(ranked, topic, name_ko, top_n=top_n)
    return ranked[:top_n]
# -*- coding: utf-8 -*-
"""[민성 Layer1~4] 재무 스코어·기술지표·융합 신호 → LLM 컨텍스트 배선.

배경(사용자 확인 요청으로 발견): `src/finance/`의 Layer1(섹터 상대 재무 스코어)/Layer2(기술지표
26종)/Layer4(시간감쇠 가중 융합)는 `data_collection/` 배치 스크립트로만 돌고 산출물이 로컬 md에만
남아(그마저 Layer2/4는 2종목뿐) **답변 생성에 전혀 도달하지 못하고 있었다** — 배당 스코어링·뉴스
감성과 동일한 "만들어놓고 배선 안 된" 패턴.

방식(사용자 결정): **DB 적재 없이 질의 시점에 직접 계산해 바로 주입**한다.
  - Layer1: `compute_layer1_score(ticker)` 실시간 호출(입력은 로컬 `KOSPI200_output/kospi200_financials`
    md — 네트워크 없음, 섹터 피어 z-score 계산).
  - Layer2: `analyze(ticker)` 실시간 호출(yfinance 2년 OHLCV) — **현재가와 기준시각도 여기서 나와
    핸드오프 남은과제 5("실시간 주가 미연동")를 함께 해소**한다.
  - 뉴스: 새로 수집하지 않고 `company_news_sentiment` **캐시를 읽기만** 한다(수집·TTL 관리는
    news_sentiment_link 소관 — 같은 fetch_company_db_context 호출에서 이 모듈보다 먼저 실행돼
    캐시가 데워진 상태).
  - Layer4: 위 세 신호를 `fuse()`(α 가중 + 시간감쇠)로 융합해 S와 참고 판정을 낸다.

주의: 융합 판정("매수 우위" 등)은 규칙 기반 **자동 참고 신호**다 — 블록에 그 성격을 명시해
LLM이 확정 결론처럼 복창하지 않게 한다(citation-check는 숫자만 보므로 이런 오귀속은 못 거른다).
실패는 티커 단위로 조용히 건너뛴다(보조 신호 — 재무/프로필/배당 컨텍스트 생성을 막으면 안 됨).
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 질의당 실시간 계산할 최대 종목 수 — Layer2가 종목당 yfinance 호출(~1–3s)이라 다기업 리포트
# (Construct는 7개 매칭)에서 전부 돌리면 지연이 커진다. 병렬(아래) + 상한으로 묶는다.
MAX_TICKERS = 5
_TICKER_WORKERS = 4


def _verdict(S: float) -> str:
    # data_collection/layer4_fuse_kospi200_score.py:71과 동일한 임계값(일관성 유지)
    return ("강한 매수" if S > 0.5 else "매수 우위" if S > 0.15
            else "강한 매도" if S < -0.5 else "매도 우위" if S < -0.15 else "중립")


def _news_signal_from_cache(db_url: str, ticker: str):
    """company_news_sentiment 캐시에서 (score, age_days, note) — 없으면 None(융합에서 제외)."""
    import psycopg2
    try:
        conn = psycopg2.connect(db_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "select sentiment, avg_age_days, n_articles, collected_at "
                    "from company_news_sentiment where ticker=%s", (ticker,))
                row = cur.fetchone()
        finally:
            conn.close()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    sentiment, avg_age, n_articles, collected_at = row
    staleness = 0.0
    if collected_at is not None:
        staleness = max((datetime.now(timezone.utc) - collected_at).total_seconds() / 86400.0, 0.0)
    return float(sentiment), float(avg_age or 0.0) + staleness, f"캐시 기사 {n_articles}건"


def compute_ticker_signals(ticker: str, name: str, db_url: str) -> dict | None:
    """한 종목의 Layer1/2/뉴스/융합 신호를 실시간 계산. 세 소스 전부 실패면 None."""
    from src.finance.layer4_fusion import SourceSignal, fuse

    signals, detail = {}, {}

    try:
        from src.finance.layer1_financial_score import compute_layer1_score
        r1 = compute_layer1_score(ticker)
        signals["fin"] = SourceSignal(score=r1.s_fin, age_days=0.0, note="실시간 계산")
        detail["fin"] = f"재무 s_fin {r1.s_fin:+.2f} (섹터 {r1.sector} 상대 z-score, 실시간 계산)"
    except Exception:
        pass

    try:
        from src.finance.layer2_technical_indicators import analyze
        t = analyze(ticker)
        signals["tech"] = SourceSignal(score=t.s_tech, age_days=0.0, note="실시간 yfinance")
        detail["tech"] = (f"기술 s_tech {t.s_tech:+.2f} (매수신호 {t.n_buy}/매도 {t.n_sell}/중립 "
                          f"{t.n_neutral}) · 현재가 {t.close:,.0f}원 (기준 {str(t.as_of)[:10]}, 실시간)")
    except Exception:
        pass

    news = _news_signal_from_cache(db_url, ticker)
    if news is not None:
        s_news, age, note = news
        signals["news"] = SourceSignal(score=s_news, age_days=age, note=note)
        detail["news"] = f"뉴스 s_news {s_news:+.2f} ({note}, {age:.1f}일 전)"

    if not signals:
        return None
    result = fuse(signals)
    used = [c.key for c in result.contributions]
    lines = [detail[k] for k in ("fin", "tech", "news") if k in detail]
    lines.append(f"융합 S ≈ {result.S:+.3f} → 참고 신호 \"{_verdict(result.S)}\" "
                 f"(사용 소스: {', '.join(used) or '없음'}"
                 + (f" / 충돌: {'; '.join(result.conflicts)}" if result.conflicts else "") + ")")
    return {"ticker": ticker, "name": name, "lines": lines}


def fetch_layer_signals_context(db_url: str, matched: list, max_tickers: int = MAX_TICKERS) -> str:
    """매칭 종목들의 종합 시그널 블록 — LLM 프롬프트에 그대로 넣는 텍스트(없으면 빈 문자열)."""
    targets = matched[:max_tickers]
    if not targets:
        return ""
    with ThreadPoolExecutor(max_workers=_TICKER_WORKERS) as ex:
        futures = [ex.submit(compute_ticker_signals, m["ticker"], m["name"], db_url) for m in targets]
        results = []
        for f in futures:
            try:
                r = f.result()
            except Exception:
                r = None
            if r:
                results.append(r)
    if not results:
        return ""
    skipped = len(matched) - len(targets)
    blocks = []
    for r in results:
        body = "\n".join(f"- {l}" for l in r["lines"])
        blocks.append(f"[{r['name']}({r['ticker']}) 종합 시그널 — Layer1~4 규칙 기반 자동 산출]\n{body}")
    note = ("\n(위 시그널은 규칙 기반 자동 산출 참고 신호로, 확정 투자 판단이 아님 — 다른 근거와 "
            "교차 검증해 서술할 것" + (f". 매칭 {len(matched)}개 중 상위 {len(targets)}개만 계산됨" if skipped > 0 else "") + ")")
    return "\n\n".join(blocks) + note

"""Layer1: 재무제표 규칙 기반 스코어링.

s_fin = (1/|J|) * Σ_{j∈J} tanh((x_j - μ_j) / σ_j) ∈ [-1, 1]

- J = {매출성장률, 영업이익률 변화, 부채비율}
- μ_j, σ_j = 동일 섹터(KOSPI200_output/kospi200_profiles의 "섹터" 값이 같은) 종목들의
  평균/표본표준편차(n-1). 대상 종목 자신도 피어 모집단에 포함된다.
- 부채비율은 낮을수록 좋으므로 z-score 부호를 반전(sign-adjust)한 뒤 tanh를 적용한다.

원본 데이터: data_collection/fetch_kospi200_financials.py / fetch_kospi200_profiles.py로
이미 받아둔 KOSPI200_output/kospi200_financials, kospi200_profiles 마크다운을 파싱해서 쓴다
(라이브 API 재호출 없이, 이미 수집된 재무데이터 population 위에서 섹터 평균/표준편차를 계산).
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FINANCIALS_DIR = REPO_ROOT / "KOSPI200_output" / "kospi200_financials"
PROFILES_DIR = REPO_ROOT / "KOSPI200_output" / "kospi200_profiles"

METRICS = ("revenue_growth", "opinc_margin_change", "debt_ratio")
LOWER_IS_BETTER = {"debt_ratio"}
METRIC_LABELS = {
    "revenue_growth": "매출성장률",
    "opinc_margin_change": "영업이익률 변화",
    "debt_ratio": "부채비율",
}


@dataclass
class FinancialMetrics:
    revenue_growth: float
    opinc_margin_change: float
    debt_ratio: float
    fiscal_year_latest: str
    fiscal_year_prev: str


@dataclass
class Layer1Result:
    ticker: str
    sector: str
    target: FinancialMetrics
    peer_metrics: dict[str, FinancialMetrics]  # code -> metrics (대상 종목 포함)
    mu_sigma: dict[str, tuple[float, float]]
    z_scores: dict[str, float]
    tanh_scores: dict[str, float]
    s_fin: float


def get_company_name(code: str) -> str | None:
    path = PROFILES_DIR / f"{code}.KS_profile.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^# (.+?) \(", text)
    return m.group(1).strip() if m else None


def get_sector(code: str) -> str | None:
    path = PROFILES_DIR / f"{code}.KS_profile.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^- 섹터:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def list_available_tickers() -> list[str]:
    """재무제표·프로필이 모두 존재하는 KOSPI200 종목 코드 목록 (6자리, .KS 제외)."""
    fin_codes = {p.name.removesuffix(".KS_financials.md") for p in FINANCIALS_DIR.glob("*.KS_financials.md")}
    prof_codes = {p.name.removesuffix(".KS_profile.md") for p in PROFILES_DIR.glob("*.KS_profile.md")}
    return sorted(fin_codes & prof_codes)


def _annual_block(text: str, section_header: str) -> str:
    """`## {section_header}` 아래 `### 연간` 서브섹션만 잘라낸다 (분기 데이터 제외)."""
    lines = text.splitlines()
    in_section = False
    in_annual = False
    collected: list[str] = []
    for line in lines:
        if line.startswith("## "):
            in_section = line.strip() == section_header
            in_annual = False
            continue
        if in_section and line.startswith("### "):
            in_annual = line.strip() == "### 연간"
            continue
        if in_section and in_annual:
            collected.append(line)
    return "\n".join(collected)


def _row_values(block: str, row_name: str) -> list[float | None]:
    """마크다운 테이블에서 `| {row_name} | v1 | v2 | ... |` 행을 찾아 값 리스트를 반환한다."""
    for line in block.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells or cells[0] != row_name:
            continue
        values: list[float | None] = []
        for cell in cells[1:]:
            cell = cell.replace(",", "")
            if cell in ("", "-", "nan", "None"):
                values.append(None)
                continue
            try:
                values.append(float(cell))
            except ValueError:
                values.append(None)
        return values
    return []


def compute_financial_metrics(code: str) -> FinancialMetrics | None:
    """최근 2개 회계연도 데이터로 매출성장률/영업이익률 변화/부채비율을 계산한다.

    필요한 행이 없거나 값이 비어있으면(재무데이터 부족) None을 반환해 상위 로직에서 제외시킨다.
    """
    path = FINANCIALS_DIR / f"{code}.KS_financials.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")

    income_block = _annual_block(text, "## 손익계산서 (Income Statement)")
    balance_block = _annual_block(text, "## 대차대조표 (Balance Sheet)")
    if not income_block or not balance_block:
        return None

    revenue = _row_values(income_block, "Total Revenue")
    opinc = _row_values(income_block, "Operating Income")
    equity = _row_values(balance_block, "Stockholders Equity")
    liabilities = _row_values(balance_block, "Total Liabilities Net Minority Interest")

    if len(revenue) < 2 or len(opinc) < 2 or len(equity) < 1 or len(liabilities) < 1:
        return None
    rev0, rev1 = revenue[0], revenue[1]
    op0, op1 = opinc[0], opinc[1]
    eq0 = equity[0]
    li0 = liabilities[0]
    if None in (rev0, rev1, op0, op1, eq0, li0):
        return None
    if rev0 == 0 or rev1 == 0 or eq0 == 0:
        return None

    dates = re.findall(r"\d{4}-\d{2}-\d{2}", income_block.splitlines()[0]) if income_block else []
    fy_latest = dates[0] if dates else "?"
    fy_prev = dates[1] if len(dates) > 1 else "?"

    return FinancialMetrics(
        revenue_growth=(rev0 - rev1) / rev1,
        opinc_margin_change=(op0 / rev0) - (op1 / rev1),
        debt_ratio=li0 / eq0,
        fiscal_year_latest=fy_latest,
        fiscal_year_prev=fy_prev,
    )


def compute_layer1_score(ticker: str) -> Layer1Result:
    code = ticker.split(".")[0]

    sector = get_sector(code)
    if sector is None:
        raise ValueError(f"{ticker}: 프로필에서 섹터 정보를 찾지 못했습니다 (kospi200_profiles에 파일이 있는지 확인).")

    peer_metrics: dict[str, FinancialMetrics] = {}
    for c in list_available_tickers():
        if get_sector(c) != sector:
            continue
        m = compute_financial_metrics(c)
        if m is not None:
            peer_metrics[c] = m

    if code not in peer_metrics:
        raise ValueError(f"{ticker}: 재무 지표를 계산할 수 없습니다 (손익계산서/대차대조표 데이터 부족).")
    if len(peer_metrics) < 3:
        raise ValueError(
            f"{ticker}: 섹터({sector}) 내 재무데이터 보유 피어가 {len(peer_metrics)}개뿐이라 "
            f"표준편차 계산에 부적합합니다 (최소 3개 필요)."
        )

    mu_sigma: dict[str, tuple[float, float]] = {}
    for metric in METRICS:
        values = [getattr(m, metric) for m in peer_metrics.values()]
        mu = statistics.mean(values)
        sigma = statistics.stdev(values)  # 표본표준편차 (n-1)
        mu_sigma[metric] = (mu, sigma)

    target = peer_metrics[code]
    z_scores: dict[str, float] = {}
    tanh_scores: dict[str, float] = {}
    for metric in METRICS:
        mu, sigma = mu_sigma[metric]
        x = getattr(target, metric)
        z = (x - mu) / sigma if sigma != 0 else 0.0
        sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
        z_scores[metric] = z
        tanh_scores[metric] = math.tanh(sign * z)

    s_fin = sum(tanh_scores.values()) / len(tanh_scores)

    return Layer1Result(
        ticker=ticker,
        sector=sector,
        target=target,
        peer_metrics=peer_metrics,
        mu_sigma=mu_sigma,
        z_scores=z_scores,
        tanh_scores=tanh_scores,
        s_fin=s_fin,
    )

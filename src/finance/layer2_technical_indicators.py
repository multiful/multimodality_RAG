"""실시간 시세(yfinance)로부터 investing.com 기술적 분석 페이지와 동일한 공개 판정
기준을 raw OHLCV에 직접 적용해 Buy/Sell/Neutral 요약을 재현한다.

investing.com은 Cloudflare Enterprise 봇 관리를 사용해 headless 브라우저 자동화를
차단한다 (Playwright, playwright-stealth, 실제 Chrome 채널, CDP 탐지 회피 특화 포크인
patchright까지 모두 "Just a moment..." 챌린지에서 막히는 것을 확인함). 대신 동일한
지표(RSI/STOCH/MACD/ADX/Williams%R/CCI/ATR/Highs-Lows/Bull-Bear Power/Ultimate
Oscillator/ROC, SMA·EMA 5/10/20/50/100/200)를 investing.com이 공개한 판정 규칙대로
계산해 매 호출마다 실시간 시세로 재현한다.

집계 규칙(investing.com 실제 표기와 대조해 역산):
- Overbought/Oversold 라벨(STOCH, STOCHRSI, Williams %R)과 변동성 라벨(ATR)은
  참고용으로만 표시되고 Buy/Sell/Neutral 집계에는 포함되지 않는다.
- RSI/MACD/ADX/CCI/Highs-Lows/Bull-Bear Power/Ultimate Oscillator/ROC(8개)만
  Buy/Sell/Neutral로 집계된다.
- 이동평균(SMA·EMA × 5/10/20/50/100/200 = 12개)은 모두 집계된다(종가 vs MA 비교).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

BUY, SELL, NEUTRAL = "Buy", "Sell", "Neutral"
OVERBOUGHT, OVERSOLD = "Overbought", "Oversold"
LESS_VOL, HIGH_VOL = "Less Volatility", "High Volatility"

MA_PERIODS = (5, 10, 20, 50, 100, 200)


@dataclass
class IndicatorRow:
    name: str
    value: float
    signal: str
    counted: bool  # True면 Buy/Sell/Neutral 집계에 포함


@dataclass
class MARow:
    period: int
    sma: float
    sma_signal: str
    ema: float
    ema_signal: str


@dataclass
class TechnicalSummary:
    ticker: str
    as_of: pd.Timestamp
    close: float
    indicators: list[IndicatorRow] = field(default_factory=list)
    mas: list[MARow] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def _counted_signals(self) -> list[str]:
        signals = [row.signal for row in self.indicators if row.counted]
        for ma in self.mas:
            signals.append(ma.sma_signal)
            signals.append(ma.ema_signal)
        return signals

    @property
    def n_buy(self) -> int:
        return sum(1 for s in self._counted_signals() if s == BUY)

    @property
    def n_sell(self) -> int:
        return sum(1 for s in self._counted_signals() if s == SELL)

    @property
    def n_neutral(self) -> int:
        return sum(1 for s in self._counted_signals() if s == NEUTRAL)

    @property
    def s_tech(self) -> float:
        """Layer2: s_tech = (N_buy - N_sell) / (N_buy + N_sell + N_neutral)."""
        total = self.n_buy + self.n_sell + self.n_neutral
        if total == 0:
            return 0.0
        return (self.n_buy - self.n_sell) / total


def _sign_signal(value: float, buy_gt: float = 0.0) -> str:
    if value > buy_gt:
        return BUY
    if value < buy_gt:
        return SELL
    return NEUTRAL


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)  # avg_loss == 0 -> RSI 100


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 9, d_period: int = 6):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    percent_d = percent_k.rolling(d_period).mean()
    return percent_k, percent_d


def _stoch_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    rsi = _rsi(close, period)
    lowest = rsi.rolling(period).min()
    highest = rsi.rolling(period).max()
    return 100 * (rsi - lowest) / (highest - lowest)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(period).mean()
    mean_dev = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical_price - sma_tp) / (0.015 * mean_dev)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _highs_lows(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    hl = ((high - high.shift(1)) + (low - low.shift(1))) / 2
    return hl.rolling(period).mean()


def _bull_bear_power(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13) -> pd.Series:
    ema = close.ewm(span=period, adjust=False).mean()
    bull_power = high - ema
    bear_power = low - ema
    return bull_power + bear_power


def _ultimate_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, p1: int = 7, p2: int = 14, p3: int = 28
) -> pd.Series:
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)

    avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
    avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
    avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()
    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7


def _roc(close: pd.Series, period: int = 12) -> pd.Series:
    return 100 * (close - close.shift(period)) / close.shift(period)


def fetch_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        raise ValueError(f"{ticker}: yfinance에서 시세 데이터를 가져오지 못했습니다.")
    return df


def analyze(ticker: str, ohlcv: pd.DataFrame | None = None) -> TechnicalSummary:
    """실시간(또는 전달받은) OHLCV로부터 기술적 분석 요약을 계산한다."""
    df = ohlcv if ohlcv is not None else fetch_ohlcv(ticker)
    high, low, close = df["High"], df["Low"], df["Close"]

    last = lambda s: float(s.iloc[-1])  # noqa: E731

    indicators: list[IndicatorRow] = []

    rsi = _rsi(close, 14)
    rsi_v = last(rsi)
    rsi_signal = SELL if rsi_v > 70 else BUY if rsi_v < 30 else NEUTRAL
    indicators.append(IndicatorRow("RSI(14)", rsi_v, rsi_signal, counted=True))

    k, d = _stoch(high, low, close, 9, 6)
    stoch_v = last(d) if not np.isnan(last(d)) else last(k)
    stoch_signal = OVERBOUGHT if stoch_v > 80 else OVERSOLD if stoch_v < 20 else NEUTRAL
    indicators.append(IndicatorRow("STOCH(9,6)", stoch_v, stoch_signal, counted=False))

    stoch_rsi_v = last(_stoch_rsi(close, 14))
    stoch_rsi_signal = OVERBOUGHT if stoch_rsi_v > 80 else OVERSOLD if stoch_rsi_v < 20 else NEUTRAL
    indicators.append(IndicatorRow("STOCHRSI(14)", stoch_rsi_v, stoch_rsi_signal, counted=False))

    macd_line, signal_line = _macd(close, 12, 26, 9)
    macd_hist_v = last(macd_line - signal_line)
    macd_signal = _sign_signal(macd_hist_v)
    indicators.append(IndicatorRow("MACD(12,26)", macd_hist_v, macd_signal, counted=True))

    adx, plus_di, minus_di = _adx(high, low, close, 14)
    adx_v, plus_v, minus_v = last(adx), last(plus_di), last(minus_di)
    if adx_v > 20 and plus_v > minus_v:
        adx_signal = BUY
    elif adx_v > 20 and minus_v > plus_v:
        adx_signal = SELL
    else:
        adx_signal = NEUTRAL
    indicators.append(IndicatorRow("ADX(14)", adx_v, adx_signal, counted=True))

    wr_v = last(_williams_r(high, low, close, 14))
    wr_signal = OVERBOUGHT if wr_v > -20 else OVERSOLD if wr_v < -80 else NEUTRAL
    indicators.append(IndicatorRow("Williams %R", wr_v, wr_signal, counted=False))

    cci_v = last(_cci(high, low, close, 14))
    cci_signal = BUY if cci_v > 100 else SELL if cci_v < -100 else NEUTRAL
    indicators.append(IndicatorRow("CCI(14)", cci_v, cci_signal, counted=True))

    atr_series = _atr(high, low, close, 14)
    atr_v = last(atr_series)
    atr_avg_v = last(atr_series.rolling(14).mean())
    atr_signal = LESS_VOL if atr_v < atr_avg_v else HIGH_VOL
    indicators.append(IndicatorRow("ATR(14)", atr_v, atr_signal, counted=False))

    hl_v = last(_highs_lows(high, low, 14))
    hl_signal = _sign_signal(hl_v)
    indicators.append(IndicatorRow("Highs/Lows(14)", hl_v, hl_signal, counted=True))

    bbp_v = last(_bull_bear_power(high, low, close, 13))
    bbp_signal = _sign_signal(bbp_v)
    indicators.append(IndicatorRow("Bull/Bear Power(13)", bbp_v, bbp_signal, counted=True))

    uo_v = last(_ultimate_oscillator(high, low, close, 7, 14, 28))
    uo_signal = BUY if uo_v > 50 else SELL if uo_v < 50 else NEUTRAL
    indicators.append(IndicatorRow("Ultimate Oscillator", uo_v, uo_signal, counted=True))

    roc_v = last(_roc(close, 12))
    roc_signal = _sign_signal(roc_v)
    indicators.append(IndicatorRow("ROC", roc_v, roc_signal, counted=True))

    close_v = last(close)
    mas: list[MARow] = []
    for p in MA_PERIODS:
        sma_v = last(close.rolling(p).mean())
        ema_v = last(close.ewm(span=p, adjust=False).mean())
        mas.append(
            MARow(
                period=p,
                sma=sma_v,
                sma_signal=BUY if close_v > sma_v else SELL if close_v < sma_v else NEUTRAL,
                ema=ema_v,
                ema_signal=BUY if close_v > ema_v else SELL if close_v < ema_v else NEUTRAL,
            )
        )

    return TechnicalSummary(
        ticker=ticker,
        as_of=df.index[-1],
        close=close_v,
        indicators=indicators,
        mas=mas,
    )
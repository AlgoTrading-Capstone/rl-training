"""
KAMA Trend Strategy

Pine Script source: input/KAMA-Trend-Strategy.pine
Timeframe: 1h  |  Lookback: 100 bars (100 h)

Entry logic:
  Long  — fast KAMA (ER=10, fast=2, slow=10) crosses above slow KAMA
           (ER=10, fast=10, slow=30) while slow KAMA slope is rising.
  Short — fast KAMA crosses below slow KAMA while slow KAMA slope is falling
           (i.e., NOT rising).

Exit management (ATR-based stop-loss and take-profit, and swing-high/low fallback)
is NOT implemented here. It must be configured in the external execution layer.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import talib

from src.base_strategy import BaseStrategy, StrategyRecommendation, SignalType


def _compute_kama(
    src: np.ndarray, er_length: int, fast_length: int, slow_length: int
) -> np.ndarray:
    """Kaufman's Adaptive Moving Average (KAMA).

    Matches TVta.kama(source, er_length, fast_length, slow_length).
    Recursive — cannot be vectorized; runs in a single forward pass.

    Parameters
    ----------
    src        : 1-D float64 array (close prices).
    er_length  : Efficiency Ratio lookback window.
    fast_length: Fast smoothing constant period.
    slow_length: Slow smoothing constant period.
    """
    fast_sc = 2.0 / (fast_length + 1)
    slow_sc = 2.0 / (slow_length + 1)

    n = len(src)
    kama = np.full(n, np.nan)

    if n <= er_length:
        return kama

    # Seed: first computable bar is er_length
    kama[er_length] = src[er_length]

    for i in range(er_length + 1, n):
        direction = abs(src[i] - src[i - er_length])
        volatility = float(np.sum(np.abs(np.diff(src[i - er_length : i + 1]))))
        er = direction / volatility if volatility != 0.0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (src[i] - kama[i - 1])

    return kama


class KamaTrendStrategy(BaseStrategy):
    """KAMA Trend Strategy.

    Uses dual Kaufman Adaptive Moving Averages (fast + slow) for trend
    detection. A crossover of fast KAMA over slow KAMA, filtered by the
    slow KAMA slope, generates entry signals.

    Default parameters match the Pine Script inputs:
        er_len        : 10   (Efficiency Ratio window)
        fast_kama_fast: 2    (fast KAMA — fast SC period)
        fast_kama_slow: 10   (fast KAMA — slow SC period)
        slow_kama_fast: 10   (slow KAMA — fast SC period)
        slow_kama_slow: 30   (slow KAMA — slow SC period)
        atr_len       : 14   (ATR period — used for exit sizing only)

    Exit logic (ATR SL/TP, swing-based fallback) is NOT implemented.
    Configure exits in the execution layer.
    """

    def __init__(
        self,
        er_len: int = 10,
        fast_kama_fast: int = 2,
        fast_kama_slow: int = 10,
        slow_kama_fast: int = 10,
        slow_kama_slow: int = 30,
        atr_len: int = 14,
    ):
        super().__init__(
            name="KAMA Trend Strategy",
            description=(
                "Dual Kaufman Adaptive Moving Average crossover strategy. "
                "Long when fast KAMA crosses above slow KAMA and slow KAMA is rising. "
                "Short when fast KAMA crosses below slow KAMA and slow KAMA is not rising."
            ),
            timeframe="1h",
            lookback_hours=100,
        )
        self.er_len = er_len
        self.fast_kama_fast = fast_kama_fast
        self.fast_kama_slow = fast_kama_slow
        self.slow_kama_fast = slow_kama_fast
        self.slow_kama_slow = slow_kama_slow
        self.atr_len = atr_len

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        close = df["close"].to_numpy(dtype=float)

        # Warmup guard: 3 × slowest recursive period
        min_bars = 3 * max(self.er_len, self.slow_kama_slow, self.atr_len)
        if len(close) < min_bars:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # ---- Indicators ------------------------------------------------
        kama_fast = _compute_kama(close, self.er_len, self.fast_kama_fast, self.fast_kama_slow)
        kama_slow = _compute_kama(close, self.er_len, self.slow_kama_fast, self.slow_kama_slow)

        idx = len(close) - 1  # current (last complete) bar

        # Guard: need valid KAMA values at current and previous bar
        if (
            np.isnan(kama_fast[idx])
            or np.isnan(kama_fast[idx - 1])
            or np.isnan(kama_slow[idx])
            or np.isnan(kama_slow[idx - 1])
        ):
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # ---- Signal logic at current bar --------------------------------
        # kamaFastUp = kamaFast > kamaFast[1]   (unused in entry — matches Pine)
        kama_slow_up = kama_slow[idx] > kama_slow[idx - 1]

        bull_cross = kama_fast[idx] > kama_slow[idx] and kama_fast[idx - 1] <= kama_slow[idx - 1]
        bear_cross = kama_fast[idx] < kama_slow[idx] and kama_fast[idx - 1] >= kama_slow[idx - 1]

        long_entry = bull_cross and kama_slow_up
        short_entry = bear_cross and not kama_slow_up

        if long_entry:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        if short_entry:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)
        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

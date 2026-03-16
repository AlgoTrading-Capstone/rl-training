"""Evasive SuperTrend Strategy [Source Select]

Pine Script source: input/Evasive-SuperTrend-Strategy-Source-Select.pine
Timeframe: 1h  |  Lookback: 50 bars (~50 h)

Entry logic:
  Long  — SuperTrend flips from bearish (-1) to bullish (1):
          trend changes from -1 to 1 on this bar (ta.change(trend) > 0).

  Short — SuperTrend flips from bullish (1) to bearish (-1):
          trend changes from 1 to -1 on this bar (ta.change(trend) < 0).

This strategy implements a custom "Evasive" SuperTrend variant that adds:
  - Configurable source input (default: hl2)
  - Noise avoidance logic: if |close - prevLine| < atr * threshold, the band
    creeps toward price (expands slightly) rather than snapping to a new level.
  - When noisy (in a whipsaw zone), the trend line drifts by `alpha * atr`
    per bar; otherwise it uses the standard max/min clamp logic.

Exit / position management is NOT handled here.
Configure stop-loss and take-profit in the external execution layer.

Key parameters (defaults match Pine Script inputs):
    length_input    : 10  — ATR period
    multiplier_input: 3.0 — base band multiplier
    threshold_input : 1.0 — noise threshold (× ATR)
    alpha_input     : 0.5 — band expansion speed when noisy (× ATR)
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import talib

from src.base_strategy import BaseStrategy, SignalType, StrategyRecommendation


class EvasiveSuperTrendStrategySourceSelectStrategy(BaseStrategy):
    """Evasive SuperTrend with noise-avoidance band logic.

    The strategy mirrors the PineScript logic bar-by-bar via a Python for-loop
    because ``st_line`` at bar i depends on ``st_line[i-1]`` (true recursive
    state) — this cannot be vectorised without distorting the adaptive band.

    Algorithm per bar (after ATR is available):
    1. Compute upper/lower base bands from hl2 ± multiplier × ATR.
    2. If |close[i-1] - st_line[i-1]| < threshold × ATR  (noisy zone):
         - trend == 1:  st_line drifts down by alpha × ATR  (band creeps toward price)
         - trend == -1: st_line drifts up   by alpha × ATR
    3. If NOT noisy (normal zone):
         - trend == 1:  st_line = max(lower_base, st_line[i-1])  (floor ratchet up)
         - trend == -1: st_line = min(upper_base, st_line[i-1])  (ceiling ratchet down)
    4. Flip check:
         - If trend == 1  and close[i] < st_line → trend = -1, st_line = upper_base
         - If trend == -1 and close[i] > st_line → trend = +1, st_line = lower_base
    """

    # Minimum bars before signals can be evaluated.
    # 3 × ATR period (10) = 30 to allow ATR to converge.
    MIN_BARS: int = 30

    def __init__(
        self,
        length_input: int = 10,
        multiplier_input: float = 3.0,
        threshold_input: float = 1.0,
        alpha_input: float = 0.5,
    ):
        super().__init__(
            name="Evasive SuperTrend Strategy [Source Select]",
            description=(
                "Custom SuperTrend with noise-avoidance logic. Enters long when "
                "the adaptive band flips bullish, short when it flips bearish. "
                "Uses hl2 as the source by default. In the noisy zone the band "
                "creeps at alpha×ATR per bar instead of making hard level jumps."
            ),
            timeframe="1h",
            lookback_hours=50,
        )
        self.length_input = length_input
        self.multiplier_input = multiplier_input
        self.threshold_input = threshold_input
        self.alpha_input = alpha_input

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_evasive_supertrend(
        self,
        src: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the evasive SuperTrend line and trend direction arrays.

        Parameters
        ----------
        src   : Source series (default hl2 = (high + low) / 2).
        high, low, close : Standard OHLCV arrays.
        atr   : ATR array (NaN for early bars).

        Returns
        -------
        st_line : Band value for each bar (NaN until ATR is available).
        trend   : +1 (bullish) or -1 (bearish) for each bar.
        """
        n = len(src)
        st_line = np.full(n, np.nan)
        trend = np.ones(n, dtype=int)  # Pine default: var int trend = 1

        for i in range(n):
            if np.isnan(atr[i]):
                # ATR not yet available — keep defaults (NaN / 1)
                continue

            upper_base = src[i] + self.multiplier_input * atr[i]
            lower_base = src[i] - self.multiplier_input * atr[i]

            # Previous state
            if i == 0 or np.isnan(st_line[i - 1]):
                # First valid bar: initialise based on default trend (1 = bullish)
                prev_trend = 1
                prev_line = lower_base  # nz(na, lowerBase) when trend==1
            else:
                prev_trend = int(trend[i - 1])
                prev_line = st_line[i - 1]

            # Noise detection: |close[i] - prevLine| < threshold × ATR
            # Pine uses close (current bar) in the isNoisy check
            is_noisy = abs(close[i] - prev_line) < (atr[i] * self.threshold_input)

            # Carry forward trend (will flip below if condition met)
            trend[i] = prev_trend

            if prev_trend == 1:
                if is_noisy:
                    st_line[i] = prev_line - (atr[i] * self.alpha_input)
                else:
                    st_line[i] = max(lower_base, prev_line)
                # Flip check
                if close[i] < st_line[i]:
                    trend[i] = -1
                    st_line[i] = upper_base
            else:  # prev_trend == -1
                if is_noisy:
                    st_line[i] = prev_line + (atr[i] * self.alpha_input)
                else:
                    st_line[i] = min(upper_base, prev_line)
                # Flip check
                if close[i] > st_line[i]:
                    trend[i] = 1
                    st_line[i] = lower_base

        return st_line, trend

    # ------------------------------------------------------------------
    # Public run() method
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        """Evaluate the Evasive SuperTrend strategy for the current bar.

        Parameters
        ----------
        df        : OHLCV DataFrame (columns: open, high, low, close, volume, date).
                    All rolling/shift operations are backward-looking only.
        timestamp : UTC datetime for the evaluation bar.

        Returns
        -------
        StrategyRecommendation with LONG, SHORT, or HOLD signal.
        """
        if len(df) < self.MIN_BARS:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        df = df.reset_index(drop=True)

        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        # Source: hl2 (Pine default for srcInput)
        src = (high + low) / 2.0

        # ATR — TA-Lib (backward-looking, no lookahead)
        atr = talib.ATR(high, low, close, timeperiod=self.length_input)

        # Guard: ATR must be valid at the last two bars
        n = len(df)
        idx = n - 1
        if np.isnan(atr[idx]) or np.isnan(atr[idx - 1]):
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # Compute the adaptive SuperTrend over all bars
        st_line, trend = self._compute_evasive_supertrend(src, high, low, close, atr)

        # Signal: ta.change(trend) > 0  → trend flipped from -1 to +1 → LONG
        #         ta.change(trend) < 0  → trend flipped from +1 to -1 → SHORT
        trend_curr = int(trend[idx])
        trend_prev = int(trend[idx - 1])
        trend_change = trend_curr - trend_prev  # equivalent to ta.change(trend)

        if trend_change > 0:
            # Bullish flip: was -1, now +1
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        elif trend_change < 0:
            # Bearish flip: was +1, now -1
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)

        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

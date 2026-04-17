from datetime import datetime

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType


def _pivot_high(series: pd.Series, left_bars: int, right_bars: int) -> pd.Series:
    """
    Vectorised equivalent of Pine's pivothigh(leftBars, rightBars).

    A pivot high at candidate bar i is confirmed right_bars bars later:
      - series[i] >= all of series[i-left_bars : i]   (left side)
      - series[i] >= all of series[i+1 : i+right_bars+1]  (right side)
    The result value is placed at index i + right_bars to match Pine's reporting.

    Uses rolling-window operations instead of a Python loop for performance.
    """
    # Left side: candidate must equal the rolling max over (left_bars + 1) window
    left_max = series.rolling(window=left_bars + 1, min_periods=left_bars + 1).max()
    # Right side: reverse the series, rolling max, then reverse back
    right_max = series.iloc[::-1].rolling(window=right_bars + 1, min_periods=right_bars + 1).max().iloc[::-1]

    is_pivot = (series == left_max) & (series == right_max)
    result = pd.Series(np.nan, index=series.index)
    result[is_pivot] = series[is_pivot]
    # Shift by right_bars to match Pine's delayed reporting
    return result.shift(right_bars)


def _pivot_low(series: pd.Series, left_bars: int, right_bars: int) -> pd.Series:
    """
    Vectorised equivalent of Pine's pivotlow(leftBars, rightBars).
    Uses rolling-window operations instead of a Python loop for performance.
    """
    left_min = series.rolling(window=left_bars + 1, min_periods=left_bars + 1).min()
    right_min = series.iloc[::-1].rolling(window=right_bars + 1, min_periods=right_bars + 1).min().iloc[::-1]

    is_pivot = (series == left_min) & (series == right_min)
    result = pd.Series(np.nan, index=series.index)
    result[is_pivot] = series[is_pivot]
    return result.shift(right_bars)


class MonthlyReturnsInPinescriptStrategiesStrategy(BaseStrategy):

    def __init__(self):
        super().__init__(
            name="MonthlyReturnsInPinescriptStrategiesStrategy",
            description=(
                "Pivot-breakout strategy converted from Pine Script v4. "
                "Detects pivot highs/lows and arms long/short entry flags. "
                "Goes long when a pivot-high breakout is pending; "
                "goes short when a pivot-low breakdown is pending. "
                "Monthly P&L table (visual only) is excluded."
            ),
            timeframe="15m",
            lookback_hours=13,
        )
        self.left_bars = 2
        self.right_bars = 1
        # Dynamic RL warmup: 3x the pivot detection window
        self.MIN_CANDLES_REQUIRED = 3 * (self.left_bars + self.right_bars)

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # --- Pivot detection (vectorised) ---
        swh = _pivot_high(df["high"], self.left_bars, self.right_bars)
        swl = _pivot_low(df["low"], self.left_bars, self.right_bars)

        # Last confirmed pivot levels, forward-filled.
        hprice = swh.ffill()
        lprice = swl.ffill()

        high_arr = df["high"].to_numpy(dtype=float)
        low_arr = df["low"].to_numpy(dtype=float)
        hprice_arr = hprice.to_numpy(dtype=float)
        lprice_arr = lprice.to_numpy(dtype=float)

        idx = len(df) - 1
        prev_idx = idx - 1

        has_long_level = not np.isnan(hprice_arr[idx]) and not np.isnan(hprice_arr[prev_idx])
        has_short_level = not np.isnan(lprice_arr[idx]) and not np.isnan(lprice_arr[prev_idx])

        long_pulse = (
            has_long_level
            and high_arr[prev_idx] <= hprice_arr[prev_idx]
            and high_arr[idx] > hprice_arr[idx]
        )
        short_pulse = (
            has_short_level
            and low_arr[prev_idx] >= lprice_arr[prev_idx]
            and low_arr[idx] < lprice_arr[idx]
        )

        # Emit a pulse only on the breakout bar itself.
        if long_pulse and not short_pulse:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        if short_pulse and not long_pulse:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)

        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

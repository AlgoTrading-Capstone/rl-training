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
    """
    n = len(series)
    result = np.full(n, np.nan)
    arr = series.to_numpy(dtype=float)
    for i in range(left_bars, n - right_bars):
        candidate = arr[i]
        if candidate >= arr[i - left_bars: i].max() and candidate >= arr[i + 1: i + right_bars + 1].max():
            result[i + right_bars] = candidate
    return pd.Series(result, index=series.index)


def _pivot_low(series: pd.Series, left_bars: int, right_bars: int) -> pd.Series:
    """
    Vectorised equivalent of Pine's pivotlow(leftBars, rightBars).
    """
    n = len(series)
    result = np.full(n, np.nan)
    arr = series.to_numpy(dtype=float)
    for i in range(left_bars, n - right_bars):
        candidate = arr[i]
        if candidate <= arr[i - left_bars: i].min() and candidate <= arr[i + 1: i + right_bars + 1].min():
            result[i + right_bars] = candidate
    return pd.Series(result, index=series.index)


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
        # Dynamic RL warmup: 3× the pivot detection window
        self.MIN_CANDLES_REQUIRED = 3 * (self.left_bars + self.right_bars)

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # --- Pivot detection ---
        swh = _pivot_high(df["high"], self.left_bars, self.right_bars)
        swl = _pivot_low(df["low"], self.left_bars, self.right_bars)

        # hprice / lprice: last confirmed pivot price, forward-filled
        hprice = swh.ffill().fillna(0.0)
        lprice = swl.ffill().fillna(0.0)

        # --- Stateful le / se flags (depend on own previous value) ---
        swh_arr = swh.to_numpy(dtype=float)
        swl_arr = swl.to_numpy(dtype=float)
        high_arr = df["high"].to_numpy(dtype=float)
        low_arr = df["low"].to_numpy(dtype=float)
        hprice_arr = hprice.to_numpy(dtype=float)
        lprice_arr = lprice.to_numpy(dtype=float)

        n = len(df)
        le = np.zeros(n, dtype=bool)
        se = np.zeros(n, dtype=bool)

        for i in range(1, n):
            # Long entry flag
            if not np.isnan(swh_arr[i]):
                le[i] = True
            elif le[i - 1] and high_arr[i] > hprice_arr[i]:
                le[i] = False
            else:
                le[i] = le[i - 1]

            # Short entry flag
            if not np.isnan(swl_arr[i]):
                se[i] = True
            elif se[i - 1] and low_arr[i] < lprice_arr[i]:
                se[i] = False
            else:
                se[i] = se[i - 1]

        # --- Signal from the last bar ---
        if le[-1]:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        if se[-1]:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)

        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

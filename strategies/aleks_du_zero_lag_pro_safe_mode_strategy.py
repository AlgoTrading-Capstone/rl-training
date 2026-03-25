"""
[AleksDU AI] Zero-Lag Pro: Safe Mode

Pine Script source: input/AleksDU-Zero-Lag-Pro-Safe-Mode.pine
Timeframe: 4h  |  Lookback: 38 bars (152 h)

Strategy logic:
  Uses a custom Hull Moving Average (HMA) with an ATR-based noise filter to
  define a volatility corridor around the HMA.

  Long  — close crosses above (HMA + ATR_noise_filter): price breaks out of
           the upper band, signalling a new uptrend.
  Short — close crosses below (HMA - ATR_noise_filter): price breaks out of
           the lower band, signalling a new downtrend.
  Exit  — close crosses the raw HMA line in either direction.

Stop-loss and take-profit management is NOT implemented here.
Configure exits in the external execution layer.
"""

from datetime import datetime
import math

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType


class AleksDuZeroLagProSafeModeStrategy(BaseStrategy):
    """Zero-Lag Pro: Safe Mode strategy.

    Combines a Hull Moving Average (HMA) with an ATR volatility corridor.
    Signals fire when price breaks outside the noise-filtered band.

    Default parameters match the Pine Script inputs:
        hma_len          : 32   (AI-HMA length)
        filter_strength  : 0.5  (ATR multiplier for noise band width)
        atr_period       : 14   (ATR period for noise filter)
    """

    def __init__(
        self,
        hma_len: int = 32,
        filter_strength: float = 0.5,
        atr_period: int = 14,
    ):
        super().__init__(
            name="[AleksDU AI] Zero-Lag Pro: Safe Mode",
            description=(
                "Trend-following strategy using a Hull Moving Average (HMA) with an "
                "ATR-based noise filter. Enters long/short when close breaks outside "
                "the volatility corridor. Exits when close crosses back through the HMA."
            ),
            timeframe="4h",
            lookback_hours=152,
        )
        self.hma_len = hma_len
        self.filter_strength = filter_strength
        self.atr_period = atr_period
        # CRITICAL RL GUARD: dynamic warmup — max indicator lookback * 3
        self.MIN_CANDLES_REQUIRED = 3 * max(self.hma_len, self.atr_period)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hma(self, src: np.ndarray, length: int) -> np.ndarray:
        """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n)).

        Mirrors the Pine Script custom hma() function:
            hma(s, l) => ta.wma(2 * ta.wma(s, l/2) - ta.wma(s, l), math.round(math.sqrt(l)))
        """
        half_len = length // 2
        sqrt_len = int(round(math.sqrt(length)))

        wma_half = talib.WMA(src, timeperiod=half_len)
        wma_full = talib.WMA(src, timeperiod=length)
        raw = 2.0 * wma_half - wma_full
        return talib.WMA(raw, timeperiod=sqrt_len)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)

        # ---- Indicators -----------------------------------------------
        fast_ma = self._hma(close, self.hma_len)
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        noise_filter = atr * self.filter_strength

        upper_band = fast_ma + noise_filter
        lower_band = fast_ma - noise_filter

        idx = len(close) - 1  # current (last complete) bar
        prev = idx - 1

        # Guard: ensure all indicator values are available
        if (
            np.isnan(fast_ma[idx]) or np.isnan(fast_ma[prev])
            or np.isnan(upper_band[idx]) or np.isnan(upper_band[prev])
            or np.isnan(lower_band[idx]) or np.isnan(lower_band[prev])
        ):
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # ---- Signal conditions ----------------------------------------
        # Long: close crosses above upper band (noise-filtered HMA)
        long_cond = (
            close[idx] > upper_band[idx]
            and close[prev] <= upper_band[prev]
        )
        # Short: close crosses below lower band
        short_cond = (
            close[idx] < lower_band[idx]
            and close[prev] >= lower_band[prev]
        )
        # Exit: close crosses the raw HMA in either direction
        exit_cond = (
            (close[idx] > fast_ma[idx] and close[prev] <= fast_ma[prev])
            or (close[idx] < fast_ma[idx] and close[prev] >= fast_ma[prev])
        )

        # ---- Signal priority: FLAT > LONG > SHORT > HOLD ---------------
        if exit_cond:
            return StrategyRecommendation(signal=SignalType.FLAT, timestamp=timestamp)
        if long_cond:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        if short_cond:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)
        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

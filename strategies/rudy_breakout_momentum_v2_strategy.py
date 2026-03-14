"""Rudy Breakout Momentum v2 Strategy

Pine Script source: input/Rudy-Breakout-Momentum-v2.pine
Timeframe: 1D  |  Lookback: 126 bars (~3024 h)

Entry logic:
  Long — price makes a new 126-bar high (relative to previous bar's rolling high),
         fast EMA (21) is above slow EMA (50), and RSI is between 40 and 80.

Exit logic:
  FLAT — close crosses under EMA21 (close is below EMA21 on current bar but was
         at or above EMA21 on the previous bar).

Priority: FLAT (trend break exit) takes precedence over LONG entry if both
          conditions are simultaneously true on the same bar.

Position sizing, profit targets, and stop-loss levels are NOT implemented here.
Configure those in the external execution layer.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import talib

from src.base_strategy import BaseStrategy, StrategyRecommendation, SignalType


class RudyBreakoutMomentumV2Strategy(BaseStrategy):
    """Rudy Breakout Momentum v2 Strategy.

    A breakout-momentum system that enters long when price breaks out to a new
    126-bar high, trend is confirmed by EMA21 > EMA50, and RSI is in the 40-80
    range (not overbought, not oversold).

    Exits when price closes back below EMA21 (trend break signal).

    Default parameters match the Pine Script inputs:
        lookback : 126  (high lookback bars)
        ema_fast : 21   (fast EMA period)
        ema_slow : 50   (slow EMA period)
        rsi_period: 14  (RSI period)
    """

    # Minimum candles needed before the first valid signal.
    # Covers 126-bar rolling high + EMA50 warmup + RSI14 warmup + buffer.
    MIN_BARS: int = 160

    def __init__(
        self,
        lookback: int = 126,
        ema_fast: int = 21,
        ema_slow: int = 50,
        rsi_period: int = 14,
    ):
        super().__init__(
            name="Rudy Breakout Momentum v2",
            description=(
                "Breakout-momentum strategy: enters long on a new 126-bar high with "
                "EMA trend confirmation (EMA21 > EMA50) and RSI in the 40–80 range. "
                "Exits on a close below EMA21 (trend break)."
            ),
            timeframe="1D",
            lookback_hours=3024,
        )
        self.lookback = lookback
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        """Evaluate the strategy for the current bar.

        Parameters
        ----------
        df        : OHLCV DataFrame with columns: open, high, low, close, volume, date.
                    All shift() operations use positive integers (backward-looking only).
        timestamp : UTC datetime for the current evaluation bar.

        Returns
        -------
        StrategyRecommendation with LONG, FLAT, or HOLD signal.
        """
        if len(df) < self.MIN_BARS:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        close = df["close"].reset_index(drop=True)
        high = df["high"].reset_index(drop=True)

        # ---- Indicators (vectorized over full df) -----------------------

        # EMA21 and EMA50 via TA-Lib
        ema21_arr = talib.EMA(close.values.astype(float), timeperiod=self.ema_fast)
        ema50_arr = talib.EMA(close.values.astype(float), timeperiod=self.ema_slow)

        # RSI via TA-Lib
        rsi_arr = talib.RSI(close.values.astype(float), timeperiod=self.rsi_period)

        # Rolling 126-bar highest high (pandas rolling, backward-looking)
        # high126[1] in Pine = previous bar's rolling max → shift(1) here to avoid
        # including the current bar's high when evaluating whether current high
        # broke out above the prior rolling max.
        high126_shifted = high.rolling(self.lookback).max().shift(1)

        # ---- Last-bar values -------------------------------------------
        idx = len(close) - 1  # current bar index
        if idx < 1:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        close_curr = close.iloc[idx]
        close_prev = close.iloc[idx - 1]
        high_curr = high.iloc[idx]

        ema21_curr = ema21_arr[idx]
        ema21_prev = ema21_arr[idx - 1]
        ema50_curr = ema50_arr[idx]
        rsi_curr = rsi_arr[idx]
        high126_prev = high126_shifted.iloc[idx]  # previous bar's 126-high (shift(1))

        # Guard: if any indicator is NaN, not enough data yet
        if (
            np.isnan(ema21_curr)
            or np.isnan(ema21_prev)
            or np.isnan(ema50_curr)
            or np.isnan(rsi_curr)
            or np.isnan(high126_prev)
        ):
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # ---- Conditions -------------------------------------------------

        # trendUp = ema21 > ema50
        trend_up = ema21_curr > ema50_curr

        # newHigh = high >= high126[1]  (current high breaks prior rolling max)
        new_high = high_curr >= high126_prev

        # rsiOk = rsiVal < 80 and rsiVal > 40
        rsi_ok = 40.0 < rsi_curr < 80.0

        # breakoutBuy = newHigh and trendUp and rsiOk
        breakout_buy = new_high and trend_up and rsi_ok

        # trendBroken = close < ema21 and close[1] >= ema21[1]
        # i.e. close crossed under EMA21 on this bar
        trend_broken = close_curr < ema21_curr and close_prev >= ema21_prev

        # ---- Signal priority: FLAT (exit) wins over LONG (entry) --------
        if trend_broken:
            return StrategyRecommendation(signal=SignalType.FLAT, timestamp=timestamp)
        if breakout_buy:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

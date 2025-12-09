"""
Volatility System Strategy

Adapted from Freqtrade version based on https://www.tradingview.com/script/3hhs0XbR/

Logic:
    - Resample candles into 3h blocks
    - Compute ATR * 2
    - Compute absolute close-change
    - LONG when upward price movement > ATR
    - SHORT when downward movement > ATR
"""

from datetime import datetime

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from strategies.base_strategy import (BaseStrategy, SignalType, StrategyRecommendation)
from utils.resampling import resample_to_interval, resampled_merge


class VolatilitySystem(BaseStrategy):
    """
    Volatility System Strategy for Bitcoin futures trading.
    Detects breakout conditions using ATR-based volatility threshold.
    """

    # ATR(14) on 3h data needs: 14 * 3 + buffer = 52 candles minimum
    MIN_CANDLES_REQUIRED = 52

    def __init__(self):
        super().__init__(
            name="VolatilitySystem",
            description="Volatility breakout system based on ATR and candle movement.",
            timeframe="1h",
            lookback_hours=66  # 14 * 3 + 24 buffer
        )

    def _calculate_indicators(self, df: DataFrame) -> DataFrame:
        """Reproduce populate_indicators() logic from Freqtrade version."""
        resample_int = 60 * 3  # 3 hours in minutes
        resampled = resample_to_interval(df, resample_int)

        # ATR (period 14) * 2
        resampled["atr"] = ta.ATR(resampled, timeperiod=14) * 2.0

        # Close-change absolute
        resampled["close_change"] = resampled["close"].diff()
        resampled["abs_close_change"] = resampled["close_change"].abs()

        # Merge back into original df
        df = resampled_merge(df, resampled, fill_na=True)
        df["atr"] = df[f"resample_{resample_int}_atr"]
        df["close_change"] = df[f"resample_{resample_int}_close_change"]
        df["abs_close_change"] = df[f"resample_{resample_int}_abs_close_change"]

        return df

    def _generate_signal(self, df: DataFrame) -> SignalType:
        """
        Reproduce populate_entry_trend() logic.

        LONG when positive move > ATR (shifted)
        SHORT when negative move > ATR (shifted)
        Otherwise HOLD.
        """
        if len(df) < 2:
            return SignalType.HOLD

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        close_change = last_row["close_change"]
        atr_prev = prev_row["atr"]  # CRITICAL: use shifted ATR like original

        if pd.isna(close_change) or pd.isna(atr_prev):
            return SignalType.HOLD

        # LONG signal
        if close_change > atr_prev:
            return SignalType.LONG

        # SHORT signal
        if -close_change > atr_prev:
            return SignalType.SHORT

        return SignalType.HOLD

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        """Execute the VolatilitySystem logic."""

        if df is None or len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        df = df.copy()
        df = self._calculate_indicators(df)

        signal = self._generate_signal(df)

        # timestemp in exit should be the decision time (LONG/SHORT/HOLD)

        return StrategyRecommendation(signal=signal, timestamp=timestamp)
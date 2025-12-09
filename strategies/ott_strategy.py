"""
OTT (Optimized Trend Tracker) Strategy

Adapted from Freqtrade FOttStrategy.

Logic:
    - Calculate OTT indicator using CMO-based EMA
    - Generate VAR (smoothed close) and OTT lines
    - LONG when VAR crosses above OTT
    - SHORT when VAR crosses below OTT

Note: Exit signals (ADX > 60) removed as exits are handled by meta-strategy layer.
"""

from datetime import datetime

import pandas as pd
import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from strategies.base_strategy import (BaseStrategy, SignalType, StrategyRecommendation)


class OTTStrategy(BaseStrategy):
    """
    OTT (Optimized Trend Tracker) Strategy for Bitcoin trading.
    Uses CMO-based smoothing to detect trend changes.
    """

    # Rolling window is 9 periods, plus initialization buffer
    MIN_CANDLES_REQUIRED = 30

    def __init__(self):
        super().__init__(
            name="OTTStrategy",
            description="Optimized Trend Tracker using CMO-based EMA for trend detection.",
            timeframe="1h",
            lookback_hours=50  # 30 candles + buffer for 1h timeframe
        )

    def _calculate_ott(self, df: DataFrame) -> DataFrame:
        """
        Calculate OTT indicator.
        EXACT copy of original ott() method logic.
        """
        df = df.copy()

        pds = 2
        percent = 1.4
        alpha = 2 / (pds + 1)

        df["ud1"] = np.where(
            df["close"] > df["close"].shift(1), (df["close"] - df["close"].shift()), 0
        )
        df["dd1"] = np.where(
            df["close"] < df["close"].shift(1), (df["close"].shift() - df["close"]), 0
        )
        df["UD"] = df["ud1"].rolling(9).sum()
        df["DD"] = df["dd1"].rolling(9).sum()
        df["CMO"] = ((df["UD"] - df["DD"]) / (df["UD"] + df["DD"])).fillna(0).abs()

        # df['Var'] = talib.EMA(df['close'], timeperiod=5)
        df["Var"] = 0.0
        for i in range(pds, len(df)):
            var_col = df.columns.get_loc("Var")
            cmo_col = df.columns.get_loc("CMO")
            close_col = df.columns.get_loc("close")
            prev_var_col = df.columns.get_loc("Var")
            df.iloc[i, var_col] = (
                    alpha * df.iloc[i, cmo_col] * df.iloc[i, close_col]
                    + (1 - alpha * df.iloc[i, cmo_col]) * df.iloc[i - 1, prev_var_col]
            )

        df["fark"] = df["Var"] * percent * 0.01
        df["newlongstop"] = df["Var"] - df["fark"]
        df["newshortstop"] = df["Var"] + df["fark"]
        df["longstop"] = 0.0
        df["shortstop"] = 999999999999999999.0
        # df['dir'] = 1
        for i in df["UD"]:

            def maxlongstop():
                df.loc[(df["newlongstop"] > df["longstop"].shift(1)), "longstop"] = df[
                    "newlongstop"
                ]
                df.loc[(df["longstop"].shift(1) > df["newlongstop"]), "longstop"] = df[
                    "longstop"
                ].shift(1)

                return df["longstop"]

            def minshortstop():
                df.loc[
                    (df["newshortstop"] < df["shortstop"].shift(1)), "shortstop"
                ] = df["newshortstop"]
                df.loc[
                    (df["shortstop"].shift(1) < df["newshortstop"]), "shortstop"
                ] = df["shortstop"].shift(1)

                return df["shortstop"]

            df["longstop"] = np.where(
                ((df["Var"] > df["longstop"].shift(1))),
                maxlongstop(),
                df["newlongstop"],
            )

            df["shortstop"] = np.where(
                ((df["Var"] < df["shortstop"].shift(1))),
                minshortstop(),
                df["newshortstop"],
            )

        # get xover

        df["xlongstop"] = np.where(
            (
                (df["Var"].shift(1) > df["longstop"].shift(1))
                & (df["Var"] < df["longstop"].shift(1))
            ),
            1,
            0,
        )

        df["xshortstop"] = np.where(
            (
                (df["Var"].shift(1) < df["shortstop"].shift(1))
                & (df["Var"] > df["shortstop"].shift(1))
            ),
            1,
            0,
        )

        df["trend"] = 0
        df["dir"] = 0
        for i in df["UD"]:
            df["trend"] = np.where(
                ((df["xshortstop"] == 1)),
                1,
                (np.where((df["xlongstop"] == 1), -1, df["trend"].shift(1))),
            )

            df["dir"] = np.where(
                ((df["xshortstop"] == 1)),
                1,
                (np.where((df["xlongstop"] == 1), -1, df["dir"].shift(1).fillna(1))),
            )

        # get OTT

        df["MT"] = np.where(df["dir"] == 1, df["longstop"], df["shortstop"])
        df["OTT"] = np.where(
            df["Var"] > df["MT"],
            (df["MT"] * (200 + percent) / 200),
            (df["MT"] * (200 - percent) / 200),
        )
        df["OTT"] = df["OTT"].shift(2)

        return df

    def _calculate_indicators(self, df: DataFrame) -> DataFrame:
        """Reproduce populate_indicators() logic from Freqtrade version."""

        # Calculate OTT indicator (returns df with OTT and Var columns)
        df = self._calculate_ott(df)

        # Calculate ADX (not used for entry, but available for future use)
        df["adx"] = ta.ADX(df, timeperiod=14)

        return df

    def _crossed_above(self, series1: pd.Series, series2: pd.Series) -> bool:
        """
        Check if series1 crossed above series2 in the last candle.
        Equivalent to qtpylib.crossed_above().
        """
        if len(series1) < 2 or len(series2) < 2:
            return False

        # Previous: series1 was below or equal to series2
        # Current: series1 is above series2
        prev_below = series1.iloc[-2] <= series2.iloc[-2]
        curr_above = series1.iloc[-1] > series2.iloc[-1]

        return prev_below and curr_above

    def _crossed_below(self, series1: pd.Series, series2: pd.Series) -> bool:
        """
        Check if series1 crossed below series2 in the last candle.
        Equivalent to qtpylib.crossed_below().
        """
        if len(series1) < 2 or len(series2) < 2:
            return False

        # Previous: series1 was above or equal to series2
        # Current: series1 is below series2
        prev_above = series1.iloc[-2] >= series2.iloc[-2]
        curr_below = series1.iloc[-1] < series2.iloc[-1]

        return prev_above and curr_below

    def _generate_signal(self, df: DataFrame) -> SignalType:
        """
        Reproduce populate_entry_trend() logic.

        LONG when VAR crosses above OTT
        SHORT when VAR crosses below OTT
        Otherwise HOLD.

        Note: Exit conditions (ADX > 60) removed as handled by meta-strategy layer.
        """
        if len(df) < 2:
            return SignalType.HOLD

        # Check for NaN values in required columns
        last_row = df.iloc[-1]
        if pd.isna(last_row["Var"]) or pd.isna(last_row["OTT"]):
            return SignalType.HOLD

        # LONG signal: VAR crosses above OTT
        if self._crossed_above(df["Var"], df["OTT"]):
            return SignalType.LONG

        # SHORT signal: VAR crosses below OTT
        if self._crossed_below(df["Var"], df["OTT"]):
            return SignalType.SHORT

        return SignalType.HOLD

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        """Execute the OTT Strategy logic."""

        if df is None or len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        df = df.copy()
        df = self._calculate_indicators(df)

        signal = self._generate_signal(df)

        return StrategyRecommendation(signal=signal, timestamp=timestamp)

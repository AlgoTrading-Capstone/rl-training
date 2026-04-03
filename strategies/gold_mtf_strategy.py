"""
Gold MTF Dashboard Strategy

Converted from PineScript v5: "Gold MTF Dashboard FINAL NO ERROR"
Source: https://www.tradingview.com/script/6mFzFaov-Gold-MTF/

Multi-Timeframe (MTF) strategy for Gold (XAUUSD) combining:
  - EMA trend detection (9 EMA vs 21 EMA) across 3 timeframes: 1m, 3m, 5m
  - Dynamic Support/Resistance breakout via rolling lowest/highest
  - Volume confirmation filter (volume > SMA(volume,20) * 1.2)

Entry logic:
  - LONG  when price breaks above resistance with bullish trend alignment
  - SHORT when price breaks below support with bearish trend alignment

Exit logic (ATR-based SL/TP) was NOT converted.
Configure stop-loss and take-profit in the execution layer using:
  - SL = low  - ATR(14) * sl_multiplier  (default 1.0)
  - TP = SL distance * tp_multiplier      (default 2.0)
"""

from datetime import datetime

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType
from strategies.utils.resampling import resample_to_interval, resampled_merge


class GoldMtfStrategy(BaseStrategy):

    def __init__(self):
        super().__init__(
            name="GoldMtfStrategy",
            description=(
                "MTF Gold strategy: EMA trend + SR breakout + volume filter "
                "across 1m/3m/5m timeframes. "
                "Exit logic (ATR SL/TP) delegated to execution layer."
            ),
            timeframe="1m",
            lookback_hours=6,
        )

        # Indicator parameters (sourced from PineScript inputs)
        self.ema_fast_period = 9
        self.ema_slow_period = 21
        self.lookback = 20          # SR Lookback
        self.atr_period = 14
        self.vol_sma_period = 20
        self.vol_multiplier = 1.2   # Volume Multiplier

        # Higher-TF intervals in minutes (Pine: tf2="3", tf3="5")
        self.tf2_minutes = 3
        self.tf3_minutes = 5

        # CRITICAL RL GUARD: dynamic warmup — must cover the highest TF fully
        max_indicator_period = max(
            self.ema_slow_period,
            self.lookback,
            self.atr_period,
            self.vol_sma_period,
        )
        self.MIN_CANDLES_REQUIRED = 3 * max_indicator_period * self.tf3_minutes

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        # --- CRITICAL RL GUARD ---
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(SignalType.HOLD, timestamp)

        df = df.copy()

        # ------------------------------------------------------------------ #
        # TF1 (1m) — compute directly on base dataframe                       #
        # ------------------------------------------------------------------ #
        tf1_ema_fast = pd.Series(
            talib.EMA(df["close"].values, timeperiod=self.ema_fast_period),
            index=df.index,
        )
        tf1_ema_slow = pd.Series(
            talib.EMA(df["close"].values, timeperiod=self.ema_slow_period),
            index=df.index,
        )
        df["tf1_trend"] = np.where(
            tf1_ema_fast > tf1_ema_slow, 1,
            np.where(tf1_ema_fast < tf1_ema_slow, -1, 0),
        )
        df["tf1_resistance"] = df["high"].rolling(self.lookback).max()
        df["tf1_support"] = df["low"].rolling(self.lookback).min()
        df["tf1_vol_sma"] = df["volume"].rolling(self.vol_sma_period).mean()

        # [1] shift for TF1 — 1 base-bar ago (Pine: tf1Res[1], tf1Sup[1])
        df["tf1_resistance_prev"] = df["tf1_resistance"].shift(1)
        df["tf1_support_prev"] = df["tf1_support"].shift(1)

        # ------------------------------------------------------------------ #
        # TF2 (3m) — resample, compute indicators, merge back                 #
        # ------------------------------------------------------------------ #
        resampled_3 = resample_to_interval(df, f"{self.tf2_minutes}m")

        tf2_ema_fast = pd.Series(
            talib.EMA(resampled_3["close"].values, timeperiod=self.ema_fast_period),
            index=resampled_3.index,
        )
        tf2_ema_slow = pd.Series(
            talib.EMA(resampled_3["close"].values, timeperiod=self.ema_slow_period),
            index=resampled_3.index,
        )
        resampled_3["tf2_trend"] = np.where(
            tf2_ema_fast > tf2_ema_slow, 1,
            np.where(tf2_ema_fast < tf2_ema_slow, -1, 0),
        )
        resampled_3["tf2_resistance"] = resampled_3["high"].rolling(self.lookback).max()
        resampled_3["tf2_support"] = resampled_3["low"].rolling(self.lookback).min()
        resampled_3["tf2_vol_sma"] = resampled_3["volume"].rolling(self.vol_sma_period).mean()

        # [1] shift in TF2 terms — 1 three-minute-bar ago (Pine: tf2Res[1], tf2Sup[1])
        resampled_3["tf2_resistance_prev"] = resampled_3["tf2_resistance"].shift(1)
        resampled_3["tf2_support_prev"] = resampled_3["tf2_support"].shift(1)

        merged = resampled_merge(original=df, resampled=resampled_3, fill_na=True)

        # ------------------------------------------------------------------ #
        # TF3 (5m) — resample, compute indicators, merge back                 #
        # ------------------------------------------------------------------ #
        resampled_5 = resample_to_interval(df, f"{self.tf3_minutes}m")

        tf3_ema_fast = pd.Series(
            talib.EMA(resampled_5["close"].values, timeperiod=self.ema_fast_period),
            index=resampled_5.index,
        )
        tf3_ema_slow = pd.Series(
            talib.EMA(resampled_5["close"].values, timeperiod=self.ema_slow_period),
            index=resampled_5.index,
        )
        resampled_5["tf3_trend"] = np.where(
            tf3_ema_fast > tf3_ema_slow, 1,
            np.where(tf3_ema_fast < tf3_ema_slow, -1, 0),
        )
        resampled_5["tf3_resistance"] = resampled_5["high"].rolling(self.lookback).max()
        resampled_5["tf3_support"] = resampled_5["low"].rolling(self.lookback).min()
        resampled_5["tf3_vol_sma"] = resampled_5["volume"].rolling(self.vol_sma_period).mean()

        # [1] shift in TF3 terms — 1 five-minute-bar ago (Pine: tf3Res[1], tf3Sup[1])
        resampled_5["tf3_resistance_prev"] = resampled_5["tf3_resistance"].shift(1)
        resampled_5["tf3_support_prev"] = resampled_5["tf3_support"].shift(1)

        merged = resampled_merge(original=merged, resampled=resampled_5, fill_na=True)

        # ------------------------------------------------------------------ #
        # Signal extraction — evaluate on the last completed bar              #
        # ------------------------------------------------------------------ #
        last = merged.iloc[-1]
        close = last["close"]

        # TF1 scalars
        tf1_trend = last["tf1_trend"]
        tf1_res_prev = last["tf1_resistance_prev"]
        tf1_sup_prev = last["tf1_support_prev"]
        tf1_vol = last["volume"]
        tf1_vol_sma = last["tf1_vol_sma"]

        # TF2 scalars (prefixed by resampled_merge: resample_3_*)
        tf2_pfx = f"resample_{self.tf2_minutes}_"
        tf2_trend = last[f"{tf2_pfx}tf2_trend"]
        tf2_res_prev = last[f"{tf2_pfx}tf2_resistance_prev"]
        tf2_sup_prev = last[f"{tf2_pfx}tf2_support_prev"]
        tf2_vol = last[f"{tf2_pfx}volume"]
        tf2_vol_sma = last[f"{tf2_pfx}tf2_vol_sma"]

        # TF3 scalars (prefixed by resampled_merge: resample_5_*)
        tf3_pfx = f"resample_{self.tf3_minutes}_"
        tf3_trend = last[f"{tf3_pfx}tf3_trend"]
        tf3_res_prev = last[f"{tf3_pfx}tf3_resistance_prev"]
        tf3_sup_prev = last[f"{tf3_pfx}tf3_support_prev"]
        tf3_vol = last[f"{tf3_pfx}volume"]
        tf3_vol_sma = last[f"{tf3_pfx}tf3_vol_sma"]

        # NaN guard — indicators haven't converged yet
        if pd.isna(tf1_trend) or pd.isna(tf2_trend) or pd.isna(tf3_trend):
            return StrategyRecommendation(SignalType.HOLD, timestamp)

        # Volume filters per timeframe
        vol_ok_tf1 = pd.notna(tf1_vol_sma) and bool(
            tf1_vol > tf1_vol_sma * self.vol_multiplier
        )
        vol_ok_tf2 = pd.notna(tf2_vol_sma) and bool(
            tf2_vol > tf2_vol_sma * self.vol_multiplier
        )
        vol_ok_tf3 = pd.notna(tf3_vol_sma) and bool(
            tf3_vol > tf3_vol_sma * self.vol_multiplier
        )

        # ---- Entry conditions (Pine lines 55-62) ---- #
        buy_tf1 = (
            tf1_trend == 1
            and pd.notna(tf1_res_prev) and close > tf1_res_prev
            and tf2_trend == 1
            and vol_ok_tf1
        )
        sell_tf1 = (
            tf1_trend == -1
            and pd.notna(tf1_sup_prev) and close < tf1_sup_prev
            and tf2_trend == -1
            and vol_ok_tf1
        )

        buy_tf2 = (
            tf2_trend == 1
            and pd.notna(tf2_res_prev) and close > tf2_res_prev
            and tf3_trend == 1
            and vol_ok_tf2
        )
        sell_tf2 = (
            tf2_trend == -1
            and pd.notna(tf2_sup_prev) and close < tf2_sup_prev
            and tf3_trend == -1
            and vol_ok_tf2
        )

        buy_tf3 = (
            tf3_trend == 1
            and pd.notna(tf3_res_prev) and close > tf3_res_prev
            and vol_ok_tf3
        )
        sell_tf3 = (
            tf3_trend == -1
            and pd.notna(tf3_sup_prev) and close < tf3_sup_prev
            and vol_ok_tf3
        )

        if buy_tf1 or buy_tf2 or buy_tf3:
            return StrategyRecommendation(SignalType.LONG, timestamp)
        if sell_tf1 or sell_tf2 or sell_tf3:
            return StrategyRecommendation(SignalType.SHORT, timestamp)

        return StrategyRecommendation(SignalType.HOLD, timestamp)

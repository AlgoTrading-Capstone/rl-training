"""
MTF EMA Engulf + Retest - NQ 1m

Converted from Pine Script v5 strategy of the same name.

Logic overview:
  1. Multi-timeframe EMA cloud alignment (1m/5m/15m/30m/1H):
     - Bull cloud:  EMA(fast) > EMA(slow)
     - Bear cloud:  EMA(fast) < EMA(slow)
  2. An *engulfing* candle fires a watch state when >= min_align timeframes agree
     and price is above (bull) or below (bear) the 1m EMA-200 trend filter.
  3. The strategy enters on the first bar that *retests* the EMA cloud band
     (ema_bot <= price <= ema_top) without first invalidating the signal.
  4. Invalidation: close beyond signal_low (bull) or signal_high (bear).
  5. Expiry: retest window closes after retest_expiry bars with no valid retest.

External stop-loss / take-profit management is handled by the RL engine.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, SignalType, StrategyRecommendation
from utils.resampling import (
    compute_interval_minutes,
    resample_to_interval,
    resampled_merge,
)
from utils.timeframes import timeframe_to_minutes


class MtfEmaEngulfRetestNq1MStrategy(BaseStrategy):
    """MTF EMA Engulf + Retest strategy, originally designed for NQ 1-minute charts."""

    def __init__(
        self,
        ema_fast: int = 13,
        ema_slow: int = 48,
        ema_trend: int = 200,
        min_align: int = 4,
        retest_expiry: int = 8,
    ) -> None:
        super().__init__(
            name="MtfEmaEngulfRetestNq1MStrategy",
            description=(
                "Multi-timeframe EMA engulf + retest strategy for NQ 1m. "
                "Enters long/short on a retest of the EMA cloud after a confirmed "
                "engulfing candle with >= min_align timeframes aligned."
            ),
            timeframe="1m",
            lookback_hours=10,
        )
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.min_align = min_align
        self.retest_expiry = retest_expiry
        # Dynamic warmup: allow the slowest indicator (EMA-200) to fully converge.
        self.MIN_CANDLES_REQUIRED = 3 * max(self.ema_fast, self.ema_slow, self.ema_trend)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mtf_bull(
        self,
        df: pd.DataFrame,
        interval_str: str,
        base_interval_min: int,
        base_bull: pd.Series,
    ) -> pd.Series:
        """
        Return a boolean Series: True when the EMA fast > EMA slow on *interval_str*.

        Falls back to *base_bull* when the target interval is <= the base interval
        (upsampling is not supported and would introduce lookahead bias).
        """
        target_min = timeframe_to_minutes(interval_str)
        if target_min <= base_interval_min:
            return base_bull.copy()

        try:
            htf = resample_to_interval(df, interval_str)
            if len(htf) < self.ema_slow:
                return pd.Series(False, index=df.index)

            htf["ema_fast_htf"] = talib.EMA(htf["close"].values, timeperiod=self.ema_fast)
            htf["ema_slow_htf"] = talib.EMA(htf["close"].values, timeperiod=self.ema_slow)
            htf["bull"] = htf["ema_fast_htf"] > htf["ema_slow_htf"]

            bull_df = htf[["date", "bull"]].copy()
            merged = resampled_merge(original=df, resampled=bull_df, fill_na=True)

            col = f"resample_{target_min}_bull"
            if col not in merged.columns:
                return pd.Series(False, index=df.index)
            return merged[col].fillna(False).astype(bool)
        except Exception:
            return base_bull.copy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        df = df.copy()

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        # ---- Base EMAs -------------------------------------------------------
        e_fast = pd.Series(
            talib.EMA(close.values, timeperiod=self.ema_fast), index=df.index
        )
        e_slow = pd.Series(
            talib.EMA(close.values, timeperiod=self.ema_slow), index=df.index
        )
        e_trend = pd.Series(
            talib.EMA(close.values, timeperiod=self.ema_trend), index=df.index
        )

        ema_top = pd.Series(np.maximum(e_fast.values, e_slow.values), index=df.index)
        ema_bot = pd.Series(np.minimum(e_fast.values, e_slow.values), index=df.index)
        above_200 = close > e_trend

        # ---- MTF alignment ---------------------------------------------------
        base_bull = (e_fast > e_slow).fillna(False)

        base_interval_min: int
        if "date" in df.columns and len(df) >= 2:
            try:
                base_interval_min = compute_interval_minutes(df)
            except Exception:
                base_interval_min = 1
        else:
            base_interval_min = 1

        bull_1m = base_bull
        bull_5m = self._mtf_bull(df, "5m", base_interval_min, base_bull)
        bull_15m = self._mtf_bull(df, "15m", base_interval_min, base_bull)
        bull_30m = self._mtf_bull(df, "30m", base_interval_min, base_bull)
        bull_1h = self._mtf_bull(df, "1h", base_interval_min, base_bull)

        bull_count = (
            bull_1m.astype(int)
            + bull_5m.astype(int)
            + bull_15m.astype(int)
            + bull_30m.astype(int)
            + bull_1h.astype(int)
        )
        bear_count = (
            (~bull_1m).astype(int)
            + (~bull_5m).astype(int)
            + (~bull_15m).astype(int)
            + (~bull_30m).astype(int)
            + (~bull_1h).astype(int)
        )

        all_green = (bull_count >= self.min_align) & above_200.fillna(False)
        all_red = (bear_count >= self.min_align) & (~above_200.fillna(False))

        # ---- Engulf detection (raw — `not watching` guard applied in loop) ---
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_open = open_.shift(1)
        prev_close = close.shift(1)

        bull_engulf_raw = (
            all_green
            & (close > open_)
            & (
                (close > prev_high)
                | ((close > prev_open) & (open_ < prev_close) & (open_ < prev_open))
            )
        ).fillna(False)

        bear_engulf_raw = (
            all_red
            & (close < open_)
            & (
                (close < prev_low)
                | ((close < prev_open) & (open_ > prev_close) & (open_ > prev_open))
            )
        ).fillna(False)

        # ---- Iterative state machine ----------------------------------------
        # This mirrors the Pine `var`-based state machine exactly.
        # State transitions per bar:
        #   1. Evaluate retest / invalid / expired (watching must be True).
        #   2. Reset watching on any trigger.
        #   3. Check for new engulf (only when not watching).
        #   4. Increment bars_since if watching.
        n = len(df)
        signal_arr: list = [SignalType.HOLD] * n

        watching = False
        bull_mode = False
        signal_low: float = np.nan
        signal_high: float = np.nan
        bars_since: int = 0

        for i in range(self.MIN_CANDLES_REQUIRED, n):
            e_top_i = float(ema_top.iloc[i])
            e_bot_i = float(ema_bot.iloc[i])
            lo_i = float(low.iloc[i])
            hi_i = float(high.iloc[i])
            cl_i = float(close.iloc[i])

            # Engulf fires only when NOT watching (matches Pine `not watching and …`)
            bg = bool(bull_engulf_raw.iloc[i]) if not watching else False
            be = bool(bear_engulf_raw.iloc[i]) if not watching else False

            bull_entry_i = False
            bear_entry_i = False
            bull_invalid_i = False
            bear_invalid_i = False
            exprd = False

            if watching and not (np.isnan(e_top_i) or np.isnan(e_bot_i)):
                exprd = bars_since >= self.retest_expiry

                if bull_mode:
                    bull_ret = lo_i <= e_top_i and hi_i >= e_bot_i and cl_i > signal_low
                    bull_invalid_i = cl_i < signal_low
                    bull_entry_i = bull_ret and not bull_invalid_i
                else:
                    bear_ret = hi_i >= e_bot_i and lo_i <= e_top_i and cl_i < signal_high
                    bear_invalid_i = cl_i > signal_high
                    bear_entry_i = bear_ret and not bear_invalid_i

                if bull_entry_i or bear_entry_i or bull_invalid_i or bear_invalid_i or exprd:
                    watching = False
                    bull_mode = False
                    bars_since = 0

            if bull_entry_i:
                signal_arr[i] = SignalType.LONG
            elif bear_entry_i:
                signal_arr[i] = SignalType.SHORT

            # New engulf — `watching` may have just been reset above
            if not watching:
                if bg:
                    signal_low = lo_i
                    watching = True
                    bull_mode = True
                    bars_since = 0
                elif be:
                    signal_high = hi_i
                    watching = True
                    bull_mode = False
                    bars_since = 0

            if watching:
                bars_since += 1

        return StrategyRecommendation(signal=signal_arr[-1], timestamp=timestamp)

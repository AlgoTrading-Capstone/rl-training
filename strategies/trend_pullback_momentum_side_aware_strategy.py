"""
Trend Pullback Momentum Side-Aware

Pine Script source: input/Trend-Pullback-Momentum-Side-Aware.pine
Timeframe: 15m  |  Lookback: 1600 bars (400 h)

Strategy logic:
  A trend-following pullback strategy that combines:
  - Higher-timeframe (2h) EMA as a directional bias filter.
  - Base-timeframe (15m) trend EMA (21) for trend direction.
  - Pullback detection: price must have touched the EMA zone (within ATR buffer)
    within the last N bars before entry.
  - Reclaim: price reclaims the EMA from below (long) or above (short),
    with close breaking the previous bar's high/low.
  - Momentum confirmation: RSI direction and bar close direction aligned with bias.
  - Optional volume expansion and RSI threshold filters.

  Long  — bull HTF bias AND trend EMA bullish AND pullback recently touched zone
           AND close reclaims EMA AND RSI > 50 AND bullish bar AND RSI rising.
  Short — bear HTF bias AND trend EMA bearish AND pullback recently touched zone
           AND close below EMA AND RSI < 50 AND bearish bar AND RSI falling.
  Hold  — none of the above.

Stop-loss, take-profit, partial exits, and position sizing are NOT implemented
here; they are managed by the external execution engine.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType
from strategies.utils.resampling import resample_to_interval, resampled_merge


class TrendPullbackMomentumSideAwareStrategy(BaseStrategy):
    """
    Trend Pullback Momentum Side-Aware strategy.

    Uses a 2h HTF EMA bias filter combined with a 15m trend EMA pullback system,
    RSI momentum confirmation, and optional volume expansion. Designed for use
    in the RL execution engine with ATR-based dynamic warmup.
    """

    def __init__(
        self,
        # HTF bias
        use_htf_bias: bool = True,
        bias_ema_len: int = 200,
        trade_side: str = "Both",  # "Both", "Long only", "Short only"
        # Trend + pullback
        trend_ema_len: int = 21,
        require_pullback: bool = True,
        # Long settings
        long_pullback_atr_buffer: float = 0.35,
        long_reclaim_bars: int = 3,
        long_rsi_min: float = 50.0,
        # Short settings
        short_pullback_atr_buffer: float = 0.35,
        short_reclaim_bars: int = 3,
        short_rsi_max: float = 50.0,
        # Momentum confirmation
        rsi_len: int = 14,
        require_rsi_reclaim: bool = True,
        use_volume_expansion: bool = False,
        volume_len: int = 20,
        # Risk / ATR
        atr_len: int = 14,
        swing_lookback: int = 10,
    ):
        super().__init__(
            name="TrendPullbackMomentumSideAware",
            description=(
                "Trend Pullback Momentum Side-Aware: uses HTF (2h) EMA bias filter, "
                "trend EMA (21) pullback entries with ATR-zone detection and reclaim, "
                "RSI momentum confirmation, and dynamic RL warmup."
            ),
            timeframe="15m",
            lookback_hours=400,
        )

        # Store all parameters
        self.use_htf_bias = use_htf_bias
        self.bias_ema_len = bias_ema_len
        self.trade_side = trade_side
        self.trend_ema_len = trend_ema_len
        self.require_pullback = require_pullback
        self.long_pullback_atr_buffer = long_pullback_atr_buffer
        self.long_reclaim_bars = long_reclaim_bars
        self.long_rsi_min = long_rsi_min
        self.short_pullback_atr_buffer = short_pullback_atr_buffer
        self.short_reclaim_bars = short_reclaim_bars
        self.short_rsi_max = short_rsi_max
        self.rsi_len = rsi_len
        self.require_rsi_reclaim = require_rsi_reclaim
        self.use_volume_expansion = use_volume_expansion
        self.volume_len = volume_len
        self.atr_len = atr_len
        self.swing_lookback = swing_lookback

        # CRITICAL RL GUARD: dynamic warmup from the largest indicator lookback.
        # bias_ema_len (200) dominates; multiply by 3 to ensure full convergence.
        self.MIN_CANDLES_REQUIRED = 3 * max(
            self.bias_ema_len,
            self.trend_ema_len,
            self.rsi_len,
            self.atr_len,
            self.volume_len,
            self.swing_lookback,
        )

    @staticmethod
    def _bars_since(condition: pd.Series) -> pd.Series:
        """
        Vectorized equivalent of Pine's ta.barssince(condition).

        Returns the number of bars elapsed since the condition was last True.
        Returns NaN for rows before the condition has ever fired.
        """
        idx = pd.Series(np.arange(len(condition), dtype=float), index=condition.index)
        last_true_idx = idx.where(condition).ffill()
        return idx - last_true_idx

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        # --- CRITICAL RL GUARD ---
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(SignalType.HOLD, timestamp)

        df = df.copy()

        # ----------------------------------------------------------------
        # HTF BIAS — 2h EMA (barmerge.lookahead_off via resampled_merge shift)
        # biasTf = "120" → "2h"; column prefix: resample_120_
        # ----------------------------------------------------------------
        try:
            resampled_df = resample_to_interval(df, "2h")
            resampled_df["ema_bias"] = talib.EMA(
                resampled_df["close"].values, timeperiod=self.bias_ema_len
            )
            merged_df = resampled_merge(original=df, resampled=resampled_df, fill_na=True)
            htf_close = pd.Series(merged_df["resample_120_close"].values, index=df.index)
            htf_ema = pd.Series(merged_df["resample_120_ema_bias"].values, index=df.index)
        except Exception:
            # Fallback: HTF bias collapses to no filter
            htf_close = df["close"].copy()
            htf_ema = df["close"].copy()

        bull_bias = htf_close > htf_ema
        bear_bias = htf_close < htf_ema

        if self.use_htf_bias:
            long_bias_ok = bull_bias
            short_bias_ok = bear_bias
        else:
            long_bias_ok = pd.Series(True, index=df.index)
            short_bias_ok = pd.Series(True, index=df.index)

        # ----------------------------------------------------------------
        # BASE INDICATORS
        # ----------------------------------------------------------------
        trend_ema = pd.Series(
            talib.EMA(df["close"].values, timeperiod=self.trend_ema_len),
            index=df.index,
        )
        atr = pd.Series(
            talib.ATR(
                df["high"].values,
                df["low"].values,
                df["close"].values,
                timeperiod=self.atr_len,
            ),
            index=df.index,
        )
        rsi = pd.Series(
            talib.RSI(df["close"].values, timeperiod=self.rsi_len),
            index=df.index,
        )

        # ----------------------------------------------------------------
        # VOLUME FILTER (optional; default off)
        # ----------------------------------------------------------------
        if self.use_volume_expansion:
            vol_ma = pd.Series(
                talib.SMA(df["volume"].values, timeperiod=self.volume_len),
                index=df.index,
            )
            vol_ok = df["volume"] > vol_ma
        else:
            vol_ok = pd.Series(True, index=df.index)

        # ----------------------------------------------------------------
        # RSI FILTER (require RSI above/below midline)
        # ----------------------------------------------------------------
        if self.require_rsi_reclaim:
            rsi_long_ok = rsi > self.long_rsi_min
            rsi_short_ok = rsi < self.short_rsi_max
        else:
            rsi_long_ok = pd.Series(True, index=df.index)
            rsi_short_ok = pd.Series(True, index=df.index)

        # ----------------------------------------------------------------
        # TREND DIRECTION
        # trendBull = close > trendEma and trendEma > trendEma[1]
        # trendBear = close < trendEma and trendEma < trendEma[1]
        # ----------------------------------------------------------------
        trend_bull = (df["close"] > trend_ema) & (trend_ema > trend_ema.shift(1))
        trend_bear = (df["close"] < trend_ema) & (trend_ema < trend_ema.shift(1))

        # ----------------------------------------------------------------
        # PULLBACK DETECTION
        # longPullbackTouched  = low  <= trendEma + atr * longPullbackAtrBuffer
        # shortPullbackTouched = high >= trendEma - atr * shortPullbackAtrBuffer
        # longPulledBackRecently  = ta.barssince(longPullbackTouched)  <= longReclaimBars
        # shortPulledBackRecently = ta.barssince(shortPullbackTouched) <= shortReclaimBars
        # ----------------------------------------------------------------
        long_pullback_touched = df["low"] <= (trend_ema + atr * self.long_pullback_atr_buffer)
        short_pullback_touched = df["high"] >= (trend_ema - atr * self.short_pullback_atr_buffer)

        long_bars_since = self._bars_since(long_pullback_touched)
        short_bars_since = self._bars_since(short_pullback_touched)

        if self.require_pullback:
            long_pullback_ok = long_bars_since <= self.long_reclaim_bars
            short_pullback_ok = short_bars_since <= self.short_reclaim_bars
        else:
            long_pullback_ok = pd.Series(True, index=df.index)
            short_pullback_ok = pd.Series(True, index=df.index)

        # ----------------------------------------------------------------
        # RECLAIM (EMA break + prior bar high/low)
        # longReclaim  = close > trendEma and close > high[1]
        # shortReclaim = close < trendEma and close < low[1]
        # ----------------------------------------------------------------
        long_reclaim = (df["close"] > trend_ema) & (df["close"] > df["high"].shift(1))
        short_reclaim = (df["close"] < trend_ema) & (df["close"] < df["low"].shift(1))

        # ----------------------------------------------------------------
        # MOMENTUM SHIFT (bar direction + RSI direction)
        # longMomentumShift  = close > open and rsi > rsi[1]
        # shortMomentumShift = close < open and rsi < rsi[1]
        # ----------------------------------------------------------------
        long_momentum_shift = (df["close"] > df["open"]) & (rsi > rsi.shift(1))
        short_momentum_shift = (df["close"] < df["open"]) & (rsi < rsi.shift(1))

        # ----------------------------------------------------------------
        # FULL SETUP CONDITIONS
        # Session filter omitted: useSessionFilter defaults to False → inSession = True
        # ----------------------------------------------------------------
        long_setup = (
            long_bias_ok
            & trend_bull
            & vol_ok
            & rsi_long_ok
            & long_momentum_shift
            & long_reclaim
            & long_pullback_ok
        )
        short_setup = (
            short_bias_ok
            & trend_bear
            & vol_ok
            & rsi_short_ok
            & short_momentum_shift
            & short_reclaim
            & short_pullback_ok
        )

        # Apply trade_side filter
        if self.trade_side == "Long only":
            short_setup = pd.Series(False, index=df.index)
        elif self.trade_side == "Short only":
            long_setup = pd.Series(False, index=df.index)

        # ----------------------------------------------------------------
        # SIGNAL — evaluate at the last (most recent closed) bar
        # ----------------------------------------------------------------
        def _safe_bool(series: pd.Series) -> bool:
            val = series.iloc[-1]
            return bool(val) if not pd.isna(val) else False

        if _safe_bool(long_setup):
            return StrategyRecommendation(SignalType.LONG, timestamp)
        if _safe_bool(short_setup):
            return StrategyRecommendation(SignalType.SHORT, timestamp)
        return StrategyRecommendation(SignalType.HOLD, timestamp)

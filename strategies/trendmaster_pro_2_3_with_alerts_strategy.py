"""TrendMaster Pro 2.3 with Alerts Strategy

Pine Script source: input/TrendMaster-Pro-2-3-with-Alerts.pine
Timeframe: 15m  |  Lookback: 50 bars

Entry logic:
  Long  — Short MA (9) crosses over Long MA (21) AND all filters pass:
            - Bollinger Bands volatility filter: BB width > ATR * bb_multiplier
            - BB trend filter: close > BB basis (uptrend confirmation)
            - RSI > 55 (bullish momentum)
            - MACD line > signal line
            - Stochastic: k > 20 (not oversold) AND k > d (rising)
            - ADX > 25 (strong trend)

  Short — Short MA (9) crosses under Long MA (21) AND all filters pass:
            - Bollinger Bands volatility filter: BB width > ATR * bb_multiplier
            - BB trend filter: close < BB basis (downtrend confirmation)
            - RSI < 45 (bearish momentum)
            - MACD line < signal line
            - Stochastic: k < 80 (not overbought) AND k < d (falling)
            - ADX > 25 (strong trend)

Session management: The Pine Script uses session windows (Asia/London/NY AM/NY PM).
This implementation treats all bars as in-session (simplified), since the Python
engine does not have a persistent session scheduler.

Position management dropped:
- strategy.position_size, strategy.closedtrades, canGoLong()/canGoShort() state
  depend on execution-layer position tracking unavailable here. Removed entirely.
- Per-session trade counters (max trades per session) require execution state.
  Removed entirely.
- TP/SL bracket logic (_sl(), _tp(), strategy.exit()) is execution-layer concern.
  StrategyRecommendation only carries direction — not SL/TP prices.
- maCrossValue intersection calculation: visual-only feature, not used for entry.
- Band Power (maHigh/maLow) bands: indicator-only, not used for entry signal.
- Support & Resistance (pivot_point_levels): display-only, not used for entry.

Key parameters (defaults match Pine Script defaults):
    ma_type          : "SMA"  — MA type: "EMA", "SMA", or "SMMA"
    short_ma_length  : 9      — short MA period
    long_ma_length   : 21     — long MA period
    bb_length        : 20     — Bollinger Bands period
    bb_multiplier    : 2.0    — BB standard deviation multiplier
    rsi_length       : 14     — RSI period
    rsi_long_thresh  : 55     — RSI threshold for longs
    rsi_short_thresh : 45     — RSI threshold for shorts
    macd_fast        : 12     — MACD fast EMA period
    macd_slow        : 26     — MACD slow EMA period
    macd_signal      : 9      — MACD signal smoothing
    stoch_length     : 14     — Stochastic %K raw period
    stoch_smoothing  : 3      — Stochastic %K and %D smoothing
    stoch_overbought : 80     — Stochastic overbought threshold
    stoch_oversold   : 20     — Stochastic oversold threshold
    adx_length       : 14     — ADX/DMI period
    adx_threshold    : 25     — ADX minimum for trend strength
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import talib
from talib import MA_Type

from strategies.base_strategy import BaseStrategy, SignalType, StrategyRecommendation


class TrendmasterPro23WithAlertsStrategy(BaseStrategy):
    """MA crossover strategy with multi-filter confirmation (BB, RSI, MACD, Stoch, ADX).

    Translates TrendMaster Pro 2.3 with Alerts from PineScript v6.
    All signal logic is entry-only; exit management must be handled externally.
    """

    def __init__(
        self,
        ma_type: str = "SMA",
        short_ma_length: int = 9,
        long_ma_length: int = 21,
        bb_length: int = 20,
        bb_multiplier: float = 2.0,
        rsi_length: int = 14,
        rsi_long_thresh: float = 55.0,
        rsi_short_thresh: float = 45.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        stoch_length: int = 14,
        stoch_smoothing: int = 3,
        stoch_overbought: float = 80.0,
        stoch_oversold: float = 20.0,
        adx_length: int = 14,
        adx_threshold: float = 25.0,
    ):
        super().__init__(
            name="TrendMaster Pro 2.3 with Alerts",
            description=(
                "MA crossover (9/21) with Bollinger Bands volatility filter, BB trend "
                "filter, RSI, MACD, Stochastic, and ADX confirmation filters."
            ),
            timeframe="15m",
            lookback_hours=50,  # 50 bars × 15m = ~12.5 hours
        )
        self.ma_type = ma_type
        self.short_ma_length = short_ma_length
        self.long_ma_length = long_ma_length
        self.bb_length = bb_length
        self.bb_multiplier = bb_multiplier
        self.rsi_length = rsi_length
        self.rsi_long_thresh = rsi_long_thresh
        self.rsi_short_thresh = rsi_short_thresh
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stoch_length = stoch_length
        self.stoch_smoothing = stoch_smoothing
        self.stoch_overbought = stoch_overbought
        self.stoch_oversold = stoch_oversold
        self.adx_length = adx_length
        self.adx_threshold = adx_threshold

        # Dynamic warmup: driven by the slowest indicator.
        # MACD slow EMA (26) + signal smoothing (9) is usually the deepest,
        # but ADX (14) and Stochastic (14+3) also need warmup.
        # We use 3× the maximum indicator period to ensure full convergence.
        self.MIN_CANDLES_REQUIRED = 3 * max(
            self.long_ma_length,
            self.bb_length,
            self.rsi_length,
            self.macd_slow + self.macd_signal,
            self.stoch_length + self.stoch_smoothing,
            self.adx_length,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _smma(series: pd.Series, length: int) -> pd.Series:
        """Smoothed Moving Average (SMMA) — equivalent to Pine's smma() function.

        Pine Script:
            smma = na(smma[1]) ? ta.sma(src, len) : (smma[1] * (len - 1) + src) / len

        This is equivalent to an EMA with alpha = 1/length (RMA in TradingView terms).
        """
        return series.ewm(alpha=1.0 / length, adjust=False).mean()

    def _compute_ma(self, series: pd.Series, length: int) -> pd.Series:
        """Compute MA of the selected type (EMA, SMA, or SMMA)."""
        if self.ma_type == "EMA":
            return series.ewm(span=length, adjust=False).mean()
        elif self.ma_type == "SMMA":
            return self._smma(series, length)
        else:  # default: SMA
            return series.rolling(window=length).mean()

    # ------------------------------------------------------------------
    # Public run() method
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        """Evaluate the TrendMaster Pro strategy for the current bar.

        Parameters
        ----------
        df        : OHLCV DataFrame (columns: open, high, low, close, volume, date).
                    All rolling/shift operations are strictly backward-looking.
        timestamp : UTC datetime for the evaluation bar.

        Returns
        -------
        StrategyRecommendation with LONG, SHORT, or HOLD signal.
        """
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # Reset index on a local working copy to ensure positional indexing is safe.
        # Indicator results are written back to the original df by column name.
        work = df.reset_index(drop=True)
        close = work["close"]
        high = work["high"]
        low = work["low"]

        # ---- Moving Averages ------------------------------------------------
        short_ma = self._compute_ma(close, self.short_ma_length)
        long_ma = self._compute_ma(close, self.long_ma_length)

        # MA crossover on the last two bars (no lookahead — uses shift(1))
        # buyCondition  = ta.crossover(shortMA, longMA)  → prev bar short < long, current bar short > long
        # sellCondition = ta.crossunder(shortMA, longMA) → prev bar short > long, current bar short < long
        short_ma_prev = short_ma.shift(1)
        long_ma_prev = long_ma.shift(1)
        buy_cross = (short_ma_prev < long_ma_prev) & (short_ma > long_ma)
        sell_cross = (short_ma_prev > long_ma_prev) & (short_ma < long_ma)

        # ---- Bollinger Bands ------------------------------------------------
        bb_basis = close.rolling(window=self.bb_length).mean()
        bb_std = close.rolling(window=self.bb_length).std(ddof=1)
        bb_upper = bb_basis + self.bb_multiplier * bb_std
        bb_lower = bb_basis - self.bb_multiplier * bb_std

        # ATR for volatility filter (uses bb_length period, matching Pine Script)
        atr_bb = pd.Series(
            talib.ATR(
                high.values.astype(float),
                low.values.astype(float),
                close.values.astype(float),
                timeperiod=self.bb_length,
            ),
            index=work.index,
        )

        # Volatility filter: bands must be wider than ATR * multiplier
        bb_width = bb_upper - bb_lower
        volatility_filter = bb_width > (atr_bb * self.bb_multiplier)

        # BB trend filter (price relative to midline)
        bb_trend_long = close > bb_basis
        bb_trend_short = close < bb_basis

        # ---- RSI ------------------------------------------------------------
        rsi = pd.Series(
            talib.RSI(close.values.astype(float), timeperiod=self.rsi_length),
            index=work.index,
        )
        rsi_filter_long = rsi > self.rsi_long_thresh
        rsi_filter_short = rsi < self.rsi_short_thresh

        # ---- MACD -----------------------------------------------------------
        macd_line_arr, signal_line_arr, _ = talib.MACD(
            close.values.astype(float),
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )
        macd_line = pd.Series(macd_line_arr, index=work.index)
        signal_line = pd.Series(signal_line_arr, index=work.index)
        macd_filter_long = macd_line > signal_line
        macd_filter_short = macd_line < signal_line

        # ---- Stochastic -----------------------------------------------------
        # Pine: k = ta.sma(ta.stoch(close, high, low, stochLength), stochSmoothing)
        #       d = ta.sma(k, stochSmoothing)
        # talib.STOCH computes the fast %K first, then smooths to %K and %D.
        # fastk_period = stoch_length (raw stochastic window)
        # slowk_period = stoch_smoothing (smoothing for %K)
        # slowd_period = stoch_smoothing (smoothing for %D from %K)
        stoch_k_arr, stoch_d_arr = talib.STOCH(
            high.values.astype(float),
            low.values.astype(float),
            close.values.astype(float),
            fastk_period=self.stoch_length,
            slowk_period=self.stoch_smoothing,
            slowk_matype=MA_Type.SMA,
            slowd_period=self.stoch_smoothing,
            slowd_matype=MA_Type.SMA,
        )
        stoch_k = pd.Series(stoch_k_arr, index=work.index)
        stoch_d = pd.Series(stoch_d_arr, index=work.index)

        # Stochastic filter for long: k > oversold AND k > d (not overbought oversold zone, rising momentum)
        # Pine: stochFilterLong = not useStochastic or k > stochOversold and k > d
        stoch_filter_long = (stoch_k > self.stoch_oversold) & (stoch_k > stoch_d)
        # Stochastic filter for short: k < overbought AND k < d
        # Pine: stochFilterShort = not useStochastic or k < stochOverbought and k < d
        stoch_filter_short = (stoch_k < self.stoch_overbought) & (stoch_k < stoch_d)

        # ---- ADX ------------------------------------------------------------
        # Pine: [plusDI, minusDI, adx] = ta.dmi(adxLength, adxLength)
        # talib does not have a combined DMI function; use talib.ADX directly.
        adx_arr = talib.ADX(
            high.values.astype(float),
            low.values.astype(float),
            close.values.astype(float),
            timeperiod=self.adx_length,
        )
        adx = pd.Series(adx_arr, index=work.index)
        adx_filter = adx > self.adx_threshold

        # ---- Combined entry conditions --------------------------------------
        # buyConditionFiltered  = buyCondition and volatilityFilter and bbTrendFilterLong
        #                         and rsiFilterLong and macdFilterLong
        #                         and stochFilterLong and adxFilter
        # (trendFilterLong/Short disabled by default in Pine — omitted here)
        long_signal = (
            buy_cross
            & volatility_filter
            & bb_trend_long
            & rsi_filter_long
            & macd_filter_long
            & stoch_filter_long
            & adx_filter
        )
        short_signal = (
            sell_cross
            & volatility_filter
            & bb_trend_short
            & rsi_filter_short
            & macd_filter_short
            & stoch_filter_short
            & adx_filter
        )

        # ---- Evaluate last bar only -----------------------------------------
        # Guard: ensure all indicators have valid (non-NaN) values at last bar
        last_idx = len(work) - 1

        if (
            pd.isna(short_ma.iloc[last_idx])
            or pd.isna(long_ma.iloc[last_idx])
            or pd.isna(bb_basis.iloc[last_idx])
            or pd.isna(rsi.iloc[last_idx])
            or pd.isna(macd_line.iloc[last_idx])
            or pd.isna(stoch_k.iloc[last_idx])
            or pd.isna(adx.iloc[last_idx])
        ):
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # Write indicator columns back to the caller's df (aligned by position).
        # This allows tests and callers to inspect computed indicators after run().
        # We reset the index of the caller's df temporarily to align with work's index.
        orig_index = df.index
        df.reset_index(drop=True, inplace=True)
        df["short_ma"] = short_ma.values
        df["long_ma"] = long_ma.values
        df["bb_basis"] = bb_basis.values
        df["bb_upper"] = bb_upper.values
        df["bb_lower"] = bb_lower.values
        df["rsi"] = rsi.values
        df["macd_line"] = macd_line.values
        df["macd_signal"] = signal_line.values
        df["stoch_k"] = stoch_k.values
        df["stoch_d"] = stoch_d.values
        df["adx"] = adx.values
        df.index = orig_index

        # Emit signal for the last bar
        if long_signal.iloc[last_idx]:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        if short_signal.iloc[last_idx]:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)

        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

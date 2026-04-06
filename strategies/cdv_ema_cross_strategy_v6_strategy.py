"""
CDV EMA Cross Strategy v6

Converted from PineScript v6 strategy "CDV EMA Cross Strategy v6".

Strategy Overview:
    This strategy uses Cumulative Delta Volume (CDV) to construct synthetic candles,
    optionally applies Heikin Ashi smoothing to those CDV candles, then fires entries
    based on EMA crossovers on the CDV (or CDV-HA) close series.

    Signal logic:
        - LONG  when EMA1 crosses over  EMA2 (on CDV / CDV-HA close)
        - SHORT when EMA1 crosses under EMA2 (on CDV / CDV-HA close)
        - HOLD  otherwise

Exit logic note:
    The original PineScript strategy has no explicit exit management — it relies solely
    on reverse-entry signals (a new opposing crossover flips the position). The RL
    execution layer handles stop-loss and take-profit externally.

CDV calculation:
    For each bar the script allocates volume between buyers (deltaup) and sellers
    (deltadown) using a wick/body ratio heuristic (_rate function), then computes the
    running cumulative sum (cumdelta). Synthetic OHLC candles are built from consecutive
    cumdelta values, and an optional Heikin Ashi layer is applied on top.

    haopen is stateful (recursive: haopen[i] depends on haopen[i-1]) and is therefore
    computed with an explicit Python for-loop rather than a vectorized operation.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, SignalType, StrategyRecommendation


class CdvEmaCrossStrategyV6(BaseStrategy):
    """CDV EMA Cross Strategy v6 — EMA crossover on Cumulative Delta Volume candles."""

    def __init__(
        self,
        ema1_len: int = 9,
        ema2_len: int = 21,
        use_heikin_ashi: bool = True,
    ) -> None:
        super().__init__(
            name="CDV EMA Cross Strategy v6",
            description=(
                "EMA crossover strategy using Cumulative Delta Volume (CDV) synthetic "
                "candles with optional Heikin Ashi smoothing. "
                "LONG on EMA1 crossover EMA2; SHORT on EMA1 crossunder EMA2."
            ),
            timeframe="1h",
            lookback_hours=168,  # 7 days of 1-hour bars
        )

        self.ema1_len: int = ema1_len
        self.ema2_len: int = ema2_len
        self.use_heikin_ashi: bool = use_heikin_ashi

        # Dynamic RL warmup — 3x the longest indicator period for EMA convergence
        self.MIN_CANDLES_REQUIRED: int = 3 * max(self.ema1_len, self.ema2_len)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cdv(df: pd.DataFrame) -> pd.Series:
        """Compute the Cumulative Delta Volume series from OHLCV data.

        Replicates Pine's _rate() function and cumdelta = ta.cum(delta).

        _rate(cond):
            tw   = high - max(open, close)          # upper wick
            bw   = min(open, close) - low           # lower wick
            body = abs(close - open)
            ret  = 0.5 * (tw + bw + (cond ? 2*body : 0)) / (tw + bw + body)
            ret  = nz(ret) == 0 ? 0.5 : ret         # default 0.5 when denom=0
        """
        tw = df["high"] - np.maximum(df["open"], df["close"])
        bw = np.minimum(df["open"], df["close"]) - df["low"]
        body = np.abs(df["close"] - df["open"])
        denom = tw + bw + body

        # Bullish bar (_rate with cond = open <= close)
        rate_up = 0.5 * (tw + bw + 2 * body) / denom
        rate_up = rate_up.fillna(0)
        rate_up = np.where(rate_up == 0, 0.5, rate_up)

        # Bearish bar (_rate with cond = open > close)
        rate_down = 0.5 * (tw + bw) / denom
        rate_down = pd.Series(rate_down, index=df.index).fillna(0)
        rate_down = np.where(rate_down == 0, 0.5, rate_down)

        deltaup = df["volume"] * rate_up
        deltadown = df["volume"] * rate_down

        delta = np.where(df["close"] >= df["open"], deltaup, -deltadown)
        delta = pd.Series(delta, index=df.index)

        cumdelta = delta.cumsum()
        return cumdelta

    @staticmethod
    def _build_cdv_candles(cumdelta: pd.Series) -> pd.DataFrame:
        """Build synthetic OHLC candles from cumdelta.

        Pine:
            o = cumdelta[1]                        -> previous bar's cumdelta
            h = math.max(cumdelta, cumdelta[1])
            l = math.min(cumdelta, cumdelta[1])
            c = cumdelta
        """
        prev = cumdelta.shift(1)
        cdv_open = prev
        cdv_close = cumdelta
        cdv_high = np.maximum(cumdelta, prev)
        cdv_low = np.minimum(cumdelta, prev)

        return pd.DataFrame(
            {
                "cdv_open": cdv_open,
                "cdv_high": cdv_high,
                "cdv_low": cdv_low,
                "cdv_close": cdv_close,
            },
            index=cumdelta.index,
        )

    @staticmethod
    def _apply_heikin_ashi(cdv: pd.DataFrame) -> pd.DataFrame:
        """Apply Heikin Ashi transformation to CDV candles.

        haclose = (o + h + l + c) / 4
        haopen  = (haopen[1] + haclose[1]) / 2   [recursive — computed via loop]
                  initial seed: (o + c) / 2       [when haopen[1] is na]
        hahigh  = max(h, haopen, haclose)
        halow   = min(l, haopen, haclose)

        The haopen state variable is recursive (each bar depends on the prior bar's
        haopen), so it is computed with an explicit for-loop to avoid lookahead bias.
        """
        o = cdv["cdv_open"].values
        h = cdv["cdv_high"].values
        low = cdv["cdv_low"].values
        c = cdv["cdv_close"].values
        n = len(cdv)

        haclose = (o + h + low + c) / 4.0

        haopen = np.empty(n, dtype=np.float64)
        haopen[:] = np.nan

        for i in range(n):
            if i == 0 or np.isnan(haopen[i - 1]):
                # Initial seed: (o + c) / 2
                haopen[i] = (o[i] + c[i]) / 2.0
            else:
                haopen[i] = (haopen[i - 1] + haclose[i - 1]) / 2.0

        hahigh = np.maximum(h, np.maximum(haopen, haclose))
        halow = np.minimum(low, np.minimum(haopen, haclose))

        return pd.DataFrame(
            {
                "ha_open": haopen,
                "ha_high": hahigh,
                "ha_low": halow,
                "ha_close": haclose,
            },
            index=cdv.index,
        )

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        """Execute CDV EMA Cross strategy on the provided OHLCV DataFrame."""

        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # --- Step 1: Compute Cumulative Delta Volume ---
        cumdelta = self._compute_cdv(df)

        # --- Step 2: Build CDV synthetic candles ---
        cdv_candles = self._build_cdv_candles(cumdelta)

        # --- Step 3: Optionally apply Heikin Ashi ---
        if self.use_heikin_ashi:
            ha_candles = self._apply_heikin_ashi(cdv_candles)
            close_series = ha_candles["ha_close"]
        else:
            close_series = cdv_candles["cdv_close"]

        # --- Step 4: Compute EMAs ---
        ema1 = pd.Series(
            talib.EMA(close_series.values, timeperiod=self.ema1_len),
            index=df.index,
        )
        ema2 = pd.Series(
            talib.EMA(close_series.values, timeperiod=self.ema2_len),
            index=df.index,
        )

        # Persist indicator columns for inspection / test assertions
        df["cdv_close"] = cdv_candles["cdv_close"].values
        df["ema1"] = ema1.values
        df["ema2"] = ema2.values

        # --- Step 5: Detect crossover / crossunder (vectorized) ---
        # ta.crossover(a, b)  → (a > b) & (a.shift(1) <= b.shift(1))
        # ta.crossunder(a, b) → (a < b) & (a.shift(1) >= b.shift(1))
        long_cross = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        short_cross = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))

        # --- Step 6: Evaluate signal on the last bar ---
        if long_cross.iloc[-1]:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        elif short_cross.iloc[-1]:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)

        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

"""
Quant Sentiment & Spread Master — Converted from Pine Script (PG-QSD-for-Nifty-Future).
Multi-factor quantitative sentiment strategy combining price-action WSI score,
ATR-based volatility scaling, and spread conviction confirmation.
Optimised for Nifty Futures on the 15m timeframe.
"""
import math as _math
from datetime import datetime

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType


class PgQsdForNiftyFutureStrategy(BaseStrategy):
    """
    Quant Sentiment & Spread Master.

    Aggregates price-action proxies, rolling volume delta, and OI-buildup signals
    into a composite WSI sentiment score.  An ATR-ratio adaptive multiplier
    amplifies conviction during high-momentum breakouts and dampens it during
    range-bound chop.  Entries are gated by a spread conviction buffer that
    confirms price efficiency before trading.

    Signal hierarchy
    ----------------
    LONG  : WSI state is BULLISH or RISING  AND  real_spread_ratio > spread_conv_buff
    SHORT : WSI state is BEARISH or DECLINING  AND  real_spread_ratio < -spread_conv_buff
    HOLD  : otherwise

    Exit logic (not converted)
    --------------------------
    The original Pine Script used ``strategy.exit("SL", stop=...)`` with a 0.4 % hard
    stop-loss calculated from the average entry price.  Stop-loss and take-profit
    management are handled externally by the RL execution layer and must be configured
    there, not in this class.

    Position-size gating (``strategy.position_size <= 0`` / ``>= 0``) was also removed;
    the RL engine manages position state externally.
    """

    def __init__(self):
        super().__init__(
            name="PgQsdForNiftyFutureStrategy",
            description=(
                "Quant Sentiment & Spread Master: aggregates price-action PCR proxies, "
                "volume delta, and OI-buildup into a WSI sentiment score, applies ATR-based "
                "volatility scaling and spread conviction filtering for Nifty Futures entries."
            ),
            timeframe="15m",
            lookback_hours=16,
        )
        # ── Volatility Engine ──────────────────────────────────────────────
        self.atr_fast_len = 5
        self.atr_slow_len = 21
        self.vol_len = 21
        self.vol_mult = 1.1
        self.ema_len_v = 5

        # ── WSI Signal ────────────────────────────────────────────────────
        self.in_step = 100.0
        self.in_off = 300.0
        self.sig_period = 7       # WSI smoothing length (HMA)
        self.v_sens = 1.5         # ATR-ratio exponent for v_adapt
        self.rev_thresh = 5.0     # conviction buffer (%); becomes 10.0 raw units

        # ── Sentiment Weights ────────────────────────────────────────────
        self.w_opt_p = 0.12
        self.w_opt_tv = 0.18
        self.w_pcr_d = 0.10
        self.w_pcr_h = 0.125
        self.w_build_d = 0.12
        self.w_build_h = 0.18
        self.w_vol_score = 0.075
        self.w_spread_pcr = 0.10
        self.spread_ma_len = 20
        self.spread_conv_buff = 0.1

        # ── RL warmup guard ──────────────────────────────────────────────
        self.MIN_CANDLES_REQUIRED = 3 * max(
            self.atr_slow_len,   # 21
            self.vol_len,        # 21
            self.spread_ma_len,  # 20
            self.sig_period,     # 7
        )  # → 63

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _f_sym(c: pd.Series, p: pd.Series) -> np.ndarray:
        """Pine f_sym: (c - p) / (c + p), 0 when denominator is 0."""
        total = c + p
        return np.where(total != 0, (c - p) / total, 0.0)

    def _compute_hma(self, src: np.ndarray, length: int) -> np.ndarray:
        """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
        half_len = max(length // 2, 2)
        sqrt_len = max(int(_math.sqrt(length)), 2)
        wma_half = talib.WMA(src, timeperiod=half_len)
        wma_full = talib.WMA(src, timeperiod=length)
        diff = np.where(
            ~np.isnan(wma_half) & ~np.isnan(wma_full),
            2.0 * wma_half - wma_full,
            np.nan,
        )
        return talib.WMA(diff, timeperiod=sqrt_len)

    # ─────────────────────────────────────────────────────────────────────
    # Main
    # ─────────────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(SignalType.HOLD, timestamp)

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        open_ = df["open"].values.astype(float)
        volume = df["volume"].values.astype(float)

        # ── Volatility Engine ─────────────────────────────────────────────
        vh_ema = talib.EMA(close, timeperiod=self.ema_len_v)
        fast_atr = talib.ATR(high, low, close, timeperiod=self.atr_fast_len)
        slow_atr = talib.ATR(high, low, close, timeperiod=self.atr_slow_len)
        atr_ratio = np.where(slow_atr != 0, fast_atr / slow_atr, 1.0)

        vol_ma = talib.SMA(volume, timeperiod=self.vol_len)
        safe_vol_ma = np.where(np.isnan(vol_ma), 0.0, vol_ma)
        is_vol_high = volume > (safe_vol_ma * self.vol_mult)

        # vhState — stateful: update only on high-volatility bars
        vh_state = np.zeros(len(df), dtype=int)
        for i in range(1, len(df)):
            s_atr = slow_atr[i]
            if not np.isnan(s_atr) and s_atr > 0:
                if (high[i] - low[i]) / s_atr > 1.0 and is_vol_high[i]:
                    vh_state[i] = 1 if close[i] > vh_ema[i] else -1
                else:
                    vh_state[i] = vh_state[i - 1]
            else:
                vh_state[i] = vh_state[i - 1]

        v_hub = np.where(
            vh_state == 1,
            np.where(atr_ratio >= 1.0, 2.0, 1.0),
            np.where(
                vh_state == -1,
                np.where(atr_ratio >= 1.0, -1.0, -2.0),
                0.0,
            ),
        )

        # ── ATM / In-Zone filter ──────────────────────────────────────────
        atm = np.round(close / self.in_step) * self.in_step
        iz = (close >= (atm - self.in_off)) & (close <= (atm + self.in_off))

        # ── Real Spread Ratio ─────────────────────────────────────────────
        c_eff = pd.Series(np.where(iz, high - open_, 0.0), index=df.index)
        p_eff = pd.Series(np.where(iz, open_ - low, 0.0), index=df.index)
        c_sma = c_eff.rolling(self.spread_ma_len).mean()
        p_sma = p_eff.rolling(self.spread_ma_len).mean()
        real_spread_ratio = pd.Series(self._f_sym(c_sma, p_sma), index=df.index)

        # ── Sentiment Components ──────────────────────────────────────────
        # Price PCR
        call_p = pd.Series(np.where(iz & (close > open_), close, 0.0), index=df.index)
        put_p = pd.Series(np.where(iz & (close < open_), close, 0.0), index=df.index)
        v_pcr_p = pd.Series(
            self._f_sym(call_p.rolling(10).mean(), put_p.rolling(10).mean()),
            index=df.index,
        )

        # TV PCR — proximity to rolling high / low
        lowest_low_10 = df["low"].rolling(10).min()
        highest_high_10 = df["high"].rolling(10).max()
        call_tv = pd.Series(np.maximum(0.0, close - lowest_low_10.values), index=df.index)
        put_tv = pd.Series(np.maximum(0.0, highest_high_10.values - close), index=df.index)
        v_pcr_tv = pd.Series(self._f_sym(call_tv, put_tv), index=df.index)

        # Day Vol PCR — cumulative bull vs bear volume
        bull_vol = pd.Series(np.where(iz & (close > open_), volume, 0.0), index=df.index)
        bear_vol = pd.Series(np.where(iz & (close < open_), volume, 0.0), index=df.index)
        v_pcr_d = pd.Series(self._f_sym(bull_vol.cumsum(), bear_vol.cumsum()), index=df.index)

        # Hour Vol PCR — rolling 20-bar bull vs bear volume
        v_pcr_h = pd.Series(
            self._f_sym(bull_vol.rolling(20).mean(), bear_vol.rolling(20).mean()),
            index=df.index,
        )

        # OI direction — fallback: use volume as OI proxy (Pine does the same when _OI unavailable)
        oi_v = pd.Series(volume, index=df.index)
        v_oi_d = np.where(
            oi_v.diff(5) > 0,
            np.where(df["close"].diff(5) > 0, 1.0, -1.0),
            0.0,
        )
        v_oi_h = np.where(
            oi_v.diff(3) > 0,
            np.where(df["close"].diff(3) > 0, 1.0, -1.0),
            0.0,
        )

        # ── WSI Composite Score ───────────────────────────────────────────
        v_adapt = np.power(atr_ratio, self.v_sens)
        wsi_base = (
            v_pcr_p.values * self.w_opt_p
            + v_pcr_tv.values * self.w_opt_tv
            + v_pcr_d.values * self.w_pcr_d
            + v_pcr_h.values * self.w_pcr_h
            + v_oi_d * self.w_build_d
            + v_oi_h * self.w_build_h
            + (v_hub / 2.0) * self.w_vol_score
            + real_spread_ratio.values * self.w_spread_pcr
        ) * 100.0

        wsi_total = np.nan_to_num(wsi_base * v_adapt, nan=0.0)

        # ── WSI Smooth (HMA) ──────────────────────────────────────────────
        wsi_smooth = self._compute_hma(wsi_total, self.sig_period)

        # ── Reversal Buffer — stateful trend state ────────────────────────
        # Pine: track running high/low of wsi_smooth; flip tS when buffer is pierced.
        buffer_val = (self.rev_thresh / 100.0) * 200.0  # default → 10.0
        t_state = np.zeros(len(df), dtype=int)
        lh = np.full(len(df), np.nan)
        ll = np.full(len(df), np.nan)

        for i in range(len(df)):
            ws = wsi_smooth[i]
            if np.isnan(ws):
                if i > 0:
                    t_state[i] = t_state[i - 1]
                continue
            if i == 0 or np.isnan(lh[i - 1]):
                lh[i] = ws
                ll[i] = ws
                t_state[i] = 0
            else:
                cur_lh = max(lh[i - 1], ws)
                cur_ll = min(ll[i - 1], ws)
                cur_ts = t_state[i - 1]

                if (ws - cur_ll) > buffer_val:
                    cur_ts = 1
                    cur_lh = ws
                    cur_ll = ws
                if (cur_lh - ws) > buffer_val:
                    cur_ts = -1
                    cur_lh = ws
                    cur_ll = ws

                lh[i] = cur_lh
                ll[i] = cur_ll
                t_state[i] = cur_ts

        # ── Entry Conditions (last complete bar) ──────────────────────────
        last_ts = int(t_state[-1])
        last_ws = float(wsi_smooth[-1]) if not np.isnan(wsi_smooth[-1]) else 0.0
        last_spread = (
            float(real_spread_ratio.iloc[-1])
            if not np.isnan(real_spread_ratio.iloc[-1])
            else 0.0
        )

        is_bullish_or_rising = last_ts == 1
        is_bearish_or_declining = last_ts == -1

        long_gate = is_bullish_or_rising and last_spread > self.spread_conv_buff
        short_gate = is_bearish_or_declining and last_spread < -self.spread_conv_buff

        if long_gate:
            return StrategyRecommendation(SignalType.LONG, timestamp)
        if short_gate:
            return StrategyRecommendation(SignalType.SHORT, timestamp)
        return StrategyRecommendation(SignalType.HOLD, timestamp)

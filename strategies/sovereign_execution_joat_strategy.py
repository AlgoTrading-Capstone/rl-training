"""
Sovereign Execution [JOAT]

Pine Script source: input/Sovereign-Execution-JOAT.pine
Timeframe: 15m  |  Lookback: 500 bars (125 h)

Strategy logic:
  A multi-module confluence strategy combining:
  - MODULE 1: Regime Cipher — custom adaptive MA with ATR volatility bands for
    trend-state detection (bullish / bearish / neutral).
  - MODULE 2: Displacement Lens — composite oscillator (BB %B + CCI + ROC) weighted
    with volume-based directional displacement.
  - MODULE 3: Session Filter — restrict entries to Asia / London / NY sessions (UTC).
  - MODULE 4: FVG Confluence — optional Fair Value Gap detection.
  - MODULE 5: Confluence Score — weighted blend of current-TF and higher-TF (4H) bias.

  Long  — regime is trending bull AND displacement is strongly bullish AND
           confluence score >= threshold AND session filter ok AND FVG ok AND
           edge detection (first bar of condition, not a repeated signal).
  Short — regime is trending bear AND displacement is strongly bearish AND
           confluence score <= short threshold AND session filter ok AND FVG ok AND
           edge detection.
  Flat  — regime or displacement opposes the open trade direction (exit condition).
  Hold  — none of the above.

Stop-loss and take-profit management is NOT implemented here.
Configure exits in the external execution layer.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType
from utils.resampling import resample_to_interval, resampled_merge


class SovereignExecutionJoatStrategy(BaseStrategy):
    """
    Sovereign Execution [JOAT] strategy.

    Combines a custom adaptive regime MA, a displacement composite oscillator,
    session filters, optional FVG detection, and a 4H-bias confluence score to
    produce high-confidence long/short signals.

    Default parameters match the Pine Script inputs.
    """

    def __init__(
        self,
        # Execution
        risk_pct: float = 1.5,
        rr_ratio: float = 2.0,
        atr_sl_mult: float = 1.5,
        # Entry filters
        conf_thresh: int = 60,
        conf_thresh_s: int = 40,
        require_sess: bool = True,
        require_fvg: bool = False,
        # Regime Cipher
        ma_len: int = 27,
        atr_len: int = 14,
        atr_factor: float = 1.05,
        base_ma: str = "EMA",
        # Displacement
        bb_len: int = 20,
        bb_mult: float = 2.0,
        cci_len: int = 23,
        roc_len: int = 50,
        disp_smooth: int = 5,
        # Session (UTC hours)
        asia_s: int = 0,
        asia_e: int = 8,
        ldn_s: int = 8,
        ldn_e: int = 14,
        ny_s: int = 14,
        ny_e: int = 21,
        # Confluence
        swing_len: int = 10,
    ):
        super().__init__(
            name="Sovereign Execution [JOAT]",
            description=(
                "Multi-module confluence strategy: Regime Cipher (adaptive MA + ATR bands), "
                "Displacement Lens (BB/CCI/ROC composite), Session Filter, optional FVG, "
                "and a 4H HTF bias score. Fires edge-detected LONG/SHORT signals."
            ),
            timeframe="15m",
            lookback_hours=500,
        )
        # Parameters
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.atr_sl_mult = atr_sl_mult
        self.conf_thresh = conf_thresh
        self.conf_thresh_s = conf_thresh_s
        self.require_sess = require_sess
        self.require_fvg = require_fvg
        self.ma_len = ma_len
        self.atr_len = atr_len
        self.atr_factor = atr_factor
        self.base_ma = base_ma
        self.bb_len = bb_len
        self.bb_mult = bb_mult
        self.cci_len = cci_len
        self.roc_len = roc_len
        self.disp_smooth = disp_smooth
        self.asia_s = asia_s
        self.asia_e = asia_e
        self.ldn_s = ldn_s
        self.ldn_e = ldn_e
        self.ny_s = ny_s
        self.ny_e = ny_e
        self.swing_len = swing_len

        # CRITICAL RL GUARD: dynamic warmup based on the largest indicator lookback.
        # 100 is the window used for disp_std (stdev of displacement over 100 bars).
        self.MIN_CANDLES_REQUIRED = 3 * max(
            self.ma_len,
            self.atr_len,
            self.bb_len,
            self.cci_len,
            self.roc_len,
            self.swing_len,
            100,
        )

    # ------------------------------------------------------------------
    # Custom MA helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wma(src: np.ndarray, length: int) -> np.ndarray:
        """Weighted Moving Average using TA-Lib."""
        return talib.WMA(src, timeperiod=length)

    def _dema(self, src: np.ndarray, length: int) -> np.ndarray:
        """Double EMA: 2*EMA1 - EMA2."""
        e1 = talib.EMA(src, timeperiod=length)
        e2 = talib.EMA(e1, timeperiod=length)
        return 2.0 * e1 - e2

    def _tema(self, src: np.ndarray, length: int) -> np.ndarray:
        """Triple EMA: 3*EMA1 - 3*EMA2 + EMA3."""
        e1 = talib.EMA(src, timeperiod=length)
        e2 = talib.EMA(e1, timeperiod=length)
        e3 = talib.EMA(e2, timeperiod=length)
        return 3.0 * e1 - 3.0 * e2 + e3

    def _hma(self, src: np.ndarray, length: int) -> np.ndarray:
        """Hull MA: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
        half_len = max(1, round(length / 2))
        sqrt_len = max(1, round(math.sqrt(length)))
        wma_half = talib.WMA(src, timeperiod=half_len)
        wma_full = talib.WMA(src, timeperiod=length)
        return talib.WMA(2.0 * wma_half - wma_full, timeperiod=sqrt_len)

    def _rma(self, series: pd.Series, length: int) -> pd.Series:
        """Wilder's RMA (EWM with alpha = 1/length)."""
        return series.ewm(alpha=1.0 / length, adjust=False).mean()

    def _base_ma_calc(self, src: np.ndarray, length: int) -> np.ndarray:
        """Dispatch to the selected base MA type."""
        ma_type = self.base_ma.upper()
        if ma_type == "RMA":
            return pd.Series(src).ewm(alpha=1.0 / length, adjust=False).mean().to_numpy()
        elif ma_type == "SMA":
            return talib.SMA(src, timeperiod=length)
        elif ma_type == "EMA":
            return talib.EMA(src, timeperiod=length)
        elif ma_type == "WMA":
            return talib.WMA(src, timeperiod=length)
        elif ma_type == "HMA":
            return self._hma(src, length)
        elif ma_type == "DEMA":
            return self._dema(src, length)
        elif ma_type == "TEMA":
            return self._tema(src, length)
        else:
            # Fallback to EMA
            return talib.EMA(src, timeperiod=length)

    # ------------------------------------------------------------------
    # HTF scoring helper (computed on resampled 4H data)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_htf_score(
        htf_df: pd.DataFrame, ma_len: int, bb_len: int, bb_mult: float
    ) -> pd.Series:
        """
        Compute the HTF confluence score on the resampled dataframe.

        Mirrors Pine's f_htf_scores():
            t  = close > ema(close, ma_len) ? 1.0 : -1.0
            pct_b = (close - (basis - dev)) / span   (BB percent-B)
            m  = pct_b > 0.5 ? 1.0 : pct_b < 0.0 ? -1.0 : 0.0
            score = (t + m) / 2.0
        """
        close = htf_df["close"].values
        ema_v = talib.EMA(close, timeperiod=ma_len)
        t = np.where(close > ema_v, 1.0, -1.0)

        std_v = pd.Series(close).rolling(bb_len).std().values
        basis = talib.SMA(close, timeperiod=bb_len)
        dev = bb_mult * std_v
        span = 2.0 * dev
        pct_b = np.where(span != 0, (close - (basis - dev)) / span, 0.5)
        m = np.where(pct_b > 0.5, 1.0, np.where(pct_b < 0.0, -1.0, 0.0))
        score = (t + m) / 2.0
        return pd.Series(score, index=htf_df.index)

    # ------------------------------------------------------------------
    # FVG detection
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fvg(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Detect recent bull and bear Fair Value Gaps (FVG) within the last 10 bars.

        Pine logic:
          bull FVG: low[i] > high[i+2] and close[i+1] > high[i+2]  for i in 1..10
          bear FVG: high[i] < low[i+2] and close[i+1] < low[i+2]   for i in 1..10

        We vectorize by checking the worst-case (tightest) definition across
        lags 1–10 and OR them together.
        """
        recent_bull = pd.Series(False, index=df.index)
        recent_bear = pd.Series(False, index=df.index)
        for i in range(1, 11):
            # bull: low[i] > high[i+2] means low shifted by i is > high shifted by i+2
            bull_cond = (
                df["low"].shift(i) > df["high"].shift(i + 2)
            ) & (
                df["close"].shift(i + 1) > df["high"].shift(i + 2)
            )
            bear_cond = (
                df["high"].shift(i) < df["low"].shift(i + 2)
            ) & (
                df["close"].shift(i + 1) < df["low"].shift(i + 2)
            )
            recent_bull = recent_bull | bull_cond
            recent_bear = recent_bear | bear_cond
        return recent_bull, recent_bear

    # ------------------------------------------------------------------
    # Struct score (f_struct_score) — stateful, uses iterative logic
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_struct_score(
        df: pd.DataFrame, swing_len: int
    ) -> pd.Series:
        """
        Vectorized approximation of Pine's f_struct_score():
            var int os = 0
            upper_sw = ta.highest(high, swing_len)
            lower_sw = ta.lowest(low, swing_len)
            os := high[swing_len] > upper_sw ? 0 : low[swing_len] < lower_sw ? 1 : os[1]
            os == 1 ? 1.0 : -1.0

        The stateful `os` is forward-filled after each conditional update.
        """
        upper_sw = df["high"].rolling(swing_len).max()
        lower_sw = df["low"].rolling(swing_len).min()

        # high[swing_len] = high shifted by swing_len bars
        high_shifted = df["high"].shift(swing_len)
        low_shifted = df["low"].shift(swing_len)

        # Compute the conditional update: 0 if bearish break, 1 if bullish break, else carry
        # We use an iterative loop because each bar's os depends on the previous os.
        high_shifted_arr = high_shifted.values
        low_shifted_arr = low_shifted.values
        upper_sw_arr = upper_sw.values
        lower_sw_arr = lower_sw.values

        os_arr = np.zeros(len(df), dtype=float)
        for i in range(1, len(df)):
            h_s = high_shifted_arr[i]
            l_s = low_shifted_arr[i]
            u = upper_sw_arr[i]
            lo = lower_sw_arr[i]
            if np.isnan(h_s) or np.isnan(u) or np.isnan(l_s) or np.isnan(lo):
                os_arr[i] = os_arr[i - 1]
            elif h_s > u:
                os_arr[i] = 0.0
            elif l_s < lo:
                os_arr[i] = 1.0
            else:
                os_arr[i] = os_arr[i - 1]

        struct_score = np.where(os_arr == 1.0, 1.0, -1.0)
        return pd.Series(struct_score, index=df.index)

    # ------------------------------------------------------------------
    # Trend state — stateful, iterative
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_trend_state(
        close: pd.Series, band_up: pd.Series, band_lo: pd.Series
    ) -> pd.Series:
        """
        Replicate Pine's stateful trend_state:
            var int trend_state = 0
            if close > band_up: trend_state := 1
            if close < band_lo: trend_state := -1

        Each bar only updates when price breaks a band; otherwise carries forward.
        """
        close_arr = close.values
        band_up_arr = band_up.values
        band_lo_arr = band_lo.values

        ts = np.zeros(len(close), dtype=int)
        for i in range(1, len(close)):
            ts[i] = ts[i - 1]
            if not np.isnan(band_up_arr[i]) and close_arr[i] > band_up_arr[i]:
                ts[i] = 1
            elif not np.isnan(band_lo_arr[i]) and close_arr[i] < band_lo_arr[i]:
                ts[i] = -1

        return pd.Series(ts, index=close.index)

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        # CRITICAL RL GUARD
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        df = df.copy()

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ----------------------------------------------------------------
        # MODULE 1 — REGIME CIPHER
        # ----------------------------------------------------------------
        x_src = np.sqrt(close.values)
        r_coeff = 2.0 / (1.0 + self.ma_len)

        base_val = pd.Series(self._base_ma_calc(x_src, self.ma_len), index=df.index)

        # core_ma = base_val * (base_val * r_coeff + nz(base_val[1]) * (1.0 - r_coeff))
        # nz(base_val[1]) = base_val.shift(1).fillna(0.0)
        base_val_prev = base_val.shift(1).fillna(0.0)
        core_ma = base_val * (base_val * r_coeff + base_val_prev * (1.0 - r_coeff))

        atr_full_arr = talib.ATR(
            high.values, low.values, close.values, timeperiod=self.atr_len
        )
        atr_full = pd.Series(atr_full_arr, index=df.index) * self.atr_factor

        atr_half_len = max(1, round(self.atr_len / 2))
        atr_half_arr = talib.ATR(
            high.values, low.values, close.values, timeperiod=atr_half_len
        )
        atr_half = pd.Series(atr_half_arr, index=df.index) * self.atr_factor

        vol_ratio = np.where(atr_half != 0, atr_full / atr_half, 1.0)
        vol_ratio = pd.Series(vol_ratio, index=df.index)

        # regime_ma = core_ma * (1.0 - vol_ratio) + nz(core_ma[1]) * vol_ratio
        # Use iterative forward-fill for the stateful nz(core_ma[1]) term
        core_ma_arr = core_ma.values
        vol_ratio_arr = vol_ratio.values
        regime_ma_arr = np.full(len(df), np.nan)
        for i in range(len(df)):
            prev_rm = regime_ma_arr[i - 1] if i > 0 and not np.isnan(regime_ma_arr[i - 1]) else 0.0
            if np.isnan(core_ma_arr[i]) or np.isnan(vol_ratio_arr[i]):
                regime_ma_arr[i] = np.nan
            else:
                regime_ma_arr[i] = (
                    core_ma_arr[i] * (1.0 - vol_ratio_arr[i])
                    + prev_rm * vol_ratio_arr[i]
                )
        regime_ma = pd.Series(regime_ma_arr, index=df.index)

        band_up = regime_ma + atr_full
        band_lo = regime_ma - atr_full

        trend_state = self._compute_trend_state(close, band_up, band_lo)

        # Volatility regime
        bb_width = close.rolling(self.bb_len).std() * self.bb_mult * 2.0
        bb_width_ma = bb_width.rolling(50).mean()
        is_expanding = bb_width > bb_width_ma * 1.1
        is_compressing = bb_width < bb_width_ma * 0.85

        regime_trending_bull = (trend_state == 1) & (~is_compressing)
        regime_trending_bear = (trend_state == -1) & (~is_compressing)

        # ----------------------------------------------------------------
        # MODULE 2 — DISPLACEMENT LENS
        # ----------------------------------------------------------------
        # BB norm
        bb_basis = close.rolling(self.bb_len).mean()
        bb_dev = self.bb_mult * close.rolling(self.bb_len).std()
        bb_span = 2.0 * bb_dev
        pct_b = np.where(
            bb_span != 0,
            (close - (bb_basis - bb_dev)) / bb_span,
            0.5,
        )
        bb_norm = np.clip((pct_b - 0.5) * 2.0, -1.0, 1.0)
        bb_norm = pd.Series(bb_norm, index=df.index)

        # CCI norm
        cci_raw = pd.Series(
            talib.CCI(high.values, low.values, close.values, timeperiod=self.cci_len),
            index=df.index,
        )
        cci_norm = cci_raw.clip(-200.0, 200.0) / 200.0
        cci_norm = cci_norm.clip(-1.0, 1.0)

        # ROC norm
        roc_v = pd.Series(
            talib.ROC(close.values, timeperiod=self.roc_len), index=df.index
        )
        roc_std = roc_v.rolling(50).std()
        roc_norm = np.where(
            roc_std != 0,
            np.clip(roc_v / (roc_std * 2.0), -1.0, 1.0),
            0.0,
        )
        roc_norm = pd.Series(roc_norm, index=df.index)

        osc_composite = (bb_norm + cci_norm + roc_norm) / 3.0

        # Volume-weighted displacement
        body_size = (close - df["open"]).abs()
        candle_range = high - low
        body_ratio = np.where(candle_range != 0, body_size / candle_range, 0.0)
        body_ratio = pd.Series(body_ratio, index=df.index)

        vol_sma = volume.rolling(20).mean()
        vol_intensity = np.where(
            vol_sma != 0,
            np.clip(volume / vol_sma - 1.0, -1.0, 1.0),
            0.0,
        )
        vol_intensity = pd.Series(vol_intensity, index=df.index)

        disp_dir = np.where(close > df["open"], 1.0, np.where(close < df["open"], -1.0, 0.0))
        disp_dir = pd.Series(disp_dir, index=df.index)

        vol_disp = disp_dir * body_ratio * vol_intensity

        raw_osc = osc_composite * 0.7 + vol_disp * 0.3
        displacement = pd.Series(
            talib.EMA(raw_osc.values, timeperiod=self.disp_smooth), index=df.index
        )

        disp_std = displacement.rolling(100).std()
        disp_thresh_up = disp_std * 1.5
        disp_thresh_dn = -disp_std * 1.5

        strong_bull_disp = displacement > disp_thresh_up
        strong_bear_disp = displacement < disp_thresh_dn

        # ----------------------------------------------------------------
        # MODULE 3 — SESSION FILTER
        # ----------------------------------------------------------------
        dates = pd.to_datetime(df["date"], utc=True)
        utc_h = dates.dt.hour

        in_asia = (utc_h >= self.asia_s) & (utc_h < self.asia_e)
        in_ldn = (utc_h >= self.ldn_s) & (utc_h < self.ldn_e)
        in_ny = (utc_h >= self.ny_s) & (utc_h < self.ny_e)
        in_session = in_asia | in_ldn | in_ny

        session_ok = ~self.require_sess | in_session

        # ----------------------------------------------------------------
        # MODULE 4 — FVG CONFLUENCE (optional)
        # ----------------------------------------------------------------
        if self.require_fvg:
            recent_bull_fvg, recent_bear_fvg = self._compute_fvg(df)
        else:
            recent_bull_fvg = pd.Series(False, index=df.index)
            recent_bear_fvg = pd.Series(False, index=df.index)

        fvg_ok_long = ~self.require_fvg | recent_bull_fvg
        fvg_ok_short = ~self.require_fvg | recent_bear_fvg

        # ----------------------------------------------------------------
        # MODULE 5 — CONFLUENCE SCORE
        # ----------------------------------------------------------------
        # Current TF scores
        ema_ctf = pd.Series(
            talib.EMA(close.values, timeperiod=self.ma_len), index=df.index
        )
        s_trend = np.where(close > ema_ctf, 1.0, -1.0)
        s_mom = osc_composite.values
        s_vol = np.where(
            bb_width_ma != 0,
            np.clip((bb_width / bb_width_ma - 1.0) * 3.0, -1.0, 1.0),
            0.0,
        )
        s_struct = self._compute_struct_score(df, self.swing_len).values
        s_volc = (disp_dir * vol_intensity.clip(upper=1.0)).values

        ctf_score = pd.Series(
            (s_trend + s_mom + s_vol + s_struct + s_volc) / 5.0, index=df.index
        )

        # HTF score via 4H resampling (anti-lookahead: shift(1) applied via resampled_merge)
        try:
            htf_df = resample_to_interval(df, "4h")
            htf_df["htf_score"] = self._compute_htf_score(
                htf_df, self.ma_len, self.bb_len, self.bb_mult
            )
            merged = resampled_merge(original=df, resampled=htf_df, fill_na=True)
            htf_score_col = "resample_240_htf_score"
            if htf_score_col in merged.columns:
                htf_score = pd.Series(merged[htf_score_col].values, index=df.index)
            else:
                htf_score = pd.Series(0.0, index=df.index)
        except Exception:
            htf_score = pd.Series(0.0, index=df.index)

        # Eliminate lookahead: HTF value must be from the previous closed 4H bar.
        # resampled_merge already handles alignment; shift(1) ensures we never use
        # the in-progress HTF bar's value.
        htf_score = htf_score.shift(1).fillna(0.0)

        # Weighted confluence: CTF=40%, HTF=60%
        raw_conf = ctf_score * 0.4 + htf_score.fillna(0.0) * 0.6
        confluence = ((raw_conf + 1.0) / 2.0 * 100).round().clip(0, 100)

        conf_ok_long = confluence >= self.conf_thresh
        conf_ok_short = confluence <= self.conf_thresh_s

        # ----------------------------------------------------------------
        # ENTRY CONDITIONS
        # ----------------------------------------------------------------
        long_entry = (
            regime_trending_bull & conf_ok_long & strong_bull_disp & session_ok & fvg_ok_long
        )
        short_entry = (
            regime_trending_bear & conf_ok_short & strong_bear_disp & session_ok & fvg_ok_short
        )

        # Edge detection: only fire on the first bar of each new signal
        prev_long = long_entry.shift(1).fillna(False)
        prev_short = short_entry.shift(1).fillna(False)
        long_edge = long_entry & ~prev_long
        short_edge = short_entry & ~prev_short

        # ----------------------------------------------------------------
        # EXIT CONDITIONS (regime flip or opposing displacement)
        # ----------------------------------------------------------------
        exit_long = (trend_state == -1) | is_compressing | strong_bear_disp
        exit_short = (trend_state == 1) | is_compressing | strong_bull_disp

        # ----------------------------------------------------------------
        # SIGNAL — evaluate at the last bar
        # ----------------------------------------------------------------
        idx = len(df) - 1

        def _safe_bool(series: pd.Series, i: int) -> bool:
            val = series.iloc[i]
            return bool(val) if not pd.isna(val) else False

        is_long_edge = _safe_bool(long_edge, idx)
        is_short_edge = _safe_bool(short_edge, idx)
        is_exit_long = _safe_bool(exit_long, idx)
        is_exit_short = _safe_bool(exit_short, idx)

        # Priority: exit first, then directional entry, then hold
        if is_exit_long or is_exit_short:
            return StrategyRecommendation(signal=SignalType.FLAT, timestamp=timestamp)
        if is_long_edge:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        if is_short_edge:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)
        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

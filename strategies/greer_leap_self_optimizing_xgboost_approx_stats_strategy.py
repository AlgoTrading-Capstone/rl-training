"""
Greer Leap Self-Optimizing XGBoost Approx & Stats Strategy

Converted from PineScript: 'Private-DO-NOT-USE-Self-Optimizing XGBoost Approx & Stats'

Combines five normalised features (RSI, CCI, ATR, ROC, WaveTrend) with
auto-optimised rolling-correlation weights.  An optional Kalman filter
smooths the composite score; an optional daily-timeframe regime filter
gates signals to the prevailing trend direction.  Entry signals are
generated on zero-crossovers of the final composite output.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, SignalType, StrategyRecommendation
from utils.resampling import resample_to_interval, resampled_merge


class GreerLeapSelfOptimizingXgboostApproxStatsStrategy(BaseStrategy):
    """
    Self-Optimizing XGBoost Approx & Stats strategy.

    Features
    --------
    - RSI, CCI, ATR (normalised), ROC, WaveTrend (WT1)
    - Auto-weighted composite via rolling Pearson correlation with price change
    - Optional Kalman filter on the composite score
    - Optional daily-timeframe regime gate (close vs EMA)
    - Long/Short on zero-crossover of the final composite output
    """

    def __init__(self):
        super().__init__(
            name="GreerLeapSelfOptimizingXgboostApproxStatsStrategy",
            description=(
                "Converted from PineScript: Private-DO-NOT-USE-Self-Optimizing XGBoost Approx & Stats. "
                "Five-feature composite (RSI, CCI, ATR, ROC, WaveTrend) with rolling correlation "
                "auto-weighting, optional Kalman filter, and daily regime gate."
            ),
            timeframe="15m",
            lookback_hours=25,
        )

        # --- Indicator parameters ---
        self.rsi_length = 14
        self.cci_length = 20
        self.atr_length = 14
        self.roc_length = 12
        self.wt_ema_length = 10
        self.wt_channel_length = 21
        self.opt_window = 5

        # --- Optional filter flags ---
        self.use_kalman = False
        self.use_regime = True
        self.regime_ema_len = 50

        # CRITICAL RL GUARD: dynamic warmup based on the longest base-TF indicator
        self.MIN_CANDLES_REQUIRED = 3 * max(
            self.rsi_length,
            self.cci_length,
            self.atr_length,
            self.roc_length,
            self.wt_ema_length,
            self.wt_channel_length,
            self.opt_window,
        )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        # --- RL warmup guard ---
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        df = df.copy()

        # ----------------------------------------------------------------
        # 1. Feature calculations
        # ----------------------------------------------------------------

        # RSI
        df["rsi"] = talib.RSI(df["close"].values, timeperiod=self.rsi_length)

        # CCI (standard formula: uses 0.015 * mean-deviation, same as PineScript)
        df["cci"] = talib.CCI(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            timeperiod=self.cci_length,
        )

        # ATR normalised by close
        df["atr"] = talib.ATR(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            timeperiod=self.atr_length,
        )
        df["norm_atr"] = df["atr"] / df["close"] * 100

        # ROC
        df["roc"] = talib.ROC(df["close"].values, timeperiod=self.roc_length)

        # WaveTrend (WT1)
        ap = (df["high"] + df["low"] + df["close"]) / 3
        esa = pd.Series(
            talib.EMA(ap.values, timeperiod=self.wt_ema_length),
            index=df.index,
        )
        abs_diff = (ap - esa).abs()
        d = pd.Series(
            talib.EMA(abs_diff.values, timeperiod=self.wt_ema_length),
            index=df.index,
        )
        # Guard against zero / NaN denominator (mirrors Pine's implicit NZ behaviour)
        d_safe = d.replace(0, np.nan)
        ci = ((ap - esa) / (0.015 * d_safe)).fillna(0.0)
        df["wt1"] = pd.Series(
            talib.EMA(ci.values, timeperiod=self.wt_channel_length),
            index=df.index,
        )

        # ----------------------------------------------------------------
        # 2. Feature normalisation
        # ----------------------------------------------------------------

        df["rsin"] = (df["rsi"] - 50) / 50
        df["ccin"] = df["cci"].clip(-200, 200) / 200
        df["atrn"] = (df["norm_atr"] / 5).clip(0, 1)
        df["rocn"] = df["roc"].clip(-5, 5) / 5
        df["wtn"]  = df["wt1"].clip(-60, 60) / 60

        # ----------------------------------------------------------------
        # 3. Auto-optimisation (rolling Pearson correlation weights)
        # ----------------------------------------------------------------

        price_change = df["close"].diff()

        w_rsi = df["rsin"].rolling(self.opt_window).corr(price_change).fillna(0.2)
        w_cci = df["ccin"].rolling(self.opt_window).corr(price_change).fillna(0.2)
        # ATR weight is negated (higher volatility is a bearish signal for weighting)
        w_atr = df["atrn"].rolling(self.opt_window).corr(price_change).fillna(0.2) * -1
        w_roc = df["rocn"].rolling(self.opt_window).corr(price_change).fillna(0.2)
        w_wt  = df["wtn"].rolling(self.opt_window).corr(price_change).fillna(0.2)

        total_w = (
            w_rsi.abs() + w_cci.abs() + w_atr.abs() + w_roc.abs() + w_wt.abs()
        )
        numerator = (
            w_rsi * df["rsin"]
            + w_cci * df["ccin"]
            + w_atr * df["atrn"]
            + w_roc * df["rocn"]
            + w_wt  * df["wtn"]
        )
        # Avoid division by zero when all weights cancel out
        df["composite"] = numerator / total_w.replace(0, 1)

        # ----------------------------------------------------------------
        # 4. Kalman filter (stateful — requires a scalar loop)
        # ----------------------------------------------------------------

        if self.use_kalman:
            composite_arr = df["composite"].values
            kf_state = np.full(len(df), np.nan)
            kf_p = 1.0
            kalman_gain = 0.2
            kalman_noise = 0.01
            prev_state = np.nan

            for i in range(len(df)):
                val = composite_arr[i]
                if np.isnan(val):
                    continue
                if np.isnan(prev_state):
                    # First valid bar — initialise
                    kf_state[i] = val
                    prev_state = val
                else:
                    kf_p += kalman_noise
                    kalman_k = kf_p / (kf_p + kalman_gain)
                    kf_state[i] = prev_state + kalman_k * (val - prev_state)
                    prev_state = kf_state[i]
                    kf_p = (1 - kalman_k) * kf_p

            df["kf_output"] = kf_state
        else:
            df["kf_output"] = df["composite"]

        # ----------------------------------------------------------------
        # 5. Daily regime filter (MTF via resampling utilities)
        # ----------------------------------------------------------------

        if self.use_regime:
            resampled_df = resample_to_interval(df, "1d")
            resampled_df["regime_ema"] = talib.EMA(
                resampled_df["close"].values,
                timeperiod=self.regime_ema_len,
            )
            # 1d = 1440 minutes -> merged columns are prefixed resample_1440_
            df = resampled_merge(original=df, resampled=resampled_df, fill_na=True)

            r_close = df["resample_1440_close"]
            r_ema   = df["resample_1440_regime_ema"]

            in_regime = np.where(
                r_close > r_ema, 1,
                np.where(r_close < r_ema, -1, 0),
            )
            in_regime_s = pd.Series(in_regime, index=df.index)

            # Bullish regime: only allow positive composite  (clip lower=0)
            # Bearish regime: only allow negative composite  (clip upper=0)
            # Neutral regime: output is 0
            df["final_output"] = np.where(
                in_regime_s == 1,  df["kf_output"].clip(lower=0),
                np.where(
                    in_regime_s == -1, df["kf_output"].clip(upper=0),
                    0.0,
                ),
            )
        else:
            df["final_output"] = df["kf_output"]

        # ----------------------------------------------------------------
        # 6. Signal: zero-crossover of the final composite score
        # ----------------------------------------------------------------

        fo      = df["final_output"]
        fo_prev = fo.shift(1)

        long_signal  = (fo > 0) & (fo_prev <= 0)
        short_signal = (fo < 0) & (fo_prev >= 0)

        last_idx = df.index[-1]

        if long_signal.loc[last_idx]:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        if short_signal.loc[last_idx]:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)

        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

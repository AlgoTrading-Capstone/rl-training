"""
Kinetic Inertia Short 1h

Pine Script source: input/kinetic-inertia-short-1h.pine
Timeframe: 1h  |  Lookback: 21 bars (21 h)

Strategy logic:
  A short-only momentum strategy built around the physics metaphor of kinetic
  energy. It models price movement using velocity (Rate of Change), acceleration
  (first difference of velocity), and kinetic energy (0.5 * volume * velocity^2 *
  sign(velocity)).

  Short entry is triggered when ALL of the following hold on the current bar and
  did NOT hold on the previous bar (edge-trigger):
    - bearish_candle  : close < open
    - velocity > 0    : price is still technically in an upward ROC (fading momentum)
    - smooth_accel < 0: smoothed acceleration is negative (velocity decelerating)
    - kinetic_energy < smooth_ke: raw KE is below its SMA (energy draining)

  stop-loss, take-profit, and position-sizing are NOT implemented here;
  they are managed by the external execution engine (strategy.exit() dropped).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType


class KineticInertiaShort1hStrategy(BaseStrategy):
    """
    Kinetic Inertia Short 1h strategy.

    Short-only. Detects fading bullish momentum via velocity (ROC), acceleration
    (change-in-velocity) and kinetic energy (KE = 0.5 * volume * ROC^2 * sign(ROC))
    and fires a SHORT signal on the first bar where the full bearish setup triggers.

    Parameters
    ----------
    length : int
        ROC period and SMA smoothing period for kinetic energy. Default 10.
    smooth : int
        SMA period for smoothing acceleration. Default 3.
    atr_length : int
        ATR period (referenced for dynamic MIN_CANDLES_REQUIRED). Default 14.
    """

    def __init__(
        self,
        length: int = 10,
        smooth: int = 3,
        atr_length: int = 14,
    ):
        super().__init__(
            name="KineticInertiaShort1h",
            description=(
                "Short-only momentum strategy using physics-inspired kinetic energy: "
                "velocity (ROC), acceleration (diff of ROC), and KE (0.5*volume*ROC^2*sign(ROC)). "
                "Enters short on a bearish candle with decelerating but still-positive velocity "
                "and KE below its SMA. SL/TP managed externally."
            ),
            timeframe="1h",
            lookback_hours=21,
        )

        # RL-tunable parameters
        self.length = length
        self.smooth = smooth
        self.atr_length = atr_length

        # CRITICAL RL GUARD: dynamic warmup — 3x the largest indicator period.
        # Indicators: ROC(length), SMA(smooth), SMA(length), ATR(atr_length)
        self.MIN_CANDLES_REQUIRED = 3 * max(self.length, self.smooth, self.atr_length)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        # --- CRITICAL RL GUARD ---
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        df = df.copy()

        # ------------------------------------------------------------------
        # 1. VELOCITY — ta.roc(close, length)
        # ------------------------------------------------------------------
        df["velocity"] = pd.Series(
            talib.ROC(df["close"].values, timeperiod=self.length),
            index=df.index,
        )

        # ------------------------------------------------------------------
        # 2. ACCELERATION — velocity - velocity[1]
        #    smooth_accel = ta.sma(acceleration, smooth)
        # ------------------------------------------------------------------
        df["acceleration"] = df["velocity"] - df["velocity"].shift(1)
        df["smooth_accel"] = pd.Series(
            talib.SMA(df["acceleration"].values, timeperiod=self.smooth),
            index=df.index,
        )

        # ------------------------------------------------------------------
        # 3. KINETIC ENERGY — 0.5 * volume * velocity^2 * sign(velocity)
        #    smooth_ke = ta.sma(kinetic_energy, length)
        # ------------------------------------------------------------------
        df["kinetic_energy"] = (
            0.5
            * df["volume"]
            * np.power(df["velocity"], 2)
            * np.sign(df["velocity"])
        )
        df["smooth_ke"] = pd.Series(
            talib.SMA(df["kinetic_energy"].values, timeperiod=self.length),
            index=df.index,
        )

        # ------------------------------------------------------------------
        # 4. SETUP CONDITIONS (vectorized)
        #    bearish_candle   = close < open
        #    setup_condition  = velocity > 0 AND smooth_accel < 0 AND ke < smooth_ke
        #    enter_short      = bearish_candle AND setup_condition AND NOT setup_condition[1]
        # ------------------------------------------------------------------
        bearish_candle = df["close"] < df["open"]
        setup_condition = (
            (df["velocity"] > 0)
            & (df["smooth_accel"] < 0)
            & (df["kinetic_energy"] < df["smooth_ke"])
        )
        prev_setup = setup_condition.shift(1).fillna(False).astype(bool)
        enter_short = bearish_candle & setup_condition & ~prev_setup

        # ------------------------------------------------------------------
        # 5. SIGNAL — evaluate at the last (most recently closed) bar
        # ------------------------------------------------------------------
        last_val = enter_short.iloc[-1]
        if pd.isna(last_val):
            last_val = False

        if bool(last_val):
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)

        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

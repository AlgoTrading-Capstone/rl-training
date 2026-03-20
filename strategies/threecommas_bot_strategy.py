"""
3Commas Bot Strategy
Converted from Pine Script v5: "3Commas Bot" by Bj Bot

Signal logic: MA crossover (EMA21 vs EMA50) on close.
  - LONG  when EMA21 crosses above EMA50
  - SHORT when EMA21 crosses below EMA50
  - HOLD  otherwise

Stop / limit / trailing-stop / position-sizing logic from the original Pine Script
is broker-emulator logic and is NOT implemented here; only entry signals are produced.

MA types supported: EMA (default), HEMA (EMA of Heikin-Ashi open), SMA, HMA, WMA,
DEMA, VWMA, VWAP, T3.
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, SignalType, StrategyRecommendation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_ha_open(df: pd.DataFrame) -> pd.Series:
    """Vectorized Heikin-Ashi open calculation matching Pine's _haOpen()."""
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    ha_open_values = [np.nan] * len(df)
    for i in range(len(df)):
        if i == 0:
            ha_open_values[i] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
        else:
            ha_open_values[i] = (ha_open_values[i - 1] + ha_close.iloc[i - 1]) / 2.0

    return pd.Series(ha_open_values, index=df.index)


def _compute_ma(df: pd.DataFrame, ma_type: str, length: int) -> pd.Series:
    """
    Compute the requested moving average type on df['close'] (or HA-open for HEMA).
    Mirrors the Pine getMA() function.
    """
    close = df["close"]
    vol = df["volume"]

    if ma_type == "EMA":
        return pd.Series(
            talib.EMA(close.values, timeperiod=length), index=df.index
        )

    elif ma_type == "HEMA":
        ha_open = _compute_ha_open(df)
        return pd.Series(
            talib.EMA(ha_open.values, timeperiod=length), index=df.index
        )

    elif ma_type == "SMA":
        return pd.Series(
            talib.SMA(close.values, timeperiod=length), index=df.index
        )

    elif ma_type == "HMA":
        half_len = max(length // 2, 1)
        sqrt_len = max(int(math.sqrt(length)), 1)
        wma_half = talib.WMA(close.values, timeperiod=half_len)
        wma_full = talib.WMA(close.values, timeperiod=length)
        raw = pd.Series(2 * wma_half - wma_full, index=df.index)
        return pd.Series(
            talib.WMA(raw.values, timeperiod=sqrt_len), index=df.index
        )

    elif ma_type == "WMA":
        return pd.Series(
            talib.WMA(close.values, timeperiod=length), index=df.index
        )

    elif ma_type == "DEMA":
        e1 = talib.EMA(close.values, timeperiod=length)
        e2 = talib.EMA(e1, timeperiod=length)
        return pd.Series(2 * e1 - e2, index=df.index)

    elif ma_type == "VWMA":
        # VWMA = sum(close * volume, length) / sum(volume, length)
        cv = (close * vol).rolling(length).sum()
        v = vol.rolling(length).sum()
        return cv / v

    elif ma_type == "VWAP":
        typical = (df["high"] + df["low"] + close) / 3
        return (vol * typical).cumsum() / vol.cumsum()

    elif ma_type == "T3":
        ab = 0.7
        ac1 = -(ab ** 3)
        ac2 = 3 * ab ** 2 + 3 * ab ** 3
        ac3 = -6 * ab ** 2 - 3 * ab - 3 * ab ** 3
        ac4 = 1 + 3 * ab + ab ** 3 + 3 * ab ** 2
        axe1 = talib.EMA(close.values, timeperiod=length)
        axe2 = talib.EMA(axe1, timeperiod=length)
        axe3 = talib.EMA(axe2, timeperiod=length)
        axe4 = talib.EMA(axe3, timeperiod=length)
        axe5 = talib.EMA(axe4, timeperiod=length)
        axe6 = talib.EMA(axe5, timeperiod=length)
        return pd.Series(
            ac1 * axe6 + ac2 * axe5 + ac3 * axe4 + ac4 * axe3, index=df.index
        )

    else:
        # Fallback to EMA
        return pd.Series(
            talib.EMA(close.values, timeperiod=length), index=df.index
        )


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class ThreeCommasBotStrategy(BaseStrategy):
    """
    3Commas Bot Strategy

    Entry signals are generated from a dual-MA crossover system.
    Default configuration: EMA(21) vs EMA(50) on 15-minute candles.

    Parameters mirrored from Pine Script defaults:
      - MA Type 1  : EMA
      - MA Length 1: 21
      - MA Type 2  : EMA
      - MA Length 2: 50
      - Swing lookback for stop calculation: 5
      - ATR length: 14
      - Risk adjustment (RiskM): 1.0
      - Reward:Risk ratio (RnR): 1.0

    Broker-only logic NOT translated:
      - strategy.exit stop/limit/trail parameters
      - position sizing (default_qty_value)
      - max drawdown guard (setMaxDrawdown)
      - FLIP (reversal trade) management
      - 3Commas JSON alert messages
    """

    # Configurable parameters (match Pine Script defaults)
    MA_TYPE_1: str = "EMA"
    MA_TYPE_2: str = "EMA"
    MA_LENGTH_1: int = 21
    MA_LENGTH_2: int = 50
    SWING_LOOKBACK: int = 5
    ATR_LENGTH: int = 14
    RISK_M: float = 1.0
    RNR: float = 1.0

    def __init__(self):
        # lookback_hours: 50 bars * 15m = 750 min = 12.5 h → round up to 13 h
        super().__init__(
            name="3Commas Bot",
            description=(
                "Dual moving-average crossover strategy converted from the "
                "3Commas Bot Pine Script. Generates LONG on MA1 cross-above MA2, "
                "SHORT on MA1 cross-below MA2. Default: EMA(21) vs EMA(50) on 15m."
            ),
            timeframe="15m",
            lookback_hours=13,
        )
        # 3× the longest indicator period for EMA/ATR convergence
        self.MIN_CANDLES_REQUIRED = 3 * max(self.MA_LENGTH_1, self.MA_LENGTH_2, self.ATR_LENGTH)

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(SignalType.HOLD, timestamp)

        # --- Moving averages ---
        ma1 = _compute_ma(df, self.MA_TYPE_1, self.MA_LENGTH_1)
        ma2 = _compute_ma(df, self.MA_TYPE_2, self.MA_LENGTH_2)

        # --- ATR (used for stop calculation; not gating signal, but guards NaN) ---
        atr = pd.Series(
            talib.ATR(
                df["high"].values,
                df["low"].values,
                df["close"].values,
                timeperiod=self.ATR_LENGTH,
            ),
            index=df.index,
        )

        # --- Crossover / crossunder detection (vectorized, no lookahead) ---
        # ta.crossover(ma1, ma2)  -> ma1 > ma2 on current bar AND ma1 <= ma2 on previous bar
        ma1_cross_up = (ma1 > ma2) & (ma1.shift(1) <= ma2.shift(1))
        ma1_cross_dn = (ma1 < ma2) & (ma1.shift(1) >= ma2.shift(1))

        # Valid entry requires ATR to be non-NaN (mirrors "not na(atr)" in Pine)
        valid_long = ma1_cross_up & atr.notna()
        valid_short = ma1_cross_dn & atr.notna()

        # --- Evaluate signal on the last completed bar ---
        last_valid_long = valid_long.iloc[-1]
        last_valid_short = valid_short.iloc[-1]

        if last_valid_long:
            return StrategyRecommendation(SignalType.LONG, timestamp)
        elif last_valid_short:
            return StrategyRecommendation(SignalType.SHORT, timestamp)

        return StrategyRecommendation(SignalType.HOLD, timestamp)

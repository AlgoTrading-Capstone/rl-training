"""
Quant Pullback Day Trade Strategy

Pine Script source: input/Quant-Pullback-Day-Trade-Strategy.pine
Timeframe: 15m  |  Lookback: 60 bars (15 h)

Entry logic:
  Long  — bullish trend (close > EMA9 > EMA20), price within pullback zone
          (ATR-based distance to EMA9/EMA20/VWAP), RSI in 40–55 range,
          volume ≤ SMA(vol,20), bullish candle confirmation (close > open
          AND close > prior high).
  Short — bearish trend (close < EMA9 < EMA20), same pullback zone check,
          RSI in 45–60 range, bearish candle confirmation (close < open
          AND close < prior low).

Exit management (swing-based stop, ATR R:R take-profit, EMA trail) is NOT
implemented here — configure in the external execution layer.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType


class QuantPullbackDayTradeStrategy(BaseStrategy):
    """Quant Pullback Day Trade Strategy.

    Trend-following pullback strategy that enters on confirmed pullbacks
    to EMAs/VWAP within an ATR-defined zone, filtered by RSI and volume.

    Default parameters match the Pine Script inputs.
    """

    def __init__(
        self,
        ema_fast_len: int = 9,
        ema_slow_len: int = 20,
        rsi_len: int = 14,
        long_rsi_min: float = 40.0,
        long_rsi_max: float = 55.0,
        short_rsi_min: float = 45.0,
        short_rsi_max: float = 60.0,
        pullback_atr_mult: float = 0.35,
        atr_len: int = 14,
        lookback_swing: int = 5,
        use_volume_filter: bool = True,
        vol_sma_len: int = 20,
        pullback_vol_max_mult: float = 1.0,
        use_vwap: bool = True,
    ):
        super().__init__(
            name="Quant Pullback Day Trade Strategy",
            description=(
                "Trend-following pullback strategy using EMA crossover for trend, "
                "ATR-based pullback zone, RSI band filter, volume filter, and "
                "candle confirmation for entries."
            ),
            timeframe="15m",
            lookback_hours=15,
        )
        self.ema_fast_len = ema_fast_len
        self.ema_slow_len = ema_slow_len
        self.rsi_len = rsi_len
        self.long_rsi_min = long_rsi_min
        self.long_rsi_max = long_rsi_max
        self.short_rsi_min = short_rsi_min
        self.short_rsi_max = short_rsi_max
        self.pullback_atr_mult = pullback_atr_mult
        self.atr_len = atr_len
        self.lookback_swing = lookback_swing
        self.use_volume_filter = use_volume_filter
        self.vol_sma_len = vol_sma_len
        self.pullback_vol_max_mult = pullback_vol_max_mult
        self.use_vwap = use_vwap

        # Dynamic warmup: 3× the longest indicator period
        self.MIN_CANDLES_REQUIRED = 3 * max(
            self.ema_fast_len,
            self.ema_slow_len,
            self.rsi_len,
            self.atr_len,
            self.vol_sma_len,
            self.lookback_swing,
        )

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        open_ = df["open"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)

        # ---- Indicators --------------------------------------------------
        ema_fast = talib.EMA(close, timeperiod=self.ema_fast_len)
        ema_slow = talib.EMA(close, timeperiod=self.ema_slow_len)
        rsi_val = talib.RSI(close, timeperiod=self.rsi_len)
        atr_val = talib.ATR(high, low, close, timeperiod=self.atr_len)
        vol_sma = talib.SMA(volume, timeperiod=self.vol_sma_len)

        # VWAP: cumulative (typical_price * volume) / cumulative(volume)
        if self.use_vwap:
            typical_price = (high + low + close) / 3.0
            cum_tp_vol = np.cumsum(typical_price * volume)
            cum_vol = np.cumsum(volume)
            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                vwap_val = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
        else:
            vwap_val = None

        # ---- Evaluate at the last bar ------------------------------------
        idx = len(close) - 1

        # Guard against NaN indicators
        if (
            np.isnan(ema_fast[idx])
            or np.isnan(ema_slow[idx])
            or np.isnan(rsi_val[idx])
            or np.isnan(atr_val[idx])
        ):
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        c = close[idx]
        h = high[idx]
        l = low[idx]
        o = open_[idx]
        ef = ema_fast[idx]
        es = ema_slow[idx]
        rsi = rsi_val[idx]
        atr = atr_val[idx]

        # ---- Trend conditions --------------------------------------------
        bull_trend = c > ef and ef > es
        bear_trend = c < ef and ef < es

        if self.use_vwap and vwap_val is not None:
            vwap = vwap_val[idx]
            if np.isnan(vwap):
                return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)
            bull_trend = bull_trend and c > vwap
            bear_trend = bear_trend and c < vwap
        else:
            vwap = None

        # ---- Pullback zone -----------------------------------------------
        pullback_dist = atr * self.pullback_atr_mult
        near_ema_fast = abs(c - ef) <= pullback_dist
        near_ema_slow = abs(c - es) <= pullback_dist
        near_vwap = False
        if self.use_vwap and vwap is not None:
            near_vwap = abs(c - vwap) <= pullback_dist

        long_pullback_zone = near_ema_fast or near_ema_slow or near_vwap
        short_pullback_zone = long_pullback_zone

        # ---- RSI filter ---------------------------------------------------
        long_rsi_ok = self.long_rsi_min <= rsi <= self.long_rsi_max
        short_rsi_ok = self.short_rsi_min <= rsi <= self.short_rsi_max

        # ---- Volume filter ------------------------------------------------
        if self.use_volume_filter:
            vs = vol_sma[idx]
            if np.isnan(vs):
                vol_ok = True
            else:
                vol_ok = volume[idx] <= vs * self.pullback_vol_max_mult
        else:
            vol_ok = True

        # ---- Candle confirmation ------------------------------------------
        # bullConfirm = close > open and close > high[1]
        # bearConfirm = close < open and close < low[1]
        if idx < 1:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        bull_confirm = c > o and c > high[idx - 1]
        bear_confirm = c < o and c < low[idx - 1]

        # ---- Entry signals ------------------------------------------------
        long_signal = bull_trend and long_pullback_zone and long_rsi_ok and vol_ok and bull_confirm
        short_signal = bear_trend and short_pullback_zone and short_rsi_ok and vol_ok and bear_confirm

        if long_signal:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        if short_signal:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)
        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

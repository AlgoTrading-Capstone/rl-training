"""
EMA Pullback + ADX + CVD Divergence Strategy

Converted from PineScript v5.
Source: https://www.tradingview.com/script/Q9FXt5N6-EMA-Pullback-ADX-CVD-Divergence/

Logic summary:
  - BUY:  After 9 EMA crosses above 20 EMA, wait for a pullback to either EMA,
          confirm trend strength with ADX > threshold, and require a bullish CVD
          divergence (price lower low + CVD higher low).
  - SELL: After 9 EMA crosses below 20 EMA, wait for a pullback to either EMA,
          confirm trend strength with ADX > threshold, and require a bearish CVD
          divergence (price higher high + CVD lower high).
  - SL/TP: Managed externally by the RL execution engine.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType


class EmaPullbackAdxCvdDivergenceStrategy(BaseStrategy):

    def __init__(self):
        super().__init__(
            name="EmaPullbackAdxCvdDivergenceStrategy",
            description=(
                "EMA Pullback + ADX + CVD Divergence. "
                "Enters long on first pullback to 9/20 EMA after a bullish cross, "
                "with strong ADX and bullish CVD divergence (price LL, CVD HL). "
                "Enters short on first pullback to EMA after a bearish cross, "
                "with strong ADX and bearish CVD divergence (price HH, CVD LH)."
            ),
            timeframe="15m",
            lookback_hours=24,
        )
        self.ema_fast_period = 9
        self.ema_slow_period = 20
        self.adx_len = 14
        self.adx_threshold = 25
        self.divergence_lookback = 10

        # Dynamic warmup: largest indicator window drives convergence
        self.MIN_CANDLES_REQUIRED = 3 * max(
            self.ema_slow_period,
            self.adx_len,
            self.divergence_lookback,
        )

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        df = df.copy()

        # ---- EMA ----
        df['ema9'] = talib.EMA(df['close'].values, timeperiod=self.ema_fast_period)
        df['ema20'] = talib.EMA(df['close'].values, timeperiod=self.ema_slow_period)

        # ---- ADX ----
        df['adx'] = talib.ADX(
            df['high'].values, df['low'].values, df['close'].values,
            timeperiod=self.adx_len,
        )

        # ---- CVD Proxy ----
        # delta: +volume on bullish candle, -volume on bearish, 0 on doji
        df['delta'] = np.where(
            df['close'] > df['open'], df['volume'],
            np.where(df['close'] < df['open'], -df['volume'], 0.0),
        )
        df['cvd'] = df['delta'].cumsum()

        # ---- Divergence Detection ----
        # Bullish: current low breaks the lookback low (shifted by 1 to exclude current bar)
        #          AND current CVD is above the lookback CVD low
        price_ll = df['low'] < df['low'].shift(1).rolling(self.divergence_lookback).min()
        cvd_hl = df['cvd'] > df['cvd'].shift(1).rolling(self.divergence_lookback).min()
        df['bull_div'] = price_ll & cvd_hl

        # Bearish: current high breaks the lookback high AND CVD is below the lookback CVD high
        price_hh = df['high'] > df['high'].shift(1).rolling(self.divergence_lookback).max()
        cvd_lh = df['cvd'] < df['cvd'].shift(1).rolling(self.divergence_lookback).max()
        df['bear_div'] = price_hh & cvd_lh

        # ---- Trend State Machine (waitBuy / waitSell) ----
        # Vectorised forward-fill: state=1 after bullish EMA cross, -1 after bearish cross.
        # Approximates PineScript var bool with per-signal reset (first trade per trend period).
        bull_cross = (df['ema9'] > df['ema20']) & (df['ema9'].shift(1) <= df['ema20'].shift(1))
        bear_cross = (df['ema9'] < df['ema20']) & (df['ema9'].shift(1) >= df['ema20'].shift(1))

        state = pd.Series(np.nan, index=df.index)
        state[bull_cross] = 1.0
        state[bear_cross] = -1.0
        state = state.ffill().fillna(0.0)

        wait_buy = state == 1.0
        wait_sell = state == -1.0

        # ---- Pullback Conditions ----
        pullback_buy = wait_buy & ((df['low'] <= df['ema9']) | (df['low'] <= df['ema20']))
        pullback_sell = wait_sell & ((df['high'] >= df['ema9']) | (df['high'] >= df['ema20']))

        # ---- Strong Trend Filter ----
        strong_trend = df['adx'] > self.adx_threshold

        # ---- Final Entry Signals ----
        df['buy_signal'] = (
            pullback_buy & strong_trend & df['bull_div'] & (df['close'] > df['open'])
        )
        df['sell_signal'] = (
            pullback_sell & strong_trend & df['bear_div'] & (df['close'] < df['open'])
        )

        last = df.iloc[-1]

        if bool(last['buy_signal']):
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        if bool(last['sell_signal']):
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)

        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

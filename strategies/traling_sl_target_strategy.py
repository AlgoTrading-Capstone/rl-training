"""
Trailing SL Target Strategy

Converted from PineScript: Traling.SL.Target by Sharad_Gaikwad
Source: TradingView editors_pick

Entry logic: EMA(20) / EMA(50) crossover system.
- LONG when fast EMA crosses above slow EMA
- SHORT when fast EMA crosses below slow EMA

EXIT LOGIC NOT CONVERTED: The original PineScript implements a significant
trailing stop-loss and trailing target system (both %-based and fixed-point
modes). This exit management — including initial SL/target placement and
iterative trailing adjustments — depends on position state and entry price
tracking that must be configured in the execution layer. The original also
gates entries on strategy.position_size == 0 (no re-entry while in a
position); this condition has been removed as position sizing is external.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType


class TralingSLTargetStrategy(BaseStrategy):

    def __init__(self):
        super().__init__(
            name="TralingSLTarget",
            description="EMA crossover strategy (20/50) with trailing SL/target (exit logic external)",
            timeframe="15m",
            lookback_hours=48,
        )
        self.MIN_CANDLES_REQUIRED = 150  # 3 * 50 (slow EMA period)

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(SignalType.HOLD, timestamp)

        df = df.copy()

        # --- Indicator calculations ---
        fast_len = 20
        slow_len = 50

        df['fast_ema'] = talib.EMA(df['close'].values, timeperiod=fast_len)
        df['slow_ema'] = talib.EMA(df['close'].values, timeperiod=slow_len)

        # --- Crossover detection ---
        df['cross_long'] = (df['fast_ema'] > df['slow_ema']) & (df['fast_ema'].shift(1) <= df['slow_ema'].shift(1))
        df['cross_short'] = (df['fast_ema'] < df['slow_ema']) & (df['fast_ema'].shift(1) >= df['slow_ema'].shift(1))

        # --- Signal from last complete bar ---
        last = df.iloc[-1]

        if last['cross_long']:
            return StrategyRecommendation(SignalType.LONG, timestamp)
        elif last['cross_short']:
            return StrategyRecommendation(SignalType.SHORT, timestamp)

        return StrategyRecommendation(SignalType.HOLD, timestamp)

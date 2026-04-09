from datetime import datetime
import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType
from utils.resampling import resample_to_interval, resampled_merge


class XauusdM5HybridEma915PartialTpRunnerStrategy(BaseStrategy):
    """
    XAUUSD M5 Hybrid EMA 9/15 (Partial TP + Runner)

    Dual EMA trend-following strategy designed for Gold (XAUUSD) on the 5-minute
    timeframe. Uses 9/15 EMA crossover for trend direction with M15 higher-timeframe
    confirmation, filtered by rejection-candle price action. Partial TP and trailing
    runner exits are managed externally by the RL execution engine.

    Entry logic:
    - Long: EMA9 > EMA15 on M5 AND M15, plus a bullish rejection candle
      (long lower wick > 1.5× body, bullish close).
    - Short: EMA9 < EMA15 on M5 AND M15, plus a bearish rejection candle
      (long upper wick > 1.5× body, bearish close).
    """

    def __init__(self):
        super().__init__(
            name="XauusdM5HybridEma915PartialTpRunnerStrategy",
            description=(
                "XAUUSD M5 hybrid EMA 9/15 strategy with M15 HTF confirmation and "
                "rejection-candle entry filter. Partial TP + runner managed externally."
            ),
            timeframe="5m",
            lookback_hours=4,
        )
        self.ema_fast_len = 9
        self.ema_slow_len = 15
        # Dynamic RL warmup: enough M5 bars for the HTF EMAs to converge on M15 data.
        self.MIN_CANDLES_REQUIRED = 3 * max(self.ema_fast_len, self.ema_slow_len)

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # --- M5 EMAs ---
        df['ema_fast'] = talib.EMA(df['close'].values, timeperiod=self.ema_fast_len)
        df['ema_slow'] = talib.EMA(df['close'].values, timeperiod=self.ema_slow_len)

        # --- M15 HTF EMAs (request.security equivalent) ---
        resampled_df = resample_to_interval(df, "15m")
        resampled_df['ema_fast'] = talib.EMA(
            resampled_df['close'].values, timeperiod=self.ema_fast_len
        )
        resampled_df['ema_slow'] = talib.EMA(
            resampled_df['close'].values, timeperiod=self.ema_slow_len
        )
        df = resampled_merge(original=df, resampled=resampled_df, fill_na=True)

        # --- Rejection candle components (vectorized) ---
        body = np.abs(df['close'] - df['open'])
        upper_wick = df['high'] - np.maximum(df['close'], df['open'])
        lower_wick = np.minimum(df['close'], df['open']) - df['low']

        bullish_rejection = (lower_wick > body * 1.5) & (df['close'] > df['open'])
        bearish_rejection = (upper_wick > body * 1.5) & (df['close'] < df['open'])

        # --- Trend conditions (M5 + M15 alignment) ---
        bull_trend = (
            (df['ema_fast'] > df['ema_slow']) &
            (df['resample_15_ema_fast'] > df['resample_15_ema_slow'])
        )
        bear_trend = (
            (df['ema_fast'] < df['ema_slow']) &
            (df['resample_15_ema_fast'] < df['resample_15_ema_slow'])
        )

        buy_condition = bull_trend & bullish_rejection
        sell_condition = bear_trend & bearish_rejection

        # --- Signal on last completed bar ---
        if buy_condition.iloc[-1]:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        if sell_condition.iloc[-1]:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)

        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

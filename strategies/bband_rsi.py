"""
BbandRsi Strategy

Adapted from Freqtrade strategy by Gert Wohlgemuth
Original: https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/BbandRsi.cs

Logic:
    - Calculate RSI(14) on close price
    - Calculate Bollinger Bands (20, 2) on typical price (HLC/3)
    - LONG entry: RSI < 30 AND close < lower band (oversold + price below support)
    - EXIT (FLAT): RSI > 70 (overbought condition)
    - HOLD: Otherwise (no clear signal)

The strategy identifies oversold conditions with confirmation from both RSI and
Bollinger Bands for entries, and exits on overbought RSI conditions.
"""

from datetime import datetime
import pandas as pd
import talib
from strategies.base_strategy import BaseStrategy, SignalType, StrategyRecommendation


class BbandRsi(BaseStrategy):
    """
    Bollinger Bands + RSI mean-reversion strategy.
    Enters long on oversold conditions, exits on overbought signals.
    """

    # RSI(14) needs 14 periods minimum, Bollinger(20) needs 20 periods
    # Use 20 + buffer for safety
    MIN_CANDLES_REQUIRED = 30

    def __init__(self):
        super().__init__(
            name="BbandRsi",
            description="Mean-reversion strategy using RSI and Bollinger Bands on typical price",
            timeframe="1h",
            lookback_hours=40  # 30 candles + 10 hour buffer
        )

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI and Bollinger Bands indicators.

        RSI: 14-period RSI on close price
        Bollinger Bands: 20-period, 2 std dev on typical price (HLC/3)
        """
        df = df.copy()

        # RSI(14) on close
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)

        # Typical price = (High + Low + Close) / 3
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0

        # Bollinger Bands on typical price (20, 2)
        df['bb_upperband'], df['bb_middleband'], df['bb_lowerband'] = talib.BBANDS(
            df['typical_price'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0  # Simple Moving Average
        )

        return df

    def _generate_signal(self, df: pd.DataFrame) -> SignalType:
        """
        Generate trading signal based on RSI and Bollinger Bands.

        Entry Logic (LONG):
            - RSI < 30 (oversold)
            - Close < lower Bollinger Band (price below support)

        Exit Logic (FLAT):
            - RSI > 70 (overbought)

        Otherwise: HOLD
        """
        if len(df) < 1:
            return SignalType.HOLD

        last_row = df.iloc[-1]

        rsi = last_row['rsi']
        close = last_row['close']
        bb_lowerband = last_row['bb_lowerband']

        # Handle NaN values from indicator warmup period
        if pd.isna(rsi) or pd.isna(bb_lowerband):
            return SignalType.HOLD

        # Entry condition: Oversold (RSI < 30) AND price below lower band
        if rsi < 30 and close < bb_lowerband:
            return SignalType.LONG

        # Exit condition: Overbought (RSI > 70)
        if rsi > 70:
            return SignalType.FLAT

        # No clear signal
        return SignalType.HOLD

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        """
        Execute the BbandRsi strategy logic.

        Args:
            df: DataFrame with columns [date, open, high, low, close, volume]
            timestamp: Current evaluation timestamp (UTC)

        Returns:
            StrategyRecommendation with signal and timestamp
        """
        # Validate sufficient data
        if df is None or len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # Calculate indicators
        df = self._calculate_indicators(df)

        # Generate signal
        signal = self._generate_signal(df)

        return StrategyRecommendation(signal=signal, timestamp=timestamp)



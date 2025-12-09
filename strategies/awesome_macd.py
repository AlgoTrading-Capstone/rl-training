"""
Awesome MACD Strategy

Converted from Freqtrade strategy by Gert Wohlgemuth.
Originally converted from: https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/AwesomeMacd.cs

Logic:
    - Uses MACD (Moving Average Convergence Divergence) with default parameters (12, 26, 9)
    - Uses Awesome Oscillator (AO): difference between 5-period and 34-period SMA of median price
    - LONG entry: MACD > 0 AND AO crosses above zero (AO > 0 and previous AO < 0)
    - SHORT entry: MACD < 0 AND AO crosses below zero (AO < 0 and previous AO > 0)
    - Otherwise: HOLD

The strategy looks for momentum alignment between MACD and AO oscillator,
entering on crossovers of the Awesome Oscillator when MACD confirms direction.
"""

from datetime import datetime
import pandas as pd
import talib
from pandas import DataFrame

from strategies.base_strategy import BaseStrategy, SignalType, StrategyRecommendation


class AwesomeMacd(BaseStrategy):
    """
    Awesome MACD Strategy for Bitcoin futures trading.
    Combines MACD momentum indicator with Awesome Oscillator crossovers.
    """

    # AO needs 34-period SMA + MACD needs 26-period EMA + buffer for previous value check
    # 34 periods + 26 periods + 10 buffer = 70 candles minimum
    MIN_CANDLES_REQUIRED = 70

    def __init__(self):
        super().__init__(
            name="AwesomeMacd",
            description="Momentum strategy combining MACD and Awesome Oscillator crossovers.",
            timeframe="1h",
            lookback_hours=100  # 70 candles + extra buffer for safety
        )

    def _calculate_awesome_oscillator(self, df: DataFrame) -> pd.Series:
        """
        Calculate Awesome Oscillator manually.

        AO = SMA(median price, 5) - SMA(median price, 34)
        where median price = (high + low) / 2

        :param df: DataFrame with 'high' and 'low' columns
        :return: Series containing AO values
        """
        median_price = (df['high'] + df['low']) / 2.0

        # Calculate SMAs using TA-Lib
        sma_fast = talib.SMA(median_price, timeperiod=5)
        sma_slow = talib.SMA(median_price, timeperiod=34)

        # Awesome Oscillator is the difference
        ao = sma_fast - sma_slow

        return ao

    def _calculate_indicators(self, df: DataFrame) -> DataFrame:
        """
        Calculate all required indicators:
        - ADX (Average Directional Index) with period 14 - not used in signals but calculated
        - Awesome Oscillator (AO)
        - MACD with default parameters (12, 26, 9)
        """
        # ADX - calculated but not used in original entry/exit logic
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # Awesome Oscillator
        df['ao'] = self._calculate_awesome_oscillator(df)

        # MACD (returns tuple of macd, macdsignal, macdhist)
        macd, macdsignal, macdhist = talib.MACD(
            df['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        df['macd'] = macd
        df['macdsignal'] = macdsignal
        df['macdhist'] = macdhist

        return df

    def _generate_signal(self, df: DataFrame) -> SignalType:
        """
        Generate trading signal based on MACD and AO crossover logic.

        LONG entry conditions (from populate_entry_trend):
            - MACD > 0
            - AO > 0
            - Previous AO < 0 (AO crossing above zero)

        SHORT entry conditions (from populate_exit_trend, converted to SHORT):
            - MACD < 0
            - AO < 0
            - Previous AO > 0 (AO crossing below zero)

        Otherwise: HOLD
        """
        # Need at least 2 rows to check previous AO value
        if len(df) < 2:
            return SignalType.HOLD

        # Get current and previous row values
        current = df.iloc[-1]
        previous = df.iloc[-2]

        macd = current['macd']
        ao = current['ao']
        ao_prev = previous['ao']

        # Check for NaN values (can occur during indicator warmup period)
        if pd.isna(macd) or pd.isna(ao) or pd.isna(ao_prev):
            return SignalType.HOLD

        # LONG entry: MACD positive, AO crosses above zero
        if macd > 0 and ao > 0 and ao_prev < 0:
            return SignalType.LONG

        # SHORT entry: MACD negative, AO crosses below zero
        # Note: Original strategy only had exit_long, converted to SHORT for bidirectional trading
        if macd < 0 and ao < 0 and ao_prev > 0:
            return SignalType.SHORT

        # No clear signal - hold current position
        return SignalType.HOLD

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        """
        Execute the AwesomeMacd strategy logic.

        :param df: DataFrame with OHLCV data (columns: date, open, high, low, close, volume)
        :param timestamp: Current evaluation timestamp (UTC)
        :return: StrategyRecommendation with signal and timestamp
        """
        # Validate sufficient data
        if df is None or len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # Work on a copy to avoid modifying original DataFrame
        df = df.copy()

        # Calculate all indicators
        df = self._calculate_indicators(df)

        # Generate trading signal
        signal = self._generate_signal(df)

        return StrategyRecommendation(signal=signal, timestamp=timestamp)


# Allow direct execution for testing
if __name__ == "__main__":
    print("AwesomeMacd strategy module loaded successfully.")
    print("Strategy uses MACD and Awesome Oscillator crossovers for momentum trading.")
    print("Timeframe: 1h, Lookback: 100 hours")

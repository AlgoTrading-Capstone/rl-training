"""
Supertrend Strategy

Adapted from Freqtrade Supertrend strategy by @juankysoriano.

Logic:
    - Calculate 3 Supertrend indicators for LONG (BUY params) and SHORT (SELL params)
    - LONG when all 3 BUY indicators are 'up' AND volume > 0
    - SHORT when all 3 SELL indicators are 'down' AND volume > 0
    - Uses hyperopt-optimized parameters

Supertrend implementation from: https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/futures/FSupertrendStrategy.py
Supertend pinescript reference: https://www.tradingview.com/script/UJpKX1gG-SuperTrend-Multitimeframe/
"""

from datetime import datetime

import pandas as pd
import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from strategies.base_strategy import (BaseStrategy, SignalType, StrategyRecommendation)


class SupertrendStrategy(BaseStrategy):
    """
    Supertrend Strategy for Bitcoin trading.
    Uses 3 Supertrend indicators with optimized parameters for both LONG and SHORT signals.
    """

    # Supertrend needs period candles to initialize, longest period is 9
    # Original startup_candle_count was 199, using conservative 100
    MIN_CANDLES_REQUIRED = 100

    # Hyperopt-optimized BUY parameters for LONG signals
    BUY_M1 = 4
    BUY_M2 = 7
    BUY_M3 = 1
    BUY_P1 = 8
    BUY_P2 = 9
    BUY_P3 = 8

    # Hyperopt-optimized SELL parameters for SHORT signals
    #  These are separate from buy params for proper SHORT signal generation
    SELL_M1 = 1
    SELL_M2 = 2
    SELL_M3 = 4
    SELL_P1 = 8
    SELL_P2 = 9
    SELL_P3 = 8

    def __init__(self):
        super().__init__(
            name="SupertrendStrategy",
            description="Triple Supertrend strategy with LONG/SHORT signals using hyperopt-optimized parameters.",
            timeframe="1h",
            lookback_hours=124  # 100 candles + 24h buffer for 1h timeframe
        )

    def _supertrend_vectorized(self, df: DataFrame, multiplier: int, period: int) -> DataFrame:
        """
        Calculate Supertrend indicator using vectorized NumPy operations.

        This is FAST (O(N)) and mathematically equivalent to the original iterative version.
        Uses .values arrays instead of .iloc[] for performance.

        Args:
            df: DataFrame with OHLCV columns
            multiplier: ATR multiplier for bands
            period: ATR period

        Returns:
            DataFrame with ST and STX columns
        """
        df = df.copy()

        # Calculate ATR using TA-Lib
        df['TR'] = ta.TRANGE(df)
        df['ATR'] = ta.SMA(df['TR'], period)

        # Calculate basic bands
        hl2 = (df['high'] + df['low']) / 2
        df['basic_ub'] = hl2 + multiplier * df['ATR']
        df['basic_lb'] = hl2 - multiplier * df['ATR']

        # Initialize final bands as numpy arrays for vectorized operations
        basic_ub = df['basic_ub'].values
        basic_lb = df['basic_lb'].values
        close = df['close'].values

        final_ub = np.zeros(len(df))
        final_lb = np.zeros(len(df))

        # Vectorized calculation of final bands
        # Note: Still need a loop but using numpy arrays is much faster than .iloc[]
        for i in range(period, len(df)):
            # Final upper band logic
            if (basic_ub[i] < final_ub[i-1]) or (close[i-1] > final_ub[i-1]):
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]

            # Final lower band logic
            if (basic_lb[i] > final_lb[i-1]) or (close[i-1] < final_lb[i-1]):
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]

        df['final_ub'] = final_ub
        df['final_lb'] = final_lb

        # Calculate Supertrend using vectorized operations
        st = np.zeros(len(df))

        for i in range(period, len(df)):
            # Nested ternary logic from original (using numpy arrays)
            if i == period:
                st[i] = final_ub[i]
            else:
                if st[i-1] == final_ub[i-1] and close[i] <= final_ub[i]:
                    st[i] = final_ub[i]
                elif st[i-1] == final_ub[i-1] and close[i] > final_ub[i]:
                    st[i] = final_lb[i]
                elif st[i-1] == final_lb[i-1] and close[i] >= final_lb[i]:
                    st[i] = final_lb[i]
                elif st[i-1] == final_lb[i-1] and close[i] < final_lb[i]:
                    st[i] = final_ub[i]
                else:
                    st[i] = 0.0

        # Calculate trend direction vectorized
        stx = np.where(st > 0, np.where(close < st, 'down', 'up'), None)

        return DataFrame(index=df.index, data={
            'ST': st,
            'STX': stx
        })

    def _calculate_supertrend(self, df: DataFrame) -> DataFrame:
        """
        Calculate all 3 BUY supertrend indicators with simplified column names.

        This method is primarily used for testing to provide a clean interface.
        Returns DataFrame with columns: st1, st1x, st2, st2x, st3, st3x

        Note: Only calculates BUY indicators for backward compatibility with existing tests.
        For full LONG/SHORT logic, use _calculate_indicators().
        """
        df = df.copy()

        # Calculate 3 Supertrend indicators using BUY parameters
        result1 = self._supertrend_vectorized(df, self.BUY_M1, self.BUY_P1)
        df['st1'] = result1['ST']
        df['st1x'] = result1['STX']

        result2 = self._supertrend_vectorized(df, self.BUY_M2, self.BUY_P2)
        df['st2'] = result2['ST']
        df['st2x'] = result2['STX']

        result3 = self._supertrend_vectorized(df, self.BUY_M3, self.BUY_P3)
        df['st3'] = result3['ST']
        df['st3x'] = result3['STX']

        return df

    def _calculate_indicators(self, df: DataFrame) -> DataFrame:
        """
        Reproduce populate_indicators() logic from Freqtrade version.

        Calculates 3 BUY indicators (for LONG signals) and 3 SELL indicators (for SHORT signals)
        using hyperopt-optimized parameters.

        Also creates simplified column names (st1, st1x, etc.) for testing compatibility.
        """
        # Calculate 3 Supertrend indicators for LONG signals (BUY parameters)
        result1 = self._supertrend_vectorized(df, self.BUY_M1, self.BUY_P1)
        df[f'supertrend_1_buy_{self.BUY_M1}_{self.BUY_P1}'] = result1['STX']
        # Add simplified names for testing
        df['st1'] = result1['ST']
        df['st1x'] = result1['STX']

        result2 = self._supertrend_vectorized(df, self.BUY_M2, self.BUY_P2)
        df[f'supertrend_2_buy_{self.BUY_M2}_{self.BUY_P2}'] = result2['STX']
        df['st2'] = result2['ST']
        df['st2x'] = result2['STX']

        result3 = self._supertrend_vectorized(df, self.BUY_M3, self.BUY_P3)
        df[f'supertrend_3_buy_{self.BUY_M3}_{self.BUY_P3}'] = result3['STX']
        df['st3'] = result3['ST']
        df['st3x'] = result3['STX']

        # Calculate 3 Supertrend indicators for SHORT signals (SELL parameters)
        sell_result1 = self._supertrend_vectorized(df, self.SELL_M1, self.SELL_P1)
        df[f'supertrend_1_sell_{self.SELL_M1}_{self.SELL_P1}'] = sell_result1['STX']

        sell_result2 = self._supertrend_vectorized(df, self.SELL_M2, self.SELL_P2)
        df[f'supertrend_2_sell_{self.SELL_M2}_{self.SELL_P2}'] = sell_result2['STX']

        sell_result3 = self._supertrend_vectorized(df, self.SELL_M3, self.SELL_P3)
        df[f'supertrend_3_sell_{self.SELL_M3}_{self.SELL_P3}'] = sell_result3['STX']

        return df

    def _generate_signal(self, df: DataFrame) -> SignalType:
        """
        Generate trading signals based on Supertrend indicators.

        LONG: All 3 BUY indicators are 'up' AND volume > 0
        SHORT: All 3 SELL indicators are 'down' AND volume > 0
        Otherwise: HOLD
        """
        if len(df) < 1:
            return SignalType.HOLD

        last_row = df.iloc[-1]

        # Get BUY indicator columns
        buy_st1_col = f'supertrend_1_buy_{self.BUY_M1}_{self.BUY_P1}'
        buy_st2_col = f'supertrend_2_buy_{self.BUY_M2}_{self.BUY_P2}'
        buy_st3_col = f'supertrend_3_buy_{self.BUY_M3}_{self.BUY_P3}'

        # Get SELL indicator columns
        sell_st1_col = f'supertrend_1_sell_{self.SELL_M1}_{self.SELL_P1}'
        sell_st2_col = f'supertrend_2_sell_{self.SELL_M2}_{self.SELL_P2}'
        sell_st3_col = f'supertrend_3_sell_{self.SELL_M3}_{self.SELL_P3}'

        # Check if columns exist
        buy_cols_exist = (buy_st1_col in df.columns and
                         buy_st2_col in df.columns and
                         buy_st3_col in df.columns)
        sell_cols_exist = (sell_st1_col in df.columns and
                          sell_st2_col in df.columns and
                          sell_st3_col in df.columns)

        if not (buy_cols_exist and sell_cols_exist):
            return SignalType.HOLD

        # Get values
        volume = last_row['volume']

        buy_st1 = last_row[buy_st1_col]
        buy_st2 = last_row[buy_st2_col]
        buy_st3 = last_row[buy_st3_col]

        sell_st1 = last_row[sell_st1_col]
        sell_st2 = last_row[sell_st2_col]
        sell_st3 = last_row[sell_st3_col]

        # Check for NaN
        if (pd.isna(buy_st1) or pd.isna(buy_st2) or pd.isna(buy_st3) or
            pd.isna(sell_st1) or pd.isna(sell_st2) or pd.isna(sell_st3) or
            pd.isna(volume)):
            return SignalType.HOLD

        # LONG signal: All 3 BUY indicators are 'up' AND volume > 0
        if (buy_st1 == 'up' and
            buy_st2 == 'up' and
            buy_st3 == 'up' and
            volume > 0):
            return SignalType.LONG

        # SHORT signal: All 3 SELL indicators are 'down' AND volume > 0
        if (sell_st1 == 'down' and
            sell_st2 == 'down' and
            sell_st3 == 'down' and
            volume > 0):
            return SignalType.SHORT

        return SignalType.HOLD

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        """Execute the Supertrend Strategy logic."""

        if df is None or len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        df = df.copy()
        df = self._calculate_indicators(df)

        signal = self._generate_signal(df)

        return StrategyRecommendation(signal=signal, timestamp=timestamp)
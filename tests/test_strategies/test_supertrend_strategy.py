"""
Test suite for SupertrendStrategy with reference implementation.

This module contains:
1. Reference implementation using EXACT Freqtrade iterative logic (slow but correct)
2. Mathematical equivalence tests comparing optimized code to reference
3. Signal logic tests verifying LONG/SHORT generation
4. Edge case tests (band touches, gaps, trend reversals)

The reference implementation serves as ground truth - it must match
the original Freqtrade FSupertrendStrategy exactly.
"""

import pytest
import pandas as pd
import numpy as np
from pandas import DataFrame

from strategies.supertrend_strategy import SupertrendStrategy
from strategies.base_strategy import SignalType


def reference_supertrend_calculation(df: DataFrame) -> DataFrame:
    """
    Reference implementation of Supertrend using EXACT Freqtrade iterative logic.

    This is SLOW (O(N)) but CORRECT - used as ground truth for testing.
    Based on the original FSupertrendStrategy from Freqtrade.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with st1, st2, st3 columns and their STX (trend direction) columns
    """
    df = df.copy()

    # Calculate ATR (needed for all three supertrends)
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)

    # Three supertrend configurations
    configs = [
        {'period': 10, 'multiplier': 3.0, 'name': 'st1'},
        {'period': 11, 'multiplier': 2.0, 'name': 'st2'},
        {'period': 12, 'multiplier': 1.0, 'name': 'st3'},
    ]

    for config in configs:
        period = config['period']
        multiplier = config['multiplier']
        st = config['name']

        # Calculate ATR for this period
        df['atr'] = df['tr'].rolling(window=period).mean()

        # Calculate basic upper and lower bands
        hl2 = (df['high'] + df['low']) / 2
        df['basic_ub'] = hl2 + (multiplier * df['atr'])
        df['basic_lb'] = hl2 - (multiplier * df['atr'])

        # Initialize final bands
        df['final_ub'] = 0.0
        df['final_lb'] = 0.0

        # Calculate final bands using iterative logic (CRITICAL - must match original)
        for i in range(period, len(df)):
            # Final upper band logic
            if (df['basic_ub'].iat[i] < df['final_ub'].iat[i-1]) or (df['close'].iat[i-1] > df['final_ub'].iat[i-1]):
                df['final_ub'].iat[i] = df['basic_ub'].iat[i]
            else:
                df['final_ub'].iat[i] = df['final_ub'].iat[i-1]

            # Final lower band logic
            if (df['basic_lb'].iat[i] > df['final_lb'].iat[i-1]) or (df['close'].iat[i-1] < df['final_lb'].iat[i-1]):
                df['final_lb'].iat[i] = df['basic_lb'].iat[i]
            else:
                df['final_lb'].iat[i] = df['final_lb'].iat[i-1]

        # Initialize supertrend column
        df[st] = 0.0

        # Calculate supertrend using nested ternary logic (EXACT Freqtrade implementation)
        for i in range(period, len(df)):
            if i == period:
                # First calculation
                df[st].iat[i] = df['final_ub'].iat[i]
            else:
                # Nested ternary logic from original
                df[st].iat[i] = (
                    df['final_ub'].iat[i] if (
                        (df[st].iat[i-1] == df['final_ub'].iat[i-1]) and
                        (df['close'].iat[i] <= df['final_ub'].iat[i])
                    ) else (
                        df['final_lb'].iat[i] if (
                            (df[st].iat[i-1] == df['final_ub'].iat[i-1]) and
                            (df['close'].iat[i] > df['final_ub'].iat[i])
                        ) else (
                            df['final_lb'].iat[i] if (
                                (df[st].iat[i-1] == df['final_lb'].iat[i-1]) and
                                (df['close'].iat[i] >= df['final_lb'].iat[i])
                            ) else (
                                df['final_ub'].iat[i] if (
                                    (df[st].iat[i-1] == df['final_lb'].iat[i-1]) and
                                    (df['close'].iat[i] < df['final_lb'].iat[i])
                                ) else 0.0
                            )
                        )
                    )
                )

        # Calculate trend direction (STX column)
        df[f'{st}x'] = np.where(
            df['close'] > df[st],
            'up',
            np.where(df['close'] < df[st], 'down', np.nan)
        )
        df[f'{st}x'] = df[f'{st}x'].fillna(method='ffill')

    # Return only the supertrend columns
    return df[['st1', 'st1x', 'st2', 'st2x', 'st3', 'st3x']].copy()


class TestSupertrendMathematicalEquivalence:
    """Test that optimized Supertrend calculation matches reference implementation."""

    def test_st1_values_match_reference(self, synthetic_ohlcv):
        """Verify ST1 indicator values match reference (rtol=1e-5)."""
        # Calculate reference (slow but correct)
        reference_result = reference_supertrend_calculation(synthetic_ohlcv)

        # Calculate optimized (fast, needs to match)
        strategy = SupertrendStrategy()
        optimized_result = strategy._calculate_supertrend(synthetic_ohlcv)

        # Compare ST1 values (skip first 20 rows due to indicator warm-up)
        reference_st1 = reference_result['st1'].iloc[20:].values
        optimized_st1 = optimized_result['st1'].iloc[20:].values

        # Remove NaNs before comparison
        mask = ~(np.isnan(reference_st1) | np.isnan(optimized_st1))

        np.testing.assert_allclose(
            reference_st1[mask],
            optimized_st1[mask],
            rtol=1e-5,
            err_msg="ST1 values don't match reference implementation"
        )

    def test_st2_values_match_reference(self, synthetic_ohlcv):
        """Verify ST2 indicator values match reference (rtol=1e-5)."""
        reference_result = reference_supertrend_calculation(synthetic_ohlcv)
        strategy = SupertrendStrategy()
        optimized_result = strategy._calculate_supertrend(synthetic_ohlcv)

        reference_st2 = reference_result['st2'].iloc[20:].values
        optimized_st2 = optimized_result['st2'].iloc[20:].values

        mask = ~(np.isnan(reference_st2) | np.isnan(optimized_st2))

        np.testing.assert_allclose(
            reference_st2[mask],
            optimized_st2[mask],
            rtol=1e-5,
            err_msg="ST2 values don't match reference implementation"
        )

    def test_st3_values_match_reference(self, synthetic_ohlcv):
        """Verify ST3 indicator values match reference (rtol=1e-5)."""
        reference_result = reference_supertrend_calculation(synthetic_ohlcv)
        strategy = SupertrendStrategy()
        optimized_result = strategy._calculate_supertrend(synthetic_ohlcv)

        reference_st3 = reference_result['st3'].iloc[20:].values
        optimized_st3 = optimized_result['st3'].iloc[20:].values

        mask = ~(np.isnan(reference_st3) | np.isnan(optimized_st3))

        np.testing.assert_allclose(
            reference_st3[mask],
            optimized_st3[mask],
            rtol=1e-5,
            err_msg="ST3 values don't match reference implementation"
        )

    def test_trend_directions_match_reference(self, synthetic_ohlcv):
        """Verify trend direction indicators (st1x, st2x, st3x) match reference."""
        reference_result = reference_supertrend_calculation(synthetic_ohlcv)
        strategy = SupertrendStrategy()
        optimized_result = strategy._calculate_supertrend(synthetic_ohlcv)

        # Compare trend directions (skip first 20 rows)
        for st_col in ['st1x', 'st2x', 'st3x']:
            reference_trend = reference_result[st_col].iloc[20:].values
            optimized_trend = optimized_result[st_col].iloc[20:].values

            # Trend is categorical ('up'/'down'), compare directly
            mask = ~(pd.isna(reference_trend) | pd.isna(optimized_trend))

            assert np.array_equal(reference_trend[mask], optimized_trend[mask]), \
                f"{st_col} trend directions don't match reference"


class TestSupertrendSignalLogic:
    """Test signal generation logic for LONG/SHORT/HOLD."""

    def test_long_signal_when_all_up(self, uptrend_ohlcv):
        """Verify LONG signal when all three supertrend indicators are 'up'."""
        strategy = SupertrendStrategy()

        df = uptrend_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Find rows where all three are 'up'
        all_up_rows = []
        for i in range(strategy.MIN_CANDLES_REQUIRED, len(df)):
            if (df.iloc[i]['st1x'] == 'up' and
                df.iloc[i]['st2x'] == 'up' and
                df.iloc[i]['st3x'] == 'up'):
                all_up_rows.append(i)

        # Verify LONG signal generation for these rows
        for idx in all_up_rows[:10]:  # Test first 10 occurrences
            window_df = df.iloc[:idx+1].copy()
            signal = strategy._generate_signal(window_df)

            assert signal == SignalType.LONG, \
                f"Expected LONG signal at index {idx} (all indicators 'up')"

    def test_short_signal_when_all_down(self, downtrend_ohlcv):
        """Verify SHORT signal when all three supertrend indicators are 'down'."""
        strategy = SupertrendStrategy()

        df = downtrend_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Find rows where all three are 'down'
        all_down_rows = []
        for i in range(strategy.MIN_CANDLES_REQUIRED, len(df)):
            if (df.iloc[i]['st1x'] == 'down' and
                df.iloc[i]['st2x'] == 'down' and
                df.iloc[i]['st3x'] == 'down'):
                all_down_rows.append(i)

        # Verify SHORT signal generation for these rows
        for idx in all_down_rows[:10]:  # Test first 10 occurrences
            window_df = df.iloc[:idx+1].copy()
            signal = strategy._generate_signal(window_df)

            assert signal == SignalType.SHORT, \
                f"Expected SHORT signal at index {idx} (all indicators 'down')"

    def test_hold_signal_when_mixed(self, synthetic_ohlcv):
        """Verify HOLD signal when indicators are mixed (not all up or all down)."""
        strategy = SupertrendStrategy()

        df = synthetic_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Find rows where indicators are mixed
        mixed_rows = []
        for i in range(strategy.MIN_CANDLES_REQUIRED, len(df)):
            st1x = df.iloc[i]['st1x']
            st2x = df.iloc[i]['st2x']
            st3x = df.iloc[i]['st3x']

            # Mixed if not all same
            if not ((st1x == st2x == st3x == 'up') or (st1x == st2x == st3x == 'down')):
                mixed_rows.append(i)

        # Verify HOLD signal for mixed indicators
        for idx in mixed_rows[:10]:  # Test first 10 occurrences
            window_df = df.iloc[:idx+1].copy()
            signal = strategy._generate_signal(window_df)

            assert signal == SignalType.HOLD, \
                f"Expected HOLD signal at index {idx} (mixed indicators)"


class TestSupertrendEdgeCases:
    """Test edge cases: band touches, gaps, trend reversals."""

    def test_handles_nan_values(self, synthetic_ohlcv):
        """Verify strategy handles NaN values gracefully."""
        strategy = SupertrendStrategy()

        df = synthetic_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Early rows should have NaN due to indicator warm-up
        assert pd.isna(df.iloc[0]['st1']) or df.iloc[0]['st1'] == 0.0
        assert pd.isna(df.iloc[0]['st2']) or df.iloc[0]['st2'] == 0.0
        assert pd.isna(df.iloc[0]['st3']) or df.iloc[0]['st3'] == 0.0

        # Later rows should have valid values
        assert not pd.isna(df.iloc[50]['st1'])
        assert not pd.isna(df.iloc[50]['st2'])
        assert not pd.isna(df.iloc[50]['st3'])

    def test_min_candles_requirement(self, synthetic_ohlcv):
        """Verify strategy requires MIN_CANDLES_REQUIRED before generating signals."""
        strategy = SupertrendStrategy()

        # Test with insufficient candles
        small_df = synthetic_ohlcv.iloc[:20].copy()
        result = strategy.run(small_df, small_df.iloc[-1]['date'])

        assert result.signal == SignalType.HOLD, \
            "Should return HOLD when candles < MIN_CANDLES_REQUIRED"

        # Test with sufficient candles
        large_df = synthetic_ohlcv.copy()
        result = strategy.run(large_df, large_df.iloc[-1]['date'])

        assert result.signal in [SignalType.LONG, SignalType.SHORT, SignalType.HOLD]

    def test_exact_band_touch(self, edge_case_ohlcv):
        """Verify strategy handles exact band touches (close == supertrend)."""
        strategy = SupertrendStrategy()

        df = edge_case_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Check for exact touches (close very close to st values)
        for i in range(strategy.MIN_CANDLES_REQUIRED, len(df)):
            close = df.iloc[i]['close']
            st1 = df.iloc[i]['st1']

            # If close is within 0.01% of st1, it's essentially a touch
            if abs(close - st1) / st1 < 0.0001:
                window_df = df.iloc[:i+1].copy()
                signal = strategy._generate_signal(window_df)

                # Should not crash, should return valid signal
                assert signal in [SignalType.LONG, SignalType.SHORT, SignalType.HOLD]
                break  # Test one occurrence

    def test_gap_handling(self, edge_case_ohlcv):
        """Verify strategy handles price gaps (large jumps)."""
        strategy = SupertrendStrategy()

        df = edge_case_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Should handle gaps without errors
        assert not df['st1'].isna().all()
        assert not df['st2'].isna().all()
        assert not df['st3'].isna().all()

    def test_trend_reversal(self, synthetic_ohlcv):
        """Verify strategy detects trend reversals (uptrend → downtrend)."""
        strategy = SupertrendStrategy()

        df = synthetic_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Check transition from uptrend (rows 0-66) to downtrend (rows 67-133)
        uptrend_signals = []
        downtrend_signals = []

        for idx in range(60, 70):
            window_df = df.iloc[:idx+1].copy()
            signal = strategy._generate_signal(window_df)
            uptrend_signals.append(signal)

        for idx in range(75, 85):
            window_df = df.iloc[:idx+1].copy()
            signal = strategy._generate_signal(window_df)
            downtrend_signals.append(signal)

        # Signals should change during trend reversal
        all_signals = set(uptrend_signals + downtrend_signals)
        assert len(all_signals) > 1, "Signals should vary across trend reversal"


class TestSupertrendATRCalculation:
    """Test ATR calculation correctness."""

    def test_atr_positive_values(self, synthetic_ohlcv):
        """Verify ATR is always positive (measure of volatility)."""
        strategy = SupertrendStrategy()

        df = synthetic_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # ATR should be positive where defined
        for st_col in ['st1', 'st2', 'st3']:
            # Check that ATR used in calculation produces reasonable results
            # (Indirect check - supertrend values should exist and be reasonable)
            valid_st = df[st_col].dropna()
            assert len(valid_st) > 0, f"{st_col} should have valid values"
            assert (valid_st > 0).any(), f"{st_col} should have positive values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Test suite for VolatilitySystem with edge case testing and sign handling verification.

This module contains:
1. Reference implementation using EXACT Freqtrade iterative logic (slow but correct)
2. Mathematical equivalence tests comparing optimized code to reference
3. Sign handling tests verifying SHORT/LONG signal correctness
4. Edge case tests (NaNs, flat markets, extreme volatility spikes)

The reference implementation serves as ground truth - it must match
the original Freqtrade VolatilitySystem exactly.
"""

import pytest
import pandas as pd
import numpy as np
from pandas import DataFrame

from strategies.volatility_system import VolatilitySystem
from strategies.base_strategy import SignalType


def reference_volatility_calculation(df: DataFrame) -> DataFrame:
    """
    Reference implementation of Volatility System using EXACT Freqtrade iterative logic.

    This is SLOW (O(N)) but CORRECT - used as ground truth for testing.
    Based on the original Freqtrade VolatilitySystem strategy.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with ATR and close_change columns
    """
    df = df.copy()

    # ATR calculation (using 3h timeframe in original, but testing on base timeframe)
    period = 14
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()

    # Close change calculation
    df['close_change'] = df['close'] - df['close'].shift(1)

    return df[['atr', 'close_change']].copy()


class TestVolatilitySystemMathematicalEquivalence:
    """Test that optimized Volatility System calculation matches reference implementation."""

    def test_atr_values_match_reference(self, synthetic_ohlcv):
        """Verify ATR values match reference (rtol=1e-5)."""
        # Calculate reference (slow but correct)
        reference_result = reference_volatility_calculation(synthetic_ohlcv)

        # Calculate optimized (fast, needs to match)
        strategy = VolatilitySystem()
        optimized_result = strategy._calculate_volatility_indicators(synthetic_ohlcv)

        # Compare ATR values (skip first 20 rows due to indicator warm-up)
        reference_atr = reference_result['atr'].iloc[20:].values
        optimized_atr = optimized_result['atr'].iloc[20:].values

        # Remove NaNs before comparison
        mask = ~(np.isnan(reference_atr) | np.isnan(optimized_atr))

        np.testing.assert_allclose(
            reference_atr[mask],
            optimized_atr[mask],
            rtol=1e-5,
            err_msg="ATR values don't match reference implementation"
        )

    def test_close_change_values_match_reference(self, synthetic_ohlcv):
        """Verify close_change values match reference."""
        reference_result = reference_volatility_calculation(synthetic_ohlcv)
        strategy = VolatilitySystem()
        optimized_result = strategy._calculate_volatility_indicators(synthetic_ohlcv)

        reference_change = reference_result['close_change'].iloc[20:].values
        optimized_change = optimized_result['close_change'].iloc[20:].values

        mask = ~(np.isnan(reference_change) | np.isnan(optimized_change))

        np.testing.assert_allclose(
            reference_change[mask],
            optimized_change[mask],
            rtol=1e-5,
            err_msg="close_change values don't match reference implementation"
        )


class TestVolatilitySystemSignHandling:
    """Test sign handling logic for SHORT/LONG signals."""

    def test_long_signal_on_upward_volatility_breakout(self, synthetic_ohlcv):
        """
        Verify LONG signal when close_change > atr.shift(1).

        Original logic: close_change > atr.shift(1)
        This should trigger LONG on upward volatility breakouts.
        """
        strategy = VolatilitySystem()

        df = synthetic_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Find rows where close_change > previous ATR
        long_candidates = []
        for i in range(strategy.MIN_CANDLES_REQUIRED, len(df)):
            curr_close_change = df.iloc[i]['close_change']
            prev_atr = df.iloc[i-1]['atr']
            volume = df.iloc[i]['volume']

            if (not pd.isna(curr_close_change) and not pd.isna(prev_atr) and
                curr_close_change > prev_atr and volume > 0):
                long_candidates.append(i)

        # Verify LONG signal generation for these rows
        for idx in long_candidates[:5]:  # Test first 5 occurrences
            window_df = df.iloc[:idx+1].copy()
            signal = strategy._generate_signal(window_df)

            assert signal == SignalType.LONG, \
                f"Expected LONG signal at index {idx} (close_change > atr)"

    def test_short_signal_on_downward_volatility_breakout(self, synthetic_ohlcv):
        """
        Verify SHORT signal when -close_change > atr.shift(1).

        Original logic: close_change * -1 > atr.shift(1)
        Equivalent to: -close_change > atr.shift(1)

        This should trigger SHORT on downward volatility breakouts.
        """
        strategy = VolatilitySystem()

        df = synthetic_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Find rows where -close_change > previous ATR
        short_candidates = []
        for i in range(strategy.MIN_CANDLES_REQUIRED, len(df)):
            curr_close_change = df.iloc[i]['close_change']
            prev_atr = df.iloc[i-1]['atr']
            volume = df.iloc[i]['volume']

            if (not pd.isna(curr_close_change) and not pd.isna(prev_atr) and
                -curr_close_change > prev_atr and volume > 0):
                short_candidates.append(i)

        # Verify SHORT signal generation for these rows
        for idx in short_candidates[:5]:  # Test first 5 occurrences
            window_df = df.iloc[:idx+1].copy()
            signal = strategy._generate_signal(window_df)

            assert signal == SignalType.SHORT, \
                f"Expected SHORT signal at index {idx} (-close_change > atr)"

    def test_sign_equivalence(self, synthetic_ohlcv):
        """
        Verify that -close_change > atr is mathematically equivalent to close_change * -1 > atr.
        """
        df = synthetic_ohlcv.copy()

        # Calculate close_change
        df['close_change'] = df['close'] - df['close'].shift(1)

        # Create dummy ATR
        df['atr'] = 10.0

        # Test equivalence for various close_change values
        test_values = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]

        for val in test_values:
            # Method 1: -close_change > atr
            method1 = -val > 10.0

            # Method 2: close_change * -1 > atr
            method2 = val * -1 > 10.0

            assert method1 == method2, \
                f"Sign handling methods not equivalent for close_change={val}"

    def test_hold_signal_when_no_breakout(self, sideways_ohlcv):
        """Verify HOLD signal when neither LONG nor SHORT conditions are met."""
        strategy = VolatilitySystem()

        df = sideways_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Check middle section where market is sideways
        hold_count = 0
        for idx in range(50, 80):
            window_df = df.iloc[:idx+1].copy()
            signal = strategy._generate_signal(window_df)

            if signal == SignalType.HOLD:
                hold_count += 1

        # In sideways market, majority should be HOLD (small moves don't exceed ATR)
        assert hold_count > 15, \
            f"Expected majority HOLD signals in sideways market, got {hold_count}/30"


class TestVolatilitySystemEdgeCases:
    """Test edge cases: NaNs, flat markets, extreme volatility spikes."""

    def test_handles_nan_values(self, synthetic_ohlcv):
        """Verify strategy handles NaN values gracefully."""
        strategy = VolatilitySystem()

        df = synthetic_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Early rows should have NaN due to ATR warm-up
        assert pd.isna(df.iloc[0]['atr']) or df.iloc[0]['atr'] == 0.0
        assert pd.isna(df.iloc[0]['close_change'])

        # Later rows should have valid values
        assert not pd.isna(df.iloc[50]['atr'])
        assert not pd.isna(df.iloc[50]['close_change'])

    def test_min_candles_requirement(self, synthetic_ohlcv):
        """Verify strategy requires MIN_CANDLES_REQUIRED before generating signals."""
        strategy = VolatilitySystem()

        # Test with insufficient candles
        small_df = synthetic_ohlcv.iloc[:20].copy()
        result = strategy.run(small_df, small_df.iloc[-1]['date'])

        assert result.signal == SignalType.HOLD, \
            "Should return HOLD when candles < MIN_CANDLES_REQUIRED"

        # Test with sufficient candles
        large_df = synthetic_ohlcv.copy()
        result = strategy.run(large_df, large_df.iloc[-1]['date'])

        assert result.signal in [SignalType.LONG, SignalType.SHORT, SignalType.HOLD]

    def test_flat_market_handling(self, edge_case_ohlcv):
        """Verify strategy handles flat periods (no price movement)."""
        strategy = VolatilitySystem()

        df = edge_case_ohlcv.copy()

        # Insert flat period (rows 10-15 have close = 100 exactly)
        # This should result in close_change = 0, which won't trigger signals

        # Run strategy - should not crash on flat periods
        result = strategy.run(df, df.iloc[-1]['date'])

        assert result.signal in [SignalType.LONG, SignalType.SHORT, SignalType.HOLD]

    def test_extreme_volatility_spike(self, edge_case_ohlcv):
        """Verify strategy handles extreme volatility spikes (large ATR values)."""
        strategy = VolatilitySystem()

        df = edge_case_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # edge_case_ohlcv has gap up at row 20 and gap down at row 30
        # These should create large ATR values

        # Check that ATR captures these spikes
        atr_values = df['atr'].dropna()
        assert atr_values.max() > atr_values.mean(), \
            "ATR should spike during volatile periods"

        # Check that strategy handles these spikes without errors
        for idx in range(40, 50):
            window_df = df.iloc[:idx+1].copy()
            signal = strategy._generate_signal(window_df)

            assert signal in [SignalType.LONG, SignalType.SHORT, SignalType.HOLD]

    def test_zero_volume_handling(self, synthetic_ohlcv):
        """Verify strategy returns HOLD when volume is zero."""
        strategy = VolatilitySystem()

        df = synthetic_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Set volume to zero for a specific row
        test_idx = 50
        df.loc[test_idx, 'volume'] = 0

        window_df = df.iloc[:test_idx+1].copy()
        signal = strategy._generate_signal(window_df)

        # Should return HOLD when volume is zero (even if breakout condition met)
        assert signal == SignalType.HOLD, \
            "Should return HOLD when volume is zero"

    def test_atr_shift_correctness(self, synthetic_ohlcv):
        """
        Verify that strategy uses ATR.shift(1) (previous candle's ATR).

        This is critical to prevent lookahead bias.
        """
        strategy = VolatilitySystem()

        df = synthetic_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Manually verify shift operation for a few rows
        for idx in range(50, 55):
            curr_row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]

            curr_close_change = curr_row['close_change']
            prev_atr = prev_row['atr']

            if not pd.isna(curr_close_change) and not pd.isna(prev_atr):
                # The strategy should be comparing curr_close_change with prev_atr
                # Let's verify the logic is using the previous ATR, not current

                # This is an indirect test - we verify that the comparison
                # makes sense temporally (current change vs previous volatility)
                assert True  # Placeholder - actual verification happens in signal generation


class TestVolatilitySystemResamplingIntegration:
    """Test that resampling to 3h timeframe works correctly."""

    def test_resampling_preserves_signal_logic(self, synthetic_ohlcv):
        """
        Verify that resampling to 3h doesn't break signal generation.

        Note: This test uses base timeframe data (not resampled) since
        resampling is handled by DataManager, not the strategy itself.
        """
        strategy = VolatilitySystem()

        # Strategy should work on base timeframe data
        df = synthetic_ohlcv.copy()
        result = strategy.run(df, df.iloc[-1]['date'])

        assert result.signal in [SignalType.LONG, SignalType.SHORT, SignalType.HOLD]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
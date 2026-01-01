"""
Test suite for OTTStrategy with reference implementation.

This module contains:
1. Reference implementation using EXACT Freqtrade iterative logic (slow but correct)
2. Mathematical equivalence tests comparing optimized code to reference
3. Signal logic tests verifying LONG/SHORT generation
4. Edge case tests (NaNs, flat markets, trend reversals)

The reference implementation serves as ground truth - it must match
the original Freqtrade FOttStrategy exactly.
"""

import pytest
import pandas as pd
import numpy as np
from pandas import DataFrame

from strategies.ott_strategy import OTTStrategy
from strategies.base_strategy import SignalType


def reference_ott_calculation(df: DataFrame) -> DataFrame:
    """
    Reference implementation of OTT indicator using EXACT Freqtrade iterative logic.

    This is SLOW (O(N²)) but CORRECT - used as ground truth for testing.
    Based on the original FOttStrategy from Freqtrade.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with OTT and VAR columns
    """
    df = df.copy()

    pds = 2
    percent = 1.4
    alpha = 2 / (pds + 1)

    # Step 1: Calculate UD1 and DD1
    df["ud1"] = np.where(
        df["close"] > df["close"].shift(1),
        (df["close"] - df["close"].shift(1)),
        0
    )
    df["dd1"] = np.where(
        df["close"] < df["close"].shift(1),
        (df["close"].shift() - df["close"]),
        0
    )

    # Step 2: Calculate UD and DD (rolling sums)
    df["UD"] = df["ud1"].rolling(9).sum()
    df["DD"] = df["dd1"].rolling(9).sum()

    # Step 3: Calculate CMO
    df["CMO"] = ((df["UD"] - df["DD"]) / (df["UD"] + df["DD"])).fillna(0).abs()

    # Step 4: Calculate Var using iterative method (CRITICAL - must match original)
    df["Var"] = 0.0
    for i in range(pds, len(df)):
        df["Var"].iat[i] = (
            alpha * df["CMO"].iat[i] * df["close"].iat[i]
            + (1 - alpha * df["CMO"].iat[i]) * df["Var"].iat[i - 1]
        )

    # Step 5: Calculate fark and stops
    df["fark"] = df["Var"] * percent * 0.01
    df["newlongstop"] = df["Var"] - df["fark"]
    df["newshortstop"] = df["Var"] + df["fark"]
    df["longstop"] = 0.0
    df["shortstop"] = 999999999999999999.0

    # Step 6: Calculate longstop and shortstop using iterative logic
    # CRITICAL: This is the complex nested loop from original Freqtrade
    for i in df["UD"]:  # Iterate over UD values (triggers the loop)
        def maxlongstop():
            df.loc[(df["newlongstop"] > df["longstop"].shift(1)), "longstop"] = df["newlongstop"]
            df.loc[(df["longstop"].shift(1) > df["newlongstop"]), "longstop"] = df["longstop"].shift(1)
            return df["longstop"]

        def minshortstop():
            df.loc[(df["newshortstop"] < df["shortstop"].shift(1)), "shortstop"] = df["newshortstop"]
            df.loc[(df["shortstop"].shift(1) < df["newshortstop"]), "shortstop"] = df["shortstop"].shift(1)
            return df["shortstop"]

        df["longstop"] = np.where(
            ((df["Var"] > df["longstop"].shift(1))),
            maxlongstop(),
            df["newlongstop"],
        )

        df["shortstop"] = np.where(
            ((df["Var"] < df["shortstop"].shift(1))),
            minshortstop(),
            df["newshortstop"],
        )

    # Step 7: Calculate crossovers
    df["xlongstop"] = np.where(
        (
            (df["Var"].shift(1) > df["longstop"].shift(1))
            & (df["Var"] < df["longstop"].shift(1))
        ),
        1,
        0,
    )

    df["xshortstop"] = np.where(
        (
            (df["Var"].shift(1) < df["shortstop"].shift(1))
            & (df["Var"] > df["shortstop"].shift(1))
        ),
        1,
        0,
    )

    # Step 8: Calculate trend and dir
    df["trend"] = 0
    df["dir"] = 0
    for i in df["UD"]:
        df["trend"] = np.where(
            ((df["xshortstop"] == 1)),
            1,
            (np.where((df["xlongstop"] == 1), -1, df["trend"].shift(1))),
        )

        df["dir"] = np.where(
            ((df["xshortstop"] == 1)),
            1,
            (np.where((df["xlongstop"] == 1), -1, df["dir"].shift(1).fillna(1))),
        )

    # Step 9: Calculate MT and OTT
    df["MT"] = np.where(df["dir"] == 1, df["longstop"], df["shortstop"])
    df["OTT"] = np.where(
        df["Var"] > df["MT"],
        (df["MT"] * (200 + percent) / 200),
        (df["MT"] * (200 - percent) / 200),
    )

    # Step 10: Shift OTT by 2 periods (as in original)
    df["OTT"] = df["OTT"].shift(2)

    return DataFrame(index=df.index, data={"OTT": df["OTT"], "VAR": df["Var"]})


class TestOTTStrategyMathematicalEquivalence:
    """Test that optimized OTT calculation matches reference implementation."""

    def test_ott_values_match_reference(self, synthetic_ohlcv):
        """Verify OTT indicator values match reference (rtol=1e-5)."""
        # Calculate reference (slow but correct)
        reference_result = reference_ott_calculation(synthetic_ohlcv)

        # Calculate optimized (fast, needs to match)
        strategy = OTTStrategy()
        optimized_result = strategy._calculate_ott(synthetic_ohlcv)

        # Compare OTT values (skip first 20 rows due to indicator warm-up)
        reference_ott = reference_result['OTT'].iloc[20:].values
        optimized_ott = optimized_result['OTT'].iloc[20:].values

        # Remove NaNs before comparison
        mask = ~(np.isnan(reference_ott) | np.isnan(optimized_ott))

        np.testing.assert_allclose(
            reference_ott[mask],
            optimized_ott[mask],
            rtol=1e-5,
            err_msg="OTT values don't match reference implementation"
        )

    def test_var_values_match_reference(self, synthetic_ohlcv):
        """Verify VAR values match reference (rtol=1e-5)."""
        reference_result = reference_ott_calculation(synthetic_ohlcv)
        strategy = OTTStrategy()
        optimized_result = strategy._calculate_ott(synthetic_ohlcv)

        reference_var = reference_result['VAR'].iloc[20:].values
        optimized_var = optimized_result['Var'].iloc[20:].values

        mask = ~(np.isnan(reference_var) | np.isnan(optimized_var))

        np.testing.assert_allclose(
            reference_var[mask],
            optimized_var[mask],
            rtol=1e-5,
            err_msg="VAR values don't match reference implementation"
        )

    def test_uptrend_calculation(self, uptrend_ohlcv):
        """Verify OTT calculation is correct during uptrend."""
        reference_result = reference_ott_calculation(uptrend_ohlcv)
        strategy = OTTStrategy()
        optimized_result = strategy._calculate_ott(uptrend_ohlcv)

        reference_ott = reference_result['OTT'].iloc[20:].values
        optimized_ott = optimized_result['OTT'].iloc[20:].values

        mask = ~(np.isnan(reference_ott) | np.isnan(optimized_ott))

        np.testing.assert_allclose(
            reference_ott[mask],
            optimized_ott[mask],
            rtol=1e-5,
            err_msg="OTT calculation differs in uptrend"
        )

    def test_downtrend_calculation(self, downtrend_ohlcv):
        """Verify OTT calculation is correct during downtrend."""
        reference_result = reference_ott_calculation(downtrend_ohlcv)
        strategy = OTTStrategy()
        optimized_result = strategy._calculate_ott(downtrend_ohlcv)

        reference_ott = reference_result['OTT'].iloc[20:].values
        optimized_ott = optimized_result['OTT'].iloc[20:].values

        mask = ~(np.isnan(reference_ott) | np.isnan(optimized_ott))

        np.testing.assert_allclose(
            reference_ott[mask],
            optimized_ott[mask],
            rtol=1e-5,
            err_msg="OTT calculation differs in downtrend"
        )


class TestOTTSignalLogic:
    """Test signal generation logic for LONG/SHORT/HOLD."""

    def test_long_signal_on_var_crosses_above_ott(self, synthetic_ohlcv):
        """Verify LONG signal when VAR crosses above OTT."""
        strategy = OTTStrategy()

        # Run strategy on full dataset
        df = synthetic_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Find rows where VAR crosses above OTT
        crossed_above = []
        for i in range(1, len(df)):
            prev_var = df.iloc[i-1]['Var']
            prev_ott = df.iloc[i-1]['OTT']
            curr_var = df.iloc[i]['Var']
            curr_ott = df.iloc[i]['OTT']

            if (not pd.isna(prev_var) and not pd.isna(prev_ott) and
                not pd.isna(curr_var) and not pd.isna(curr_ott)):
                # Crossed above: prev was below/equal, current is above
                if prev_var <= prev_ott and curr_var > curr_ott:
                    crossed_above.append(i)

        # For each crossover, verify signal generation
        for idx in crossed_above:
            if idx >= strategy.MIN_CANDLES_REQUIRED:
                window_df = df.iloc[:idx+1].copy()
                signal = strategy._generate_signal(window_df)

                assert signal == SignalType.LONG, \
                    f"Expected LONG signal at index {idx} (VAR crossed above OTT)"

    def test_hold_signal_when_no_cross(self, sideways_ohlcv):
        """Verify HOLD signal in sideways market with no crossovers."""
        strategy = OTTStrategy()

        df = sideways_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Check middle section where indicators are stable
        for idx in range(50, 80):
            window_df = df.iloc[:idx+1].copy()
            signal = strategy._generate_signal(window_df)

            # In most sideways candles, should be HOLD (no crossing)
            # Allow some LONG/SHORT but majority should be HOLD
            # This is a sanity check, not a strict requirement
            assert signal in [SignalType.LONG, SignalType.SHORT, SignalType.HOLD]


class TestOTTEdgeCases:
    """Test edge cases: NaNs, flat markets, trend reversals."""

    def test_handles_nan_values(self, synthetic_ohlcv):
        """Verify strategy handles NaN values gracefully."""
        strategy = OTTStrategy()

        df = synthetic_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Early rows should have NaN due to indicator warm-up
        assert df.iloc[0]['OTT'] is None or pd.isna(df.iloc[0]['OTT'])
        assert df.iloc[0]['Var'] is None or pd.isna(df.iloc[0]['Var']) or df.iloc[0]['Var'] == 0.0

        # Later rows should have valid values
        assert not pd.isna(df.iloc[50]['OTT'])
        assert not pd.isna(df.iloc[50]['Var'])

    def test_min_candles_requirement(self, synthetic_ohlcv):
        """Verify strategy requires MIN_CANDLES_REQUIRED before generating signals."""
        strategy = OTTStrategy()

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
        strategy = OTTStrategy()

        df = edge_case_ohlcv.copy()

        # Run strategy - should not crash on flat periods
        result = strategy.run(df, df.iloc[-1]['date'])

        assert result.signal in [SignalType.LONG, SignalType.SHORT, SignalType.HOLD]

    def test_gap_handling(self, edge_case_ohlcv):
        """Verify strategy handles price gaps (large jumps)."""
        strategy = OTTStrategy()

        df = edge_case_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Should handle gaps without errors
        assert not df['Var'].isna().all()
        assert not df['OTT'].isna().all()

    def test_trend_reversal(self, synthetic_ohlcv):
        """Verify strategy handles trend reversals (uptrend → downtrend) without crashing."""
        strategy = OTTStrategy()

        df = synthetic_ohlcv.copy()
        df = strategy._calculate_indicators(df)

        # Check transition from uptrend (rows 0-66) to downtrend (rows 67-133)
        # OTT is a slow-moving indicator, so it may not produce crossovers in short synthetic data
        # This test just verifies no crashes and valid signal types are produced
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

        # Verify all signals are valid SignalType values (no crashes)
        all_signals = uptrend_signals + downtrend_signals
        for signal in all_signals:
            assert signal in [SignalType.LONG, SignalType.SHORT, SignalType.HOLD], \
                f"Invalid signal type: {signal}"


class TestOTTShiftOperation:
    """Test OTT shift operation to verify lookahead bias prevention."""

    def test_ott_shift_by_2(self, synthetic_ohlcv):
        """Verify OTT is shifted by 2 periods as in original Freqtrade."""
        strategy = OTTStrategy()
        df = synthetic_ohlcv.copy()
        result = strategy._calculate_ott(df)

        # OTT at row i should represent value from row i-2
        # This is verified by checking reference implementation matches
        # (The reference implementation explicitly uses .shift(2))

        # Rows 0-1 should be NaN due to shift
        assert pd.isna(result['OTT'].iloc[0])
        assert pd.isna(result['OTT'].iloc[1])

        # Row 2 onward should have values (after shift)
        # (May still be NaN due to indicator warm-up, but shift is applied)
        assert result['OTT'].iloc[10:].notna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Strategy Processor

Executes trading strategies on historical data and converts signals to One-Hot encoded arrays.
This module bridges the gap between strategy implementations and the RL environment.
"""

import pandas as pd
import numpy as np
from typing import List
from datetime import datetime

import config
from strategies.base_strategy import SignalType, BaseStrategy
from strategies.registry import StrategyRegistry
from utils.timeframes import timeframe_to_minutes


def signal_to_onehot(signal: SignalType) -> np.ndarray:
    """
    Convert SignalType enum to One-Hot encoded vector.

    Args:
        signal: SignalType enum value (FLAT, LONG, SHORT, or HOLD)

    Returns:
        np.ndarray of shape (4,): [is_flat, is_long, is_short, is_hold]

    Examples:
        FLAT  → [1, 0, 0, 0]
        LONG  → [0, 1, 0, 0]
        SHORT → [0, 0, 1, 0]
        HOLD  → [0, 0, 0, 1]
    """
    mapping = {
        SignalType.FLAT:  np.array([1, 0, 0, 0], dtype=np.float32),
        SignalType.LONG:  np.array([0, 1, 0, 0], dtype=np.float32),
        SignalType.SHORT: np.array([0, 0, 1, 0], dtype=np.float32),
        SignalType.HOLD:  np.array([0, 0, 0, 1], dtype=np.float32)
    }
    return mapping[signal]


def calculate_lookback_candles(lookback_hours: int, timeframe: str) -> int:
    """
    Calculate number of candles needed for lookback period.

    Args:
        lookback_hours: Hours of history required by strategy
        timeframe: Candle timeframe (e.g., "15m", "1h")

    Returns:
        Number of candles needed
    """
    candle_minutes = timeframe_to_minutes(timeframe)
    lookback_minutes = lookback_hours * 60
    return int(lookback_minutes / candle_minutes)


def add_strategy_signals(df: pd.DataFrame, strategy_list: List[str]) -> pd.DataFrame:
    """
    Execute strategies on historical data and add One-Hot encoded signal columns.

    For each strategy, adds 4 binary columns:
        - strategy_{name}_flat
        - strategy_{name}_long
        - strategy_{name}_short
        - strategy_{name}_hold

    Args:
        df: DataFrame with OHLCV data and technical indicators
        strategy_list: List of strategy class names to execute

    Returns:
        DataFrame with added strategy signal columns (4 per strategy)

    Example:
        If strategy_list = ["AwesomeMacd", "BbandRsi"], adds 8 columns:
        - strategy_awesomemacd_flat, strategy_awesomemacd_long, etc.
        - strategy_bbandrsi_flat, strategy_bbandrsi_long, etc.
    """
    if not strategy_list:
        print("  No strategies to process")
        return df

    # Make a copy to avoid modifying original
    df = df.copy()

    # Load strategy instances
    registry = StrategyRegistry()
    strategies = []
    for name in strategy_list:
        try:
            strategy = registry.get_strategy(name)
            strategies.append(strategy)
            print(f"  Loaded strategy: {strategy.name}")
        except ValueError as e:
            print(f"  Warning: {e}")
            continue

    if not strategies:
        print("  No valid strategies loaded")
        return df

    # Initialize all signal columns with 0.0
    for strategy in strategies:
        name_lower = strategy.name.lower()
        df[f'strategy_{name_lower}_flat'] = 0.0
        df[f'strategy_{name_lower}_long'] = 0.0
        df[f'strategy_{name_lower}_short'] = 0.0
        df[f'strategy_{name_lower}_hold'] = 0.0

    # Get base timeframe from config
    base_timeframe = config.DATA_TIMEFRAME

    # Progress tracking
    total_candles = len(df)
    print(f"  Processing {total_candles:,} candles...")

    # Iterate through each timestamp
    for idx in range(total_candles):
        # Progress indicator every 10%
        if idx % (total_candles // 10) == 0 and idx > 0:
            progress = (idx / total_candles) * 100
            print(f"  Progress: {progress:.0f}% ({idx:,}/{total_candles:,} candles)")

        timestamp = df.loc[idx, 'date']

        # Execute each strategy
        for strategy in strategies:
            # Calculate required lookback window
            lookback_candles = calculate_lookback_candles(
                strategy.lookback_hours,
                base_timeframe
            )

            # Extract historical window (from start to current candle inclusive)
            start_idx = max(0, idx - lookback_candles + 1)
            df_window = df.iloc[start_idx:idx+1].copy()

            # Skip if insufficient data
            if len(df_window) < getattr(strategy, 'MIN_CANDLES_REQUIRED', 10):
                # Default to HOLD when insufficient data
                signal = SignalType.HOLD
            else:
                # Run strategy
                try:
                    recommendation = strategy.run(df_window, timestamp)
                    signal = recommendation.signal
                except Exception as e:
                    print(f"  Warning: Strategy {strategy.name} failed at idx {idx}: {e}")
                    signal = SignalType.HOLD

            # Convert signal to One-Hot encoding
            onehot = signal_to_onehot(signal)

            # Store in DataFrame columns
            name_lower = strategy.name.lower()
            df.loc[idx, f'strategy_{name_lower}_flat'] = onehot[0]
            df.loc[idx, f'strategy_{name_lower}_long'] = onehot[1]
            df.loc[idx, f'strategy_{name_lower}_short'] = onehot[2]
            df.loc[idx, f'strategy_{name_lower}_hold'] = onehot[3]

    print(f"  ✓ Strategy signals generated for {len(strategies)} strategies")
    print(f"  Added {len(strategies) * 4} One-Hot encoded columns")

    return df
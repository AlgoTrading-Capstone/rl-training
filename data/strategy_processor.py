"""
Strategy Processor - Utility Functions

Provides utility functions for strategy signal processing.
Main strategy execution is now handled by DataManager (data/data_manager.py).
"""

import numpy as np

from strategies.base_strategy import SignalType
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

    Example:
        >>> calculate_lookback_candles(100, "1h")
        100
        >>> calculate_lookback_candles(100, "15m")
        400
    """
    candle_minutes = timeframe_to_minutes(timeframe)
    lookback_minutes = lookback_hours * 60
    return int(lookback_minutes / candle_minutes)
"""
Shared fixtures for strategy testing.

Provides high-quality synthetic OHLCV data with distinct market regimes:
- Uptrend: Clear bullish movement
- Downtrend: Clear bearish movement
- Sideways/Chop: Mean-reverting consolidation

All fixtures use deterministic generation for reproducible tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


@pytest.fixture
def synthetic_ohlcv():
    """
    Generate 200 candles of synthetic OHLCV data with three market regimes.

    Structure:
    - Rows 0-66: Uptrend (100 → 150, +50%)
    - Rows 67-133: Downtrend (150 → 100, -33%)
    - Rows 134-199: Sideways (100 ± 5%, choppy)

    Returns:
        pd.DataFrame with columns: date, open, high, low, close, volume
    """
    np.random.seed(42)  # Reproducible tests

    dates = pd.date_range('2020-01-01', periods=200, freq='1h')

    # Regime 1: Uptrend (67 candles)
    uptrend = np.linspace(100, 150, 67)
    # Add small random noise (±1%) to make it realistic
    uptrend += np.random.normal(0, 0.5, 67)

    # Regime 2: Downtrend (67 candles)
    downtrend = np.linspace(150, 100, 67)
    downtrend += np.random.normal(0, 0.5, 67)

    # Regime 3: Sideways/Chop (66 candles)
    sideways = np.ones(66) * 100
    sideways += np.random.normal(0, 2, 66)  # ±2% noise

    # Combine regimes
    close = np.concatenate([uptrend, downtrend, sideways])

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, 200)))  # Up to 2% above close
    low = close * (1 - np.abs(np.random.normal(0, 0.01, 200)))   # Up to 2% below close
    open_price = close * (1 + np.random.normal(0, 0.005, 200))   # ±0.5% from close

    # Ensure OHLC relationship is valid: high >= max(open, close), low <= min(open, close)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    # Volume: Random but consistent magnitude
    volume = np.random.uniform(900, 1100, 200)

    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df


@pytest.fixture
def uptrend_ohlcv():
    """
    Generate 100 candles of pure uptrend data.
    Useful for testing LONG signal generation.
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='1h')

    close = np.linspace(100, 200, 100)
    close += np.random.normal(0, 0.5, 100)

    high = close * 1.01
    low = close * 0.99
    open_price = close * 1.005

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.ones(100) * 1000

    return pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def downtrend_ohlcv():
    """
    Generate 100 candles of pure downtrend data.
    Useful for testing SHORT signal generation.
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='1h')

    close = np.linspace(200, 100, 100)
    close += np.random.normal(0, 0.5, 100)

    high = close * 1.01
    low = close * 0.99
    open_price = close * 1.005

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.ones(100) * 1000

    return pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def sideways_ohlcv():
    """
    Generate 100 candles of choppy sideways data.
    Useful for testing HOLD signal generation in range-bound markets.
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='1h')

    close = np.ones(100) * 100
    close += np.random.normal(0, 2, 100)  # ±2% random walk

    high = close * 1.01
    low = close * 0.99
    open_price = close * 1.005

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.ones(100) * 1000

    return pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def edge_case_ohlcv():
    """
    Generate edge case data for testing:
    - Exact band touches (price = band level)
    - Gaps (large price jumps)
    - Flat periods (no movement)
    - Extreme volatility spikes
    """
    dates = pd.date_range('2020-01-01', periods=100, freq='1h')

    close = np.ones(100) * 100

    # Add specific edge cases
    close[10:15] = 100  # Flat period (exactly 100)
    close[20] = 120     # Gap up (+20%)
    close[30] = 80      # Gap down (-20%)
    close[40:50] = np.linspace(100, 150, 10)  # Sharp rally
    close[50:60] = np.linspace(150, 50, 10)   # Sharp crash
    close[70] = close[69]  # Doji (no close change)

    high = close * 1.02
    low = close * 0.98
    open_price = close.copy()

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.ones(100) * 1000

    return pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
"""
Resampling utilities for OHLCV datasets.

Provides helpers to:
- Resample base-timeframe candles into higher intervals.
- Merge resampled data back into the original dataframe without lookahead bias.

Used by multi-timeframe trading strategies.
"""

from __future__ import annotations
from typing import Union
import pandas as pd
from pandas import DataFrame, DatetimeIndex

from .timeframes import timeframe_to_minutes


def compute_interval_minutes(dataframe: DataFrame) -> int:
    """
    Infer the candle interval (in minutes) from the 'date' column of a dataframe.

    The function assumes:
    - The 'date' column exists.
    - Rows are ordered by time.
    - The interval between the first two rows is representative for the whole series.

    Returns:
        Interval length in minutes as an integer.
    """
    if "date" not in dataframe.columns:
        raise ValueError("DataFrame must contain a 'date' column to compute interval.")

    if len(dataframe) < 2:
        raise ValueError("DataFrame must contain at least 2 rows to compute interval.")

    dates = pd.to_datetime(dataframe["date"])
    delta = dates.iloc[1] - dates.iloc[0]
    minutes = int(delta.total_seconds() // 60)

    if minutes <= 0:
        raise ValueError(f"Non-positive interval inferred from dataframe: {minutes} minutes.")

    return minutes


def minutes_to_timedelta(minutes: int) -> pd.Timedelta:
    """
    Convert a number of minutes into a pandas Timedelta.
    """
    return pd.to_timedelta(minutes, unit="m")


def resample_to_interval(dataframe: DataFrame, interval: Union[int, str]) -> DataFrame:
    """
    Resample an OHLCV dataframe to a higher timeframe.

    The input dataframe is expected to contain at least the following columns:
    - 'date' (datetime-like or string)
    - 'open', 'high', 'low', 'close', 'volume'

    Args:
        dataframe: Original candle dataframe at a base timeframe.
        interval: Target interval. Either:
            - int: number of minutes (e.g. 180 for 3 hours), or
            - str: timeframe string supported by timeframe_to_minutes (e.g. "3m", "1h").

    Returns:
        A new dataframe resampled to the desired interval, with OHLCV aggregated.
    """
    if isinstance(interval, str):
        interval_minutes = timeframe_to_minutes(interval)
    else:
        interval_minutes = int(interval)

    if interval_minutes <= 0:
        raise ValueError(f"Interval must be positive, got: {interval_minutes}")

    df = dataframe.copy()

    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column for resampling.")

    # Ensure datetime index with UTC normalization
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index(DatetimeIndex(df["date"]))

    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Resample to the left border because 'date' represents candle open time
    resampled = (
        df.resample(f"{interval_minutes}min", label="left", closed="left")
          .agg(ohlc_dict)
          .dropna()
    )

    resampled.reset_index(inplace=True) # 'date' becomes a normal column again

    return resampled


def resampled_merge(
    original: DataFrame,
    resampled: DataFrame,
    fill_na: bool = True,
) -> DataFrame:
    """
    Merge a resampled OHLCV dataset back into the original base timeframe dataset.

    This function is designed to avoid lookahead bias:
    - The resampled candles are time-shifted so that their computed indicators align
      with the correct base timeframe candles without using future information.

    Args:
        original: Original candle dataframe (faster timeframe, e.g. 1h),
                  must contain a 'date' column.
        resampled: Resampled dataframe (slower timeframe, e.g. 3h),
                   must contain a 'date' column.
        fill_na: If True, forward-fill missing values after the merge.

    Returns:
        A merged dataframe where all columns from `resampled` are joined onto `original`.
        Columns from the resampled frame are prefixed with
        `resample_{interval_minutes}_`.
    """
    if "date" not in original.columns or "date" not in resampled.columns:
        raise ValueError("Both original and resampled dataframes must contain a 'date' column.")

    original_int = compute_interval_minutes(original)
    resampled_int = compute_interval_minutes(resampled)

    if original_int >= resampled_int:
        # We only support downsampling (e.g. 1h -> 3h), not upsampling.
        raise ValueError(
            "Tried to merge a faster or equal timeframe into a slower timeframe. "
            "Upsampling is not supported."
        )

    original_df = original.copy()
    resampled_df = resampled.copy()

    # Normalize dates to UTC-aware datetimes
    original_df["date"] = pd.to_datetime(original_df["date"], utc=True)
    resampled_df["date"] = pd.to_datetime(resampled_df["date"], utc=True)

    # Shift resampled timestamps slightly so they align correctly with base candles.
    # This mirrors Freqtrade's behavior and avoids a one-candle delay.
    resampled_df["date_merge"] = (
        resampled_df["date"]
        + minutes_to_timedelta(resampled_int)
        - minutes_to_timedelta(original_int)
    )

    # Prefix all resampled columns with the resampled interval for clarity.
    prefixed_columns = {
        col: f"resample_{resampled_int}_{col}"
        for col in resampled_df.columns
    }
    resampled_df = resampled_df.rename(columns=prefixed_columns)

    merge_key = f"resample_{resampled_int}_date_merge"

    merged = pd.merge(
        original_df,
        resampled_df,
        how="left",
        left_on="date",
        right_on=merge_key,
    )

    # Drop the technical merge column — it is no longer needed.
    merged = merged.drop(columns=[merge_key])

    if fill_na:
        merged = merged.ffill()

    return merged
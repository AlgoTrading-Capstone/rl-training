"""
Feature Store Loader

Loads pre-computed parquet features and filters to active strategies.
Designed for teammates to quickly access cached data without rerunning the pipeline.

Usage (CLI):
    python -m data.load_features
    python -m data.load_features --strategies Ema5BreakoutTargetShiftingMtfStrategy
    python -m data.load_features --info

Usage (import):
    from data.load_features import load_feature_store
    df = load_feature_store(strategies=["Ema5BreakoutTargetShiftingMtfStrategy"])
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def _get_processed_path() -> Path:
    """Return path to the processed parquet cache."""
    pair_clean = config.TRADING_PAIR.replace("/", "_")
    filename = f"{config.EXCHANGE_NAME}_{pair_clean}_{config.DATA_TIMEFRAME}_processed.parquet"
    return Path(config.DATA_ROOT_PATH) / "training_data" / "processed" / filename


def _core_columns() -> List[str]:
    """Columns always included (OHLCV + indicators + turbulence)."""
    return [
        "date", "open", "high", "low", "close", "volume", "tic",
        *config.INDICATORS,
        "turbulence",
        "vix",
    ]


def load_feature_store(
    strategies: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load pre-computed features from the parquet cache.

    Args:
        strategies: List of strategy names to include. None = use config.STRATEGY_LIST.
                   Empty list = no strategy columns.
        start_date: Optional start date filter (YYYY-MM-DD or DD-MM-YYYY).
        end_date: Optional end date filter.

    Returns:
        DataFrame with OHLCV + indicators + filtered strategy signals.

    Raises:
        FileNotFoundError: If the parquet cache doesn't exist.
    """
    path = _get_processed_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No cached features found at {path}.\n"
            f"Run: python -m data.precompute_features"
        )

    df = pd.read_parquet(path)

    # Filter date range if requested
    if start_date or end_date:
        df["date"] = pd.to_datetime(df["date"])
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date, dayfirst=True)
            except ValueError:
                start_dt = pd.to_datetime(start_date)
            df = df[df["date"] >= start_dt]
        if end_date:
            try:
                end_dt = pd.to_datetime(end_date, dayfirst=True)
            except ValueError:
                end_dt = pd.to_datetime(end_date)
            df = df[df["date"] <= end_dt]

    # Select columns: core + requested strategies
    if strategies is None:
        strategies = config.STRATEGY_LIST

    keep_cols = []
    for col in _core_columns():
        if col in df.columns:
            keep_cols.append(col)

    for name in strategies:
        name_lower = name.lower()
        for suffix in ("flat", "long", "short", "hold"):
            col = f"strategy_{name_lower}_{suffix}"
            if col in df.columns:
                keep_cols.append(col)

    # Also keep any other columns that aren't strategy columns (preserve extras)
    return df[keep_cols].reset_index(drop=True)


def print_info():
    """Print metadata about the cached feature store."""
    path = _get_processed_path()
    if not path.exists():
        print(f"No cache found at: {path}")
        print("Run: python -m data.precompute_features")
        return

    df = pd.read_parquet(path)
    dates = pd.to_datetime(df["date"])

    # Detect strategy columns
    strat_cols = [c for c in df.columns if c.startswith("strategy_")]
    strat_names = sorted({c.rsplit("_", 1)[0].replace("strategy_", "") for c in strat_cols})

    print("=== Feature Store Info ===")
    print(f"  Path:       {path}")
    print(f"  Size:       {path.stat().st_size / 1e6:.1f} MB")
    print(f"  Shape:      {df.shape}")
    print(f"  Date range: {dates.min()} to {dates.max()}")
    print(f"  Columns:    {len(df.columns)} total")
    print(f"  Strategies: {len(strat_names)} ({len(strat_cols)} signal columns)")
    print()
    print("  Available strategies:")
    for name in strat_names:
        print(f"    - {name}")


def main():
    parser = argparse.ArgumentParser(description="Load pre-computed features.")
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated strategy names to include (default: all from config).",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print info about the cached feature store and exit.",
    )
    parser.add_argument(
        "--start-date", type=str, default=None, help="Filter start date."
    )
    parser.add_argument(
        "--end-date", type=str, default=None, help="Filter end date."
    )
    args = parser.parse_args()

    if args.info:
        print_info()
        return

    strategies = args.strategies.split(",") if args.strategies else None

    df = load_feature_store(
        strategies=strategies,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()

"""
Feature Store Pre-computation Script

Runs the full data pipeline (download -> indicators -> strategies -> save parquet)
WITHOUT starting training. This eliminates the 6+ hour preprocessing bottleneck
on subsequent runs.

Usage:
    python -m data.precompute_features --start-date 01-01-2018 --end-date 31-12-2023
    python -m data.precompute_features  # Uses config defaults
"""

import argparse
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from data.data_manager import DataManager
from utils.logger import RLLogger


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute strategy signals and save to parquet cache."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (DD-MM-YYYY). Defaults to a wide range for maximum coverage.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (DD-MM-YYYY). Defaults to today.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cache exists.",
    )
    args = parser.parse_args()

    from datetime import date as _date
    start_date = args.start_date or "01-01-2017"
    end_date = args.end_date or _date.today().strftime("%d-%m-%Y")

    logger = RLLogger(run_path=None, log_level=config.LOG_LEVEL)
    logger.info("=== Feature Store Pre-computation ===")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Strategies: {len(config.STRATEGY_LIST)} enabled")

    manager = DataManager(
        exchange=config.EXCHANGE_NAME,
        trading_pair=config.TRADING_PAIR,
        base_timeframe=config.DATA_TIMEFRAME,
        logger=logger,
    )

    strategy_list = config.STRATEGY_LIST if config.ENABLE_STRATEGIES else []

    t0 = time.time()

    df = manager.get_processed_data(
        start_date=start_date,
        end_date=end_date,
        strategy_list=strategy_list,
        force_redownload=args.force,
    )

    elapsed = time.time() - t0

    processed_path = (
        Path(config.DATA_ROOT_PATH)
        / "training_data"
        / "processed"
        / manager._get_processed_filename()
    )

    logger.info("")
    logger.info("=== Pre-computation Complete ===")
    logger.info(f"  Rows:     {len(df):,}")
    logger.info(f"  Columns:  {len(df.columns)}")
    logger.info(f"  Shape:    {df.shape}")
    logger.info(f"  Time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Saved to: {processed_path}")
    logger.info("")
    logger.info("To share with teammates, copy the folder:")
    logger.info(f"  {Path(config.DATA_ROOT_PATH) / 'training_data'}")


if __name__ == "__main__":
    main()
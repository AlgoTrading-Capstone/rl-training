"""
Download historical Bitcoin market data using CCXT
Run this script once to download training data

This is a standalone script for data exploration and testing.
For integrated training runs, use main.py instead.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

import config
from data.data_processor import DataProcessor

# ============================================================
# Backward compatibility: use default paths if config variables removed
# ============================================================
DEFAULT_DATA_PATH = "raw_data"
DEFAULT_YEARS_BACK = 6

# Check if config has the old variables, otherwise use defaults
# Can be overridden via environment variables:
#   RL_DATA_PATH=mydata RL_DATA_YEARS_BACK=2 python data/download_data.py
DATA_PATH = getattr(config, 'DATA_PATH', os.getenv("RL_DATA_PATH", DEFAULT_DATA_PATH))
DATA_YEARS_BACK = getattr(config, 'DATA_YEARS_BACK', int(os.getenv("RL_DATA_YEARS_BACK", DEFAULT_YEARS_BACK)))


def download_training_data():
    """
    Download historical Bitcoin data for training
    Uses settings from config.py
    Creates timestamped folders for each download
    """
    print("=" * 60)
    print("Bitcoin RL Training - Data Download")
    print("=" * 60)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * DATA_YEARS_BACK)

    # Format dates
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str_download = end_date.strftime('%Y-%m-%d')
    timestamp = end_date.strftime('%Y-%m-%d_%H-%M')

    print(f"\nConfiguration:")
    print(f"   Exchange: {config.EXCHANGE_NAME}")
    print(f"   Trading Pair: {config.TRADING_PAIR}")
    print(f"   Timeframe: {config.DATA_TIMEFRAME}")
    print(f"   Period: {start_date_str} to {end_date_str_download}")
    print(f"   Years: {DATA_YEARS_BACK}")
    print(f"   Download ID: {timestamp}")
    print(f"   Data path: {DATA_PATH}")

    # Create timestamped folders
    downloads_path = Path(DATA_PATH) / "downloads" / timestamp
    processed_path = Path(DATA_PATH) / "processed" / timestamp

    # Check for previous downloads BEFORE creating new folders
    downloads_root = Path(DATA_PATH) / "downloads"
    if downloads_root.exists():
        # Filter out the current timestamp folder (in case it already exists)
        previous_downloads = sorted([d for d in downloads_root.iterdir() if d.is_dir() and d.name != timestamp])
        if previous_downloads:
            print(f"\nPrevious downloads found: {len(previous_downloads)}")
            for prev in previous_downloads[-3:]:
                print(f"   - {prev.name}")
    downloads_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    # Initialize processor
    processor = DataProcessor(
        data_source=config.EXCHANGE_NAME,
        tech_indicator=config.INDICATORS
    )

    # Download data
    print(f"\n{'=' * 60}")
    print("STEP 1: Downloading OHLCV Data")
    print(f"{'=' * 60}")

    df = processor.download_data(
        ticker_list=[config.TRADING_PAIR],
        start_date=start_date_str,
        end_date=end_date_str_download,
        time_interval=config.DATA_TIMEFRAME
    )

    # Save raw data
    raw_filename = f"{config.TRADING_PAIR.replace('/', '_')}_{config.DATA_TIMEFRAME}_raw.parquet"
    raw_file_path = downloads_path / raw_filename

    df.to_parquet(raw_file_path, engine='pyarrow', compression='snappy')
    file_size_mb = os.path.getsize(raw_file_path) / (1024 * 1024)

    print(f"\nRaw data saved:")
    print(f"   Folder: {downloads_path}")
    print(f"   File: {raw_filename}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Rows: {len(df):,}")

    # Clean data
    print(f"\n{'=' * 60}")
    print("STEP 2: Cleaning Data")
    print(f"{'=' * 60}")

    df = processor.clean_data(df)

    # Add technical indicators
    print(f"\n{'=' * 60}")
    print("STEP 3: Adding Technical Indicators")
    print(f"{'=' * 60}")

    df = processor.add_technical_indicator(df, config.INDICATORS)

    # Add turbulence
    if config.ENABLE_TURBULENCE:
        df = processor.add_turbulence(df)

    # Add VIX
    if config.ENABLE_VIX:
        df = processor.add_vix(df)

    # Add strategy signals
    if config.ENABLE_STRATEGIES and config.STRATEGY_LIST:
        print(f"\n{'=' * 60}")
        print("STEP 4: Generating Strategy Signals")
        print(f"{'=' * 60}")
        df = processor.add_strategy_signals(df, config.STRATEGY_LIST)

    # Save processed data
    processed_filename = f"{config.TRADING_PAIR.replace('/', '_')}_{config.DATA_TIMEFRAME}_processed.parquet"
    processed_file_path = processed_path / processed_filename

    df.to_parquet(processed_file_path, engine='pyarrow', compression='snappy')
    file_size_mb = os.path.getsize(processed_file_path) / (1024 * 1024)

    print(f"\nProcessed data saved:")
    print(f"   Folder: {processed_path}")
    print(f"   File: {processed_filename}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")

    # Display summary
    print(f"\n{'=' * 60}")
    print("Download Complete!")
    print(f"{'=' * 60}")
    print(f"\nData Summary:")
    print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Total Candles: {len(df):,}")
    print(f"   Indicators: {', '.join(config.INDICATORS)}")
    print(f"\nFiles created:")
    print(f"   Downloads: {downloads_path}")
    print(f"      - {raw_filename}")
    print(f"   Processed: {processed_path}")
    print(f"      - {processed_filename}")
    print(f"\nReady for training!")


if __name__ == "__main__":
    try:
        download_training_data()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
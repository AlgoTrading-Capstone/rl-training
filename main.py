"""
Main orchestrator for RL training/testing pipeline.
"""

import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

from utils.user_input import collect_user_input
import config
from data.data_processor import DataProcessor


def download_and_process_data(metadata: dict, run_path: str) -> dict:
    """
    Download and process market data for training.

    Args:
        metadata: Metadata dict from collect_user_input()
        run_path: Path to results folder (e.g., "results/model_OmerPC")

    Returns:
        Updated metadata with data paths and processing info

    Raises:
        RuntimeError: If download or processing fails
    """
    print("\n" + "=" * 60)
    print("DATA DOWNLOAD AND PROCESSING")
    print("=" * 60)

    # Setup paths and timestamp
    timestamp = datetime.utcnow().strftime('%Y-%m-%d_%H-%M')
    base_data_path = Path(run_path) / "data"
    downloads_path = base_data_path / "downloads" / timestamp
    processed_path = base_data_path / "processed" / timestamp

    # Create directories
    downloads_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    # Convert dates from DD-MM-YYYY to YYYY-MM-DD
    start_dt = datetime.strptime(metadata['start_date'], '%d-%m-%Y')
    end_dt = datetime.strptime(metadata['end_date'], '%d-%m-%Y')
    start_date_str = start_dt.strftime('%Y-%m-%d')
    end_date_str = end_dt.strftime('%Y-%m-%d')

    print(f"\nConfiguration:")
    print(f"  Exchange: {config.EXCHANGE_NAME}")
    print(f"  Trading Pair: {config.TRADING_PAIR}")
    print(f"  Timeframe: {config.DATA_TIMEFRAME}")
    print(f"  Date Range: {start_date_str} to {end_date_str}")
    print(f"  Download ID: {timestamp}")

    # Initialize processor
    processor = DataProcessor(
        data_source=config.EXCHANGE_NAME,
        tech_indicator=config.INDICATORS
    )

    # STEP 1: Download OHLCV data
    try:
        print("\n" + "=" * 60)
        print("STEP 1: Downloading OHLCV Data")
        print("=" * 60)

        df = processor.download_data(
            ticker_list=[config.TRADING_PAIR],
            start_date=start_date_str,
            end_date=end_date_str,
            time_interval=config.DATA_TIMEFRAME
        )
    except Exception as e:
        raise RuntimeError(f"Data download failed: {e}") from e

    # Save raw data
    raw_filename = f"{config.TRADING_PAIR.replace('/', '_')}_{config.DATA_TIMEFRAME}_raw.parquet"
    raw_file_path = downloads_path / raw_filename
    df.to_parquet(raw_file_path, engine='pyarrow', compression='snappy')

    raw_size_mb = raw_file_path.stat().st_size / (1024 * 1024) # Convert to MB
    print(f"\nRaw data saved:")
    print(f"  File: {raw_file_path}")
    print(f"  Size: {raw_size_mb:.2f} MB")
    print(f"  Rows: {len(df):,}")

    # STEP 2: Clean data
    print("\n" + "=" * 60)
    print("STEP 2: Cleaning Data")
    print("=" * 60)
    df = processor.clean_data(df)

    # STEP 3: Add technical indicators
    print("\n" + "=" * 60)
    print("STEP 3: Adding Technical Indicators")
    print("=" * 60)
    df = processor.add_technical_indicator(df, config.INDICATORS)

    # STEP 4: Add turbulence (optional)
    if config.ENABLE_TURBULENCE:
        df = processor.add_turbulence(df)

    # STEP 5: Add VIX (optional)
    if config.ENABLE_VIX:
        df = processor.add_vix(df)

    # STEP 6: Add strategy signals (optional)
    if config.ENABLE_STRATEGIES and config.STRATEGY_LIST:
        print("\n" + "=" * 60)
        print("STEP 4: Generating Strategy Signals")
        print("=" * 60)
        df = processor.add_strategy_signals(df, config.STRATEGY_LIST)

    # Save processed data
    processed_filename = f"{config.TRADING_PAIR.replace('/', '_')}_{config.DATA_TIMEFRAME}_processed.parquet"
    processed_file_path = processed_path / processed_filename
    df.to_parquet(processed_file_path, engine='pyarrow', compression='snappy')

    processed_size_mb = processed_file_path.stat().st_size / (1024 * 1024)
    print(f"\nProcessed data saved:")
    print(f"  File: {processed_file_path}")
    print(f"  Size: {processed_size_mb:.2f} MB")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # Update metadata
    metadata.update({
        'data_download_timestamp': timestamp,
        'data_download_status': 'completed',
        'raw_data_path': str(raw_file_path),
        'processed_data_path': str(processed_file_path),
        'total_candles': len(df),
        'date_range_actual': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat()
        },
        'indicators': config.INDICATORS,
        'turbulence_enabled': config.ENABLE_TURBULENCE,
        'vix_enabled': config.ENABLE_VIX,
        'strategies_enabled': config.ENABLE_STRATEGIES,
        'strategy_list': config.STRATEGY_LIST if config.ENABLE_STRATEGIES else []
    })

    # Save updated metadata
    metadata_file = Path(run_path) / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 60)
    print("DATA DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nData Summary:")
    print(f"  Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total Candles: {len(df):,}")
    print(f"  Indicators: {', '.join(config.INDICATORS)}")

    return metadata


def load_processed_data(run_path: str) -> tuple:
    """
    Load most recent processed data from run folder.

    Args:
        run_path: Path to results folder (e.g., "results/model_OmerPC")

    Returns:
        Tuple of (price_array, tech_array, turbulence_array, signal_array)
        Note: turbulence_array is not used by BitcoinTradingEnv but returned for compatibility

    Raises:
        FileNotFoundError: If no processed data found
        ValueError: If data format is invalid
    """
    print("\n" + "=" * 60)
    print("LOADING PROCESSED DATA")
    print("=" * 60)

    # Find most recent processed folder
    processed_base = Path(run_path) / "data" / "processed"

    if not processed_base.exists():
        raise FileNotFoundError(
            f"No processed data found in {processed_base}\n"
            f"Have you run data download? The folder should have been created."
        )

    # Get all timestamp folders, sort descending (most recent first)
    timestamp_folders = sorted(
        [d for d in processed_base.iterdir() if d.is_dir()],
        reverse=True
    )

    if not timestamp_folders:
        raise FileNotFoundError(
            f"No processed data folders found in {processed_base}\n"
            f"Expected format: YYYY-MM-DD_HH-MM/"
        )

    latest_folder = timestamp_folders[0]
    print(f"\nUsing data from: {latest_folder.name}")

    # Find Parquet file
    parquet_files = list(latest_folder.glob("*_processed.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {latest_folder}\n"
            f"Expected file pattern: *_processed.parquet"
        )

    if len(parquet_files) > 1:
        print(f"Warning: Multiple Parquet files found, using: {parquet_files[0].name}")

    data_file = parquet_files[0]

    # Load DataFrame
    print(f"Loading: {data_file}")
    df = pd.read_parquet(data_file)
    print(f"  Loaded {len(df):,} candles")
    print(f"  Columns: {len(df.columns)}")

    # Convert to arrays
    processor = DataProcessor()
    price_array, tech_array, turbulence_array, signal_array = processor.df_to_array(
        df=df,
        if_vix=config.ENABLE_VIX
    )

    # Validate arrays
    if price_array.shape[0] == 0:
        raise ValueError("Price array is empty - data conversion failed")

    print(f"\nArrays created:")
    print(f"  Price:      {price_array.shape} - [open, high, low, close]")
    print(f"  Tech:       {tech_array.shape} - indicators + VIX")
    print(f"  Turbulence: {turbulence_array.shape} - market stress (not used by env)")
    print(f"  Signals:    {signal_array.shape} - strategy One-Hot vectors")

    return price_array, tech_array, turbulence_array, signal_array


def main():
    print("=" * 60)
    print("RL Training Execution")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: Collect user input + create run folder + metadata
    # --------------------------------------------------------
    metadata, run_path = collect_user_input()

    # --------------------------------------------------------
    # STEP 2: Download and process data
    # --------------------------------------------------------
    try:
        metadata = download_and_process_data(metadata, run_path)
    except RuntimeError as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR: {e}")
        print(f"{'=' * 60}")
        print("\nTraining aborted due to data download failure.")
        print(f"Partial data may be available in: {run_path}/data/")
        return

    # --------------------------------------------------------
    # STEP 3: Load processed data into arrays
    # --------------------------------------------------------
    try:
        price_array, tech_array, turbulence_array, signal_array = load_processed_data(run_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR: {e}")
        print(f"{'=' * 60}")
        print("\nTraining aborted due to data loading failure.")
        return

    # --------------------------------------------------------
    # STEP 4: TODO - Initialize RL environment
    # --------------------------------------------------------
    # NOTE: BitcoinTradingEnv takes only 3 arrays (price, tech, signal)
    # turbulence_array is not used by the current environment implementation

    # from bitcoin_env import BitcoinTradingEnv
    # train_env = BitcoinTradingEnv(price_array, tech_array, signal_array, mode="train")
    # test_env = BitcoinTradingEnv(price_array, tech_array, signal_array, mode="test")

    print("\n" + "=" * 60)
    print("DATA READY FOR TRAINING")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Initialize BitcoinTradingEnv with (price_array, tech_array, signal_array)")
    print("  2. Setup RL agent (ElegantRL/FinRL)")
    print("  3. Train agent")
    print("  4. Evaluate and save results")

    # --------------------------------------------------------
    # STEP 5: TODO - Train agent
    # --------------------------------------------------------

    # --------------------------------------------------------
    # STEP 6: TODO - Evaluate and save results
    # --------------------------------------------------------

    print("\n=== Training Session Complete ===\n")


if __name__ == "__main__":
    main()
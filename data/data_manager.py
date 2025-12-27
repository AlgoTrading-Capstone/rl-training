"""
DataManager - Unified data management system for Bitcoin RL training

Handles:
- Smart incremental downloads (only fetch missing data gaps)
- Multi-timeframe strategy support with automatic resampling
- Parallel strategy execution using ProcessPoolExecutor
- Robust validation and error handling
"""

import json
import shutil
import time
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import ccxt
import numpy as np
import pandas as pd

import config
from data.data_processor import DataProcessor
from data.strategy_processor import signal_to_onehot, calculate_lookback_candles
from strategies.base_strategy import SignalType
from strategies.registry import StrategyRegistry
from utils.timeframes import timeframe_to_minutes
from utils.resampling import resample_to_interval, resampled_merge
from utils.progress import ProgressTracker
from utils.date_display import format_date_range_for_display


# ============================================================================
# Custom Exceptions
# ============================================================================

class DataManagerError(Exception):
    """Base exception for DataManager errors"""
    pass


class DownloadError(DataManagerError):
    """Failed to download market data"""
    pass


class StorageError(DataManagerError):
    """Failed to read/write storage"""
    pass


class StrategyExecutionError(DataManagerError):
    """Strategy execution failed"""
    pass


class ValidationError(DataManagerError):
    """Data validation failed"""
    pass


# ============================================================================
# Top-Level Worker Function (MUST be outside class for Windows pickling)
# ============================================================================

def _process_single_strategy_with_resampling(
    strategy_name: str,
    df: pd.DataFrame,
    base_timeframe: str,
    strategy_timeframe: str,
    lookback_hours: int
) -> Tuple[str, Dict[str, np.ndarray]]:
    """
    Execute strategy with automatic timeframe adaptation.

    CRITICAL: This function MUST be top-level for ProcessPoolExecutor pickling.
    Windows requires this constraint.

    Args:
        strategy_name: Name of strategy to execute
        df: Base timeframe DataFrame
        base_timeframe: Base data timeframe (e.g., "15m")
        strategy_timeframe: Strategy's required timeframe (e.g., "1h")
        lookback_hours: Hours of historical data required

    Returns:
        Tuple of (strategy_name, signal_arrays_dict) where signal_arrays_dict contains:
        {
            'flat': np.ndarray,
            'long': np.ndarray,
            'short': np.ndarray,
            'hold': np.ndarray
        }
    """
    # Load strategy instance
    registry = StrategyRegistry()
    strategy = registry.get_strategy(strategy_name)

    # Determine if resampling needed
    base_tf_min = timeframe_to_minutes(base_timeframe)
    strategy_tf_min = timeframe_to_minutes(strategy_timeframe)

    if strategy_tf_min == base_tf_min:
        # No resampling needed
        df_for_calc = df.copy()
        needs_merge = False
    elif strategy_tf_min > base_tf_min:
        # Resample to higher timeframe
        df_for_calc = resample_to_interval(df, strategy_timeframe)
        needs_merge = True
    else:
        # Cannot upsample
        raise ValueError(
            f"Strategy {strategy_name} requires {strategy_timeframe} "
            f"but base data is {base_timeframe}. Cannot upsample."
        )

    # Initialize signal arrays
    total_candles = len(df_for_calc)
    signal_arrays = {
        'flat': np.zeros(total_candles, dtype=np.float32),
        'long': np.zeros(total_candles, dtype=np.float32),
        'short': np.zeros(total_candles, dtype=np.float32),
        'hold': np.zeros(total_candles, dtype=np.float32)
    }

    # Calculate lookback
    lookback_candles = calculate_lookback_candles(
        lookback_hours,
        strategy_timeframe
    )

    # Process each candle
    for idx in range(total_candles):
        timestamp = df_for_calc.loc[idx, 'date']
        start_idx = max(0, idx - lookback_candles + 1)
        df_window = df_for_calc.iloc[start_idx:idx+1].copy()

        # Run strategy
        if len(df_window) >= strategy.MIN_CANDLES_REQUIRED:
            try:
                recommendation = strategy.run(df_window, timestamp)
                signal = recommendation.signal
            except Exception as e:
                print(f"  Warning: {strategy_name} failed at {timestamp}: {e}")
                signal = SignalType.HOLD
        else:
            signal = SignalType.HOLD

        # Convert to One-Hot
        onehot = signal_to_onehot(signal)
        signal_arrays['flat'][idx] = onehot[0]
        signal_arrays['long'][idx] = onehot[1]
        signal_arrays['short'][idx] = onehot[2]
        signal_arrays['hold'][idx] = onehot[3]

    # Merge back to base timeframe if resampled
    if needs_merge:
        signal_df = pd.DataFrame({
            'date': df_for_calc['date'],
            'flat': signal_arrays['flat'],
            'long': signal_arrays['long'],
            'short': signal_arrays['short'],
            'hold': signal_arrays['hold']
        })

        # CRITICAL: Use resampled_merge (prevents lookahead bias)
        merged = resampled_merge(df, signal_df, fill_na=True)

        # Extract aligned signals with prefix
        prefix = f"resample_{strategy_tf_min}_"
        signal_arrays = {
            'flat': merged[f'{prefix}flat'].values,
            'long': merged[f'{prefix}long'].values,
            'short': merged[f'{prefix}short'].values,
            'hold': merged[f'{prefix}hold'].values
        }

    return strategy_name, signal_arrays


# ============================================================================
# DataManager Class
# ============================================================================

class DataManager:
    """
    Unified data management system for Bitcoin RL training.

    Handles incremental downloads, feature engineering, strategy signals,
    validation, and array conversion for RL environment.

    Usage:
        manager = DataManager(storage_path="results/my_model_OmerPC/data")
        arrays = manager.get_arrays_for_training(
            start_date="2023-01-01",
            end_date="2024-01-01",
            strategy_list=["AwesomeMacd", "BbandRsi"]
        )
    """

    def __init__(
        self,
        exchange: str = None,
        trading_pair: str = None,
        base_timeframe: str = None,
        storage_path: str = None,
        logger=None  # Required parameter
    ):
        """
        Initialize DataManager.

        Args:
            exchange: CCXT exchange name (default: config.EXCHANGE_NAME)
            trading_pair: Trading symbol (default: config.TRADING_PAIR)
            base_timeframe: Base data timeframe (default: config.DATA_TIMEFRAME)
            storage_path: Root storage path (default: config.DATA_ROOT_PATH)
                         DEPRECATED - use config.DATA_ROOT_PATH instead
            logger: Logger instance (REQUIRED)
        """
        # Logger is required - validate it's provided
        if logger is None:
            raise ValueError("logger parameter is required. Pass RLLogger instance from main.py")

        from utils.logger import LogComponent
        self.logger = logger.for_component(LogComponent.DATA)

        self.exchange = exchange or config.EXCHANGE_NAME
        self.trading_pair = trading_pair or config.TRADING_PAIR
        self.base_timeframe = base_timeframe or config.DATA_TIMEFRAME

        # Use shared data root (new architecture)
        # storage_path parameter is deprecated but kept for backward compatibility
        self.storage_path = Path(storage_path or config.DATA_ROOT_PATH)

        # Create complete directory structure for Feature Store
        # training_data/ - PORTABLE folder (copy to other machines)
        (self.storage_path / "training_data" / "raw").mkdir(parents=True, exist_ok=True)
        (self.storage_path / "training_data" / "processed").mkdir(parents=True, exist_ok=True)
        # archived/ - LOCAL HISTORY folder (stays on this machine)
        (self.storage_path / "archived" / "raw").mkdir(parents=True, exist_ok=True)
        (self.storage_path / "archived" / "processed").mkdir(parents=True, exist_ok=True)

        # Initialize DataProcessor (wrapper with CcxtProcessor + ExternalDataManager)
        self.processor = DataProcessor(data_source=self.exchange)



    # ========================================================================
    # Phase 1: Smart Incremental Download
    # ========================================================================

    def _get_storage_filename(self) -> str:
        """
        Generate storage filename based on configuration.

        Returns: "exchange_pair_timeframe.parquet"
        Example: "binance_BTC_USDT_15m.parquet"
        """
        # Clean pair name (replace / with _)
        pair_clean = self.trading_pair.replace('/', '_')
        return f"{self.exchange}_{pair_clean}_{self.base_timeframe}.parquet"

    def _get_processed_filename(self) -> str:
        """
        Generate processed data filename (with indicators + strategies).

        Returns: "exchange_pair_timeframe_processed.parquet"
        Example: "binance_BTC_USDT_15m_processed.parquet"
        """
        base = self._get_storage_filename()
        return base.replace(".parquet", "_processed.parquet")

    def _validate_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> bool:
        """
        Check if DataFrame covers requested date range (superset match).

        Args:
            df: DataFrame with 'date' column
            start_date: Start date "DD-MM-YYYY" or "YYYY-MM-DD"
            end_date: End date "DD-MM-YYYY" or "YYYY-MM-DD"

        Returns:
            True if DataFrame covers at least the requested range
        """
        if 'date' not in df.columns or len(df) == 0:
            return False

        # Convert dates to ISO format (handles both DD-MM-YYYY and YYYY-MM-DD)
        start_iso = self.processor._convert_to_iso_date(start_date)
        end_iso = self.processor._convert_to_iso_date(end_date)

        # Parse to datetime with explicit ISO format
        requested_start = pd.to_datetime(start_iso, format="ISO8601")
        requested_end = pd.to_datetime(end_iso, format="ISO8601")

        # Get actual range in DataFrame
        df_start = df['date'].min()
        df_end = df['date'].max()

        # Superset match: DataFrame must cover AT LEAST the requested range
        return df_start <= requested_start and df_end >= requested_end

    def _filter_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Filter DataFrame to exact requested date range.

        Args:
            df: DataFrame with 'date' column
            start_date: Start date "DD-MM-YYYY" or "YYYY-MM-DD"
            end_date: End date "DD-MM-YYYY" or "YYYY-MM-DD"

        Returns:
            Filtered DataFrame
        """
        # Convert dates to ISO format
        start_iso = self.processor._convert_to_iso_date(start_date)
        end_iso = self.processor._convert_to_iso_date(end_date)

        # Parse to datetime with explicit ISO format
        requested_start = pd.to_datetime(start_iso, format="ISO8601")
        requested_end = pd.to_datetime(end_iso, format="ISO8601")

        # Filter
        mask = (df['date'] >= requested_start) & (df['date'] <= requested_end)
        filtered = df[mask].copy()

        self.logger.debug(f"Filtered {len(df)} rows → {len(filtered)} rows (requested range)")

        return filtered

    def _archive_file(self, filepath: Path) -> None:
        """
        Move existing file to archived/ with timestamp prefix.

        Example:
          data/processed/binance_BTC_USDT_15m_processed.parquet
          → data/archived/processed/20251213_1430_binance_BTC_USDT_15m_processed.parquet

        Args:
            filepath: Path to file to archive
        """
        if not filepath.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        archived_dir = self.storage_path / "archived" / filepath.parent.name
        archived_path = archived_dir / f"{timestamp}_{filepath.name}"

        shutil.move(str(filepath), str(archived_path))
        self.logger.info(f"Archived: {filepath.name} → archived/{filepath.parent.name}/{archived_path.name}")

    def _save_processed_data(self, df: pd.DataFrame, filepath: Path) -> None:
        """
        Save processed DataFrame with archiving.

        Args:
            df: DataFrame to save
            filepath: Target file path
        """
        # Archive existing processed data before overwriting
        if filepath.exists():
            self._archive_file(filepath)

        df.to_parquet(filepath, index=False)
        self.logger.info(f"Saved processed data: {filepath}")

    def _get_data_gap(
        self,
        existing_df: pd.DataFrame,
        requested_start: str,
        requested_end: str
    ) -> List[Tuple[str, str]]:
        """
        Calculate missing date ranges.

        Cases:
        1. Request fully within existing -> []
        2. Request extends before -> gap at start
        3. Request extends after -> gap at end
        4. Request extends both sides -> 2 gaps

        Args:
            existing_df: Existing data (only 'date' column needed)
            requested_start: Requested start date "YYYY-MM-DD"
            requested_end: Requested end date "YYYY-MM-DD"

        Returns:
            List of (start_date, end_date) tuples for missing gaps
        """
        gaps = []

        existing_start = existing_df['date'].min()
        existing_end = existing_df['date'].max()

        req_start = pd.to_datetime(requested_start, format="ISO8601")
        req_end = pd.to_datetime(requested_end, format="ISO8601")

        # Gap before existing data
        if req_start < existing_start:
            gaps.append((
                req_start.strftime('%Y-%m-%d'),
                (existing_start - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            ))

        # Gap after existing data
        if req_end > existing_end:
            gaps.append((
                (existing_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                req_end.strftime('%Y-%m-%d')
            ))

        return gaps

    def _download_with_retry(
        self,
        start: str,
        end: str,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Download data with exponential backoff retry logic.

        Args:
            start: Start date "YYYY-MM-DD"
            end: End date "YYYY-MM-DD"
            max_retries: Maximum retry attempts

        Returns:
            DataFrame with OHLCV data

        Raises:
            DownloadError: If download fails after all retries
        """
        for attempt in range(max_retries):
            try:
                df = self.processor.download_data(
                    ticker_list=[self.trading_pair],
                    start_date=start,
                    end_date=end,
                    time_interval=self.base_timeframe
                )
                return df
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                if attempt == max_retries - 1:
                    raise DownloadError(
                        f"Download failed after {max_retries} attempts: {e}"
                    ) from e

                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                self.logger.warning(f"Retry {attempt+1}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)

    def _save_raw_data(self, df: pd.DataFrame, filepath: Path) -> Path:
        """
        Save raw data to Parquet file.

        Args:
            df: DataFrame to save
            filepath: Target file path

        Returns:
            Path where data was saved

        Raises:
            StorageError: If save fails
        """
        try:
            df.to_parquet(filepath, index=False)
            self.logger.debug(f"Saved to: {filepath}")
            return filepath
        except Exception as e:
            raise StorageError(f"Failed to save data to {filepath}: {e}") from e

    def get_or_download_data(
        self,
        start_date: str,
        end_date: str,
        force_redownload: bool = False
    ) -> pd.DataFrame:
        """
        Smart incremental download - only fetches missing data gaps.

        Algorithm:
        1. Check if file exists
        2. If not exists OR force_redownload: Download full range
        3. If exists: Load date column only, detect gaps, download gaps
        4. Merge, deduplicate, save
        5. Return requested date range

        Args:
            start_date: Start date "YYYY-MM-DD"
            end_date: End date "YYYY-MM-DD"
            force_redownload: If True, ignore existing data and redownload

        Returns:
            DataFrame with OHLCV data for requested range
        """
        filepath = self.storage_path / "training_data" / "raw" / self._get_storage_filename()

        # Case 1: No existing data or force redownload
        if not filepath.exists() or force_redownload:
            self.logger.info(f"Downloading full range: {start_date} to {end_date}")
            df = self._download_with_retry(start_date, end_date)
            self._save_raw_data(df, filepath)
            return df

        # Case 2: Incremental update
        self.logger.info(f"Checking existing data: {filepath}")

        # OPTIMIZATION: Load only date column to check gaps
        try:
            existing_dates = pd.read_parquet(filepath, columns=['date'])
        except Exception as e:
            self.logger.warning(f"Failed to read existing data ({e})")
            self.logger.info("Redownloading full range...")
            df = self._download_with_retry(start_date, end_date)
            self._save_raw_data(df, filepath)
            return df

        # Detect gaps
        gaps = self._get_data_gap(existing_dates, start_date, end_date)

        if not gaps:
            # No missing data - load full data and filter to requested range
            self.logger.info("No missing data - using existing")
            existing_df = pd.read_parquet(filepath)
            mask = (
                (existing_df['date'] >= start_date) &
                (existing_df['date'] <= end_date)
            )
            return existing_df[mask].reset_index(drop=True)

        # Download gaps
        self.logger.info(f"Found {len(gaps)} gap(s) to download")
        gap_dfs = []
        for gap_start, gap_end in gaps:
            self.logger.info(f"Gap: {gap_start} to {gap_end}")
            gap_df = self._download_with_retry(gap_start, gap_end)

            # Skip empty downloads (e.g., future dates)
            if len(gap_df) == 0:
                self.logger.warning(f"No data available for {gap_start} to {gap_end} (skipping)")
                continue

            gap_dfs.append(gap_df)

        # Load full existing data for merging
        existing_df = pd.read_parquet(filepath)

        # Merge and deduplicate
        self.logger.info("Merging with existing data...")
        combined = pd.concat([existing_df] + gap_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'tic'])
        combined = combined.sort_values('date').reset_index(drop=True)

        # Save updated data
        self._save_raw_data(combined, filepath)

        # Return requested range
        mask = (
            (combined['date'] >= start_date) &
            (combined['date'] <= end_date)
        )
        return combined[mask].reset_index(drop=True)

    # ========================================================================
    # Phase 2: Feature Engineering
    # ========================================================================

    def add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators, turbulence, and external data using existing methods.

        Uses CcxtProcessor methods:
        - add_technical_indicator()
        - add_turbulence() if config.ENABLE_TURBULENCE
        - add_external_data() for external assets (VIX, etc.)

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            DataFrame with technical features added
        """
        # Clean data first
        df = self.processor.clean_data(df)

        # Add technical indicators
        df = self.processor.add_technical_indicator(df, config.INDICATORS)

        # Add turbulence (market stress indicator)
        if config.ENABLE_TURBULENCE:
            df = self.processor.add_turbulence(df)

        # Add external market data (VIX, etc.)
        # Enabled assets are filtered inside add_external_data()
        df = self.processor.add_external_data(df)

        return df

    # ========================================================================
    # Phase 3: Parallel Strategy Execution
    # ========================================================================

    def _load_strategy_registry(self) -> Dict:
        """
        Load strategy metadata from strategies/strategies_registry.json

        Returns:
            Dictionary with strategy metadata:
            {
                "AwesomeMacd": {
                    "enabled": true,
                    "timeframe": "1h",
                    "lookback_hours": 100,
                    ...
                }
            }
        """
        registry_path = Path("strategies") / "strategies_registry.json"
        try:
            with open(registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise StorageError(
                f"Failed to load strategy registry from {registry_path}: {e}"
            ) from e

    def _detect_existing_strategies(self, df: pd.DataFrame) -> Set[str]:
        """
        Scan DataFrame columns for existing strategy signals.

        Pattern: strategy_{name}_flat, strategy_{name}_long, etc.

        Args:
            df: DataFrame to scan

        Returns:
            Set of strategy names (lowercase) that exist in DataFrame
        """
        existing = set()

        for col in df.columns:
            if col.startswith('strategy_'):
                parts = col.split('_')
                if len(parts) >= 3:
                    name = parts[1]  # Extract strategy name
                    existing.add(name)

        return existing

    def _validate_strategy_columns(
        self,
        df: pd.DataFrame,
        strategy_name: str
    ) -> bool:
        """
        Validate strategy columns contain valid One-Hot encoded data.

        Checks 20 rows at start and 20 rows at end for:
        1. All values must be 0.0 or 1.0
        2. Exactly one signal must be 1.0 per row (sum == 1.0)

        Args:
            df: DataFrame to validate
            strategy_name: Name of strategy to check

        Returns:
            True if validation passes, False otherwise
        """
        name_lower = strategy_name.lower()
        cols = [
            f'strategy_{name_lower}_flat',
            f'strategy_{name_lower}_long',
            f'strategy_{name_lower}_short',
            f'strategy_{name_lower}_hold'
        ]

        # Check all columns exist
        if not all(col in df.columns for col in cols):
            return False

        # Sample 20 rows from start and 20 from end
        sample_indices = list(range(min(20, len(df))))  # Start
        sample_indices += list(range(max(0, len(df) - 20), len(df)))  # End

        sample = df.iloc[sample_indices][cols]

        # Check values are binary (0 or 1)
        if not ((sample == 0.0) | (sample == 1.0)).all().all():
            return False

        # Check exactly one signal per row (sum == 1.0)
        row_sums = sample.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            return False

        return True

    def _filter_strategies_to_execute(
        self,
        df: pd.DataFrame,
        requested: List[str],
        existing: Set[str],
        force: bool
    ) -> List[str]:
        """
        Determine which strategies need execution based on existing data.

        Args:
            df: DataFrame to check
            requested: List of requested strategy names
            existing: Set of existing strategy names
            force: If True, execute all strategies regardless

        Returns:
            List of strategy names to execute
        """
        if force:
            print("  Force recalculate enabled - executing all strategies")
            return requested

        to_execute = []
        for strategy_name in requested:
            name_lower = strategy_name.lower()

            if name_lower not in existing:
                to_execute.append(strategy_name)
                print(f"  Strategy {strategy_name}: Not found - will execute")
            else:
                # Validate existing data
                if self._validate_strategy_columns(df, strategy_name):
                    print(f"  Strategy {strategy_name}: Valid - skipping")
                else:
                    to_execute.append(strategy_name)
                    print(f"  Strategy {strategy_name}: Invalid data - will recalculate")

        return to_execute

    def _execute_strategies_parallel(
        self,
        df: pd.DataFrame,
        strategy_names: List[str]
    ) -> pd.DataFrame:
        """
        Execute strategies using ProcessPoolExecutor for parallel processing.

        IMPORTANT: Uses concurrent.futures.ProcessPoolExecutor (NOT threads)
        because strategy calculation is CPU-bound and needs to bypass GIL.

        Args:
            df: Base timeframe DataFrame
            strategy_names: List of strategy names to execute

        Returns:
            DataFrame with strategy signal columns added
        """
        # Load strategy metadata from JSON
        registry = self._load_strategy_registry()

        # Initialize columns
        for strategy_name in strategy_names:
            name_lower = strategy_name.lower()
            df[f'strategy_{name_lower}_flat'] = 0.0
            df[f'strategy_{name_lower}_long'] = 0.0
            df[f'strategy_{name_lower}_short'] = 0.0
            df[f'strategy_{name_lower}_hold'] = 0.0

        # Determine worker count
        max_workers = config.MAX_STRATEGY_WORKERS
        if max_workers is None:
            max_workers = max(1, mp.cpu_count() // 2)

        use_parallel = len(strategy_names) > 1 and max_workers > 1

        if use_parallel:
            self.logger.debug(f"Using {max_workers} parallel workers")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks
                futures = {}
                for name in strategy_names:
                    strategy_meta = registry[name]
                    future = executor.submit(
                        _process_single_strategy_with_resampling,
                        name,
                        df,
                        self.base_timeframe,
                        strategy_meta['timeframe'],
                        strategy_meta['lookback_hours']
                    )
                    futures[future] = name

                # Collect results with progress bar
                pbar = ProgressTracker.process_items(
                    total=len(strategy_names),
                    desc="Executing strategies",
                    unit="strategy"
                )
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        strategy_name, signal_arrays = future.result()
                        name_lower = strategy_name.lower()
                        df[f'strategy_{name_lower}_flat'] = signal_arrays['flat']
                        df[f'strategy_{name_lower}_long'] = signal_arrays['long']
                        df[f'strategy_{name_lower}_short'] = signal_arrays['short']
                        df[f'strategy_{name_lower}_hold'] = signal_arrays['hold']
                        pbar.set_postfix_str(f"{strategy_name}")
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(f"Strategy {name} failed: {e}")
                        # Set all HOLD as fallback
                        name_lower = name.lower()
                        df[f'strategy_{name_lower}_hold'] = 1.0
                        pbar.update(1)
                pbar.close()
        else:
            # Sequential execution
            print("  Using sequential execution")
            for name in strategy_names:
                try:
                    strategy_meta = registry[name]
                    strategy_name, signal_arrays = _process_single_strategy_with_resampling(
                        name,
                        df,
                        self.base_timeframe,
                        strategy_meta['timeframe'],
                        strategy_meta['lookback_hours']
                    )
                    name_lower = strategy_name.lower()
                    df[f'strategy_{name_lower}_flat'] = signal_arrays['flat']
                    df[f'strategy_{name_lower}_long'] = signal_arrays['long']
                    df[f'strategy_{name_lower}_short'] = signal_arrays['short']
                    df[f'strategy_{name_lower}_hold'] = signal_arrays['hold']
                    print(f"    Completed: {strategy_name}")
                except Exception as e:
                    print(f"    Failed: {name} - {e}")
                    name_lower = name.lower()
                    df[f'strategy_{name_lower}_hold'] = 1.0

        # CRITICAL: Apply forward fill then backward fill after merging all strategies
        # This ensures RL agent on 15m timeframe sees last known macro signal
        # instead of NaN (e.g., 1h strategy signal applied to all 15m candles)
        # Forward fill propagates signals forward in time
        # Backward fill handles any NaN at the beginning (before first valid signal)
        strategy_cols = [col for col in df.columns if col.startswith('strategy_')]
        if strategy_cols:
            df[strategy_cols] = df[strategy_cols].ffill().bfill()

            # Final safety check: replace any remaining NaN with HOLD signal
            # This should rarely happen, but ensures valid One-Hot encoding
            for strategy_name in strategy_names:
                name_lower = strategy_name.lower()
                cols = [
                    f'strategy_{name_lower}_flat',
                    f'strategy_{name_lower}_long',
                    f'strategy_{name_lower}_short',
                    f'strategy_{name_lower}_hold'
                ]
                # Check if any NaN remains
                if df[cols].isna().any().any():
                    print(f"  Warning: Found NaN in {strategy_name} signals - setting to HOLD")
                    df[f'strategy_{name_lower}_flat'] = df[f'strategy_{name_lower}_flat'].fillna(0.0)
                    df[f'strategy_{name_lower}_long'] = df[f'strategy_{name_lower}_long'].fillna(0.0)
                    df[f'strategy_{name_lower}_short'] = df[f'strategy_{name_lower}_short'].fillna(0.0)
                    df[f'strategy_{name_lower}_hold'] = df[f'strategy_{name_lower}_hold'].fillna(1.0)

        return df

    def add_strategy_signals(
        self,
        df: pd.DataFrame,
        strategy_list: List[str] = None,
        force_recalculate: bool = False
    ) -> pd.DataFrame:
        """
        Add strategy signals to DataFrame.

        Uses incremental logic:
        - Detects existing strategy columns
        - Validates existing data
        - Only executes missing or invalid strategies

        Args:
            df: DataFrame with base features
            strategy_list: List of strategy names to execute (default: config.STRATEGY_LIST)
            force_recalculate: If True, recalculate all strategies

        Returns:
            DataFrame with strategy signal columns added
        """
        if strategy_list is None:
            strategy_list = config.STRATEGY_LIST

        if not strategy_list:
            print("  No strategies requested")
            return df

        print(f"  Requested strategies: {', '.join(strategy_list)}")

        # Detect existing strategies
        existing = self._detect_existing_strategies(df)
        if existing:
            print(f"  Existing strategies: {', '.join(existing)}")

        # Filter strategies to execute
        to_execute = self._filter_strategies_to_execute(
            df, strategy_list, existing, force_recalculate
        )

        if not to_execute:
            print("  All strategies already exist and are valid")
            return df

        print(f"  Executing {len(to_execute)} strategy(s): {', '.join(to_execute)}")

        # Execute strategies
        df = self._execute_strategies_parallel(df, to_execute)

        return df

    # ========================================================================
    # Phase 4: Validation
    # ========================================================================

    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean data using existing CcxtProcessor.clean_data()

        This uses the exact clean_data logic:
        - Remove duplicates by (date, tic)
        - Sort by date and tic
        - Forward-fill then backward-fill missing values
        - Clip negative prices/volumes
        - Validate OHLC relationships

        Args:
            df: DataFrame to validate

        Returns:
            Cleaned and validated DataFrame
        """
        return self.processor.clean_data(df)

    # ========================================================================
    # Phase 5: Output
    # ========================================================================

    def to_arrays(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert DataFrame to numpy arrays for RL environment.

        Delegates to CcxtProcessor.df_to_array() method.

        Args:
            df: Fully processed DataFrame

        Returns:
            Tuple of (price_array, tech_array, turbulence_array, signal_array, datetime_array)
        """
        # Check if VIX is enabled in external assets
        vix_enabled = any(
            asset.get('enabled', False) and asset.get('col_name') == 'vix'
            for asset in config.EXTERNAL_ASSETS
        )

        price_array, tech_array, turbulence_array, signal_array = self.processor.df_to_array(
            df=df,
            if_vix=vix_enabled
        )

        datetime_array = df["date"].to_numpy()

        return price_array, tech_array, turbulence_array, signal_array, datetime_array

    # ========================================================================
    # High-Level API (Main Entry Points)
    # ========================================================================

    def get_processed_data(
        self,
        start_date: str,
        end_date: str,
        strategy_list: List[str] = None,
        force_redownload: bool = False
    ) -> pd.DataFrame:
        """
        Smart waterfall: processed cache → incremental update → raw fallback → save processed

        Flow:
        1. Try loading processed cache
        2. If found: validate date range + incrementally update missing indicators/strategies
        3. If not found or invalid: fallback to raw data + full processing
        4. Save processed data for future use

        Args:
            start_date: Start date "DD-MM-YYYY" or "YYYY-MM-DD"
            end_date: End date "DD-MM-YYYY" or "YYYY-MM-DD"
            strategy_list: List of strategy names (default: config.STRATEGY_LIST)
            force_redownload: If True, ignore existing data and redownload

        Returns:
            Fully processed DataFrame ready for array conversion
        """
        updated = False  # Track if DataFrame was modified

        # STEP A: Try loading from processed cache
        if config.USE_PREPROCESSED_DATA and not force_redownload:
            processed_path = self.storage_path / "training_data" / "processed" / self._get_processed_filename()

            if processed_path.exists():
                self.logger.info(f"Found cached data at {processed_path.name}")
                df = pd.read_parquet(processed_path)

                # Validate date range (superset match)
                if self._validate_date_range(df, start_date, end_date):
                    self.logger.debug("Date range validated. Checking features...")

                    # 1. INDICATOR CHECK
                    missing_indicators = [ind for ind in config.INDICATORS if ind not in df.columns]
                    if missing_indicators:
                        self.logger.info(f"Missing indicators: {missing_indicators}")
                        self.logger.debug("Calculating missing indicators...")
                        df = self.processor.add_technical_indicator(df, missing_indicators)
                        updated = True
                    else:
                        self.logger.debug(f"All indicators present ({len(config.INDICATORS)} total)")

                    # 2. STRATEGY CHECK (Smart Incremental Update)
                    if config.ENABLE_STRATEGIES and strategy_list:
                        existing_strategies = self._detect_existing_strategies(df)
                        self.logger.debug(f"Existing strategies: {existing_strategies}")

                        # Use existing filter logic to determine what needs execution
                        strategies_to_execute = self._filter_strategies_to_execute(
                            df,
                            strategy_list,
                            existing_strategies,
                            force=False
                        )

                        if strategies_to_execute:
                            self.logger.debug(f"Executing {len(strategies_to_execute)} missing/invalid strategy(s)...")
                            df = self._execute_strategies_parallel(df, strategies_to_execute)
                            updated = True
                        else:
                            self.logger.debug(f"All strategies valid ({len(existing_strategies)} total)")

                    # 3. AUTO-SAVE if updated
                    if updated:
                        self.logger.debug(f"Features updated. Saving...")
                        self._save_processed_data(df, processed_path)

                    # Filter to requested range and return
                    self.logger.info("Loading from cache (fast path)")
                    df = self._filter_date_range(df, start_date, end_date)
                    return df  # IMMEDIATE RETURN
                else:
                    self.logger.warning("Date range insufficient. Falling back to raw data...")

        # STEP B: Fallback to raw data + full processing
        print("\n=== Phase 1: Downloading data ===")
        raw_path = self.storage_path / "training_data" / "raw" / self._get_storage_filename()
        if raw_path.exists():
            print(f"[INFO] Found existing Raw Data at {raw_path}. Checking for gaps...")

        df = self.get_or_download_data(start_date, end_date, force_redownload)
        print(f"Downloaded {len(df)} rows")

        print("\n=== Phase 2: Adding technical indicators ===")
        df = self.add_base_features(df)
        print(f"Added {len(config.INDICATORS)} indicators")

        if config.ENABLE_STRATEGIES and strategy_list:
            print("\n=== Phase 3: Executing strategies ===")
            df = self.add_strategy_signals(df, strategy_list)
            n_strategies = len([col for col in df.columns if col.startswith('strategy_')]) // 4
            print(f"Added signals from {n_strategies} strategy(s)")

        print("\n=== Phase 4: Validating and cleaning ===")
        df = self.validate_and_clean(df)
        print(f"Final shape: {df.shape}")

        # STEP C: Save processed data for future use
        if config.USE_PREPROCESSED_DATA:
            processed_path = self.storage_path / "training_data" / "processed" / self._get_processed_filename()
            self._save_processed_data(df, processed_path)

        return df

    def get_arrays(
        self,
        start_date: str,
        end_date: str,
        strategy_list: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete pipeline: download -> features -> strategies -> arrays

        Args:
            start_date: Start date "YYYY-MM-DD"
            end_date: End date "YYYY-MM-DD"
            strategy_list: List of strategy names (default: config.STRATEGY_LIST)

        Returns:
            Tuple of (price_array, tech_array, turbulence_array, signal_array, datetime_array)
        """
        df = self.get_processed_data(start_date, end_date, strategy_list)

        print("\n=== Phase 5: Converting to arrays ===")
        arrays = self.to_arrays(df)
        price_array, tech_array, turbulence_array, signal_array, datetime_array = arrays
        print(f"  Price array shape: {price_array.shape} (OHLCV)")
        print(f"  Tech array shape: {tech_array.shape} (indicators only)")
        print(f"  Turbulence array shape: {turbulence_array.shape} (turbulence + VIX)")
        print(f"  Signal array shape: {signal_array.shape} (strategy signals)")
        print(f"  Datetime array: {datetime_array.shape[0]} timestamps (date-time info)")

        return arrays
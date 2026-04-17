"""
Data processor for Bitcoin market data using CCXT
Adapted from FinRL's DataProcessor pattern for cryptocurrency trading
"""

from __future__ import annotations

import ccxt
import warnings
import numpy as np
import pandas as pd
import talib
from datetime import datetime
import config
import yfinance as yf
from utils.progress import ProgressTracker
from data.external_manager import ExternalDataManager
from utils.logger import RLLogger, LogComponent
from utils.date_display import format_date_range_for_display


def _strategy_signal_columns(strategy_list: list[str]) -> tuple[list[str], list[str]]:
    """Build strategy signal columns in canonical block order."""
    strategy_cols: list[str] = []
    resolved_names: list[str] = []

    for strategy_name in strategy_list:
        name_lower = strategy_name.lower()
        strategy_cols.extend([
            f"strategy_{name_lower}_flat",
            f"strategy_{name_lower}_long",
            f"strategy_{name_lower}_short",
            f"strategy_{name_lower}_hold",
        ])
        resolved_names.append(strategy_name)

    return strategy_cols, resolved_names

class CcxtProcessor:
    """
    Processor for downloading and processing Bitcoin market data via CCXT
    Supports any CCXT exchange (Binance, Kraken, etc.)
    """

    def __init__(self, exchange_name: str = None, logger=None):
        """
        Initialize CCXT exchange connection

        Args:
            exchange_name: CCXT exchange identifier (default from config)
            logger: Logger instance (optional, but recommended)
        """
        self.exchange_name = exchange_name or config.EXCHANGE_NAME
        self.logger = logger  # May be None initially
        self._initialize_exchange()

    def _initialize_exchange(self) -> None:
        """Initialize CCXT exchange with proper settings"""
        exchange_class = getattr(ccxt, self.exchange_name) #Dynamically gets the exchange class from CCXT library

        self.exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if ':' in config.TRADING_PAIR else 'spot'
            }
        }) #This dictionary gets passed to the exchange constructor and configures it.

        # Exchange initialized (print removed to avoid interfering with user prompts)

    def _convert_to_iso_date(self, date_str: str) -> str:
        """
        Convert date string to ISO 8601 format (YYYY-MM-DD)
        Handles both DD-MM-YYYY and YYYY-MM-DD formats

        Args:
            date_str: Date string in DD-MM-YYYY or YYYY-MM-DD format

        Returns:
            Date string in YYYY-MM-DD format
        """
        # Try to parse DD-MM-YYYY format first (user input format)
        try:
            dt = datetime.strptime(date_str, "%d-%m-%Y")
            iso_date = dt.strftime("%Y-%m-%d")
            if self.logger:
                self.logger.debug(f"Date conversion: '{date_str}' (DD-MM-YYYY) → {iso_date} ({dt.strftime('%B %d, %Y')})")
            return iso_date
        except ValueError:
            pass

        # Try YYYY-MM-DD format (already ISO)
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            iso_date = dt.strftime("%Y-%m-%d")
            if self.logger:
                self.logger.debug(f"Date already in ISO format: '{date_str}' → {iso_date}")
            return iso_date
        except ValueError:
            raise ValueError(
                f"Invalid date format: '{date_str}'. "
                f"Expected DD-MM-YYYY or YYYY-MM-DD"
            )

    def download_data(
            self,
            ticker_list: list[str],
            start_date: str,
            end_date: str,
            time_interval: str
    ) -> pd.DataFrame:
        """
        Download OHLCV market data from exchange

        Args:
            ticker_list: List of trading symbols (e.g., ['BTC/USDT'])
            start_date: Start date in 'DD-MM-YYYY' or 'YYYY-MM-DD' format
            end_date: End date in 'DD-MM-YYYY' or 'YYYY-MM-DD' format
            time_interval: Candle timeframe (e.g., '15m', '1h', '1d')

        Returns:
            DataFrame with columns: [date, open, high, low, close, volume, tic]
        """
        all_data = []

        for symbol in ticker_list:
            # Convert dates to ISO format first for internal processing
            start_iso = self._convert_to_iso_date(start_date)
            end_iso = self._convert_to_iso_date(end_date)

            # Display with DD-MM-YYYY format for user-facing logs
            date_range_display = format_date_range_for_display(start_iso, end_iso)

            if self.logger:
                self.logger.info(f"Downloading {symbol} data (Timeframe: {time_interval}, Period: {date_range_display})")

            since = self.exchange.parse8601(f"{start_iso}T00:00:00Z")
            until = self.exchange.parse8601(f"{end_iso}T23:59:59Z")

            # Fetch data in chunks
            candles = self._fetch_ohlcv_chunks(symbol, time_interval, since, until)

            # Convert to DataFrame format
            for candle in candles:
                all_data.append({
                    'timestamp': candle[0],
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5],
                    'tic': symbol
                })

            if self.logger:
                self.logger.info(f"Downloaded {len(candles)} candles for {symbol}")

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Handle empty downloads (e.g., future dates)
        if len(df) == 0:
            if self.logger:
                self.logger.warning("No data available for requested range")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'tic'])

        # Add human-readable date
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Reorder columns (FinRL format)
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'tic']] #FinRL environment code expects: date, open, high, low, close, volume, tic

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True) #drop=True means "don't keep the old index as a column"

        # Log summary with date formatters
        if self.logger:
            actual_range = format_date_range_for_display(
                df['date'].min().strftime('%Y-%m-%d'),
                df['date'].max().strftime('%Y-%m-%d')
            )
            self.logger.info(f"Total data points: {len(df)}, Date range: {actual_range}")

        return df

    def _fetch_ohlcv_chunks(
            self,
            symbol: str,
            timeframe: str,
            since: int,
            until: int
    ) -> list:
        """
        Fetch OHLCV data in chunks to handle exchange limits

        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
            since: Start timestamp (ms)
            until: End timestamp (ms)

        Returns:
            List of OHLCV candles
        """
        all_candles = []
        current_since = since

        # Determine chunk size based on exchange
        limit = 1000 if self.exchange_name == 'binance' else 720

        # Estimate total chunks for progress bar
        # Calculate milliseconds per candle based on timeframe
        timeframe_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }.get(timeframe, 15 * 60 * 1000)  # Default to 15m

        estimated_candles = (until - since) / timeframe_ms
        estimated_chunks = max(1, int(estimated_candles / limit))

        # Create progress bar
        pbar = ProgressTracker.download_chunks(total=estimated_chunks, desc=f"Downloading {symbol}")

        while current_since < until:
            try:
                # Fetch one chunk
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit
                )

                if not candles:
                    break

                # Add to collection
                all_candles.extend(candles)

                # Update progress bar
                pbar.update(1)

                # Move to next chunk
                current_since = candles[-1][0] + 1

                # Stop if we've reached the end date
                if current_since >= until:
                    break

            except ccxt.NetworkError as e:
                if self.logger:
                    self.logger.warning(f"Network error: {e}. Retrying...")
                continue

            except ccxt.ExchangeError as e:
                if self.logger:
                    self.logger.error(f"Exchange error: {e}")
                break

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Unexpected error: {e}")
                break

        # Ensure progress bar reaches 100% and close
        if pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)
        pbar.close()

        return all_candles

    #TODO: this function need to be tested
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate market data
        """
        if self.logger:
            self.logger.info("Cleaning data...")

        initial_rows = len(df)

        # Remove duplicates

        # Validate OHLC relationships
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
            ]

        final_rows = len(df)
        removed = initial_rows - final_rows

        if self.logger:
            self.logger.info(f"Cleaned data: {final_rows} rows (removed {removed} invalid rows)")

        return df

    def add_technical_indicator(
            self,
            df: pd.DataFrame,
            tech_indicator_list: list[str]
    ) -> pd.DataFrame:
        """
        Add technical indicators using TA-Lib

        Args:
            df: OHLCV DataFrame
            tech_indicator_list: List of indicator names

        Returns:
            DataFrame with added technical indicators
        """
        if self.logger:
            self.logger.info(f"Calculating {len(tech_indicator_list)} technical indicators...")

        df = df.copy()

        # Process each ticker separately
        for tic in df['tic'].unique(): #Get unique tickers in the DataFrame
            mask = df['tic'] == tic #Create a boolean mask for the current ticker
            tic_data = df[mask].copy() #Extract data for the current ticker

            # Calculate each indicator
            for indicator in tech_indicator_list:
                indicator_lower = indicator.lower()

                if indicator_lower == 'macd':
                    macd, signal, hist = talib.MACD(
                        tic_data['close'],
                        fastperiod=12,
                        slowperiod=26,
                        signalperiod=9
                    )
                    df.loc[mask, 'macd'] = macd

                elif indicator_lower == 'rsi_30':
                    df.loc[mask, 'rsi_30'] = talib.RSI(tic_data['close'], timeperiod=30)

                elif indicator_lower == 'dx_30':
                    df.loc[mask, 'dx_30'] = talib.DX(
                        tic_data['high'],
                        tic_data['low'],
                        tic_data['close'],
                        timeperiod=30
                    )

                elif indicator_lower == 'close_30_sma':
                    df.loc[mask, 'close_30_sma'] = talib.SMA(tic_data['close'], timeperiod=30)

                elif indicator_lower == 'close_60_sma':
                    df.loc[mask, 'close_60_sma'] = talib.SMA(tic_data['close'], timeperiod=60)

                elif indicator_lower == 'boll_ub':
                    upper, middle, lower = talib.BBANDS(
                        tic_data['close'],
                        timeperiod=20,
                        nbdevup=2,
                        nbdevdn=2
                    )
                    df.loc[mask, 'boll_ub'] = upper

                elif indicator_lower == 'boll_lb':
                    upper, middle, lower = talib.BBANDS(
                        tic_data['close'],
                        timeperiod=20,
                        nbdevup=2,
                        nbdevdn=2
                    )
                    df.loc[mask, 'boll_lb'] = lower

        if self.logger:
            self.logger.info("Technical indicators calculated")

        return df

    def add_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add turbulence index (market stress indicator)
        Controlled by config.ENABLE_TURBULENCE
        """
        if not config.ENABLE_TURBULENCE:
            if self.logger:
                self.logger.info("Turbulence calculation disabled (config.ENABLE_TURBULENCE = False)")
            return df

        if self.logger:
            self.logger.info("Calculating turbulence index...")
        df = df.copy()

        df['turbulence'] = df.groupby('tic')['close'].transform(
            lambda x: x.pct_change().rolling(window=20).std()
        )
        df['turbulence'] = df['turbulence'].fillna(0)

        if self.logger:
            self.logger.info("Turbulence index calculated")
        return df



    def df_to_array(
            self,
            df: pd.DataFrame,
            tech_indicator_list: list[str],
            if_vix: bool,
            strategy_list: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """
        Convert DataFrame to numpy arrays for RL environment

        Args:
            df: Processed DataFrame with all features
            tech_indicator_list: List of technical indicators to include
            if_vix: Whether to include VIX in the output

        Returns:
            Tuple of (price_array, tech_array, turbulence_array, signal_array, strategy_names)

        Notes:
            Datetime information is NOT returned here.
            It is handled at DataManager level as a separate datetime_array
            for logging and backtesting purposes.
        """
        # Price array: [open, high, low, close, volume]
        price_array = df[['open', 'high', 'low', 'close', 'volume']].values

        # Technical indicators array (VIX belongs in turbulence_array, not here)
        tech_array = df[tech_indicator_list].values if tech_indicator_list else np.array([])

        # Defensive check: tech_array should have no NaN after warmup trimming.
        # If NaN survive, it indicates a data gap or missing warmup — log a
        # warning so it doesn't silently produce misleading zeros (e.g. RSI=0).
        if tech_indicator_list and np.isnan(tech_array).any():
            nan_counts = np.isnan(tech_array).sum(axis=0)
            details = ", ".join(
                f"{name}: {int(cnt)}"
                for name, cnt in zip(tech_indicator_list, nan_counts) if cnt > 0
            )
            warnings.warn(
                f"tech_array contains NaN AFTER warmup trimming — replaced with 0.0. "
                f"This should not happen. Per-indicator NaN counts: [{details}]"
            )
        tech_array = np.nan_to_num(tech_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Turbulence array: [turbulence, vix] (both optional based on config)
        turb_cols = []
        if 'turbulence' in df.columns:
            turb_cols.append('turbulence')
        if if_vix and 'vix' in df.columns:
            turb_cols.append('vix')

        # Create 2D array with shape (T, N) where N = number of enabled features
        if turb_cols:
            turbulence_array = df[turb_cols].values
        else:
            turbulence_array = np.zeros((len(df), 0), dtype=np.float32)

        requested_strategies = strategy_list if strategy_list is not None else config.STRATEGY_LIST
        strategy_cols: list[str] = []
        strategy_names: list[str] = []

        if config.ENABLE_STRATEGIES and requested_strategies:
            strategy_cols, strategy_names = _strategy_signal_columns(requested_strategies)
            missing_cols = [col for col in strategy_cols if col not in df.columns]
            if missing_cols:
                # The variance filter in DataManager may have removed dead strategies
                # from the DataFrame before it reaches here. Silently restrict to
                # only the strategies whose full set of 4 one-hot columns is present.
                strategy_names = [
                    name for name in strategy_names
                    if all(
                        f"strategy_{name.lower()}_{s}" in df.columns
                        for s in ("flat", "long", "short", "hold")
                    )
                ]
                strategy_cols = [c for c in strategy_cols if c in df.columns]

        if strategy_cols:
            signal_array = df[strategy_cols].values  # Shape: (T, S*4)
        else:
            signal_array = np.zeros((len(df), 0), dtype=np.float32)

        # Defensive check: turbulence_array should NEVER have NaNs at this point
        if np.isnan(turbulence_array).any():
            nan_count = np.isnan(turbulence_array).sum()
            raise ValueError(
                f"Turbulence array contains {nan_count} NaN values. "
                f"VIX or turbulence data incomplete. Check add_vix() and add_turbulence()."
            )

        turbulence_array = np.nan_to_num(turbulence_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Shapes printed in data_manager.py (removed duplicate)

        return price_array, tech_array, turbulence_array, signal_array, strategy_names


class DataProcessor:
    """
    Unified data processor wrapper (FinRL-compatible interface)
    """

    def __init__(
            self,
            data_source: str = None,
            tech_indicator: list[str] = None,
            vix: bool = None,
            **kwargs
    ):
        """
        Initialize data processor

        Args:
            data_source: Exchange name (default from config)
            tech_indicator: List of technical indicators
            vix: Whether to include VIX proxy
        """
        self.data_source = data_source or config.EXCHANGE_NAME
        self.processor = CcxtProcessor(self.data_source)
        self.tech_indicator_list = tech_indicator or config.INDICATORS
        logger = RLLogger("DataProcessor")
        self.external_manager = ExternalDataManager(config.EXTERNAL_ASSETS, logger)

    def download_data(
            self,
            ticker_list: list[str],
            start_date: str,
            end_date: str,
            time_interval: str
    ) -> pd.DataFrame:
        """Download market data"""
        return self.processor.download_data(ticker_list, start_date, end_date, time_interval)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean market data"""
        return self.processor.clean_data(df)

    def add_technical_indicator(
            self,
            df: pd.DataFrame,
            tech_indicator_list: list[str]
    ) -> pd.DataFrame:
        """Add technical indicators"""
        self.tech_indicator_list = tech_indicator_list
        return self.processor.add_technical_indicator(df, tech_indicator_list)

    def add_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add turbulence index"""
        return self.processor.add_turbulence(df)

    def add_external_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add external market data (VIX, etc.) using ExternalDataManager.

        External assets are configured via config.EXTERNAL_ASSETS.
        Each asset must have 'enabled': True to be processed.

        Returns:
            DataFrame with external data columns added (e.g., 'vix')
        """
        # Check if any external assets are enabled
        enabled_assets = [a for a in config.EXTERNAL_ASSETS if a.get('enabled', False)]
        if not enabled_assets:
            return df

        try:
            df = self.external_manager.enrich_data(df)

            # Validate that expected columns were added
            for asset in enabled_assets:
                col_name = asset.get('col_name')
                if col_name and col_name not in df.columns:
                    raise ValueError(f"Expected column '{col_name}' not found after enrichment")

            return df

        except Exception as e:
            error_msg = f"Failed to add external data: {e}"
            # Use logger if available, otherwise print
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            else:
                print(f"ERROR: {error_msg}")

            # Re-raise to fail fast (don't silently continue without external data)
            raise RuntimeError(error_msg) from e

    def df_to_array(
            self,
            df: pd.DataFrame,
            if_vix: bool,
            strategy_list: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Convert DataFrame to arrays"""
        return self.processor.df_to_array(
            df,
            self.tech_indicator_list,
            if_vix,
            strategy_list=strategy_list,
        )

    def _convert_to_iso_date(self, date_str: str) -> str:
        """Convert date string to ISO format (delegates to CcxtProcessor)"""
        return self.processor._convert_to_iso_date(date_str)

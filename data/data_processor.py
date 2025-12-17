"""
Data processor for Bitcoin market data using CCXT
Adapted from FinRL's DataProcessor pattern for cryptocurrency trading
"""

from __future__ import annotations

import ccxt
import numpy as np
import pandas as pd
import talib
from datetime import datetime
import config
from utils.progress import ProgressTracker


class CcxtProcessor:
    """
    Processor for downloading and processing Bitcoin market data via CCXT
    Supports any CCXT exchange (Binance, Kraken, etc.)
    """

    def __init__(self, exchange_name: str = None):
        """
        Initialize CCXT exchange connection

        Args:
            exchange_name: CCXT exchange identifier (default from config)
        """
        self.exchange_name = exchange_name or config.EXCHANGE_NAME
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

        print(f"{self.exchange_name.capitalize()} exchange initialized")

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
            print(f"   [DATE DEBUG] Input: '{date_str}' (DD-MM-YYYY) → Parsed as: {iso_date} ({dt.strftime('%B %d, %Y')})")
            return iso_date
        except ValueError:
            pass

        # Try YYYY-MM-DD format (already ISO)
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            iso_date = dt.strftime("%Y-%m-%d")
            print(f"   [DATE DEBUG] Input: '{date_str}' (YYYY-MM-DD) → Already ISO: {iso_date}")
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
            print(f"\n Downloading {symbol} data...")
            print(f"   Timeframe: {time_interval}")
            print(f"   Period: {start_date} to {end_date}")

            # Convert dates to timestamps
            # Handle both DD-MM-YYYY and YYYY-MM-DD formats
            start_iso = self._convert_to_iso_date(start_date)
            end_iso = self._convert_to_iso_date(end_date)

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

            print(f" Downloaded {len(candles)} candles for {symbol}")

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Handle empty downloads (e.g., future dates)
        if len(df) == 0:
            print(" Warning: No data available for requested range")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'tic'])

        # Add human-readable date
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Reorder columns (FinRL format)
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'tic']] #FinRL environment code expects: date, open, high, low, close, volume, tic

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True) #drop=True means "don't keep the old index as a column"

        print(f"\n Total data points: {len(df)}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

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
                print(f"\n   Network error: {e}. Retrying...")
                continue

            except ccxt.ExchangeError as e:
                print(f"\n   Exchange error: {e}")
                break

            except Exception as e:
                print(f"\n   Unexpected error: {e}")
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
        print("\n  Cleaning data...")

        initial_rows = len(df)

        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'tic'])

        # Sort by date and ticker
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)

        # Handle missing values by forward-fill then backward-fill
        df = df.ffill().bfill()

        # Ensure no negative prices or volumes
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].clip(lower=0)
        df['volume'] = df['volume'].clip(lower=0)

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

        print(f"  Cleaned data: {final_rows} rows (removed {removed} invalid rows)")

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
        print(f"\n  Calculating {len(tech_indicator_list)} technical indicators...")

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

        print(f"  Technical indicators calculated")

        return df

    def add_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add turbulence index (market stress indicator)
        Controlled by config.ENABLE_TURBULENCE
        """
        if not config.ENABLE_TURBULENCE:
            print("  Turbulence calculation disabled (config.ENABLE_TURBULENCE = False)")
            return df

        print("\n  Calculating turbulence index...")
        df = df.copy()

        df['turbulence'] = df.groupby('tic')['close'].transform(
            lambda x: x.pct_change().rolling(window=20).std()
        )
        df['turbulence'] = df['turbulence'].fillna(0)

        print("  Turbulence index calculated")
        return df

    def add_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add real VIX (CBOE Volatility Index) data from S&P 500

        Downloads actual VIX from Yahoo Finance
        - VIX only trades during S&P 500 market hours
        - Weekends/holidays use last available VIX value (forward-fill)
        - Daily VIX value applied to all candles on that day

        Controlled by config.ENABLE_VIX
        """
        if not config.ENABLE_VIX:
            print("  VIX disabled")
            return df

        print(f"\n  Downloading real VIX ({config.VIX_SYMBOL})...")

        try:
            import yfinance as yf

            # Get date range from our data
            start_date = df['date'].min()
            end_date = df['date'].max()

            print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            # Download VIX data
            # IMPORTANT: yfinance end parameter is EXCLUSIVE, so add 1 day
            end_date_exclusive = end_date + pd.Timedelta(days=1)

            vix_data = yf.download(
                config.VIX_SYMBOL,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date_exclusive.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True  # Explicitly set to avoid warning
            )

            if vix_data.empty:
                raise ValueError("No VIX data returned from Yahoo Finance")

            # Handle MultiIndex columns (yfinance recent versions)
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_data.columns = vix_data.columns.get_level_values(0)

            # Prepare VIX data for merge
            vix_df = pd.DataFrame({
                'date': vix_data.index,
                'vix': vix_data['Close'].values
            })

            # Convert to datetime
            vix_df['date'] = pd.to_datetime(vix_df['date'])
            df['date'] = pd.to_datetime(df['date'])

            # VIX is daily, Bitcoin is 15-minute candles
            # Match by date only (not time)
            df['date_only'] = df['date'].dt.date

            # Shift VIX dates forward by 1 day so VIX from day N is applied to day N+1
            # This prevents look-ahead bias (candles on day N use VIX from N-1 close)
            vix_df['date_only'] = (pd.to_datetime(vix_df['date']) + pd.Timedelta(days=1)).dt.date

            # Merge VIX data
            df = df.merge(
                vix_df[['date_only', 'vix']],
                on='date_only',
                how='left'
            )

            # Debug: Verify no look-ahead bias
            print(f"  VIX date shift applied (prevents look-ahead bias)")
            if len(df) > 0:
                sample = df[['date', 'vix']].head(5)
                print(f"  First 5 rows after merge:")
                for _, row in sample.iterrows():
                    vix_val = row['vix'] if pd.notna(row['vix']) else 'NaN'
                    print(f"    {row['date'].strftime('%Y-%m-%d %H:%M')} → VIX={vix_val}")

            # Backward fill FIRST - handles case where VIX starts after Bitcoin data
            df['vix'] = df['vix'].bfill()

            # Forward fill SECOND - propagates VIX through weekends/holidays
            df['vix'] = df['vix'].ffill()

            # Drop temporary column
            df = df.drop('date_only', axis=1)

            # Statistics and validation
            vix_coverage = (df['vix'].notna().sum() / len(df)) * 100
            avg_vix = df['vix'].mean()
            max_vix = df['vix'].max()
            min_vix = df['vix'].min()

            print(f"   Real VIX added")
            print(f"   Coverage: {vix_coverage:.1f}%")
            print(f"   Average VIX: {avg_vix:.2f}")
            print(f"   Range: {min_vix:.2f} - {max_vix:.2f}")

            # Moderate validation: warn < 95%, error < 50%
            if vix_coverage < 95.0:
                print(f"  ⚠️  WARNING: VIX coverage is {vix_coverage:.1f}% (< 95%)")
                print(f"      Some Bitcoin candles may have interpolated VIX data")

            if vix_coverage < 50.0:
                raise ValueError(
                    f"VIX coverage critically low ({vix_coverage:.1f}%). "
                    f"Check date range alignment with VIX availability."
                )

            # Final check: ensure NO NaNs remain after fill operations
            if df['vix'].isna().any():
                nan_count = df['vix'].isna().sum()
                raise ValueError(
                    f"VIX column contains {nan_count} NaN values after fill operations. "
                    f"VIX data may be completely unavailable for your date range."
                )

            return df

        except ImportError:
            raise ImportError(
                "yfinance package required for VIX data.\n"
                "Install with: pip install yfinance"
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to download VIX data: {e}\n"
                f"Check your internet connection and try again."
            )

    def df_to_array(
            self,
            df: pd.DataFrame,
            tech_indicator_list: list[str],
            if_vix: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert DataFrame to numpy arrays for RL environment

        Args:
            df: Processed DataFrame with all features
            tech_indicator_list: List of technical indicators to include
            if_vix: Whether to include VIX in the output

        Returns:
            Tuple of (price_array, tech_array, turbulence_array, signal_array)

        Notes:
            Datetime information is NOT returned here.
            It is handled at DataManager level as a separate datetime_array
            for logging and backtesting purposes.
        """
        print("\n Converting to numpy arrays...")

        # Price array: [open, high, low, close, volume]
        price_array = df[['open', 'high', 'low', 'close', 'volume']].values

        # Technical indicators array (VIX belongs in turbulence_array, not here)
        tech_array = df[tech_indicator_list].values if tech_indicator_list else np.array([])

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

        # Strategy signal array (One-Hot encoded)
        # Columns follow pattern: strategy_{name}_flat, strategy_{name}_long, etc.
        strategy_cols = sorted([col for col in df.columns if col.startswith('strategy_')])

        if strategy_cols and config.ENABLE_STRATEGIES:
            signal_array = df[strategy_cols].values  # Shape: (T, S*4)
            # Already binary (0 or 1), no need for nan_to_num
        else:
            # No strategies enabled - return empty array
            signal_array = np.zeros((len(df), 0), dtype=np.float32)

        # Handle NaN and Inf values in numeric arrays
        tech_array = np.nan_to_num(tech_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Defensive check: turbulence_array should NEVER have NaNs at this point
        if np.isnan(turbulence_array).any():
            nan_count = np.isnan(turbulence_array).sum()
            raise ValueError(
                f"Turbulence array contains {nan_count} NaN values. "
                f"VIX or turbulence data incomplete. Check add_vix() and add_turbulence()."
            )

        turbulence_array = np.nan_to_num(turbulence_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Shapes printed in data_manager.py (removed duplicate)

        return price_array, tech_array, turbulence_array, signal_array


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
        self.vix = vix if vix is not None else False

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

    def add_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VIX proxy"""
        return self.processor.add_vix(df)

    def df_to_array(
            self,
            df: pd.DataFrame,
            if_vix: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert DataFrame to arrays"""
        return self.processor.df_to_array(df, self.tech_indicator_list, if_vix)
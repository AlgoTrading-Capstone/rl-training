"""
External Data Manager (Daily -> Intraday Upsampling)
Fetches Daily VIX data from Yahoo Finance and broadcasts it
to match the Bitcoin timeframe (e.g., 15m) using Forward Fill.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import List, Dict
from utils.logger import RLLogger, LogComponent
from utils.formatting import Formatter


class ExternalDataManager:
    def __init__(self, assets_config: List[Dict], logger: RLLogger):
        self.logger = logger.for_component(LogComponent.DATA) # Data-related logs
        self.assets_config = [asset for asset in assets_config if asset.get('enabled', False)]

    def enrich_data(self, main_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point: Merges external data into the Bitcoin DataFrame.
        """
        if not self.assets_config or main_df.empty:
            return main_df

        # CRITICAL: Validate main dataframe has 'date' column
        if 'date' not in main_df.columns:
            raise ValueError(
                "ExternalDataManager requires 'date' column in main_df. "
                f"Found columns: {list(main_df.columns)}"
            )

        # Ensure UTC for alignment
        if main_df['date'].dt.tz is None: #tz means timezone, check if timezone is none
            main_df['date'] = main_df['date'].dt.tz_localize('UTC') #localize means assign timezone
        else:
            main_df['date'] = main_df['date'].dt.tz_convert('UTC')

        # We need the index to be datetime for reindexing to work
        temp_df = main_df.set_index('date').copy()

        with self.logger.phase("Enriching External Data", 1, len(self.assets_config)):
            for asset in self.assets_config:
                temp_df = self._process_daily_asset(temp_df, asset)

        # Restore structure
        main_df = temp_df.reset_index()

        # Convert back to naive if needed (optional, keeps consistency)
        main_df['date'] = main_df['date'].dt.tz_localize(None)

        return main_df

    def _process_daily_asset(self, main_df: pd.DataFrame, asset: Dict) -> pd.DataFrame:
        ticker = asset['ticker']
        col_name = asset['col_name']
        local_path = Path(asset['local_path'])

        self.logger.info(f"Processing {ticker} (Daily -> Upsampling)...")

        # 1. Fetch Full History (1990+)
        daily_df = self._get_daily_data(ticker, local_path)

        if daily_df.empty:
            return main_df

        # -----------------------------------------------------------
        # [STEP 1] Fill Calendar Gaps
        # -----------------------------------------------------------
        self.logger.debug(f"Resampling {ticker} to full calendar days...")
        daily_df = daily_df.resample('D').ffill()

        # -----------------------------------------------------------
        # [STEP 2] Prevent Lookahead Bias (Shift)
        # -----------------------------------------------------------
        self.logger.debug(f"Applying 1-day shift to {ticker}...")
        daily_df['price'] = daily_df['price'].shift(1)

        # -----------------------------------------------------------
        # [STEP 3] Upsampling Logic
        # -----------------------------------------------------------
        self.logger.debug(f"Broadcasting daily {ticker} to intraday timeframe...")
        aligned_series = daily_df['price'].reindex(main_df.index, method='ffill')



        #-----------------------------------------------------------
        # [STEP 4] Insert into Main DataFrame
        #-----------------------------------------------------------
        main_df[col_name] = aligned_series

        return main_df

    def _get_daily_data(self, ticker: str, local_path: Path) -> pd.DataFrame:
        """
        Fetches daily data from cache or Yahoo Finance.
        """
        # A. Try Loading Cache
        if local_path.exists():
            try:
                self.logger.debug(f"Loading {ticker} from cache: {local_path}")
                df = pd.read_csv(local_path)
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                df = df.set_index('Date').sort_index()
                # Check if cache is recent (optional optimization for later)
                return df.rename(columns={'Close': 'price'})[['price']]
            except Exception as e:
                self.logger.warning(f"Corrupt cache for {ticker}: {e}. Redownloading.")

        # B. Download from Yahoo
        self.logger.info(f"Downloading full history for {ticker}...")
        try:
            # Download full history to ensure coverage
            # Note: auto_adjust=True renames 'Adj Close' to 'Close'
            # multi_level_index=False prevents MultiIndex columns like ('Close', '^VIX')
            data = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=True, multi_level_index=False)

            if data.empty:
                self.logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Ensure UTC Index
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            else:
                data.index = data.index.tz_convert('UTC')

            # Prepare DataFrame for caching with proper structure
            cache_df = pd.DataFrame({
                'Date': data.index,
                'Close': data['Close'].values
            })

            # Save to Cache
            local_path.parent.mkdir(parents=True, exist_ok=True)
            cache_df.to_csv(local_path, index=False)
            self.logger.success(f"Cached {ticker} data to {local_path}")

            # Return with proper index
            return pd.DataFrame({'price': data['Close']}, index=data.index)

        except Exception as e:
            msg = Formatter.error_context(f"YFinance Failed for {ticker}", "Check internet connection")
            self.logger.error(msg)
            self.logger.debug(f"Exception details: {e}")
            return pd.DataFrame()
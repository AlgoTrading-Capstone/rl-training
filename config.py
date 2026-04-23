"""
Configuration file
"""

# RL algorithm selection: "PPO", "SAC"
# PPO: on-policy, more stable, slower, good baseline
# SAC: off-policy, more sample-efficient, higher variance
RL_MODEL = "PPO"

# Discount factor (gamma) for future rewards - determines how much the agent values long-term gains versus immediate rewards.
# Typical range: 0.970 – 0.995
GAMMA = 0.990

# Learning rate for neural network optimization- Controls how aggressively the agent updates its policy/value networks.
# Typical ranges:
# PPO: 1e-4 – 3e-4
# SAC: 3e-4 – 1e-3
LEARNING_RATE = 3e-4

# Hidden layer dimensions for Actor & Critic networks.
# IMPORTANT: First layer should be >= state_dim
# Recommended defaults:
# Medium state (~15-33 strategies): [128, 128]
# Large  state (33+ strategies): [256, 256]
NET_DIMS = [256, 256]

# Maximum number of environment interaction steps for training.
# This value does NOT represent the length of the dataset. Instead, it controls how many TOTAL steps the agent is allowed to interact with the environment across ALL episodes.
# TOTAL_TRAINING_STEPS ≈ (number of desired passes over the data) × max_step
# Example:
# One year of data with 15-minute candles: ~35,000 candles -> max_step ≈ 35,000
# TOTAL_TRAINING_STEPS = 300,000
# 300,000 / 35,000 ≈ 8–9 full episodes
# The agent sees the same historical year ~9 times
# ---------------------
# Practical Guidelines:
# ---------------------
# 1–2 episodes   : Too little for learning
# 5–10 episodes  : Reasonable initial experiments
# 10–20 episodes : Serious training
# 30+ episodes   : Risk of overfitting (unless strongly regularized)
# ------
# Notes:
# ------
# PPO usually requires more total steps than off-policy methods (e.g. SAC)
# Increasing TOTAL_TRAINING_STEPS increases training time linearly
TOTAL_TRAINING_STEPS = int(5e5)

# Initial cash balance in USD at the beginning of each training/test episode (one full simulation run over a selected historical time window).
INITIAL_BALANCE = 100_000

# Maximum leverage multiplier allowed for both long and short.
# Example: 2.0 means the agent may take positions up to 2x its current equity.
LEVERAGE_LIMIT = 2.0

# Minimum and maximum stop-loss distance as a fraction of entry price.
# Example: 0.01 = 1%, 0.05 = 5%.
MIN_STOP_LOSS_PCT = 0.01
MAX_STOP_LOSS_PCT = 0.05

# Deadzone for exposure changes:
# If the requested change in normalized exposure is smaller than this threshold, we treat the action as HOLD and skip trading (reduces churning and fees).
EXPOSURE_DEADZONE = 0.10  # 10% change in target exposure

# Deadzone for stop-loss updates (as fraction of current price).
# A stop update only occurs if |desired_stop - current_stop| > STOP_UPDATE_DEADZONE_PCT * price.
# Prevents noisy micro-adjustments to the stop level.
STOP_UPDATE_DEADZONE_PCT = 0.002  # 0.2% of price

# Hard cap on absolute BTC position size (long or short).
# Prevents the agent from taking excessive exposure, even when leverage is available.
MAX_POSITION_BTC = 4.0

# Transaction fee applied to each executed MARKET (taker) order (as a fraction).
# On Kraken Futures a typical taker fee is approximately 0.05% → TRANSACTION_FEE = 0.0005
TRANSACTION_FEE = 0.0005

# Reward function selection: "log_return", "asymmetric", "stop_aware_drawdown"
REWARD_FUNCTION = "stop_aware_drawdown"

# Downside weight for asymmetric reward function. Higher values increase the penalty for negative returns.
DOWNSIDE_WEIGHT = 2.0

# ============================================================
# Stop-Aware Drawdown Reward Configuration
# ============================================================

# Number of bars to look back for counting recent stop-loss events.
STOP_CLUSTER_WINDOW_BARS = 20

# Base penalty per stop event, multiplied by the number of recent stops in the window.
STOP_CLUSTER_PENALTY = 0.005

# Number of bars after a stop within which a same-side re-entry is penalized.
SAME_SIDE_REENTRY_WINDOW_BARS = 10

# Flat penalty for re-entering the same side within the window after a stop.
SAME_SIDE_REENTRY_PENALTY = 0.01

# Drawdown fraction below which no penalty is applied.
DRAWDOWN_PENALTY_THRESHOLD = 0.05

# Linear weight for drawdown penalty above the threshold.
DRAWDOWN_PENALTY_WEIGHT = 0.10

# Percentage of the dataset used for training, the remainder is used for testing.
# Example: 0.75 = 75% train, 25% test.
TRAIN_TEST_SPLIT = 0.75

# Random seed for reproducible training.
# When SEED is set to an integer value, all random number generators (NumPy, Python, PyTorch, and the environment) are initialized deterministically, producing consistent results across runs.
# Set SEED = 42 to enforce deterministic behavior across runs.
# Set SEED = None to disable seeding entirely (non-deterministic).
SEED = 42

# Exchange used for market data (CCXT identifier)
EXCHANGE_NAME = "binance"

# Trading pair used by the system (CCXT format)
TRADING_PAIR = "BTC/USDT"

# Candle timeframe to fetch from the exchange.
# Defines the resolution for both Crypto and External data.
# 15-minute candles offer a strong balance between noise reduction and responsiveness for crypto RL tasks.
DATA_TIMEFRAME = "15m"

# List of technical indicators used by the environment.
# These correspond to FinRL's BTC example and must remain in this order because normalization is index-based.
INDICATORS = [
    "macd",          # [0] Moving Average Convergence Divergence (momentum indicator)
    "boll_ub",       # [1] Upper Bollinger Band (volatility measurement)
    "boll_lb",       # [2] Lower Bollinger Band (volatility measurement)
    "rsi_30",        # [3] 30-period Relative Strength Index (overbought/oversold)
    "dx_30",         # [4] 30-period Directional Movement Index (trend strength)
    "close_30_sma",  # [5] Simple Moving Average over 30 closes (short-term trend)
    "close_60_sma",  # [6] Simple Moving Average over 60 closes (mid-term trend)
]

# Number of candles each indicator needs before producing a valid (non-NaN) value.
# Used by DataManager to compute the warmup period when fetching extra historical data.
# If an indicator is added/removed above, this map MUST be updated to match.
INDICATOR_WARMUP_CANDLES = {
    "macd": 34,
    "boll_ub": 20,
    "boll_lb": 20,
    "rsi_30": 31,
    "dx_30": 31,
    "close_30_sma": 30,
    "close_60_sma": 60,
}

# Turbulence calculation (market stress indicator)
ENABLE_TURBULENCE = True

# -----------------------------------------------------------
# External Assets Registry (Daily Upsampling Strategy)
# -----------------------------------------------------------
EXTERNAL_ASSETS = [
    {
        "enabled": False,
        "ticker": "^VIX",
        "col_name": "vix",

        # Source is now direct API, not hybrid
        "source": "yfinance",

        # We fetch DAILY data (stable), but map it to DATA_TIMEFRAME (15m)
        "source_interval": "1d",

        # Local cache file (saved in the folder you created)
        "local_path": "data/download_data/external/vix_daily_cache.csv",
    }
]

# Slippage applied to market orders as a fraction of price (e.g. 0.0001 = 0.01%).
# Sampled per trade from a lognormal distribution: lognormal(ln(SLIPPAGE_MEAN), SLIPPAGE_STD).
SLIPPAGE_MEAN = 0.0003
SLIPPAGE_STD = 0.25

# Name of the machine where training is performed.
TRAINING_MACHINE_NAME = "OmerPC"

# Device for running backtests.
# Options:
# "cpu"        - Recommended default (stable, deterministic).
# "cuda"       - Use default GPU (cuda:0) if available.
# "cuda:<id>"  - Use a specific GPU (e.g., "cuda:1").
BACKTEST_DEVICE = "cpu"

# Force-close any open position at the final candle of backtests.
# Options:
# True  : realize PnL + fees/slippage and close the last trade in trades.csv
# False : keep mark-to-market valuation only (position may remain open in trades.csv)
BACKTEST_FORCE_CLOSE = True

# Annual risk-free rate (decimal) used for Sharpe/Sortino calculations.
# In crypto markets it is common to set this to 0.0 due to the lack of a true risk-free benchmark.
RISK_FREE_RATE = 0.0

# Directory for saving training results, including: trained agent models, evaluation metrics, plots and cumulative return charts
# Data will be stored per-run in: results/{model_name}_{machine_name}/data/
RESULTS_PATH = "results"

# Logging configuration
# Console log level determines what appears in terminal output (INFO shows major milestones, DEBUG shows everything)
LOG_LEVEL = "INFO"

# File log level for detailed debugging (logs are saved to {run_path}/logs/)
FILE_LOG_LEVEL = "DEBUG"

# ============================================================
# Strategy Integration Configuration
# ============================================================

# Enable/disable strategy signals in the RL pipeline
# When True, strategies generate One-Hot encoded signals added to state space
# When False, environment runs with no strategy features (backward compatible)
ENABLE_STRATEGIES = True

# List of strategies to include (names must match class names in strategies/registry.py)
# Each enabled strategy adds 4 dimensions to signal_ary (One-Hot: [FLAT, LONG, SHORT, HOLD])
# Empty list = no strategies (signal_ary will be empty)
# Only the 4 strategies that passed the 5% signal-activity variance filter.
# Dead strategies are archived in archive_strategies/ — see strategy_post_mortem_analysis.md
# State space impact: 4 strategies × 4 one-hot signals = 16 strategy features
STRATEGY_LIST = [
    "SupertrendStrategy",                          # 44.77% active — triple Supertrend agreement
    "PgQsdForNiftyFutureStrategy",                 # 29.31% active — WSI composite score + HMA
    "MonthlyReturnsInPinescriptStrategiesStrategy", # 18.19% active — pivot high/low breakout
    "TrendPullbackMomentumSideAwareStrategy",      # 13.87% active — HTF EMA + ATR zone + RSI
    "Ema5BreakoutTargetShiftingMtfStrategy",
]

# Maximum number of parallel workers for strategy signal processing
# None = use half of available CPU cores
# Set to 1 to disable parallel processing (sequential mode)
MAX_STRATEGY_WORKERS = None

# Maximum seconds a single strategy is allowed to run before being timed out.
# Strategies exceeding this limit will have their signals set to HOLD.
STRATEGY_TIMEOUT_SECONDS = 600  # 10 minutes

# ============================================================
# Feature Store Configuration
# ============================================================

# Enable loading of preprocessed data (processed DataFrames with all indicators and strategies)
# When True: system tries to load from training_data/processed/ before recalculating
# When False: always recalculates all features from raw data
USE_PREPROCESSED_DATA = True

# Root directory for persistent data storage (shared across all training runs)
# Structure:
#   data/download_data/
#   ├── training_data/          <-- PORTABLE (copy this to other machines)
#   │   ├── raw/
#   │   └── processed/
#   └── archived/               <-- LOCAL HISTORY (stays on this machine)
#       ├── raw/
#       └── processed/
DATA_ROOT_PATH = "data/download_data"

# ============================================================================
# Date Format Standards
# ============================================================================

# User display format (console output via formatter)
USER_DATE_FORMAT = "%d-%m-%Y"                    # DD-MM-YYYY
USER_DATETIME_FORMAT = "%d-%m-%Y %H:%M:%S"      # DD-MM-YYYY HH:MM:SS

# Internal format (storage, processing, DEBUG logs)
INTERNAL_ISO_FORMAT = "%Y-%m-%d"                 # YYYY-MM-DD (ISO 8601)
INTERNAL_ISO_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"  # YYYY-MM-DD HH:MM:SS
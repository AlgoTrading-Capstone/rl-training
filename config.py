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
NET_DIMS = [128, 128]

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
TOTAL_TRAINING_STEPS = int(3e5)

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

# Hard cap on absolute BTC position size (long or short).
# Prevents the agent from taking excessive exposure, even when leverage is available.
MAX_POSITION_BTC = 1.0

# Transaction fee applied to each executed MARKET (taker) order (as a fraction).
# On Kraken Futures a typical taker fee is approximately 0.05% → TRANSACTION_FEE = 0.0005
TRANSACTION_FEE = 0.0005

# Reward function selection: "log_return", "asymmetric"
REWARD_FUNCTION = "asymmetric"

# Downside weight for asymmetric reward function. Higher values increase the penalty for negative returns.
DOWNSIDE_WEIGHT = 2.0

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
# 15-minute candles offer a strong balance between noise reduction and responsiveness for crypto RL tasks.
DATA_TIMEFRAME = "15m"

# Date range is now specified by user input (start_date and end_date)

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

# Turbulence calculation (market stress indicator)
ENABLE_TURBULENCE = True

# VIX - CBOE Volatility Index (real data from S&P 500)
ENABLE_VIX = True
VIX_SYMBOL = "^VIX"

# Average slippage applied to market orders as a fraction of price (e.g. 0.0001 = 0.01%).
SLIPPAGE_MEAN = 0.0001

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
STRATEGY_LIST = [
    "AwesomeMacd",         # Momentum strategy using MACD + Awesome Oscillator
    "BbandRsi",            # Mean-reversion using Bollinger Bands + RSI
    "OTTStrategy",         # Optimized Trend Tracker using CMO-based EMA
    "SupertrendStrategy",  # Triple Supertrend with optimized parameters
    "VolatilitySystem"     # ATR-based volatility breakout system
]  # Empty by default - configure which strategies to enable

# Maximum number of parallel workers for strategy signal processing
# None = use half of available CPU cores
# Set to 1 to disable parallel processing (sequential mode)
MAX_STRATEGY_WORKERS = None

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
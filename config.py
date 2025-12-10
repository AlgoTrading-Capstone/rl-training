"""
Configuration file
"""

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

# Discount factor (gamma) for future rewards - determines how much the agent values long-term gains versus immediate rewards.
# Typical range: 0.970 – 0.995
GAMMA = 0.990

# Number of candles between agent decisions.
# Example:
# DECISION_INTERVAL = 1 - agent acts every candle
# DECISION_INTERVAL = 2 - agent acts every 2 candles
# DECISION_INTERVAL = 4 - agent acts every 4 candles, etc.
# Decision frequency of 15 minutes (or every candle for 15m data) is recommended for balanced responsiveness.
DECISION_INTERVAL = 1

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

# Number of years of historical data to download.
# 5–7 years generally provide enough variety across market regimes (bull, bear, sideways).
DATA_YEARS_BACK = 6

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

# Execution slippage applied to market orders (fractional impact on price)
# SLIPPAGE_MEAN: average slippage as a fraction of price (e.g. 0.0001 = 0.01%)
# SLIPPAGE_STD:  standard deviation of slippage noise (e.g. 0.00005 = ±0.005%)
SLIPPAGE_MEAN = 0.0001
SLIPPAGE_STD = 0.00005

# Name of the machine where training is performed.
TRAINING_MACHINE_NAME = "OmerPC"

# Root folder for storing downloaded or cached market data.
DATA_PATH = "raw_data"

# Directory for saving training results, including: trained agent models, evaluation metrics, plots and cumulative return charts
RESULTS_PATH = "results"

# Folder for all log files
LOG_PATH = "logs"

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
    # "AwesomeMacd",         # Momentum strategy using MACD + Awesome Oscillator
    # "BbandRsi",            # Mean-reversion using Bollinger Bands + RSI
    # "OTTStrategy",         # Optimized Trend Tracker using CMO-based EMA
    # "SupertrendStrategy",  # Triple Supertrend with optimized parameters
    # "VolatilitySystem"     # ATR-based volatility breakout system
]  # Empty by default - configure which strategies to enable
"""
Configuration file
"""

# Initial cash balance in USD at the beginning of each training/test episode (one full simulation run over a selected historical time window).
INITIAL_BALANCE = 100_000

# Maximum leverage multiplier allowed for both long and short.
# Example: 2.0 means the agent may take positions up to 2x its current equity.
LEVERAGE_LIMIT = 2.0

# Maximum debt allowed (in USD).
# The agent may let the account balance drop to -MAX_DEBT, but no further. Set to 0 to forbid debt entirely.
MAX_DEBT = 20_000

# Transaction fee applied to each executed MARKET (taker) order (as a fraction).
# On Kraken Futures a typical taker fee is approximately 0.05% → TRANSACTION_FEE = 0.0005
TRANSACTION_FEE = 0.0005

# Discount factor (gamma) for future rewards - determines how much the agent values long-term gains versus immediate rewards.
# Typical range: 0.970 – 0.995
GAMMA = 0.990

# Number of candles between agent decisions.
# Example:
# DECISION_INTERVAL = 1 → agent acts every candle
# DECISION_INTERVAL = 2 → agent acts every 2 candles
# DECISION_INTERVAL = 4 → agent acts every 4 candles, etc.
# Decision frequency of 15 minutes (or every candle for 15m data) is recommended for balanced responsiveness.
DECISION_INTERVAL = 1

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

# Root folder for storing downloaded or cached market data.
DATA_PATH = "data"

# Directory for saving training results, including: trained agent models, evaluation metrics, plots and cumulative return charts
RESULTS_PATH = "results"

# Folder for all log files
LOG_PATH = "logs"
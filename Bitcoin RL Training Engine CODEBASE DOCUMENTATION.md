# Bitcoin RL Training Engine - Codebase Documentation

## 1. Project Overview

This project is a standalone Reinforcement Learning (RL) training component within a larger Bitcoin trading system.
Its sole responsibility is to train, evaluate, and generate RL trading models using historical market data and precomputed strategy signals.

The project does not perform live trading, does not include a user interface, and does not operate as a server.
It is CLI-based training environment focused exclusively on model development and experimentation.

The training pipeline leverages historical Bitcoin market data, technical indicators, turbulence metrics, and multi-timeframe strategy signals to construct an RL environment in which agents learn trading behavior through simulation. The primary outputs of this project are trained model artifacts, evaluation metrics, and backtest results, which can later be consumed by other components of the overall system.

The project is intentionally self-contained and portable, allowing it to be cloned and executed on any developer machine. This enables:

- Parallel model training across multiple machines
- Independent experimentation by different developers
- Future execution in cloud-based compute environments

In later stages, this training component is expected to integrate with the system’s central server, enabling automated delivery of trained model artifacts and evaluation results for deployment and monitoring. At present, all outputs remain local to the training environment.

**Core Functionality:**
- Downloads Bitcoin OHLCV data from cryptocurrency exchanges (Binance/Kraken) via CCXT
- Processes raw data through a 5-phase pipeline: download → feature engineering → strategy execution → validation → array conversion
- Trains deep RL agents (PPO or SAC) using the ElegantRL framework
- Backtests trained models on separate time periods with comprehensive metrics
- Generates performance visualizations comparing agent performance to buy-and-hold benchmarks

**Supported Execution Modes:**
1. **TRAIN_AND_BACKTEST**: Train a new model and automatically backtest on a different date range
2. **TRAIN_ONLY**: Train a model without backtesting
3. **BACKTEST_ONLY**: Backtest an existing trained model on new data

## 2. High-Level Architecture

The system follows a modular, pipeline-based architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Main Entry Point                         │
│                           (main.py)                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├──► User Input Collection (utils/user_input.py)
                     │    • Run mode selection
                     │    • Model name validation
                     │    • Date range specification
                     │
                     ├──► Data Management (data/data_manager.py)
                     │    ┌──────────────────────────────────────┐
                     │    │ Phase 1: Smart Incremental Download  │
                     │    │ Phase 2: Feature Engineering         │
                     │    │ Phase 3: Parallel Strategy Execution │
                     │    │ Phase 4: Validation & Cleaning       │
                     │    │ Phase 5: Array Conversion            │
                     │    └──────────────────────────────────────┘
                     │
                     ├──► RL Training (main.py → elegantrl/)
                     │    • BitcoinTradingEnv (bitcoin_env.py)
                     │    • Trade Engine (trade_engine.py)
                     │    • ElegantRL Training Loop
                     │    • Model Checkpointing (act.pth, cri.pth)
                     │
                     └──► Backtesting (backtesting/)
                          • Load trained actor
                          • Run episode with policy
                          • Log steps/trades/state
                          • Compute metrics
                          • Generate plots
```

### Directory Structure

```
rl-training/
├── main.py                          # Entry point
├── config.py                        # Central configuration
├── bitcoin_env.py                   # RL environment (Gymnasium)
├── trade_engine.py                  # Trading logic (positions, stops, fees)
├── reward_functions.py              # Reward function implementations
├── rl_configs.py                    # ElegantRL configuration builder
│
├── data/                            # Data management
│   ├── data_manager.py              # 5-phase pipeline orchestrator
│   ├── data_processor.py            # CCXT integration + feature engineering
│   ├── strategy_processor.py        # Strategy signal generation
│   ├── external_manager.py          # VIX data integration
│   └── download_data/               # Feature Store (cached data)
│       ├── training_data/           # PORTABLE: copy to other machines
│       │   ├── raw/                 # Cached OHLCV data
│       │   └── processed/           # Cached processed data
│       └── archived/                # LOCAL HISTORY: previous versions
│
├── strategies/                      # Trading strategy implementations
│   ├── base_strategy.py             # Abstract base class + signal types
│   ├── registry.py                  # Strategy registration
│   ├── awesome_macd.py              # MACD + Awesome Oscillator
│   ├── bband_rsi.py                 # Bollinger Bands + RSI
│   ├── ott_strategy.py              # Optimized Trend Tracker
│   ├── supertrend_strategy.py       # Supertrend indicator
│   └── volatility_system.py         # ATR-based volatility breakout
│
├── adapters/                        # Framework adapters
│   └── elegantrl_bitcoin_env.py     # ElegantRL environment wrapper
│
├── backtesting/                     # Backtest execution + metrics
│   ├── backtest_runner.py           # Main backtest orchestrator
│   ├── step_logger.py               # Step-level logging (steps.csv)
│   ├── trade_tracker.py             # Trade-level logging (trades.csv)
│   ├── state_debug_logger.py        # State debugging
│   ├── metrics_manager.py           # Performance metrics computation
│   └── plots/                       # Plot generation
│       ├── plot_runner.py           # Plot orchestration
│       └── benchmark_equity_plot.py # Agent vs BTC benchmark
│
├── elegantrl/                       # ElegantRL framework (vendored)
│   ├── train/
│   │   ├── run.py                   # Training loop
│   │   ├── config.py                # Config dataclass
│   │   ├── evaluator.py             # Evaluation during training
│   │   └── replay_buffer.py         # Experience replay
│   └── agents/
│       ├── AgentPPO.py              # PPO implementation
│       └── AgentSAC.py              # SAC implementation
│
├── utils/                           # Utilities
│   ├── logger.py                    # Loguru-based logging
│   ├── user_input.py                # Interactive user prompts
│   ├── metadata.py                  # Metadata file management
│   ├── normalization.py             # State normalization
│   ├── formatting.py                # Console output formatting
│   ├── progress.py                  # Progress bars (tqdm)
│   ├── resampling.py                # Timeframe resampling
│   ├── timeframes.py                # Timeframe conversion
│   └── date_display.py              # Date formatting utilities
│
├── tests/                           # Test suite
│   └── unit/
│       └── date_utils/              # Date utility tests
│
└── results/                         # Training run outputs
    └── {model_name}_{machine}/
        ├── metadata.json            # Run configuration snapshot
        ├── elegantrl/               # Training outputs
        │   ├── act.pth              # Trained actor checkpoint
        │   ├── cri.pth              # Trained critic checkpoint
        │   └── recorder.pth         # Training metrics
        └── backtests/               # Backtest outputs
            └── {backtest_id}/
                ├── steps.csv        # Step-by-step log
                ├── trades.csv       # Trade-by-trade log
                ├── summary.json     # Summary statistics
                ├── metrics.json     # Performance metrics
                └── plots/           # Generated plots (PNG)
```

## 3. Folder Responsibilities

### Root Directory Files

**main.py**
- Entry point for the entire system
- Orchestrates three execution pipelines: TRAIN_AND_BACKTEST, TRAIN_ONLY, BACKTEST_ONLY
- Initializes DataManager, logger, and user input collection
- Calls `run_training_pipeline()` and `run_backtest_pipeline()`

**config.py**
- Central configuration file containing all system parameters
- RL algorithm selection (PPO/SAC), hyperparameters (gamma, learning rate, network dimensions)
- Trading parameters (leverage, fees, slippage, stop-loss ranges, exposure deadzone)
- Data source configuration (exchange, trading pair, timeframe)
- Technical indicators list (order-sensitive for normalization)
- Strategy configuration (enable flag, strategy list, parallel workers)
- Feature Store settings (cache enable/disable, storage path)

**bitcoin_env.py**
- Custom RL environment implementing Gymnasium interface
- Manages environment state: balance, holdings, entry price, stop-loss level
- State space: balance + OHLCV + technical indicators + turbulence/VIX + strategy signals + holdings
- Action space: [a_pos, a_sl] where a_pos = desired exposure, a_sl = stop-loss tightness
- Splits data into train/test based on TRAIN_TEST_SPLIT ratio
- Handles intra-candle stop-loss triggers (checks high/low prices)
- Delegates trade execution to trade_engine.py

**trade_engine.py**
- Pure trading logic module (no RL dependencies)
- Defines dataclasses: PositionState, TradeConfig, TradeResult
- Maps actions to target exposure with leverage constraints
- Enforces hard BTC position cap (MAX_POSITION_BTC)
- Applies exposure deadzone to reduce churning
- Computes delta BTC and executes trades with fees/slippage
- Updates entry price and stop-loss on new positions or flips
- `close_position()` function for forced exits (stop-loss or end-of-episode)

**reward_functions.py**
- Implements reward functions used by the environment
- `reward_log_return()`: Logarithmic return r = ln(new / old)
- `reward_asymmetric_drawdown_penalty()`: Penalizes losses linearly by DOWNSIDE_WEIGHT
- REWARD_REGISTRY dictionary maps config.REWARD_FUNCTION to implementations

**rl_configs.py**
- Builds ElegantRL Config objects for PPO and SAC
- Validates network dimensions (NET_DIMS[0] >= state_dim)
- Sets algorithm-specific hyperparameters (horizon_len, ratio_clip for PPO; buffer_size, batch_size for SAC)
- Configures train/eval environments with data arrays
- Enriches metadata.json with training configuration via `enrich_metadata_with_training_config()`

### data/

**data_manager.py**
- Unified data management orchestrator
- **Phase 1**: Smart incremental download - detects missing date gaps, downloads only missing periods, merges with existing cache
- **Phase 2**: Feature engineering - cleans data, adds technical indicators (TA-Lib), calculates turbulence, integrates VIX
- **Phase 3**: Parallel strategy execution - loads strategies from registry, resamples to each strategy's timeframe, executes in parallel via ProcessPoolExecutor, merges signals back to base timeframe
- **Phase 4**: Validation - removes duplicates, validates OHLC relationships, forward/backward fills NaN values
- **Phase 5**: Array conversion - converts DataFrame to numpy arrays (price_array, tech_array, turbulence_array, signal_array, datetime_array)
- Implements two-tier caching: training_data/ (portable) + archived/ (local history)

**data_processor.py**
- CCXT integration for exchange connectivity
- CcxtProcessor class handles data download via exchange APIs
- Converts dates between user format (DD-MM-YYYY) and ISO format (YYYY-MM-DD)
- Downloads OHLCV data with exponential backoff retry logic
- Adds technical indicators using TA-Lib (MACD, Bollinger Bands, RSI, DX, SMAs)
- Calculates turbulence index (market stress indicator)
- Integrates external data (VIX) via ExternalDataManager
- `clean_data()`: removes duplicates, validates prices, handles missing values
- `df_to_array()`: converts DataFrame to numpy arrays for RL environment

**strategy_processor.py**
- Converts strategy signals (SignalType enum) to One-Hot encoded vectors
- `signal_to_onehot()`: maps FLAT/LONG/SHORT/HOLD → [1,0,0,0] / [0,1,0,0] / [0,0,1,0] / [0,0,0,1]
- `calculate_lookback_candles()`: computes number of candles needed based on lookback_hours and timeframe

**external_manager.py**
- Manages external market data (VIX from Yahoo Finance)
- Downloads daily data and upsamples to intraday timeframe
- Applies 1-day shift to prevent lookahead bias
- Forward-fills weekends/holidays
- Caches data locally in data/download_data/external/

### strategies/

**base_strategy.py**
- Abstract base class defining strategy interface
- SignalType enum: LONG, SHORT, FLAT, HOLD
- StrategyRecommendation namedtuple: (signal, timestamp)
- BaseStrategy abstract class requires implementation of `run(df, timestamp)` method
- Each strategy defines: name, description, timeframe, lookback_hours

**registry.py**
- Centralized Python registry mapping strategy names to classes
- `StrategyRegistry.get_strategy(name)` returns strategy instance
- Loads strategies dynamically for parallel execution

**Strategy Implementations** (awesome_macd.py, bband_rsi.py, ott_strategy.py, supertrend_strategy.py, volatility_system.py)
- Each strategy inherits from BaseStrategy
- Implements `_calculate_indicators()` for technical indicator computation
- Implements `_generate_signal()` to produce SignalType based on conditions
- Defines MIN_CANDLES_REQUIRED and lookback_hours
- Runs on configurable timeframe (e.g., "1h", "4h") with automatic resampling

### adapters/

**elegantrl_bitcoin_env.py**
- Adapter between BitcoinTradingEnv (domain environment) and ElegantRL API
- Wraps reset() to return (state, info) format expected by ElegantRL
- Wraps step() to return (next_state, reward, terminated, truncated, info)
- Converts torch tensors to numpy if needed
- Exposes state_dim, action_dim, max_step properties

### backtesting/

**backtest_runner.py**
- Main backtest orchestrator
- Validates backtest compatibility against training metadata (indicators, turbulence/VIX flags, strategy set+order, state/action dimensions)
- Loads trained actor checkpoint (act.pth)
- Runs full episode with policy inference
- Logs steps, trades, and state data via dedicated loggers
- Handles end-of-episode finalization (mark-to-market valuation, optional force-close)
- Writes summary.json
- Calls metrics_manager and plot_runner

**step_logger.py**
- Logs complete step-level data during backtest
- Writes steps.csv with columns: step_idx, timestamp, state (all normalized features), price_vec, tech_vec, turb_vec, sig_vec, actions, trade_executed, effective_delta_btc, balance, holdings, equity, reward, stop_triggered, done

**trade_tracker.py**
- Tracks individual trades (entry/exit points, P&L, duration)
- Writes trades.csv with columns: entry_time, exit_time, entry_price, exit_price, holdings, pnl_btc, pnl_usd, duration_candles, stop_triggered, forced_close
- Handles partial fills and position flips

**state_debug_logger.py**
- Logs detailed state information for debugging
- Writes state_debug.csv with raw and normalized values

**metrics_manager.py**
- Computes comprehensive performance metrics from steps.csv, trades.csv, summary.json
- Writes metrics.json with six categories:
  - Run Metadata: run_id, timeframe, date range, initial capital
  - Performance: returns (MTM/realized), CAGR, volatility, Sharpe/Sortino ratios, max drawdown, Calmar ratio
  - Risk & Behavior: exposure time %, avg/max exposure %, turnover (BTC), stop-loss trigger statistics
  - Trade-Level: num_trades, win_rate, profit_factor, avg/median P&L, avg win/loss, max win/loss, avg duration
  - Reward Diagnostics: sum/mean/std of rewards, reward-return correlation
  - Data Quality: equity min/max, NaN/Inf counts, bankruptcy flag
- Handles timeframe-specific annualization factors

**plots/plot_runner.py**
- Orchestrates plot generation pipeline
- Loads backtest artifacts (steps.csv, metrics.json, summary.json)
- Dispatches to individual plot modules
- Saves plots as PNG files under backtest/plots/
- Fail-soft error handling (continues on individual plot failures)

**plots/benchmark_equity_plot.py**
- Generates agent equity vs BTC buy-and-hold comparison plot
- Includes metadata subtitle: backtest ID, date range, agent name

### elegantrl/

**train/run.py**
- Main training loop implementation
- Initializes agent, environments, evaluator, replay buffer
- Runs training for TOTAL_TRAINING_STEPS
- Saves checkpoints: act.pth (actor), cri.pth (critic), recorder.pth (metrics)

**train/config.py**
- Config dataclass holding all training parameters
- Populated by rl_configs.py

**train/evaluator.py**
- Evaluates agent on test environment during training
- Tracks cumulative returns and episode rewards

**train/replay_buffer.py**
- Experience replay buffer for off-policy algorithms (SAC)
- Stores transitions (state, action, reward, next_state, done)

**agents/AgentPPO.py**
- Proximal Policy Optimization implementation
- On-policy algorithm with rollout buffer
- Clip-based surrogate objective
- Requires horizon_len, repeat_times, ratio_clip, lambda_gae_adv, lambda_entropy

**agents/AgentSAC.py**
- Soft Actor-Critic implementation
- Off-policy algorithm with replay buffer
- Entropy-regularized objective with automatic temperature tuning
- Requires buffer_size, batch_size, soft_update_tau, alpha

### utils/

**logger.py**
- Loguru-based logging system
- RLLogger class with component-routed file logs
- LogComponent enum: MAIN, DATA, STRATEGY, TRAINING, BACKTEST
- Creates separate log files per component (logs/training.log, logs/data_pipeline.log, logs/strategy_execution.log, logs/backtest.log)
- Colorized console output (INFO+ to stderr)
- `logger.phase()` context manager for phase timing and failure detection
- `for_component()` creates immutable logger views

**user_input.py**
- Interactive user prompts using Rich library
- `collect_run_mode()`: asks user to select TRAIN_AND_BACKTEST / TRAIN_ONLY / BACKTEST_ONLY
- `collect_train_and_backtest_input()`: collects model name, description, training dates, backtest dates
- `collect_train_only_input()`: collects training metadata without backtest
- `collect_backtest_only_input()`: selects existing model and collects backtest dates
- Date validation (DD-MM-YYYY format)
- Overlap detection between training and backtest periods

**metadata.py**
- Metadata file creation and management
- `create_metadata_file()`: creates initial metadata.json from user input
- `enrich_metadata_with_training_config()`: adds RL, environment, data, strategies sections after training config is resolved
- `load_metadata()`: reads existing metadata.json
- `append_backtest_metadata()`: appends backtest entry to metadata.json

**normalization.py**
- State normalization for neural network input
- Balance: divided by INITIAL_BALANCE
- Prices (OHLC): divided by current close (relative scaling)
- Volume: log-scaled log1p(vol) / 20
- Technical indicators: indicator-specific normalization (price-based / scaled / oscillators)
- Turbulence/VIX: scaled and squashed with tanh
- Strategy signals: passed as-is (already binary 0/1)
- Holdings: divided by MAX_POSITION_BTC
- `inverse_normalize_state()`: converts normalized state back to human-readable values (for debugging)

**formatting.py**
- Console output formatting utilities
- `Formatter.error_context()`: wraps error messages with context boxes
- `Formatter.display_training_config()`: pretty-prints training configuration
- `Formatter.format_date_range_duration()`: formats duration between dates

**progress.py**
- Progress bar wrapper using tqdm
- `ProgressTracker.process_items()`: creates progress bar for processing loops
- Configured for non-blocking stderr output (compatible with Loguru)

**resampling.py**
- Timeframe resampling utilities
- `resample_to_interval()`: aggregates candles to higher timeframe (15m → 1h, 4h, etc.)
- `resampled_merge()`: merges resampled data back to base timeframe (prevents lookahead bias by forward-filling)

**timeframes.py**
- Timeframe conversion helpers
- `timeframe_to_minutes()`: converts timeframe string to minutes (e.g., "15m" → 15, "1h" → 60, "4h" → 240)

**date_display.py**
- Date formatting utilities
- `format_date_range_for_display()`: formats date ranges for console output

## 4. Python File Responsibilities

### Root Files

#### main.py

**Purpose**: Main orchestrator for RL training and backtesting pipelines

**Key Functions**:
- `main()`: Entry point - initializes DataManager, collects user input, routes to appropriate pipeline
- `run_training_pipeline()`: Executes 5 steps - data preparation, dimension calculation, RL config build, agent training, artifact persistence
- `run_backtest_pipeline()`: Executes backtest - creates backtest ID/directory, loads model metadata, loads backtest data, runs backtest, appends metadata

**Called By**: Direct execution (`if __name__ == "__main__"`)

**Calls**:
- utils/user_input.py: `collect_run_mode()`, `collect_train_and_backtest_input()`, `collect_train_only_input()`, `collect_backtest_only_input()`
- data/data_manager.py: `DataManager.get_arrays()`
- rl_configs.py: `build_elegantrl_config()`
- elegantrl/train/run.py: `train_agent()`
- backtesting/backtest_runner.py: `run_backtest()`
- utils/metadata.py: `create_metadata_file()`, `load_metadata()`, `append_backtest_metadata()`
- utils/formatting.py: `Formatter.display_training_config()`, `Formatter.error_context()`

**Artifacts Created**:
- `results/{model_name}_{machine}/` directory
- `metadata.json` - Run configuration snapshot
- `logs/` - Component-routed log files

#### config.py

**Purpose**: Central configuration file

**Contains**:
- RL algorithm parameters (RL_MODEL, GAMMA, LEARNING_RATE, NET_DIMS, TOTAL_TRAINING_STEPS, SEED)
- Trading parameters (INITIAL_BALANCE, LEVERAGE_LIMIT, MAX_POSITION_BTC, MIN/MAX_STOP_LOSS_PCT, EXPOSURE_DEADZONE, TRANSACTION_FEE, SLIPPAGE_MEAN)
- Data parameters (EXCHANGE_NAME, TRADING_PAIR, DATA_TIMEFRAME, INDICATORS list, ENABLE_TURBULENCE, EXTERNAL_ASSETS)
- Strategy parameters (ENABLE_STRATEGIES, STRATEGY_LIST, MAX_STRATEGY_WORKERS)
- Feature Store parameters (USE_PREPROCESSED_DATA, DATA_ROOT_PATH)
- Environment parameters (TRAIN_TEST_SPLIT, REWARD_FUNCTION, DOWNSIDE_WEIGHT)
- System parameters (TRAINING_MACHINE_NAME, BACKTEST_DEVICE, BACKTEST_FORCE_CLOSE, RISK_FREE_RATE)
- Path parameters (RESULTS_PATH, LOG_LEVEL, FILE_LOG_LEVEL)
- Date format parameters (USER_DATE_FORMAT, INTERNAL_ISO_FORMAT)

**Used By**: Nearly all modules import values from config

#### bitcoin_env.py

**Purpose**: Custom RL environment implementing Gymnasium interface

**Key Class**: `BitcoinTradingEnv`

**Constructor Arguments**:
- `price_ary`, `tech_ary`, `turbulence_array`, `signal_ary`, `datetime_ary` - numpy arrays from data pipeline
- `mode` - "train", "test", or "backtest"

**Key Methods**:
- `reset()`: Resets environment to initial state, returns normalized state vector
- `step(action)`: Executes one timestep - checks stop-loss, applies action via trade_engine, computes reward, advances time, returns (next_state, reward, done, info)

**State Composition** (1D vector):
- 1 value: normalized balance
- 5 values: normalized OHLCV
- 7 values: normalized technical indicators
- 1-2 values: normalized turbulence + optional VIX
- 4×N values: strategy signals (One-Hot encoded)
- 1 value: normalized holdings

**Action Space**: [a_pos, a_sl] both in [-1, 1]

**Called By**:
- adapters/elegantrl_bitcoin_env.py: Wraps this class
- backtesting/backtest_runner.py: Creates instance in "backtest" mode

**Calls**:
- trade_engine.py: `apply_action()`, `close_position()`, `compute_equity()`
- reward_functions.py: Reward function from REWARD_REGISTRY
- utils/normalization.py: `normalize_state()`

#### trade_engine.py

**Purpose**: Pure trading logic (no RL dependencies)

**Key Dataclasses**:
- `PositionState`: balance, holdings, entry_price, stop_price
- `TradeConfig`: leverage_limit, max_position_btc, min/max_stop_pct, exposure_deadzone, fee_rate
- `TradeResult`: new_state, effective_delta_btc, trade_executed

**Key Functions**:
- `compute_equity()`: Returns balance + holdings × price
- `clip_exposure()`: Clips action to [-1, +1]
- `target_notional_from_action()`: Maps a_pos to USD notional exposure
- `limit_btc_position()`: Enforces hard BTC position cap
- `compute_delta_btc()`: Calculates BTC to buy/sell
- `apply_trade()`: Executes trade with fees/slippage, updates balance/holdings
- `compute_stop_price()`: Calculates stop-loss price based on a_sl
- `apply_action()`: Main entry point - computes target exposure, applies deadzone, executes trade, updates entry/stop
- `close_position()`: Forces position closure (for stop-loss or end-of-episode)

**Called By**:
- bitcoin_env.py: `step()` method
- backtesting/backtest_runner.py: End-of-episode force-close

#### reward_functions.py

**Purpose**: Reward function implementations

**Functions**:
- `reward_log_return()`: r = ln(new / old), penalty -1.0 for bankruptcy
- `reward_asymmetric_drawdown_penalty()`: Positive returns rewarded as-is, negative returns penalized linearly by DOWNSIDE_WEIGHT

**REWARD_REGISTRY**: Dictionary mapping config.REWARD_FUNCTION to function implementations

**Called By**:
- bitcoin_env.py: Loads reward function in constructor

#### rl_configs.py

**Purpose**: Builds ElegantRL Config objects

**Key Function**: `build_elegantrl_config()`

**Arguments**:
- Data arrays: price_array, tech_array, turbulence_array, signal_array, datetime_array
- Dimensions: state_dim, action_dim, train_max_step, eval_max_step
- Path: run_path

**Returns**: ElegantRL Config object ready for `train_agent()`

**Key Operations**:
- Validates NET_DIMS[0] >= state_dim
- Selects agent class (AgentPPO or AgentSAC) based on config.RL_MODEL
- Builds train_env_args and eval_env_args dictionaries
- Sets algorithm-specific hyperparameters (PPO_CONFIG or SAC_CONFIG)
- Configures reproducibility (random_seed)
- Sets training control parameters (break_step, gamma, learning_rate)
- Calls `enrich_metadata_with_training_config()` to save full config snapshot

**Called By**:
- main.py: `run_training_pipeline()`

**Calls**:
- utils/metadata.py: `enrich_metadata_with_training_config()`

### data/ Directory

#### data/data_manager.py

**Purpose**: Unified data management orchestrator

**Key Class**: `DataManager`

**Constructor Arguments**:
- `exchange`, `trading_pair`, `base_timeframe` - defaults from config
- `storage_path` - deprecated, uses config.DATA_ROOT_PATH
- `logger` - REQUIRED, RLLogger instance

**Key Methods**:

**Phase 1 - Smart Incremental Download**:
- `get_or_download_data()`: Loads cached raw data if exists, detects missing gaps via `_get_data_gap()`, downloads only gaps, merges and deduplicates
- `_download_with_retry()`: Downloads data with exponential backoff
- `_save_raw_data()`: Saves to Parquet format
- `_archive_file()`: Moves old files to archived/ with timestamp

**Phase 2 - Feature Engineering**:
- `add_base_features()`: Delegates to DataProcessor - cleans data, adds technical indicators, turbulence, VIX

**Phase 3 - Parallel Strategy Execution**:
- `add_strategy_signals()`: Detects existing strategies, validates data, filters strategies to execute, calls `_execute_strategies_parallel()`
- `_execute_strategies_parallel()`: Uses ProcessPoolExecutor to run strategies in parallel, calls `_process_single_strategy_with_resampling()` worker function, forward/backward fills NaN values

**Phase 4 - Validation**:
- `validate_and_clean()`: Delegates to DataProcessor.clean_data()

**Phase 5 - Array Conversion**:
- `to_arrays()`: Delegates to DataProcessor.df_to_array(), returns (price_array, tech_array, turbulence_array, signal_array, datetime_array)

**High-Level API**:
- `get_processed_data()`: Smart waterfall - tries processed cache → incremental update → raw fallback → save processed
- `get_arrays()`: Complete pipeline - calls `get_processed_data()` then `to_arrays()`

**Called By**:
- main.py: Instantiates DataManager, calls `get_arrays()`

**Calls**:
- data/data_processor.py: CcxtProcessor methods
- strategies/registry.py: `StrategyRegistry.get_strategy()`
- utils/resampling.py: `resample_to_interval()`, `resampled_merge()`
- utils/timeframes.py: `timeframe_to_minutes()`

**Artifacts Created/Modified**:
- `data/download_data/training_data/raw/{exchange}_{pair}_{timeframe}.parquet`
- `data/download_data/training_data/processed/{exchange}_{pair}_{timeframe}_processed.parquet`
- `data/download_data/archived/raw/{timestamp}_{filename}.parquet`
- `data/download_data/archived/processed/{timestamp}_{filename}.parquet`

**Top-Level Worker Function**:
- `_process_single_strategy_with_resampling()`: Must be top-level for Windows ProcessPoolExecutor pickling

#### data/data_processor.py

**Purpose**: CCXT integration and feature engineering

**Key Classes**:
- `CcxtProcessor`: Downloads data via CCXT, adds technical indicators, cleans data
- `DataProcessor`: Wrapper around CcxtProcessor + ExternalDataManager

**CcxtProcessor Methods**:
- `__init__()`: Initializes CCXT exchange connection
- `download_data()`: Downloads OHLCV data from exchange for given ticker list, date range, timeframe
- `_convert_to_iso_date()`: Converts DD-MM-YYYY or YYYY-MM-DD to ISO format
- `clean_data()`: Removes duplicates, sorts, forward/backward fills NaN, clips negative prices, validates OHLC relationships
- `add_technical_indicator()`: Adds TA-Lib indicators (MACD, Bollinger Bands, RSI, DX, SMAs)
- `add_turbulence()`: Calculates market stress indicator based on volatility
- `add_external_data()`: Integrates VIX via ExternalDataManager
- `df_to_array()`: Converts DataFrame to numpy arrays (price_array, tech_array, turbulence_array, signal_array)

**Called By**:
- data/data_manager.py: DataProcessor instance used throughout pipeline

**Calls**:
- ccxt library: Exchange API calls
- talib library: Technical indicator calculations
- data/external_manager.py: `ExternalDataManager.add_vix()`

#### data/external_manager.py

**Purpose**: Manages external market data (VIX from Yahoo Finance)

**Key Class**: `ExternalDataManager`

**Key Methods**:
- `add_vix()`: Downloads daily VIX data, applies 1-day shift (prevents lookahead bias), forward-fills to intraday timeframe, caches locally

**Called By**:
- data/data_processor.py: `CcxtProcessor.add_external_data()`

**Artifacts Created**:
- `data/download_data/external/vix_daily_cache.csv`

### strategies/ Directory

#### strategies/base_strategy.py

**Purpose**: Abstract base class for all strategies

**Key Classes/Enums**:
- `SignalType` (Enum): LONG, SHORT, FLAT, HOLD
- `StrategyRecommendation` (NamedTuple): (signal, timestamp)
- `BaseStrategy` (ABC): Abstract class with `run()` method

**Used By**: All strategy implementations inherit from BaseStrategy

#### strategies/registry.py

**Purpose**: Centralized strategy registration

**Key Class**: `StrategyRegistry`

**Methods**:
- `get_strategy(name)`: Returns strategy instance by name

**Used By**:
- data/data_manager.py: `_process_single_strategy_with_resampling()` worker function

#### Strategy Implementations

Files: `awesome_macd.py`, `bband_rsi.py`, `ott_strategy.py`, `supertrend_strategy.py`, `volatility_system.py`

**Common Structure**:
- Inherits from BaseStrategy
- Defines: name, description, timeframe, lookback_hours, MIN_CANDLES_REQUIRED
- Implements: `_calculate_indicators()`, `_generate_signal()`, `run()`

**Called By**:
- data/data_manager.py: Loaded dynamically via registry during parallel strategy execution

### adapters/ Directory

#### adapters/elegantrl_bitcoin_env.py

**Purpose**: Adapter between BitcoinTradingEnv and ElegantRL API

**Key Class**: `ElegantRLBitcoinEnv`

**Methods**:
- `__init__()`: Wraps BitcoinTradingEnv or creates from arrays
- `reset()`: Returns (state, info) format expected by ElegantRL
- `step()`: Returns (next_state, reward, terminated, truncated, info)
- `_to_numpy()`: Converts torch tensors to numpy

**Used By**:
- rl_configs.py: Set as `env_class` and `eval_env_class` in Config
- elegantrl/train/run.py: Training loop instantiates this environment

### backtesting/ Directory

#### backtesting/backtest_runner.py

**Purpose**: Main backtest orchestrator

**Key Function**: `run_backtest()`

**Arguments**:
- `model_metadata`: Loaded from metadata.json
- `act_path`: Path to act.pth checkpoint
- Data arrays: price_array, tech_array, turbulence_array, signal_array, datetime_array
- `out_dir`: Output directory for backtest results
- `backtest_config`: Date range and overlap flag

**Execution Steps**:
1. Validates inputs and prepares output directory
2. Builds backtest environment (mode="backtest")
3. Validates compatibility - checks indicators, turbulence/VIX flags, strategy set+order, state/action dimensions
4. Loads actor policy from act.pth
5. Runs episode - policy inference, state snapshot before step, environment step, logging via StepLogger/TradeTracker/StateDebugLogger
6. End-of-episode finalization - computes mark-to-market equity, optional force-close, writes summary.json
7. Calls `compute_and_write_metrics()` to generate metrics.json
8. Calls `generate_backtest_plots()` to create visualizations

**Called By**:
- main.py: `run_backtest_pipeline()`

**Calls**:
- bitcoin_env.py: Creates BitcoinTradingEnv instance
- backtesting/step_logger.py: `StepLogger.log_step()`
- backtesting/trade_tracker.py: `TradeTracker.on_step()`
- backtesting/state_debug_logger.py: `StateDebugLogger.log_step()`
- backtesting/metrics_manager.py: `compute_and_write_metrics()`
- backtesting/plots/plot_runner.py: `generate_backtest_plots()`
- elegantrl/agents/: Loads agent class to reconstruct policy network

**Artifacts Created**:
- `backtests/{backtest_id}/steps.csv`
- `backtests/{backtest_id}/trades.csv`
- `backtests/{backtest_id}/state_debug.csv`
- `backtests/{backtest_id}/summary.json`

#### backtesting/step_logger.py

**Purpose**: Logs complete step-level data

**Key Class**: `StepLogger`

**Methods**:
- `open()`: Opens steps.csv for writing with header row
- `log_step()`: Writes one row with all step data (step_idx, timestamp, state vector, price/tech/turbulence/signals, actions, trade info, balance, holdings, equity, reward, flags)
- `close()`: Closes file handle

**Called By**:
- backtesting/backtest_runner.py: Logs every step during episode

**Artifacts Created**:
- `steps.csv` - Complete step-by-step log

#### backtesting/trade_tracker.py

**Purpose**: Tracks individual trades

**Key Class**: `TradeTracker`

**Methods**:
- `open()`: Opens trades.csv for writing
- `on_step()`: Called each step - detects position changes, logs trade entries/exits
- `finalize_end()`: Handles end-of-episode if position still open
- `close()`: Closes file handle

**Trade Detection Logic**:
- Entry: holdings change from 0 to non-zero
- Exit: holdings change from non-zero to 0
- Flip: holdings change sign (long to short or vice versa) - logs exit of old trade + entry of new trade

**Called By**:
- backtesting/backtest_runner.py: Called every step and at end-of-episode

**Artifacts Created**:
- `trades.csv` - Per-trade log with entry/exit times, prices, P&L, duration

#### backtesting/metrics_manager.py

**Purpose**: Computes comprehensive performance metrics

**Key Function**: `compute_and_write_metrics()`

**Input Artifacts**:
- Loads `steps.csv`, `trades.csv`, `summary.json`

**Metric Categories Computed**:
1. Run Metadata: run_id, timeframe, date range, initial capital
2. Performance: returns (MTM/realized), CAGR, volatility, Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio
3. Risk & Behavior: exposure time %, avg/max exposure %, turnover (BTC), stop-loss trigger count/rate
4. Trade-Level: num_trades, win_rate, profit_factor, avg/median P&L, avg win/loss, max win/loss, avg duration
5. Reward Diagnostics: sum/mean/std of rewards, reward-return correlation
6. Data Quality: equity min/max, NaN/Inf counts, bankruptcy flag

**Key Operations**:
- Reads steps_df from steps.csv
- Reads trades_df from trades.csv
- Computes annualization factor based on timeframe (TIMEFRAME_TO_STEPS_PER_YEAR)
- Calculates returns (equity_curve.pct_change())
- Computes volatility (std × sqrt(steps_per_year))
- Computes Sharpe ratio ((CAGR - risk_free_rate) / volatility)
- Computes maximum drawdown (rolling max - current / rolling max)
- Sanitizes NaN/Inf values for JSON serialization

**Called By**:
- backtesting/backtest_runner.py: After episode completion

**Artifacts Created**:
- `metrics.json` - Comprehensive performance metrics

#### backtesting/plots/plot_runner.py

**Purpose**: Orchestrates plot generation

**Key Function**: `generate_backtest_plots()`

**Operations**:
- Loads steps.csv, metrics.json, summary.json
- Calls individual plot modules (benchmark_equity_plot.py, etc.)
- Saves plots as PNG files under backtest/plots/
- Fail-soft error handling (continues on individual failures)

**Called By**:
- backtesting/backtest_runner.py: After metrics computation

**Artifacts Created**:
- `backtest/plots/*.png` files

#### backtesting/plots/benchmark_equity_plot.py

**Purpose**: Generates agent vs BTC buy-and-hold comparison plot

**Key Function**: Generates equity curve comparison

**Plot Contents**:
- Agent equity over time (from steps.csv)
- BTC buy-and-hold equity over time (computed from price data)
- Metadata subtitle: backtest ID, date range, agent name

**Called By**:
- backtesting/plots/plot_runner.py

**Artifacts Created**:
- `agent_vs_btc_benchmark.png`

### utils/ Directory

#### utils/logger.py

**Purpose**: Loguru-based logging system

**Key Classes**:
- `LogComponent` (Enum): MAIN, DATA, STRATEGY, TRAINING, BACKTEST
- `RLLogger`: Main logger with component routing and phase tracking
- `_LoggerView`: Immutable logger proxy for component-specific logging

**RLLogger Methods**:
- `__init__()`: Initializes handlers (console + per-component file logs), removes old handlers to prevent duplicates
- `for_component()`: Returns _LoggerView for specific component
- `phase()`: Context manager for phase timing (auto-logs start, completion time, or failure)
- `info()`, `debug()`, `warning()`, `error()`, `exception()`, `success()`: Logging methods

**Log Files Created**:
- `logs/training.log` - MAIN + TRAINING components
- `logs/data_pipeline.log` - DATA component
- `logs/strategy_execution.log` - STRATEGY component
- `logs/backtest.log` - BACKTEST component

**Console Output**: Colorized INFO+ to stderr (tqdm-compatible)

**Used By**: Nearly all modules - passed as logger parameter

#### utils/user_input.py

**Purpose**: Interactive user prompts

**Key Functions**:
- `collect_run_mode()`: Prompts for TRAIN_AND_BACKTEST / TRAIN_ONLY / BACKTEST_ONLY
- `collect_train_and_backtest_input()`: Collects model name, description, training dates, backtest dates, creates run folder, returns (metadata, backtest_config, run_path)
- `collect_train_only_input()`: Collects training metadata, returns (metadata, run_path)
- `collect_backtest_only_input()`: Selects existing model, collects backtest dates, returns (backtest_config, run_path)
- `input_model_name()`: Validates alphanumeric + underscore only
- `input_date()`: Validates DD-MM-YYYY format
- `collect_training_date_range()`: Collects and confirms training dates
- `collect_backtest_date_range()`: Collects and confirms backtest dates, checks overlap with training
- `select_existing_model_run()`: Lists trained models with metadata.json and act.pth, prompts selection

**Called By**:
- main.py: `main()` function

#### utils/metadata.py

**Purpose**: Metadata file management

**Key Functions**:
- `create_metadata_file()`: Creates initial metadata.json from user input (called once from main.py)
- `enrich_metadata_with_training_config()`: Adds RL, env_spec, environment, data, strategies sections (called from rl_configs.py)
- `load_metadata()`: Loads metadata.json from run directory
- `append_backtest_metadata()`: Appends backtest entry to metadata["backtests"] list

**Metadata Sections**:
- model_name, machine_name, description, created_at, results_path, run_mode
- training: start_date, end_date, train_test_split
- rl: model, gamma, learning_rate, net_dims, total_training_steps, seed, algorithm_params
- env_spec: state_dim, action_dim
- environment: initial_balance, leverage_limit, stops, deadzone, fees, slippage, reward_function
- data: exchange, trading_pair, timeframe, indicators, enable_turbulence, enable_vix, external_assets
- strategies: enabled, strategy_list
- backtests: list of backtest entries (id, created_at, start_date, end_date, overlaps_training, output_dir)

**Called By**:
- main.py: `create_metadata_file()`, `append_backtest_metadata()`
- rl_configs.py: `enrich_metadata_with_training_config()`
- backtesting/backtest_runner.py: `load_metadata()`

**Artifacts Created/Modified**:
- `results/{model_name}_{machine}/metadata.json`

#### utils/normalization.py

**Purpose**: State normalization for neural network

**Key Functions**:
- `normalize_state()`: Converts raw state to normalized float32 vector
- `inverse_normalize_state()`: Converts normalized state back to human-readable values (for debugging)

**Normalization Scheme**:
- Balance: / INITIAL_BALANCE
- OHLC: / close (relative scaling)
- Volume: log1p(vol) / 20
- MACD: (val / close) × 50
- Bollinger Bands, SMAs: / close
- RSI, DX: / 100
- Turbulence: tanh(val × 20)
- VIX: tanh(val / 100)
- Strategy signals: as-is (binary 0/1)
- Holdings: / MAX_POSITION_BTC

**Called By**:
- bitcoin_env.py: `reset()`, `step()`

#### utils/formatting.py

**Purpose**: Console output formatting

**Key Class**: `Formatter`

**Methods**:
- `error_context()`: Wraps error messages with separator boxes
- `display_training_config()`: Pretty-prints training configuration
- `format_date_range_duration()`: Formats duration between dates (e.g., "1 year 2 months 15 days")

**Called By**:
- main.py: Error formatting, config display
- utils/user_input.py: Duration formatting

#### utils/progress.py

**Purpose**: Progress bar wrapper

**Key Class**: `ProgressTracker`

**Methods**:
- `process_items()`: Creates tqdm progress bar for loops

**Called By**:
- data/data_manager.py: Strategy execution progress
- data/data_processor.py: Download progress

#### utils/resampling.py

**Purpose**: Timeframe resampling

**Key Functions**:
- `resample_to_interval()`: Aggregates candles to higher timeframe (15m → 1h, 4h, etc.)
- `resampled_merge()`: Merges resampled data back to base timeframe (prevents lookahead bias)

**Called By**:
- data/data_manager.py: `_process_single_strategy_with_resampling()`

#### utils/timeframes.py

**Purpose**: Timeframe conversion

**Key Function**:
- `timeframe_to_minutes()`: Converts timeframe string to minutes

**Called By**:
- data/data_manager.py: Strategy resampling logic

#### utils/date_display.py

**Purpose**: Date formatting for display

**Called By**:
- utils/user_input.py: Date range display

## 5. End-to-End Execution Flow

### 5.1 Entry Point and Mode Selection

**File**: `main.py`

1. **System Initialization** (main() function):
   - Creates initial RLLogger instance (run_path=None) for console-only logging
   - Initializes DataManager with exchange, trading pair, timeframe from config
   - Creates directory structure: `data/download_data/training_data/raw/`, `data/download_data/training_data/processed/`, `data/download_data/archived/raw/`, `data/download_data/archived/processed/`

2. **Run Mode Selection** (`collect_run_mode()`):
   - Displays menu: TRAIN_AND_BACKTEST (1), TRAIN_ONLY (2), BACKTEST_ONLY (3)
   - User selects mode via console input
   - Returns run_mode string

### 5.2 TRAIN_AND_BACKTEST Flow

**Input Collection** (`collect_train_and_backtest_input()`):

1. **Model Name Collection**:
   - Prompts for model name (alphanumeric + underscore validation)
   - Creates run folder: `results/{model_name}_{machine_name}/`
   - Raises FileExistsError if folder already exists
   - **Artifact Created**: `results/{model_name}_{machine_name}/` directory

2. **Description Collection**:
   - Prompts for optional description

3. **Training Date Range Collection**:
   - Displays train/test split info (e.g., "75% training, 25% testing")
   - Prompts for start date (DD-MM-YYYY format)
   - Prompts for end date (DD-MM-YYYY format)
   - Validates start < end
   - Confirms with user

4. **Backtest Date Range Collection**:
   - Prompts for backtest start date
   - Prompts for backtest end date
   - Validates start < end
   - **Overlap Detection**: Checks if backtest range overlaps with training range
   - If overlap detected: Warns user about data leakage risk, asks for confirmation
   - Confirms with user

5. **Metadata Assembly**:
   - Creates metadata dictionary with: model_name, machine_name, description, created_at (UTC timestamp), results_path, run_mode="TRAIN_AND_BACKTEST", training {start_date, end_date, train_test_split}
   - Creates backtest_config dictionary with: start_date, end_date, overlaps_training flag
   - Returns (metadata, backtest_config, run_path)

**Logger Re-initialization**:
- Creates new RLLogger instance with run_path for file logging
- Sets up component-routed log files: `logs/training.log`, `logs/data_pipeline.log`, `logs/strategy_execution.log`, `logs/backtest.log`
- Updates DataManager's logger to new instance via `manager.logger = active_logger.for_component(LogComponent.DATA)`

**Configuration Display**:
- Calls `Formatter.display_training_config(metadata, active_logger)` to print training config to console

**Metadata File Creation**:
- Calls `create_metadata_file(metadata, run_path)`
- **Artifact Created**: `results/{model_name}_{machine}/metadata.json` (initial version with user input only)

**Training Pipeline Execution** (`run_training_pipeline()`):

**Phase 1: Data Preparation** (Step 1 of 5):

1. **Extract Training Dates**:
   - Reads start_date, end_date from metadata["training"]
   - Loads strategy list from config.STRATEGY_LIST if config.ENABLE_STRATEGIES else []

2. **Get Arrays** (`manager.get_arrays()`):
   - Calls `get_processed_data()` internally which follows smart waterfall:

   **Path A: Processed Cache Hit** (if `config.USE_PREPROCESSED_DATA=True` and cache exists):
   - Reads processed file: `data/download_data/training_data/processed/{exchange}_{pair}_{timeframe}_processed.parquet`
   - Validates date range (superset match)
   - Checks for missing indicators (compares config.INDICATORS vs DataFrame columns)
   - Checks for missing/invalid strategies (compares config.STRATEGY_LIST vs DataFrame columns, validates One-Hot encoding)
   - If any features missing/invalid: Incrementally adds them and saves updated processed file
   - **Artifact Modified**: `data/download_data/training_data/processed/` (if incremental update)
   - Filters to requested date range
   - Returns DataFrame immediately (fast path)

   **Path B: Raw Data + Full Processing** (if no processed cache or date range insufficient):

   **Phase 1: Download** (`get_or_download_data()`):
   - Checks for raw cache: `data/download_data/training_data/raw/{exchange}_{pair}_{timeframe}.parquet`
   - If exists: Loads date column only, detects gaps via `_get_data_gap()`, downloads missing gaps via `_download_with_retry()`, merges with existing data, deduplicates, sorts
   - If not exists or force_redownload: Downloads full range
   - **Artifacts Created/Modified**:
     - `data/download_data/training_data/raw/{exchange}_{pair}_{timeframe}.parquet`
     - `data/download_data/archived/raw/{timestamp}_{filename}.parquet` (if file was replaced)

   **Phase 2: Feature Engineering** (`add_base_features()`):
   - Cleans data (removes duplicates, validates OHLC, forward/backward fills NaN)
   - Adds technical indicators via TA-Lib (MACD, Bollinger Bands, RSI, DX, SMAs)
   - Adds turbulence index if config.ENABLE_TURBULENCE
   - Adds VIX data via ExternalDataManager if enabled in config.EXTERNAL_ASSETS
   - **Artifact Created**: `data/download_data/external/vix_daily_cache.csv` (if VIX enabled and cache missing)

   **Phase 3: Strategy Execution** (`add_strategy_signals()`):
   - Loads strategy metadata from strategies/strategies_registry.json
   - Detects existing strategy columns in DataFrame
   - Validates existing data (checks One-Hot encoding validity)
   - Filters to strategies needing execution (missing or invalid)
   - If multiple strategies: Uses ProcessPoolExecutor with max_workers from config.MAX_STRATEGY_WORKERS
   - For each strategy:
     - Checks if resampling needed (compares base_timeframe vs strategy timeframe)
     - If resampling needed: Resamples DataFrame to strategy timeframe via `resample_to_interval()`
     - Calculates lookback candles based on lookback_hours and timeframe
     - Processes each candle: creates window, runs strategy.run(), converts signal to One-Hot
     - If resampled: Merges signals back to base timeframe via `resampled_merge()` (prevents lookahead bias)
   - Forward/backward fills NaN values in strategy columns
   - Sets NaN to HOLD signal as fallback

   **Phase 4: Validation** (`validate_and_clean()`):
   - Delegates to DataProcessor.clean_data() for final validation

   **Phase 5: Save Processed Data**:
   - Saves to `data/download_data/training_data/processed/{exchange}_{pair}_{timeframe}_processed.parquet`
   - **Artifacts Created/Modified**:
     - `data/download_data/training_data/processed/{filename}.parquet`
     - `data/download_data/archived/processed/{timestamp}_{filename}.parquet` (if file was replaced)

3. **Array Conversion** (`to_arrays()`):
   - Extracts price array (OHLCV) - shape: (T, 5)
   - Extracts tech array (indicators + VIX if enabled) - shape: (T, 7-8)
   - Extracts turbulence array (turbulence + VIX if both enabled) - shape: (T, 1-2)
   - Extracts signal array (strategy One-Hot signals) - shape: (T, 4×N_strategies)
   - Extracts datetime array - shape: (T,)
   - Returns (price_array, tech_array, turbulence_array, signal_array, datetime_array)

**Phase 2: Dimension Calculation** (Step 2 of 5):

1. **Extract Dimensions**:
   - price_dim = price_array.shape[1] = 5 (OHLCV)
   - tech_dim = tech_array.shape[1] = 7 or 8 (indicators + optional VIX)
   - turb_dim = turbulence_array.shape[1] = 0, 1, or 2 (turbulence + optional VIX)
   - sig_dim = signal_array.shape[1] = 4 × len(STRATEGY_LIST)

2. **Calculate State Dimension**:
   - state_dim = 1 (balance) + price_dim + tech_dim + turb_dim + sig_dim + 1 (holdings)
   - Example: state_dim = 1 + 5 + 7 + 2 + 20 + 1 = 36 (for 5 strategies, turbulence + VIX enabled)

3. **Action Dimension**:
   - action_dim = 2 (a_pos, a_sl)

4. **Calculate Train/Test Split**:
   - total_steps = price_array.shape[0]
   - split_idx = int(total_steps × config.TRAIN_TEST_SPLIT)
   - train_max_step = split_idx (first 75% of data)
   - eval_max_step = total_steps - split_idx (last 25% of data)

**Phase 3: RL Configuration Build** (Step 3 of 5):

1. **Build ElegantRL Config** (`build_elegantrl_config()`):
   - Validates NET_DIMS[0] >= state_dim (raises ValueError if violated)
   - Selects agent class: AgentPPO if config.RL_MODEL=="PPO" else AgentSAC
   - Validates PPO horizon_len < train_max_step (raises ValueError if violated)
   - Creates train_env_args dict: env_name, num_envs, max_step=train_max_step, state_dim, action_dim, if_discrete=False, arrays (price, tech, turbulence, signal, datetime), mode="train"
   - Creates eval_env_args dict: same as train but max_step=eval_max_step, mode="test"
   - Creates Config object: agent_class, env_class=ElegantRLBitcoinEnv, env_args
   - Sets Config.cwd = `results/{model_name}_{machine}/elegantrl/`
   - Sets Config.if_remove = False (never delete run folder)
   - Sets Config.random_seed = config.SEED
   - Sets Config.break_step = config.TOTAL_TRAINING_STEPS
   - Sets Config.gamma = config.GAMMA
   - Sets Config.learning_rate = config.LEARNING_RATE
   - Sets Config.net_dims = config.NET_DIMS
   - Sets algorithm-specific params (PPO_CONFIG or SAC_CONFIG)
   - Sets Config.eval_env_class = ElegantRLBitcoinEnv, eval_env_args
   - Sets Config.eval_per_step = 20000, eval_times = 8

2. **Enrich Metadata** (`enrich_metadata_with_training_config()`):
   - Loads existing metadata.json
   - Adds "rl" section: model, gamma, learning_rate, net_dims, total_training_steps, seed, algorithm_params
   - Adds "env_spec" section: state_dim, action_dim (model contract)
   - Adds "environment" section: initial_balance, leverage_limit, stops, deadzone, fees, slippage, reward_function, downside_weight
   - Adds "data" section: exchange, trading_pair, timeframe, indicators, enable_turbulence, enable_vix, external_assets
   - Adds "strategies" section: enabled, strategy_list
   - Saves enriched metadata.json
   - **Artifact Modified**: `results/{model_name}_{machine}/metadata.json` (now contains full training config)

**Phase 4: Agent Training** (Step 4 of 5):

1. **Train Agent** (`train_agent(erl_config)`):
   - ElegantRL training loop (elegantrl/train/run.py)
   - Initializes agent (PPO or SAC) with net_dims, state_dim, action_dim
   - Creates training and evaluation environments (ElegantRLBitcoinEnv wrapping BitcoinTradingEnv)
   - Training loop runs for TOTAL_TRAINING_STEPS:
     - **For PPO**: Collects rollouts (horizon_len steps), computes advantages (GAE), updates policy/value networks (repeat_times gradient passes), clips policy updates (ratio_clip)
     - **For SAC**: Samples transitions from replay buffer (batch_size), updates actor/critic/temperature networks, soft-updates target networks (tau)
   - Periodically evaluates on test environment (eval_per_step interval)
   - Saves checkpoints every N steps

2. **Artifacts Created**:
   - `results/{model_name}_{machine}/elegantrl/act.pth` - Trained actor network (policy)
   - `results/{model_name}_{machine}/elegantrl/cri.pth` - Trained critic network (value/Q-function)
   - `results/{model_name}_{machine}/elegantrl/recorder.pth` - Training metrics (rewards, episode returns)

**Backtest Pipeline Execution** (`run_backtest_pipeline()`):

**Phase 5: Backtest Setup**:

1. **Verify Checkpoint**:
   - Checks `results/{model_name}_{machine}/elegantrl/act.pth` exists
   - Raises FileNotFoundError if missing

2. **Create Backtest ID**:
   - Generates timestamp-based ID: `bt_YYYYMMDD_HHMMSS` (UTC)
   - Creates directory: `results/{model_name}_{machine}/backtests/{backtest_id}/`
   - **Artifact Created**: Backtest output directory

3. **Load Model Metadata**:
   - Loads `metadata.json` from run_path
   - Extracts training configuration for compatibility validation

**Phase 6: Backtest Data Preparation** (Step 1 of 3):

1. **Load Strategy List**:
   - Reads config.STRATEGY_LIST if config.ENABLE_STRATEGIES else []

2. **Get Arrays** (`manager.get_arrays()`):
   - Follows same 5-phase pipeline as training data preparation
   - Uses backtest date range (from backtest_config)
   - Returns (price_array, tech_array, turbulence_array, signal_array, datetime_array) for backtest period

**Phase 7: Backtest Execution** (Step 2 of 3):

1. **Build Backtest Environment**:
   - Creates BitcoinTradingEnv with mode="backtest" (uses full dataset without train/test split)

2. **Validate Compatibility**:
   - **Indicator Set**: Compares trained_indicators vs current_indicators (raises ValueError if mismatch)
   - **Turbulence/VIX Flags**: Compares ENABLE_TURBULENCE, VIX enabled status (raises ValueError if mismatch)
   - **Strategy Set**: Compares trained strategy enabled flag, strategy set (raises ValueError if mismatch)
   - **Strategy Order**: Compares trained strategy list order (raises ValueError if different - critical for signal_vec layout)
   - **State/Action Dimensions**: Compares env.state_dim vs metadata["env_spec"]["state_dim"] (raises ValueError if mismatch)

3. **Load Actor Policy**:
   - Loads agent class (AgentPPO or AgentSAC) based on metadata["rl"]["model"]
   - Loads act.pth checkpoint to device (config.BACKTEST_DEVICE)
   - Sets policy to eval mode

4. **Initialize Loggers**:
   - Creates StepLogger (writes steps.csv)
   - Creates StateDebugLogger (writes state_debug.csv)
   - Creates TradeTracker (writes trades.csv)
   - Opens all file handles

5. **Run Episode**:
   - Resets environment: `state = env.reset()`
   - Loop until done:
     - **Policy Inference**: Converts state to tensor, runs policy forward pass, clips actions to [-1, 1], extracts (a_pos, a_sl)
     - **State Snapshot** (before step): Copies current_price, current_tech, current_turbulence, current_signal, current_datetime, holdings_before
     - **Environment Step**: `next_state, reward, done, info = env.step([a_pos, a_sl])`
       - Inside env.step():
         - Extracts OHLC from current_price
         - Computes old_equity for reward calculation
         - **Stop-Loss Check**: If stop_price exists, checks if low <= stop_price (long) or high >= stop_price (short)
         - If stop triggered: Calls `close_position()` at stop_exec_price (min(open, stop) for long, max(open, stop) for short)
         - Else: Calls `apply_action()` which:
           - Computes equity, target notional exposure from a_pos
           - Limits BTC position to MAX_POSITION_BTC
           - Applies exposure deadzone (returns TradeResult with no trade if change < EXPOSURE_DEADZONE)
           - Computes delta_btc, applies trade with fees/slippage
           - Updates entry_price and stop_price if new position or flip
         - Computes new_equity, calculates reward via reward function
         - Checks bankruptcy (equity <= 0) or end of data (step_idx == max_step - 1)
         - Advances time: step_idx += 1, loads next candle data
         - Builds next_state via `normalize_state()`
     - **Step Logging**: Calls `step_logger.log_step()` with full state data
     - **Debug Logging**: Calls `debug_logger.log_step()` with raw state data
     - **Trade Tracking**: Calls `trade_tracker.on_step()` to detect position changes and log trades
     - Advance state: `state = next_state`

6. **End-of-Episode Finalization**:
   - Computes final_equity_mtm (mark-to-market at last close)
   - Checks bankruptcy (final_equity_mtm <= 0)
   - If position still open and config.BACKTEST_FORCE_CLOSE=True:
     - Calls `close_position()` to realize P&L
     - Logs forced close trade
     - Sets final_equity_realized
   - Else if position still open:
     - Calls `trade_tracker.finalize_end()` to close open trade in trades.csv
   - Closes all logger file handles

7. **Write Summary**:
   - Creates summary dict: initial_equity, final_equity_mtm, final_equity_realized, episode_return_mtm, episode_return_realized, bankrupt, force_close
   - **Artifact Created**: `backtests/{backtest_id}/summary.json`

8. **Artifacts Created**:
   - `backtests/{backtest_id}/steps.csv` - Complete step log (step_idx, timestamp, state vector, price/tech/turb/sig, actions, trade info, equity, reward, flags)
   - `backtests/{backtest_id}/trades.csv` - Trade log (entry/exit times, prices, P&L, duration, stop_triggered)
   - `backtests/{backtest_id}/state_debug.csv` - Debug state log
   - `backtests/{backtest_id}/summary.json` - Episode summary

**Phase 8: Metrics Computation**:

1. **Load Artifacts**:
   - Reads steps.csv into DataFrame
   - Reads trades.csv into DataFrame
   - Reads summary.json

2. **Compute Metrics** (`compute_and_write_metrics()`):
   - Extracts timeframe from model_metadata, resolves steps_per_year
   - **Run Metadata**: Computes run_id, timeframe, start_date, end_date, duration, num_steps, initial_capital
   - **Performance Metrics**:
     - Calculates returns: steps_df["equity"].pct_change()
     - Computes CAGR: (final_equity / initial_equity) ^ (steps_per_year / num_steps) - 1
     - Computes volatility: returns.std() × sqrt(steps_per_year)
     - Computes Sharpe ratio: (CAGR - risk_free_rate) / volatility
     - Computes Sortino ratio: (CAGR - risk_free_rate) / downside_deviation
     - Computes max drawdown: max((rolling_max - equity) / rolling_max)
     - Computes Calmar ratio: CAGR / abs(max_drawdown)
   - **Risk & Behavior Metrics**:
     - Exposure time %: (steps with holdings != 0) / total_steps
     - Avg/max exposure %: mean/max of abs(holdings × price / equity)
     - Turnover: sum(abs(delta_btc))
     - Stop-loss triggers: count + rate
   - **Trade-Level Metrics**:
     - Num trades, win rate (wins / total)
     - Profit factor: sum(winning P&L) / abs(sum(losing P&L))
     - Avg/median P&L, avg win/loss, max win/loss, avg duration
   - **Reward Diagnostics**:
     - Sum/mean/std of rewards
     - Correlation between rewards and returns
   - **Data Quality**:
     - Equity min/max
     - NaN/Inf counts
     - Bankruptcy flag
   - Sanitizes NaN/Inf for JSON serialization

3. **Write Metrics**:
   - **Artifact Created**: `backtests/{backtest_id}/metrics.json`

**Phase 9: Plot Generation**:

1. **Generate Plots** (`generate_backtest_plots()`):
   - Loads steps.csv, metrics.json, summary.json
   - Calls `benchmark_equity_plot.py`:
     - Extracts agent equity from steps_df
     - Computes BTC buy-and-hold equity: initial_equity × (price_t / price_0)
     - Plots both curves with legend
     - Adds title with backtest ID, date range, agent name
     - Saves as PNG
   - **Artifact Created**: `backtests/{backtest_id}/plots/agent_vs_btc_benchmark.png`

**Phase 10: Metadata Update** (Step 3 of 3):

1. **Create Backtest Entry**:
   - Creates backtest_entry dict: id, created_at (UTC), start_date, end_date, overlaps_training, output_dir

2. **Append to Metadata**:
   - Loads metadata.json
   - Appends backtest_entry to metadata["backtests"] list
   - Saves metadata.json
   - **Artifact Modified**: `results/{model_name}_{machine}/metadata.json` (now includes backtest entry)

**Session Complete**:
- Logs success message
- All artifacts persisted

### 5.3 TRAIN_ONLY Flow

**Differences from TRAIN_AND_BACKTEST**:
- Only executes Phases 1-4 (skips Backtest Pipeline)
- `collect_train_only_input()` collects training metadata without backtest dates
- Returns (metadata, run_path) without backtest_config
- metadata["run_mode"] = "TRAIN_ONLY"
- No backtest artifacts created

### 5.4 BACKTEST_ONLY Flow

**Differences from TRAIN_AND_BACKTEST**:
- Skips Training Pipeline (Phases 1-4)
- Only executes Backtest Pipeline (Phases 5-10)
- `collect_backtest_only_input()`:
  - Lists existing trained models with metadata.json and act.pth
  - User selects model by number
  - Loads model_metadata from selected run
  - Collects backtest dates with overlap detection against training dates
  - Returns (backtest_config, run_path)
- No new training artifacts created
- Uses existing act.pth from selected model

## 6. Technologies & Libraries

### Core ML/RL
- **numpy** (≥1.24.0): Array operations, numerical computing
- **pandas** (≥2.0.0): DataFrame operations, data manipulation
- **torch** (≥2.0.0): PyTorch for neural networks (ElegantRL dependency)
- **stable-baselines3** (≥2.0.0): RL algorithms library
- **finrl** (≥0.3.0): Financial RL framework

### Exchange Connectivity
- **ccxt** (≥4.0.0): Cryptocurrency exchange API library (supports Binance, Kraken, etc.)

### Technical Analysis
- **TA-Lib** (≥0.4.28): Technical indicator calculation library (MACD, RSI, Bollinger Bands, etc.)

### Data Storage
- **pyarrow** (≥12.0.0): Parquet file format support
- **fastparquet** (≥2023.4.0): Alternative Parquet implementation

### Data Acquisition
- **yfinance** (≥0.2.0): Yahoo Finance data (VIX integration)

### Utilities
- **python-dateutil** (≥2.8.2): Date parsing and manipulation
- **tqdm** (≥4.65.0): Progress bars
- **loguru** (≥0.7.0): Advanced logging with component routing
- **rich** (≥13.7.0): Terminal UI (styled prompts, panels)

### Framework Components
- **ElegantRL**: Vendored RL framework (modified)
  - Supports PPO (Proximal Policy Optimization)
  - Supports SAC (Soft Actor-Critic)
  - Custom Config dataclass for training configuration
  - Evaluator for test environment evaluation
  - Replay buffer for off-policy algorithms

### Python Standard Library
- **multiprocessing**: Parallel strategy execution (ProcessPoolExecutor)
- **concurrent.futures**: Process pool management
- **pathlib**: Path operations
- **json**: Metadata serialization
- **datetime**: Timestamp management
- **dataclasses**: Configuration structures
- **enum**: Signal types, log components
- **typing**: Type hints (NamedTuple, Dict, List, Optional, Tuple)
- **abc**: Abstract base classes (BaseStrategy)

### Key Design Patterns
- **Pipeline Architecture**: 5-phase data processing pipeline
- **Strategy Pattern**: Abstract BaseStrategy with multiple implementations
- **Adapter Pattern**: ElegantRLBitcoinEnv adapts BitcoinTradingEnv to ElegantRL API
- **Factory Pattern**: StrategyRegistry for dynamic strategy loading
- **Observer Pattern**: Component-routed logging via LogComponent
- **Dataclass Pattern**: PositionState, TradeConfig, TradeResult, Config
- **Context Manager**: logger.phase() for automatic timing/failure detection
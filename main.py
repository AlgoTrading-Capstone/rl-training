"""
Main orchestrator for RL training and backtesting pipeline.
"""

from pathlib import Path
from datetime import datetime

import config
from data.data_manager import DataManager
from utils.user_input import collect_run_mode, collect_train_and_backtest_input, collect_train_only_input, collect_backtest_only_input
from utils.metadata import create_metadata_file, load_metadata, append_backtest_metadata
from utils.logger import RLLogger, LogComponent
from utils.formatting import Formatter
from rl_configs import build_elegantrl_config
from elegantrl.train.run import train_agent
from backtesting.backtest_runner import run_backtest


def run_training_pipeline(
    metadata: dict,
    run_path: Path,
    manager: DataManager,
    logger: RLLogger,
) -> None:
    """
    Execute the full RL training pipeline using ElegantRL.

    Responsibilities:
    - Load and clean training data
    - Calculate state/action dimensions and split sizes
    - Build ElegantRL training configuration
    - Train and evaluate RL agent using ElegantRL
    """

    # --------------------------------------------------------
    # STEP 2: Load and clean training data
    # --------------------------------------------------------
    with logger.phase("Data Preparation", 1, 5):
        try:
            # Get strategy list from config
            strategy_list = config.STRATEGY_LIST if config.ENABLE_STRATEGIES else []

            # Extract training date range from metadata
            train_start = metadata["training"]["start_date"]
            train_end = metadata["training"]["end_date"]

            logger.info(f"Loading data for training: {train_start} to {train_end}")

            # Fetch arrays for training
            price_array, tech_array, turbulence_array, signal_array, datetime_array = manager.get_arrays(
                start_date=train_start,
                end_date=train_end,
                strategy_list=strategy_list,
            )

            logger.info(f"Loaded {price_array.shape[0]} timesteps for training")

        except Exception as e:
            error_msg = Formatter.error_context(
                f"ERROR DURING DATA PREPARATION: {e}",
                f"Partial data may be available in: {config.DATA_ROOT_PATH}"
            )
            logger.exception(error_msg)  # Logs error + traceback
            raise
    # --------------------------------------------------------
    # STEP 3: Calculate state/action dimensions and split sizes
    # --------------------------------------------------------
    with logger.phase("Dimension Calculation", 2, 5):
        try:
            price_dim = price_array.shape[1]
            tech_dim = tech_array.shape[1]
            turb_dim = turbulence_array.shape[1]
            sig_dim = signal_array.shape[1]

            # State = balance + price features + indicators + turbulence & VIX + strategies signals + position
            state_dim = 1 + price_dim + tech_dim + turb_dim + sig_dim + 1

            # Action = position size + stop-loss
            action_dim = 2

            # Calculate Max Steps for train/test split
            total_steps = price_array.shape[0]
            split_idx = int(total_steps * config.TRAIN_TEST_SPLIT)

            # Train gets the first chunk, Test gets the remainder
            train_max_step = split_idx
            eval_max_step = total_steps - split_idx

            logger.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")
            logger.info(f"Train steps: {train_max_step}, Eval steps: {eval_max_step}")

        except Exception as e:
            error_msg = Formatter.error_context(
                f"ERROR DURING DIMENSION CALCULATION: {e}",
                "Training aborted during dimension calculation."
            )
            logger.exception(error_msg)  # Logs error + traceback
            raise

    # --------------------------------------------------------
    # STEP 4: Build ElegantRL training configuration
    # --------------------------------------------------------
    with logger.phase("RL Configuration Build", 3, 5):
        try:
            logger.info(f"Building {config.RL_MODEL} configuration")

            erl_config = build_elegantrl_config(
                price_array=price_array,
                tech_array=tech_array,
                turbulence_array=turbulence_array,
                signal_array=signal_array,
                datetime_array=datetime_array,
                state_dim=state_dim,
                action_dim=action_dim,
                train_max_step=train_max_step,
                eval_max_step=eval_max_step,
                run_path=run_path,
            )

            logger.info("ElegantRL configuration built successfully")

        except Exception as e:
            error_msg = Formatter.error_context(
                f"ERROR DURING RL CONFIG BUILD: {e}",
                "Check rl_configs.py and config.py for invalid settings."
            )
            logger.exception(error_msg)  # Logs error + traceback
            raise

    # --------------------------------------------------------
    # STEP 5: Train and evaluate RL agent using ElegantRL
    # --------------------------------------------------------
    with logger.phase("Agent Training", 4, 5):
        try:
            logger.info(f"Starting {config.RL_MODEL} training (max steps: {config.TOTAL_TRAINING_STEPS})")
            train_agent(erl_config)
            logger.info("Training completed successfully")

        except KeyboardInterrupt:
            logger.warning("TRAINING INTERRUPTED BY USER")
            logger.info("Graceful shutdown requested. Partial results may be saved.")
            raise

        except Exception as e:
            error_msg = Formatter.error_context(
                f"ERROR DURING TRAINING: {e}",
                "Check logs and hyperparameters for instability."
            )
            logger.exception(error_msg)  # Logs error + traceback
            raise


def run_backtest_pipeline(
    backtest_config: dict,
    run_path: Path,
    manager: DataManager,
    logger: RLLogger,
) -> None:
    """
    Execute a single backtest on an existing trained model.

    Responsibilities:
    - Create a unique backtest ID and output directory
    - Load model metadata
    - Load backtest data
    - Invoke backtest_runner.run_backtest
    - Append backtest entry to metadata.json
    """

    # CRITICAL: Define act_path FIRST to ensure it's in scope for all operations
    act_path = run_path / "elegantrl" / "act.pth"

    # Verify checkpoint exists immediately
    if not act_path.exists():
        error_msg = Formatter.error_context(
            f"ERROR: Model checkpoint not found at {act_path}",
            "Expected act.pth from previous training run."
        )
        logger.error(error_msg)
        raise FileNotFoundError(f"Missing model checkpoint: {act_path}")

    logger.info(f"Using actor checkpoint: {act_path}")

    # --------------------------------------------------------
    # STEP 6: Create backtest ID & output directory
    # --------------------------------------------------------
    try:
        backtest_id = datetime.utcnow().strftime("bt_%Y%m%d_%H%M%S")

        backtests_root = run_path / "backtests"
        backtest_dir = backtests_root / backtest_id

        # Create directories
        backtests_root.mkdir(exist_ok=True)
        backtest_dir.mkdir(exist_ok=False)

        logger.info(f"Backtest ID: {backtest_id}")
        logger.info(f"Output directory: {backtest_dir}")

    except Exception as e:
        error_msg = Formatter.error_context(
            f"ERROR CREATING BACKTEST DIRECTORY: {e}",
            "Backtest aborted during setup."
        )
        logger.exception(error_msg)  # Logs error + traceback
        raise

    # --------------------------------------------------------
    # STEP 7: Load model metadata
    # --------------------------------------------------------
    try:
        model_metadata = load_metadata(run_path)
        logger.debug("Model metadata loaded successfully")

    except Exception as e:
        error_msg = Formatter.error_context(
            f"ERROR LOADING MODEL METADATA FOR BACKTEST: {e}",
            f"Expected metadata.json in: {run_path}"
        )
        logger.exception(error_msg)  # Logs error + traceback
        raise

    # --------------------------------------------------------
    # STEP 8: Load backtest data
    # --------------------------------------------------------
    with logger.phase("Backtest Data Preparation", 1, 3):
        try:
            strategy_list = config.STRATEGY_LIST if config.ENABLE_STRATEGIES else []

            price_array, tech_array, turbulence_array, signal_array, datetime_array = manager.get_arrays(
                start_date=backtest_config["start_date"],
                end_date=backtest_config["end_date"],
                strategy_list=strategy_list,
            )

            logger.info(f"Loaded {price_array.shape[0]} timesteps for backtesting")

        except Exception as e:
            error_msg = Formatter.error_context(
                f"ERROR DURING BACKTEST DATA PREPARATION: {e}",
                "Backtest aborted due to data processing failure."
            )
            logger.exception(error_msg)  # Logs error + traceback
            raise

    # --------------------------------------------------------
    # STEP 9: Run backtest
    # --------------------------------------------------------
    with logger.phase("Backtest Execution", 2, 3):
        try:
            logger.info("Running backtest with trained model")

            run_backtest(
                model_metadata=model_metadata,
                act_path=act_path,
                price_array=price_array,
                tech_array=tech_array,
                turbulence_array=turbulence_array,
                signal_array=signal_array,
                datetime_array=datetime_array,
                out_dir=backtest_dir,
            )

            logger.info(f"Backtest results saved to: {backtest_dir}")

        except Exception as e:
            error_msg = Formatter.error_context(
                f"ERROR DURING BACKTEST EXECUTION: {e}",
                "Check model compatibility and data validity."
            )
            logger.exception(error_msg)  # Logs error + traceback
            raise
    # --------------------------------------------------------
    # STEP 10: Append backtest metadata
    # --------------------------------------------------------
    with logger.phase("Metadata Update", 3, 3):
        try:
            backtest_entry = {
                "id": backtest_id,
                "created_at": datetime.utcnow().isoformat(),
                "start_date": backtest_config["start_date"],
                "end_date": backtest_config["end_date"],
                "overlaps_training": backtest_config["overlaps_training"],
                "output_dir": f"backtests/{backtest_id}",
            }

            append_backtest_metadata(run_path, backtest_entry)

            logger.success("Backtest completed and metadata updated")

        except Exception as e:
            error_msg = Formatter.error_context(
                f"ERROR UPDATING BACKTEST METADATA: {e}",
                f"Metadata file: {run_path / 'metadata.json'}"
            )
            logger.exception(error_msg)  # Logs error + traceback
            raise

def main():
    # Initialize active logger reference (before run_path is created)
    active_logger = RLLogger(run_path=None, log_level=config.LOG_LEVEL)
    active_logger.info("=" * 60)
    active_logger.info("RL Pipeline Execution")
    active_logger.info("=" * 60)

    # --------------------------------------------------------
    # STEP 0: Initialize shared DataManager
    # --------------------------------------------------------
    try:
        active_logger.info("Initializing DataManager")
        manager = DataManager(
            exchange=config.EXCHANGE_NAME,
            trading_pair=config.TRADING_PAIR,
            base_timeframe=config.DATA_TIMEFRAME,
            logger=active_logger,
        )

    except Exception as e:
        error_msg = Formatter.error_context(
            f"ERROR INITIALIZING DATA MANAGER: {e}",
            "Check config.py for valid exchange/pair/timeframe settings."
        )
        active_logger.exception(error_msg)
        return

    # --------------------------------------------------------
    # STEP 1: Determine run mode and collect user input
    # --------------------------------------------------------
    try:
        run_mode = collect_run_mode()

        if run_mode == "TRAIN_AND_BACKTEST":
            metadata, backtest_config, run_path = collect_train_and_backtest_input()

            # Re-initialize logger with run_path for file logging
            active_logger = RLLogger(
                run_path=run_path,
                log_level=config.LOG_LEVEL,
                file_log_level=config.FILE_LOG_LEVEL,
                component=LogComponent.MAIN
            )

            # Update DataManager's logger to use the new file-logging instance
            manager.logger = active_logger.for_component(LogComponent.DATA)

            # Display configuration
            active_logger.info(Formatter.config_table(metadata))
            create_metadata_file(metadata, run_path)
            run_training_pipeline(metadata, run_path, manager, active_logger)
            run_backtest_pipeline(backtest_config, run_path, manager, active_logger)

        elif run_mode == "TRAIN_ONLY":
            metadata, run_path = collect_train_only_input()

            # Re-initialize logger with run_path for file logging
            active_logger = RLLogger(
                run_path=run_path,
                log_level=config.LOG_LEVEL,
                file_log_level=config.FILE_LOG_LEVEL,
                component=LogComponent.MAIN
            )

            # Update DataManager's logger to use the new file-logging instance
            manager.logger = active_logger.for_component(LogComponent.DATA)

            # Display configuration
            active_logger.info(Formatter.config_table(metadata))

            create_metadata_file(metadata, run_path)
            run_training_pipeline(metadata, run_path, manager, active_logger)

        elif run_mode == "BACKTEST_ONLY":
            backtest_config, run_path = collect_backtest_only_input()

            # Re-initialize logger with run_path for file logging
            active_logger = RLLogger(
                run_path=run_path,
                log_level=config.LOG_LEVEL,
                file_log_level=config.FILE_LOG_LEVEL,
                component=LogComponent.MAIN
            )

            # Update DataManager's logger to use the new file-logging instance
            manager.logger = active_logger.for_component(LogComponent.DATA)

            run_backtest_pipeline(backtest_config, run_path, manager, active_logger)

        active_logger.success("\n=== Session Complete ===\n")

    except KeyboardInterrupt:
        active_logger.warning("\nExecution interrupted by user.")

    except Exception:
        active_logger.error("\nPipeline failed. See detailed logs above.")


if __name__ == "__main__":
    main()
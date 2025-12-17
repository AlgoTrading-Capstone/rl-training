"""
Main orchestrator for RL training and backtesting pipeline.
"""

import config
from data.data_manager import DataManager
from utils.user_input import (
    collect_run_mode,
    collect_train_and_backtest_input,
    collect_train_only_input,
    collect_backtest_only_input,
)
from rl_configs import build_elegantrl_config
from elegantrl.train.run import train_agent
from pathlib import Path
from utils.metadata import create_metadata_file
from utils.logger import RLLogger, LogComponent
from utils.formatting import Formatter


def run_training_pipeline(metadata, run_path, manager, logger):
    """
    Execute the full RL training pipeline using ElegantRL.
    """

    # --------------------------------------------------------
    # STEP 2: Load and clean training data
    # --------------------------------------------------------
    with logger.phase("Data Preparation", 1, 5):
        try:
            # Get processed arrays using smart incremental download
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
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
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
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
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
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
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
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            raise


def run_backtest_pipeline(metadata, run_path, manager, logger):
    """
    Execute backtesting pipeline.
    """

    # --------------------------------------------------------
    # STEP 6: Validate trained actor checkpoint exists
    # --------------------------------------------------------
    model_run_path = Path(metadata.get("model_run_path", run_path))  # TODO
    act_path = model_run_path / "elegantrl" / "act.pth"

    if not act_path.is_file():
        raise FileNotFoundError(
            f"Missing ElegantRL actor checkpoint: {act_path}\n"
            f"Run mode: {metadata.get('mode')}\n"
            f"Tip: Train first, or verify the ElegantRL output folder and checkpoint name."
        )

    logger.info(f"Using actor checkpoint: {act_path}")

    # --------------------------------------------------------
    # STEP 7: Load and clean backtest data
    # --------------------------------------------------------
    with logger.phase("Backtest Data Preparation", 1, 1):
        try:
            strategy_list = config.STRATEGY_LIST if config.ENABLE_STRATEGIES else []

            bt_start = metadata["backtest"]["start_date"]
            bt_end = metadata["backtest"]["end_date"]

            logger.info(f"Loading data for backtest: {bt_start} to {bt_end}")

            price_array, tech_array, turbulence_array, signal_array, datetime_array = manager.get_arrays(
                start_date=bt_start,
                end_date=bt_end,
                strategy_list=strategy_list,
            )

            logger.info(f"Loaded {price_array.shape[0]} timesteps for backtesting")

            # Placeholder – real backtest logic will be added later
            logger.success("Backtest pipeline scaffold completed successfully")

        except Exception as e:
            error_msg = Formatter.error_context(
                f"ERROR DURING BACKTEST DATA PREPARATION: {e}",
                "Backtest aborted due to data processing failure."
            )
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            raise


def main():
    # Initialize temporary logger (before run_path is created)
    temp_logger = RLLogger(run_path=None, log_level=config.LOG_LEVEL)
    temp_logger.info("=" * 60)
    temp_logger.info("RL Pipeline Execution")
    temp_logger.info("=" * 60)

    # --------------------------------------------------------
    # STEP 0: Initialize shared DataManager
    # --------------------------------------------------------
    temp_logger.info("Initializing DataManager")
    manager = DataManager(
        exchange=config.EXCHANGE_NAME,
        trading_pair=config.TRADING_PAIR,
        base_timeframe=config.DATA_TIMEFRAME,
    )

    # --------------------------------------------------------
    # STEP 1: Determine run mode and collect user input
    # --------------------------------------------------------
    run_mode = collect_run_mode()

    try:
        if run_mode == "TRAIN_AND_BACKTEST":
            metadata, run_path = collect_train_and_backtest_input()

            # Re-initialize logger with run_path for file logging
            logger = RLLogger(
                run_path=run_path,
                log_level=config.LOG_LEVEL,
                file_log_level=config.FILE_LOG_LEVEL,
                component=LogComponent.MAIN
            )

            # Display configuration
            logger.info(Formatter.config_table(metadata))

            create_metadata_file(metadata, run_path)
            run_training_pipeline(metadata, run_path, manager, logger)
            run_backtest_pipeline(metadata, run_path, manager, logger)

        elif run_mode == "TRAIN_ONLY":
            metadata, run_path = collect_train_only_input()

            # Re-initialize logger with run_path for file logging
            logger = RLLogger(
                run_path=run_path,
                log_level=config.LOG_LEVEL,
                file_log_level=config.FILE_LOG_LEVEL,
                component=LogComponent.MAIN
            )

            # Display configuration
            logger.info(Formatter.config_table(metadata))

            create_metadata_file(metadata, run_path)
            run_training_pipeline(metadata, run_path, manager, logger)

        elif run_mode == "BACKTEST_ONLY":
            metadata, run_path = collect_backtest_only_input()

            # Re-initialize logger with run_path for file logging
            logger = RLLogger(
                run_path=run_path,
                log_level=config.LOG_LEVEL,
                file_log_level=config.FILE_LOG_LEVEL,
                component=LogComponent.MAIN
            )

            run_backtest_pipeline(metadata, run_path, manager, logger)

        logger.success("\n=== Session Complete ===\n")

    except KeyboardInterrupt:
        if 'logger' in locals():
            logger.warning("\nExecution interrupted by user.")
        else:
            temp_logger.warning("\nExecution interrupted by user.")

    except Exception:
        if 'logger' in locals():
            logger.error("\nPipeline failed. See detailed logs above.")
        else:
            temp_logger.error("\nPipeline failed. See detailed logs above.")


if __name__ == "__main__":
    main()
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


def run_training_pipeline(metadata, run_path, manager):
    """
    Execute the full RL training pipeline using ElegantRL.
    """

    # --------------------------------------------------------
    # STEP 2: Load and clean training data
    # --------------------------------------------------------
    try:
        # Get processed arrays using smart incremental download
        strategy_list = config.STRATEGY_LIST if config.ENABLE_STRATEGIES else []

        # Extract training date range from metadata
        train_start = metadata["training"]["start_date"]
        train_end = metadata["training"]["end_date"]

        # Fetch arrays for training
        price_array, tech_array, turbulence_array, signal_array, datetime_array = manager.get_arrays(
            start_date=train_start,
            end_date=train_end,
            strategy_list=strategy_list,
        )

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR DURING DATA PREPARATION: {e}")
        print(f"{'=' * 60}")
        print("\nTraining aborted due to data processing failure.")
        print(f"Partial data may be available in: {config.DATA_ROOT_PATH}")
        import traceback
        traceback.print_exc()
        raise

    # --------------------------------------------------------
    # STEP 3: Calculate state/action dimensions and split sizes
    # --------------------------------------------------------
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

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR DURING DIMENSION CALCULATION: {e}")
        print(f"{'=' * 60}")
        print("\nTraining aborted during dimension calculation.")
        import traceback
        traceback.print_exc()
        raise

    # --------------------------------------------------------
    # STEP 4: Build ElegantRL training configuration
    # --------------------------------------------------------
    try:
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

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR DURING RL CONFIG BUILD: {e}")
        print(f"{'=' * 60}")
        print("\nTraining aborted during RL configuration build.")
        print("Check rl_configs.py and config.py for invalid settings.")
        import traceback
        traceback.print_exc()
        raise

    # --------------------------------------------------------
    # STEP 5: Train and evaluate RL agent using ElegantRL
    # --------------------------------------------------------
    try:
        train_agent(erl_config)

    except KeyboardInterrupt:
        print(f"\n{'=' * 60}")
        print("TRAINING INTERRUPTED BY USER")
        print(f"{'=' * 60}")
        print("\nGraceful shutdown requested. Partial results may be saved.")
        raise

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR DURING TRAINING: {e}")
        print(f"{'=' * 60}")
        print("\nTraining aborted due to runtime error.")
        print("Check logs and hyperparameters for instability.")
        import traceback
        traceback.print_exc()
        raise


def run_backtest_pipeline(metadata, run_path, manager):
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

    print(f"[INFO] Using actor checkpoint: {act_path}")

    # --------------------------------------------------------
    # STEP 7: Load and clean backtest data
    # --------------------------------------------------------
    try:
        strategy_list = config.STRATEGY_LIST if config.ENABLE_STRATEGIES else []

        bt_start = metadata["backtest"]["start_date"]
        bt_end = metadata["backtest"]["end_date"]

        price_array, tech_array, turbulence_array, signal_array, datetime_array = manager.get_arrays(
            start_date=bt_start,
            end_date=bt_end,
            strategy_list=strategy_list,
        )

        # Placeholder – real backtest logic will be added later

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR DURING BACKTEST DATA PREPARATION: {e}")
        print(f"{'=' * 60}")
        import traceback
        traceback.print_exc()
        raise

    print("\nBacktest pipeline scaffold completed successfully.\n")


def main():
    print("=" * 60)
    print("RL Pipeline Execution")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 0: Initialize shared DataManager
    # --------------------------------------------------------
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
            create_metadata_file(metadata, run_path)
            run_training_pipeline(metadata, run_path, manager)
            run_backtest_pipeline(metadata, run_path, manager)

        elif run_mode == "TRAIN_ONLY":
            metadata, run_path = collect_train_only_input()
            create_metadata_file(metadata, run_path)
            run_training_pipeline(metadata, run_path, manager)

        elif run_mode == "BACKTEST_ONLY":
            metadata, run_path = collect_backtest_only_input()
            run_backtest_pipeline(metadata, run_path, manager)

        print("\n=== Session Complete ===\n")

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")

    except Exception:
        print("\nPipeline failed. See detailed logs above.")


if __name__ == "__main__":
    main()
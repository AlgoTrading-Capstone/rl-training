"""
Main orchestrator for RL training and backtesting pipeline.
"""

from pathlib import Path
from datetime import datetime

import config
from data.data_manager import DataManager
from utils.user_input import collect_run_mode, collect_train_and_backtest_input, collect_train_only_input, collect_backtest_only_input
from rl_configs import build_elegantrl_config
from elegantrl.train.run import train_agent
from utils.metadata import create_metadata_file, load_metadata, append_backtest_metadata
from backtesting.backtest_runner import run_backtest


def run_training_pipeline(
    metadata: dict,
    run_path: Path,
    manager: DataManager,
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
    try:
        # Get strategy list from config
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


def run_backtest_pipeline(
    backtest_config: dict,
    run_path: Path,
    manager: DataManager,
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

    # --------------------------------------------------------
    # STEP 6: Create backtest ID & output directory (fail-fast)
    # --------------------------------------------------------
    try:
        backtest_id = datetime.utcnow().strftime("bt_%Y%m%d_%H%M%S")

        backtests_root = run_path / "backtests"
        backtest_dir = backtests_root / backtest_id

        # Create directories
        backtests_root.mkdir(exist_ok=True)
        backtest_dir.mkdir(exist_ok=False)

        print(f"[INFO] Backtest ID: {backtest_id}")
        print(f"[INFO] Output directory: {backtest_dir}")

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR CREATING BACKTEST OUTPUT DIRECTORY: {e}")
        print(f"{'=' * 60}")
        import traceback
        traceback.print_exc()
        raise

    # --------------------------------------------------------
    # STEP 7: Load model metadata
    # --------------------------------------------------------
    try:
        model_metadata = load_metadata(run_path)

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR LOADING MODEL METADATA FOR BACKTEST: {e}")
        print(f"{'=' * 60}")
        import traceback
        traceback.print_exc()
        raise

    # --------------------------------------------------------
    # STEP 8: Load backtest data
    # --------------------------------------------------------
    try:
        strategy_list = config.STRATEGY_LIST if config.ENABLE_STRATEGIES else []

        price_array, tech_array, turbulence_array, signal_array, datetime_array = manager.get_arrays(
            start_date=backtest_config["start_date"],
            end_date=backtest_config["end_date"],
            strategy_list=strategy_list,
        )

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR DURING BACKTEST DATA PREPARATION: {e}")
        print(f"{'=' * 60}")
        import traceback
        traceback.print_exc()
        raise

    # --------------------------------------------------------
    # STEP 9: Run backtest
    # --------------------------------------------------------
    try:
        print("[INFO] Running backtest...")

        run_backtest(
            model_metadata=model_metadata,
            act_path=run_path / "elegantrl" / "act.pth",
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            signal_array=signal_array,
            datetime_array=datetime_array,
            out_dir=backtest_dir,
        )

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR DURING BACKTEST EXECUTION: {e}")
        print(f"{'=' * 60}")
        import traceback
        traceback.print_exc()
        raise

    # --------------------------------------------------------
    # STEP 10: Append backtest metadata
    # --------------------------------------------------------
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

        print("[INFO] Backtest completed and metadata updated.\n")

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR UPDATING BACKTEST METADATA: {e}")
        print(f"{'=' * 60}")
        import traceback
        traceback.print_exc()
        raise


def main():
    print("=" * 60)
    print("RL Pipeline Execution")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 0: Initialize shared DataManager
    # --------------------------------------------------------
    try:
        manager = DataManager(
            exchange=config.EXCHANGE_NAME,
            trading_pair=config.TRADING_PAIR,
            base_timeframe=config.DATA_TIMEFRAME,
        )

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR INITIALIZING DATA MANAGER: {e}")
        print(f"{'=' * 60}")
        import traceback
        traceback.print_exc()
        return

    # --------------------------------------------------------
    # STEP 1: Determine run mode and collect user input
    # --------------------------------------------------------
    try:
        run_mode = collect_run_mode()

        if run_mode == "TRAIN_AND_BACKTEST":
            metadata, backtest_config, run_path = collect_train_and_backtest_input()
            create_metadata_file(metadata, run_path)
            run_training_pipeline(metadata, run_path, manager)
            run_backtest_pipeline(backtest_config, run_path, manager)

        elif run_mode == "TRAIN_ONLY":
            metadata, run_path = collect_train_only_input()
            create_metadata_file(metadata, run_path)
            run_training_pipeline(metadata, run_path, manager)

        elif run_mode == "BACKTEST_ONLY":
            backtest_config, run_path = collect_backtest_only_input()
            run_backtest_pipeline(backtest_config, run_path, manager)

        print("\n=== Session Complete ===\n")

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")

    except Exception:
        print("\nPipeline failed. See detailed logs above.")


if __name__ == "__main__":
    main()
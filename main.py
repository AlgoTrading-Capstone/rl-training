"""
Main orchestrator for RL training/testing pipeline.
"""
import config
from data.data_manager import DataManager
from utils.user_input import collect_user_input
from bitcoin_env import BitcoinTradingEnv
from rl_configs import build_elegantrl_config
from elegantrl.train.run import train_agent


def main():
    print("=" * 60)
    print("RL Training Execution")
    print("=" * 60)

    # -------------------------------------------------------
    # STEP 1: Collect user input + create run folder + metadata
    # --------------------------------------------------------
    metadata, run_path = collect_user_input()

    # --------------------------------------------------------
    # STEP 2: Initialize DataManager and process data
    # --------------------------------------------------------
    try:
        # Initialize DataManager (uses shared data/ directory from config.DATA_ROOT_PATH)
        manager = DataManager(
            exchange=config.EXCHANGE_NAME,
            trading_pair=config.TRADING_PAIR,
            base_timeframe=config.DATA_TIMEFRAME
        )

        # Get processed arrays using smart incremental download
        strategy_list = config.STRATEGY_LIST if config.ENABLE_STRATEGIES else []
        price_array, tech_array, turbulence_array, signal_array = manager.get_arrays_for_training(
            start_date=metadata['start_date'],
            end_date=metadata['end_date'],
            strategy_list=strategy_list
        )

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR: {e}")
        print(f"{'=' * 60}")
        print("\nTraining aborted due to data processing failure.")
        print(f"Partial data may be available in: {config.DATA_ROOT_PATH}")
        import traceback
        traceback.print_exc()
        return

    # --------------------------------------------------------
    # STEP 3: Calculate Dimensions
    # --------------------------------------------------------
    try:
        price_dim = price_array.shape[1]
        tech_dim = tech_array.shape[1]
        turb_dim = turbulence_array.shape[1]
        sig_dim = signal_array.shape[1]

        # State includes: balance, price features, indicators, turbulence, strategy signals, position size
        state_dim = 1 + price_dim + tech_dim + turb_dim + sig_dim + 1

        # Action space: [a_pos, a_sl]
        action_dim = 2

        # Calculate Max Steps (Train/Test Split)
        total_data_points = price_array.shape[0]
        split_idx = int(total_data_points * config.TRAIN_TEST_SPLIT)

        # Train gets the first chunk, Test gets the remainder
        train_max_step = split_idx
        eval_max_step = total_data_points - split_idx

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR: {e}")
        print(f"{'=' * 60}")
        print("\nTraining aborted during dimension calculation.")
        import traceback
        traceback.print_exc()
        return

    # --------------------------------------------------------
    # STEP 4: Build ElegantRL training configuration
    # --------------------------------------------------------
    try:
        erl_config = build_elegantrl_config(
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            signal_array=signal_array,
            state_dim=state_dim,
            action_dim=action_dim,
            train_max_step=train_max_step,
            eval_max_step=eval_max_step,
            run_path=run_path,
        )

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR: {e}")
        print(f"{'=' * 60}")
        print("\nTraining aborted during RL configuration build.")
        print("Check rl_configs.py and config.py for invalid or inconsistent settings.")
        import traceback
        traceback.print_exc()
        return

    # --------------------------------------------------------
    # STEP 5: Train and test RL agent using ElegantRL
    # --------------------------------------------------------
    try:
        train_agent(erl_config)

    except KeyboardInterrupt:
        print(f"\n{'=' * 60}")
        print("TRAINING INTERRUPTED BY USER")
        print(f"{'=' * 60}")
        print("Graceful shutdown requested. Partial results may be saved.")
        return

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR DURING TRAINING: {e}")
        print(f"{'=' * 60}")
        print("\nTraining aborted due to runtime error.")
        print("Check logs and hyperparameters for instability or misconfiguration.")
        import traceback
        traceback.print_exc()
        return

    # --------------------------------------------------------
    # STEP 6: TODO - results
    # --------------------------------------------------------

    print("\n=== Training Session Complete ===\n")


if __name__ == "__main__":
    main()
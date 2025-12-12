"""
Main orchestrator for RL training/testing pipeline.
"""

import config
from data.data_manager import DataManager
from utils.user_input import collect_user_input


def main():
    print("=" * 60)
    print("RL Training Execution")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: Collect user input + create run folder + metadata
    # --------------------------------------------------------
    metadata, run_path = collect_user_input()

    # --------------------------------------------------------
    # STEP 2: Initialize DataManager and process data
    # --------------------------------------------------------
    try:
        # Initialize DataManager with integrated mode (results/{model}/data/)
        manager = DataManager(
            exchange=config.EXCHANGE_NAME,
            trading_pair=config.TRADING_PAIR,
            base_timeframe=config.DATA_TIMEFRAME,
            storage_path=f"{run_path}/data"
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
        print(f"Partial data may be available in: {run_path}/data/")
        import traceback
        traceback.print_exc()
        return

    # --------------------------------------------------------
    # STEP 4: TODO - Initialize RL environment
    # --------------------------------------------------------
    # NOTE: BitcoinTradingEnv takes only 3 arrays (price, tech, signal)
    # turbulence_array is not used by the current environment implementation

    # from bitcoin_env import BitcoinTradingEnv
    # train_env = BitcoinTradingEnv(price_array, tech_array, signal_array, mode="train")
    # test_env = BitcoinTradingEnv(price_array, tech_array, signal_array, mode="test")

    print("\n" + "=" * 60)
    print("DATA READY FOR TRAINING")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Initialize BitcoinTradingEnv with (price_array, tech_array, signal_array)")
    print("  2. Setup RL agent (ElegantRL/FinRL)")
    print("  3. Train agent")
    print("  4. Evaluate and save results")

    # --------------------------------------------------------
    # STEP 5: TODO - Train agent
    # --------------------------------------------------------

    # --------------------------------------------------------
    # STEP 6: TODO - Evaluate and save results
    # --------------------------------------------------------

    print("\n=== Training Session Complete ===\n")


if __name__ == "__main__":
    main()
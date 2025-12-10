"""
Main orchestrator for RL training/testing pipeline.
"""

from data.data_pipeline import download_and_process_data, load_processed_data
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
    # STEP 2: Download and process data
    # --------------------------------------------------------
    try:
        metadata = download_and_process_data(metadata, run_path)
    except RuntimeError as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR: {e}")
        print(f"{'=' * 60}")
        print("\nTraining aborted due to data download failure.")
        print(f"Partial data may be available in: {run_path}/data/")
        return

    # --------------------------------------------------------
    # STEP 3: Load processed data into arrays
    # --------------------------------------------------------
    try:
        price_array, tech_array, turbulence_array, signal_array = load_processed_data(run_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR: {e}")
        print(f"{'=' * 60}")
        print("\nTraining aborted due to data loading failure.")
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
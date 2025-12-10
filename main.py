"""
Main orchestrator for RL training/testing pipeline.
"""

from utils.user_input import collect_user_input


def main():
    print("=" * 60)
    print("RL Training Execution")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: Collect user input + create run folder + metadata
    # --------------------------------------------------------
    metadata, run_path = collect_user_input()


if __name__ == "__main__":
    main()
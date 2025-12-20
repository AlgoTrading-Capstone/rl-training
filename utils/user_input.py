"""
Collect user input for RL training and backtesting runs.
"""

import os
from typing import Optional, Dict, Any
import re
from pathlib import Path
from datetime import datetime
import config
from config import RESULTS_PATH, TRAINING_MACHINE_NAME, TRAIN_TEST_SPLIT
from utils.metadata import load_metadata


def input_model_name():
    """Ask for model name and validate format."""
    while True:
        name = input("Enter model name (contain only: letters, digits and underscore): ").strip()
        if not name:
            print("Model name cannot be empty.")
            continue

        if re.fullmatch(r"[A-Za-z0-9_]+", name):
            return name

        print("Invalid model name. Use only letters, numbers, and underscore.")


def input_description():
    """Ask for an optional short description."""
    return input("Enter short description (optional): ").strip()


def input_date(prompt):
    """Ask for a date in DD-MM-YYYY format and return a datetime object."""
    while True:
        raw = input(f"{prompt} (DD-MM-YYYY): ").strip()
        try:
            return datetime.strptime(raw, "%d-%m-%Y")
        except ValueError:
            print("Invalid date format. Expected DD-MM-YYYY.")


def create_run_folder(model_name):
    """
    Create a run folder based on model name and machine name.

    Raises:
        FileExistsError: If the run folder already exists.
    """
    folder_name = f"{model_name}_{TRAINING_MACHINE_NAME}"
    run_path = os.path.join(RESULTS_PATH, folder_name)

    if os.path.exists(run_path):
        raise FileExistsError(
            f"A run with name '{folder_name}' already exists.\n"
            f"Please choose a different model name."
        )

    os.makedirs(run_path)
    return run_path


def collect_model_and_run_path():
    """
    Ask for model name and create a run folder (fail fast if it already exists).

    Returns:
        model_name: str
        run_path: str
    """
    while True:
        model_name = input_model_name()
        try:
            run_path = create_run_folder(model_name)
            return model_name, run_path
        except FileExistsError as e:
            print(f"\n{e}\n")


def collect_training_date_range():
    """
    Collect and confirm training date range.

    Returns:
        train_start_dt: datetime
        train_end_dt: datetime
    """
    print("\nEnter TRAINING date range.")
    print(f"The environment will automatically split this range into:")
    print(f" - Training: {int(TRAIN_TEST_SPLIT * 100)}%")
    print(f" - Testing : {int((1 - TRAIN_TEST_SPLIT) * 100)}%\n")

    while True:
        start_dt = input_date("Training start date")
        end_dt = input_date("Training end date")

        if start_dt >= end_dt:
            print("Training start date must be earlier than end date.\n")
            continue

        duration_days = (end_dt - start_dt).days
        print(f"\n   Training period:")
        print(f"   Start: {start_dt.strftime('%Y-%m-%d')}")
        print(f"   End:   {end_dt.strftime('%Y-%m-%d')}")
        print(f"   Duration: {duration_days} days (~{duration_days / 30:.1f} months)\n")

        confirm = input("   Is this correct? (yes/no): ").strip().lower()
        if confirm in ("yes", "y"):
            return start_dt, end_dt

        print("\n   Let's try again...\n")


def collect_backtest_date_range(
    train_start_dt: Optional[datetime] = None,
    train_end_dt: Optional[datetime] = None,
) -> tuple[datetime, datetime, bool]:
    """
    Collect and confirm backtest date range.
    Optionally checks overlap with a training range.

    Returns:
        bt_start_dt: datetime
        bt_end_dt: datetime
        overlap: bool
    """
    print("\nEnter BACKTEST date range.")
    print("This range will be used ONLY for backtesting the model.\n")

    while True:
        bt_start_dt = input_date("Backtest start date")
        bt_end_dt = input_date("Backtest end date")

        if bt_start_dt >= bt_end_dt:
            print("Backtest start date must be earlier than end date.\n")
            continue

        overlap = False
        if train_start_dt and train_end_dt:
            overlap = not (bt_end_dt <= train_start_dt or bt_start_dt >= train_end_dt)

            if overlap:
                print("\n[WARNING] Backtest date range OVERLAPS with training period!")
                print(f"   Training: {train_start_dt.strftime('%Y-%m-%d')} → {train_end_dt.strftime('%Y-%m-%d')}")
                print(f"   Backtest: {bt_start_dt.strftime('%Y-%m-%d')} → {bt_end_dt.strftime('%Y-%m-%d')}")
                print("\nRunning backtest on overlapping data may cause data leakage.")

                proceed = input("\nDo you want to continue anyway? (yes/no): ").strip().lower()
                if proceed not in ("yes", "y"):
                    print("\n   Please enter backtest dates again.\n")
                    continue

        duration_days = (bt_end_dt - bt_start_dt).days
        print(f"\n   Backtest period:")
        print(f"   Start: {bt_start_dt.strftime('%Y-%m-%d')}")
        print(f"   End:   {bt_end_dt.strftime('%Y-%m-%d')}")
        print(f"   Duration: {duration_days} days (~{duration_days / 30:.1f} months)\n")

        confirm = input("   Is this correct? (yes/no): ").strip().lower()
        if confirm in ("yes", "y"):
            return bt_start_dt, bt_end_dt, overlap

        print("\n   Let's try again...\n")


def select_existing_model_run() -> tuple[Path, Dict[str, Any]]:
    """
    Let user select an existing trained model run that contains:
    - metadata.json
    - elegantrl/act.pth

    Returns:
        model_run_path: Path
        model_metadata: dict
    """
    valid_runs = []

    for d in Path(RESULTS_PATH).iterdir():
        if not d.is_dir():
            continue

        if (d / "metadata.json").exists() and (d / "elegantrl" / "act.pth").exists():
            valid_runs.append(d)

    if not valid_runs:
        raise RuntimeError(
            "No valid trained models found.\n"
            "Expected each run to contain metadata.json and elegantrl/act.pth."
        )

    print("\nAvailable trained models:")
    for idx, run in enumerate(valid_runs, 1):
        print(f"{idx}. {run}")

    while True:
        choice = input("\nSelect model by number: ").strip()

        if not choice.isdigit() or not (1 <= int(choice) <= len(valid_runs)):
            print("Invalid selection. Please enter a valid number.")
            continue

        selected_run = valid_runs[int(choice) - 1]
        run_path = selected_run

        metadata = load_metadata(run_path)

        if "training" not in metadata:
            raise ValueError("Selected model metadata does not contain training information.")

        return run_path, metadata


def collect_run_mode() -> str:
    """
    Ask user which execution mode to run.

    Returns:
        run_mode (str)
    """
    print("Select execution mode:")
    print("1 - Train new model and run backtest automatically")
    print("2 - Train new model only")
    print("3 - Run backtest on existing model\n")

    valid_choices = {
        "1": "TRAIN_AND_BACKTEST",
        "2": "TRAIN_ONLY",
        "3": "BACKTEST_ONLY",
    }

    while True:
        choice = input("Enter choice (1/2/3): ").strip()
        if choice in valid_choices:
            return valid_choices[choice]

        print("Invalid selection. Please enter 1, 2, or 3.\n")


def collect_train_and_backtest_input() -> tuple[Dict[str, Any], Dict[str, Any], Path]:
    """Collect metadata for RL training followed by backtesting."""
    model_name, run_path = collect_model_and_run_path()
    description = input_description()

    train_start_dt, train_end_dt = collect_training_date_range()
    bt_start_dt, bt_end_dt, overlap = collect_backtest_date_range(
        train_start_dt=train_start_dt,
        train_end_dt=train_end_dt
    )

    metadata = {
        "model_name": model_name,
        "machine_name": TRAINING_MACHINE_NAME,
        "description": description,
        "created_at": datetime.utcnow().isoformat(),
        "results_path": run_path,
        "training": {
            "start_date": train_start_dt.strftime("%d-%m-%Y"),
            "end_date": train_end_dt.strftime("%d-%m-%Y"),
            "train_test_split": TRAIN_TEST_SPLIT,
        },
    }

    backtest_config = {
        "start_date": bt_start_dt.strftime("%d-%m-%Y"),
        "end_date": bt_end_dt.strftime("%d-%m-%Y"),
        "overlaps_training": overlap,
    }

    return metadata, backtest_config, run_path


def collect_train_only_input() -> tuple[Dict[str, Any], Path]:
    """Collect metadata for RL training only."""
    model_name, run_path = collect_model_and_run_path()
    description = input_description()
    train_start_dt, train_end_dt = collect_training_date_range()

    metadata = {
        "model_name": model_name,
        "machine_name": TRAINING_MACHINE_NAME,
        "description": description,
        "created_at": datetime.utcnow().isoformat(),
        "results_path": run_path,
        "training": {
            "start_date": train_start_dt.strftime("%d-%m-%Y"),
            "end_date": train_end_dt.strftime("%d-%m-%Y"),
            "train_test_split": TRAIN_TEST_SPLIT,
        },
    }

    return metadata, run_path


def collect_backtest_only_input() -> tuple[Dict[str, Any], Path]:
    """Collect metadata for backtesting an existing trained model."""
    run_path, model_metadata = select_existing_model_run()

    train_meta = model_metadata["training"]
    train_start_dt = datetime.strptime(train_meta["start_date"], "%d-%m-%Y")
    train_end_dt = datetime.strptime(train_meta["end_date"], "%d-%m-%Y")

    bt_start_dt, bt_end_dt, overlap = collect_backtest_date_range(
        train_start_dt=train_start_dt,
        train_end_dt=train_end_dt
    )

    backtest_config = {
        "start_date": bt_start_dt.strftime("%d-%m-%Y"),
        "end_date": bt_end_dt.strftime("%d-%m-%Y"),
        "overlaps_training": overlap,
    }

    return backtest_config, run_path
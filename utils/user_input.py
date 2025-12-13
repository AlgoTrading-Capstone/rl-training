"""
Collect user input for RL training run metadata.
"""

import re
import os
import json
from datetime import datetime
from config import RESULTS_PATH, TRAINING_MACHINE_NAME, TRAIN_TEST_SPLIT


def input_model_name():
    """Ask for model name and validate format."""
    while True:
        name = input("Enter model name (contain only: letters, digits and underscore): ").strip()
        if not name:
            print("Model name cannot be empty.")
            continue

        # Allowed: letters, digits, underscore
        if re.fullmatch(r"[A-Za-z0-9_]+", name):
            return name

        print("Invalid model name. Use only letters, numbers, and underscore.")


def input_description():
    """Optional description."""
    return input("Enter short description (optional): ").strip()


def input_date(prompt):
    """
    Ask for a date with DD-MM-YYYY format, validate, and return datetime object.
    """
    while True:
        raw = input(f"{prompt} (DD-MM-YYYY): ").strip()
        try:
            return datetime.strptime(raw, "%d-%m-%Y")
        except ValueError:
            print("Invalid date format. Expected DD-MM-YYYY.")


def create_run_folder(model_name):
    """
    Creates a run folder based on model name + machine name.
    Returns the path.
    """
    folder_name = f"{model_name}_{TRAINING_MACHINE_NAME}"
    run_path = os.path.join(RESULTS_PATH, folder_name)
    os.makedirs(run_path, exist_ok=True)
    return run_path


def collect_user_input():
    """
    Collect metadata for this RL training run.
    Returns:
        metadata: dict
        run_path: str
    """

    model_name = input_model_name()
    description = input_description()

    print("\nEnter full session date range.")
    print("This range will be used for BOTH training and testing.")
    print(f"The environment will automatically split the range into:")
    print(f" - Training: {int(TRAIN_TEST_SPLIT * 100)}%")
    print(f" - Testing : {int((1 - TRAIN_TEST_SPLIT) * 100)}%\n")

    while True:
        start_dt = input_date("Start date")
        end_dt = input_date("End date")

        # Validate chronological order
        if start_dt >= end_dt:
            print("Start date must be earlier than end date. Please try again.\n")
            continue

        # Show parsed dates for confirmation
        print(f"\n   Parsed dates:")
        print(f"   Start: {start_dt.strftime('%B %d, %Y')} ({start_dt.strftime('%Y-%m-%d')})")
        print(f"   End:   {end_dt.strftime('%B %d, %Y')} ({end_dt.strftime('%Y-%m-%d')})")

        # Calculate duration
        duration_days = (end_dt - start_dt).days
        print(f"   Duration: {duration_days} days (~{duration_days/30:.1f} months)\n")

        # Confirm with user
        confirm = input("   Is this correct? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            break
        else:
            print("\n   Let's try again...\n")

    # Format dates for metadata
    start_str = start_dt.strftime("%d-%m-%Y")
    end_str = end_dt.strftime("%d-%m-%Y")

    # Create folder
    run_path = create_run_folder(model_name)

    # Build metadata object
    metadata = {
        "model_name": model_name,
        "machine_name": TRAINING_MACHINE_NAME,
        "description": description,
        "start_date": start_str,
        "end_date": end_str,
        "created_at": datetime.utcnow().isoformat(),
        "results_path": run_path,
        "data_base_path": os.path.join(run_path, "data"),
        "data_download_status": "pending",
    }

    # Save metadata.json
    metadata_file = os.path.join(run_path, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nRun folder created: {run_path}")
    print(f"Metadata saved to: {metadata_file}\n")

    return metadata, run_path
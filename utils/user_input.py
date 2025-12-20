"""
Collect user input for RL training and backtesting runs.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config import RESULTS_PATH, TRAINING_MACHINE_NAME, TRAIN_TEST_SPLIT
from utils.metadata import load_metadata

# Initialize rich console for styled output
console = Console()


def styled_header(text: str) -> None:
    """Display styled header matching logger.py format."""
    console.print(Panel(
        Text(text, style="bold cyan"),
        border_style="cyan",
        padding=(0, 2),
        expand=False
    ))


def styled_prompt(prompt_text: str, style: str = "yellow") -> str:
    """Get user input with styled prompt."""
    return console.input(f"[{style}]{prompt_text}[/{style}] ")


def styled_info(text: str) -> None:
    """Display info message matching logger.py INFO level."""
    console.print(f"[cyan]ℹ[/cyan] {text}")


def styled_warning(text: str) -> None:
    """Display warning message matching logger.py WARNING level."""
    console.print(f"[yellow]⚠[/yellow] {text}")


def styled_error(text: str) -> None:
    """Display error message matching logger.py ERROR level."""
    console.print(f"[red]✗[/red] {text}")


def styled_success(text: str) -> None:
    """Display success message matching logger.py SUCCESS level."""
    console.print(f"[green]✓[/green] {text}")


def input_model_name() -> str:
    """Ask for model name and validate format."""
    while True:
        name = styled_prompt("Enter model name (letters, digits, underscore only):")
        if not name:
            styled_error("Model name cannot be empty.")
            continue

        if re.fullmatch(r"[A-Za-z0-9_]+", name):
            return name

        styled_error("Invalid model name. Use only letters, numbers, and underscore.")
    raise RuntimeError("Should be unreachable")

def input_description() -> str:
    """Ask for an optional short description."""
    return styled_prompt("Enter short description (optional):", style="cyan")


def input_date(prompt: str) -> datetime:
    """Ask for a date in DD-MM-YYYY format and return a datetime object."""
    while True:
        raw = styled_prompt(f"{prompt} (DD-MM-YYYY):")
        try:
            return datetime.strptime(raw, "%d-%m-%Y")
        except ValueError:
            styled_error("Invalid date format. Expected DD-MM-YYYY.")


def create_run_folder(model_name: str) -> Path:
    """
    Create a run folder based on model name and machine name.

    Raises:
        FileExistsError: If the run folder already exists.
    """
    folder_name = f"{model_name}_{TRAINING_MACHINE_NAME}"
    # Using pathlib for consistency
    run_path = Path(RESULTS_PATH) / folder_name

    if run_path.exists():
        raise FileExistsError(
            f"A run with name '{folder_name}' already exists.\n"
            f"Please choose a different model name."
        )

    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def collect_model_and_run_path() -> Tuple[str, Path]:
    """
    Ask for model name and create a run folder (fail fast if it already exists).

    Returns:
        model_name: str
        run_path: Path
    """
    while True:
        model_name = input_model_name()
        try:
            run_path = create_run_folder(model_name)
            return model_name, run_path
        except FileExistsError as e:
            console.print(f"\n[red]{e}[/red]\n")


def collect_training_date_range() -> tuple[datetime, datetime]:
    """
    Collect and confirm training date range.

    Returns:
        train_start_dt: datetime
        train_end_dt: datetime
    """
    console.print()
    styled_header("Enter TRAINING Date Range")
    styled_info(f"The environment will automatically split this range into:")
    console.print(f"  [cyan]•[/cyan] Training: [bold]{int(TRAIN_TEST_SPLIT * 100)}%[/bold]")
    console.print(f"  [cyan]•[/cyan] Testing:  [bold]{int((1 - TRAIN_TEST_SPLIT) * 100)}%[/bold]")
    console.print()

    while True:
        start_dt = input_date("Training start date")
        end_dt = input_date("Training end date")

        if start_dt >= end_dt:
            styled_error("Training start date must be earlier than end date.")
            console.print()
            continue

        duration_days = (end_dt - start_dt).days
        console.print(f"\n[cyan]Training period:[/cyan]")
        console.print(f"  Start:    {start_dt.strftime('%Y-%m-%d')}")
        console.print(f"  End:      {end_dt.strftime('%Y-%m-%d')}")
        console.print(f"  Duration: {duration_days} days (~{duration_days / 30:.1f} months)\n")

        confirm = styled_prompt("Is this correct? (yes/no):", style="green")
        if confirm.lower() in ("yes", "y"):
            return start_dt, end_dt

        styled_info("Let's try again...")
        console.print()

    raise RuntimeError("Should be unreachable")

def collect_backtest_date_range(
    train_start_dt: Optional[datetime] = None,
    train_end_dt: Optional[datetime] = None,
) -> Tuple[datetime, datetime, bool]:
    """
    Collect and confirm backtest date range.
    Optionally checks overlap with a training range.

    Returns:
        bt_start_dt: datetime
        bt_end_dt: datetime
        overlap: bool
    """
    console.print()
    styled_header("Enter BACKTEST Date Range")
    styled_info("This range will be used ONLY for backtesting the model.")
    console.print()

    while True:
        bt_start_dt = input_date("Backtest start date")
        bt_end_dt = input_date("Backtest end date")

        if bt_start_dt >= bt_end_dt:
            styled_error("Backtest start date must be earlier than end date.")
            console.print()
            continue

        overlap = False
        if train_start_dt and train_end_dt:
            overlap = not (bt_end_dt <= train_start_dt or bt_start_dt >= train_end_dt)

            if overlap:
                console.print()
                styled_warning("[WARNING] Backtest date range OVERLAPS with training period!")
                console.print(f"  Training: {train_start_dt.strftime('%Y-%m-%d')} → {train_end_dt.strftime('%Y-%m-%d')}")
                console.print(f"  Backtest: {bt_start_dt.strftime('%Y-%m-%d')} → {bt_end_dt.strftime('%Y-%m-%d')}")
                console.print("\n[yellow]⚠[/yellow] Running backtest on overlapping data may cause data leakage.\n")

                proceed = styled_prompt("Do you want to continue anyway? (yes/no):", style="yellow")
                if proceed.lower() not in ("yes", "y"):
                    styled_info("Please enter backtest dates again.")
                    console.print()
                    continue

        duration_days = (bt_end_dt - bt_start_dt).days
        console.print(f"\n[cyan]Backtest period:[/cyan]")
        console.print(f"  Start:    {bt_start_dt.strftime('%Y-%m-%d')}")
        console.print(f"  End:      {bt_end_dt.strftime('%Y-%m-%d')}")
        console.print(f"  Duration: {duration_days} days (~{duration_days / 30:.1f} months)\n")

        confirm = styled_prompt("Is this correct? (yes/no):", style="green")
        if confirm.lower() in ("yes", "y"):
            return bt_start_dt, bt_end_dt, overlap

        styled_info("Let's try again...")
        console.print()

    raise RuntimeError("Should be unreachable")


def select_existing_model_run() -> Tuple[Path, Dict[str, Any]]:
    """
    Let user select an existing trained model run that contains:
    - metadata.json
    - elegantrl/act.pth

    Returns:
        model_run_path: Path
        model_metadata: dict
    """
    valid_runs = []

    results_dir = Path(RESULTS_PATH)
    if not results_dir.exists():
        styled_error(f"Results directory not found: {RESULTS_PATH}")
        raise FileNotFoundError(f"Results directory not found: {RESULTS_PATH}")

    for d in results_dir.iterdir():
        if not d.is_dir():
            continue

        if (d / "metadata.json").exists() and (d / "elegantrl" / "act.pth").exists():
            valid_runs.append(d)

    if not valid_runs:
        raise RuntimeError(
            "No valid trained models found.\n"
            "Expected each run to contain metadata.json and elegantrl/act.pth."
        )

    console.print()
    styled_header("Available Trained Models")
    for idx, run in enumerate(valid_runs, 1):
        console.print(f"  [cyan]{idx}.[/cyan] {run.name}")

    while True:
        console.print()
        choice = styled_prompt("Select model by number:")

        if not choice.isdigit() or not (1 <= int(choice) <= len(valid_runs)):
            styled_error("Invalid selection. Please enter a valid number.")
            continue

        selected_run = valid_runs[int(choice) - 1]
        run_path = selected_run

        metadata = load_metadata(run_path)

        if "training" not in metadata:
            raise ValueError("Selected model metadata does not contain training information.")

        return run_path, metadata

    raise RuntimeError("Should be unreachable")


def collect_run_mode() -> str:
    """
    Ask user which execution mode to run.

    Returns:
        run_mode (str)
    """
    console.print()
    styled_header("Select Execution Mode")
    console.print("  [cyan]1[/cyan] - Train new model and run backtest automatically")
    console.print("  [cyan]2[/cyan] - Train new model only")
    console.print("  [cyan]3[/cyan] - Run backtest on existing model")
    console.print()

    valid_choices = {
        "1": "TRAIN_AND_BACKTEST",
        "2": "TRAIN_ONLY",
        "3": "BACKTEST_ONLY",
    }

    while True:
        choice = styled_prompt("Enter choice (1/2/3):")
        if choice in valid_choices:
            return valid_choices[choice]

        styled_error("Invalid selection. Please enter 1, 2, or 3.")
        console.print()


def _build_base_metadata(
    model_name: str,
    description: str,
    run_path: Path,
    train_start: datetime,
    train_end: datetime
) -> Dict[str, Any]:
    """
    Helper function to construct the common metadata dictionary.
    This prevents code duplication between 'train_only' and 'train_and_backtest' modes.
    """
    # Use timezone-aware UTC time to fix the warning
    current_time = datetime.now(timezone.utc).isoformat()

    return {
        "model_name": model_name,
        "machine_name": TRAINING_MACHINE_NAME,
        "description": description,
        "created_at": current_time,
        "results_path": str(run_path),
        "training": {
            "start_date": train_start.strftime("%d-%m-%Y"),
            "end_date": train_end.strftime("%d-%m-%Y"),
            "train_test_split": TRAIN_TEST_SPLIT,
        },
    }


def collect_train_and_backtest_input() -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    """Collect metadata for RL training followed by backtesting."""
    model_name, run_path = collect_model_and_run_path()
    description = input_description()

    train_start_dt, train_end_dt = collect_training_date_range()
    bt_start_dt, bt_end_dt, overlap = collect_backtest_date_range(
        train_start_dt=train_start_dt,
        train_end_dt=train_end_dt
    )

    metadata = _build_base_metadata(
        model_name, description, run_path, train_start_dt, train_end_dt
    )
    metadata["run_mode"] = "TRAIN_AND_BACKTEST"

    backtest_config = {
        "start_date": bt_start_dt.strftime("%d-%m-%Y"),
        "end_date": bt_end_dt.strftime("%d-%m-%Y"),
        "overlaps_training": overlap,
    }

    return metadata, backtest_config, run_path


def collect_train_only_input() -> Tuple[Dict[str, Any], Path]:
    """Collect metadata for RL training only."""
    model_name, run_path = collect_model_and_run_path()
    description = input_description()
    train_start_dt, train_end_dt = collect_training_date_range()

    metadata = _build_base_metadata(
        model_name, description, run_path, train_start_dt, train_end_dt
    )
    metadata["run_mode"] = "TRAIN_ONLY"

    return metadata, run_path


def collect_backtest_only_input() -> Tuple[Dict[str, Any], Path]:
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
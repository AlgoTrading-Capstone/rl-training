"""
Run artifact helpers — extracted from main.py for cleanliness.

Provides:
- ensure_feature_cache:        auto-build processed parquet if missing (prompts user)
- finalize_training_signal_artifacts:  merge per-worker signal CSVs into one file
- log_run_artifact_summary:    print all run outputs with paths
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.load_features import _get_processed_path
from utils.logger import RLLogger


def ensure_feature_cache(
    manager,
    logger: RLLogger,
    metadata: dict,
    strategy_list: list[str] | None = None,
) -> None:
    """Ensure the processed parquet cache exists, prompting the user if missing."""
    import sys
    from loguru import logger as _loguru
    from utils.user_input import styled_prompt, styled_info, console
    from rich.panel import Panel
    from rich.text import Text
    from rich import box

    # Flush loguru's enqueued background messages before showing an interactive
    # prompt.  Without this, enqueued log lines (e.g. training-config summary)
    # race with the Rich console and appear mid-prompt.
    _loguru.complete()

    cache_path = _get_processed_path()

    if cache_path.exists():
        logger.info(f"Feature cache found: {cache_path}")
        return

    # Cache missing — show warning and prompt user
    warning_text = Text()
    warning_text.append("[ALERT] ", style="bold bright_green")
    warning_text.append("FEATURE CACHE NOT FOUND\n\n", style="bright_green")
    warning_text.append(f"Expected: {cache_path}\n\n", style="bold white")
    warning_text.append(
        "First-time data processing can take 6+ hours.\n"
        "To pre-generate the cache (recommended), run:\n\n",
        style="bright_green",
    )
    warning_text.append(
        "  python -m data.precompute_features\n\n",
        style="bold white",
    )
    warning_text.append(
        "If you continue now, processing will run inline before training starts.",
        style="bright_green",
    )

    console.print(Panel(
        warning_text,
        border_style="bright_green",
        padding=(1, 2),
        title="[bright_green]CACHE MISSING[/bright_green]",
        title_align="left",
        box=box.HEAVY,
    ))
    console.print()

    console.print(
        "  [bright_green]>>[/bright_green] [bold white]y[/bold white] — process data now and continue to training (slow on first run)\n"
        "  [bright_green]>>[/bright_green] [bold white]n[/bold white] — abort and let me run precompute_features first (recommended)"
    )
    console.print()

    choice = styled_prompt("Your choice (y/n):")
    if choice.lower() not in ("y", "yes"):
        styled_info("Aborting. Run:  python -m data.precompute_features")
        sys.exit(0)

    # User chose to proceed — build cache inline
    from datetime import date as _date
    today = _date.today().strftime("%d-%m-%Y")
    logger.info("Building feature cache inline (this may take a while)...")
    manager.get_processed_data(
        start_date="01-01-2017",
        end_date=today,
        strategy_list=strategy_list,
    )
    logger.info(f"Feature cache built: {cache_path}")


def finalize_training_signal_artifacts(
    run_path: Path, logger: RLLogger
) -> Path | None:
    """Merge per-worker strategy-signal CSVs from a training run into one file."""
    worker_logs = sorted(run_path.glob("strategy_signals_train_worker_*.csv"))
    if not worker_logs:
        logger.warning(
            f"No training worker signal logs found in {run_path.resolve()} "
            f"(expected pattern: strategy_signals_train_worker_*.csv)"
        )
        return None

    frames = []
    for worker_log in worker_logs:
        try:
            frames.append(pd.read_csv(worker_log))
        except Exception as exc:
            logger.warning(f"Skipping unreadable worker signal log {worker_log.resolve()}: {exc}")

    if not frames:
        logger.warning("Training worker signal logs were found, but none could be read")
        return None

    merged_df = pd.concat(frames, ignore_index=True)
    if "timestamp" in merged_df.columns:
        merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"], errors="coerce")
    if "step" in merged_df.columns:
        merged_df["step"] = pd.to_numeric(merged_df["step"], errors="coerce")
    if "worker_id" in merged_df.columns:
        merged_df["worker_id"] = merged_df["worker_id"].astype(str)

    sort_cols = [col for col in ("timestamp", "step", "worker_id") if col in merged_df.columns]
    if sort_cols:
        merged_df = merged_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    merged_csv = run_path / "strategy_signals_train_log.csv"
    merged_df.to_csv(merged_csv, index=False)
    logger.info(f"Merged training signal CSV: {merged_csv.resolve()}")
    return merged_csv


def log_run_artifact_summary(
    run_path: Path,
    logger: RLLogger,
    parquet_cache: Path | None = None,
) -> None:
    """Log the most useful run outputs with direct paths for quick access."""
    artifact_candidates: list[tuple[str, Path | None]] = []

    if parquet_cache is not None:
        artifact_candidates.append(("Processed feature cache", parquet_cache))

    artifact_candidates.extend([
        ("Run directory", run_path),
        ("Training signal CSV", run_path / "strategy_signals_train_log.csv"),
        ("Eval signal CSV", run_path / "strategy_signals_log.csv"),
        ("Learning curve", run_path / "elegantrl" / "LearningCurve.jpg"),
    ])

    backtests_dir = run_path / "backtests"
    if backtests_dir.exists():
        latest_backtest = max(
            (path for path in backtests_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            default=None,
        )
        if latest_backtest is not None:
            artifact_candidates.extend([
                ("Latest backtest", latest_backtest),
                ("Backtest benchmark plot", latest_backtest / "plots" / "agent_vs_btc_benchmark.png"),
                ("Backtest trades CSV", latest_backtest / "trades.csv"),
                ("Backtest steps CSV", latest_backtest / "steps.csv"),
            ])

    existing = [(label, path.resolve()) for label, path in artifact_candidates if path is not None and path.exists()]
    if not existing:
        return

    logger.info("=== Run Artifacts ===")
    for label, path in existing:
        logger.info(f"{label}: {path}")
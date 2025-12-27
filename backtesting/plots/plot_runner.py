"""
Backtest plot runner.

Responsibilities:
- Load backtest artifacts (CSV / JSON)
- Prepare shared data for plotting
- Dispatch plot generation to individual plot modules
- Persist all plots as PNG files under out_dir / plots

This module contains NO plotting logic itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import json
import pandas as pd
from datetime import datetime

from backtesting.plots.benchmark_equity_plot import plot_agent_vs_btc_benchmark


# ============================================================
# Public API
# ============================================================

def generate_backtest_plots(
    *,
    out_dir: Path,
    model_metadata: Dict[str, Any],
    backtest_config: Dict[str, Any],
) -> None:
    """
    Generate all backtest plots for a single backtest run.

    Parameters
    ----------
    out_dir:
        Backtest output directory containing:
        - steps.csv
        - metrics.json
        - summary.json
        - trades.csv (optional)

    model_metadata:
        Full model metadata dict (used for titles, annotations, timeframe, etc.)

    Notes
    -----
    - All plots are saved as PNG under out_dir / "plots"
    - Fail-fast on missing critical artifacts
    - Fail-soft on individual plot failures
    """

    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # STEP 1: Resolve artifact paths
    # --------------------------------------------------------
    steps_path = out_dir / "steps.csv"
    metrics_path = out_dir / "metrics.json"
    summary_path = out_dir / "summary.json"
    trades_path = out_dir / "trades.csv"

    # --------------------------------------------------------
    # STEP 2: Validate required artifacts
    # --------------------------------------------------------
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing required file: {steps_path}")

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing required file: {metrics_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing required file: {summary_path}")

    # --------------------------------------------------------
    # STEP 3: Load artifacts
    # --------------------------------------------------------
    steps_df = _load_steps(steps_path)
    metrics = _load_json(metrics_path)
    summary = _load_json(summary_path)

    trades_df: Optional[pd.DataFrame] = None
    if trades_path.exists():
        trades_df = pd.read_csv(trades_path)

    # --------------------------------------------------------
    # STEP 4: Prepare shared data
    # --------------------------------------------------------
    _prepare_steps_df(steps_df)
    subtitle = _build_common_subtitle(model_metadata=model_metadata, backtest_config=backtest_config)

    # --------------------------------------------------------
    # STEP 5: Dispatch plots
    # --------------------------------------------------------
    _run_plots(
        steps_df=steps_df,
        trades_df=trades_df,
        metrics=metrics,
        summary=summary,
        model_metadata=model_metadata,
        plots_dir=plots_dir,
        subtitle=subtitle,
    )


# ============================================================
# Internal helpers
# ============================================================

def _load_steps(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("steps.csv is empty; cannot generate plots.")

    if "timestamp" not in df.columns:
        raise ValueError("steps.csv missing required column: timestamp")

    return df


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_steps_df(steps_df: pd.DataFrame) -> None:
    """
    Prepare steps_df in-place for plotting.

    Actions:
    - Convert timestamp to datetime
    - Sort by time
    - Enforce numeric dtypes where required
    """

    steps_df["timestamp"] = pd.to_datetime(steps_df["timestamp"], errors="coerce")
    steps_df.sort_values("timestamp", inplace=True)
    steps_df.reset_index(drop=True, inplace=True)


def _build_common_subtitle(
    *,
    model_metadata: Dict[str, Any],
    backtest_config: Dict[str, Any],
) -> str:
    """
    Build a unified subtitle string for all backtest plots.
    """

    # Metadata date format: DD-MM-YYYY
    DATE_FMT_META = "%d-%m-%Y"
    # Display format: DD/MM/YY
    DATE_FMT_DISPLAY = "%d/%m/%y"

    start_dt = datetime.strptime(backtest_config["start_date"], DATE_FMT_META)
    end_dt = datetime.strptime(backtest_config["end_date"], DATE_FMT_META)

    # Calculate number of months between start and end dates
    n_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)

    agent_name = model_metadata.get("model_name", "Agent")

    return (
        f"Backtest ID: {backtest_config.get('id', 'N/A')} | "
        f"Date Range: {start_dt.strftime(DATE_FMT_DISPLAY)} - {end_dt.strftime(DATE_FMT_DISPLAY)} ({n_months} Months) | "
        f"Agent Name: {agent_name} |"
    )


def _run_plots(
    *,
    steps_df: pd.DataFrame,
    trades_df: Optional[pd.DataFrame],
    metrics: Dict[str, Any],
    summary: Dict[str, Any],
    model_metadata: Dict[str, Any],
    plots_dir: Path,
    subtitle: str,
) -> None:
    """
    Invoke all plot modules.

    Each plot is isolated so a failure in one plot
    does not prevent others from being generated.
    """

    _safe_plot(
        plot_agent_vs_btc_benchmark,
        steps_df=steps_df,
        plots_dir=plots_dir,
        summary=summary,
        subtitle=subtitle,
    )


def _safe_plot(plot_func, **kwargs) -> None:
    """
    Execute a plot function safely.

    Any exception is swallowed to allow other plots to continue.
    """
    try:
        plot_func(**kwargs)
    except Exception:
        # Intentionally silent – plotting must never break the pipeline
        pass
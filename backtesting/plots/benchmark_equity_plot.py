"""
Benchmark equity comparison plot.

Compares:
- Agent equity curve
- BTC Buy & Hold equity curve

Both are normalized to the same initial equity.
Includes LONG / SHORT regime shading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates


# ============================================================
# Public API
# ============================================================

def plot_agent_vs_btc_benchmark(
    *,
    steps_df: pd.DataFrame,
    plots_dir: Path,
    summary: Dict[str, Any],
    subtitle: str,
) -> None:
    """
    Plot Agent equity vs BTC Buy & Hold equity.

    Output:
        plots/agent_vs_btc_equity.png
    """

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------
    required_cols = {"timestamp", "equity", "holdings", "close"}
    missing = required_cols - set(steps_df.columns)
    if missing:
        raise ValueError(f"steps_df missing required columns: {missing}")

    # --------------------------------------------------------
    # Extract data
    # --------------------------------------------------------
    time = pd.to_datetime(steps_df["timestamp"])
    agent_equity = steps_df["equity"].astype(float)
    price = steps_df["close"].astype(float)
    holdings = steps_df["holdings"].astype(float)

    initial_equity = float(summary["initial_equity"])
    initial_price = float(price.iloc[0])

    btc_equity = initial_equity * (price / initial_price)

    # --------------------------------------------------------
    # Performance numbers (%)
    # --------------------------------------------------------
    agent_return_pct = (agent_equity.iloc[-1] / initial_equity - 1.0) * 100.0
    btc_return_pct = (btc_equity.iloc[-1] / initial_equity - 1.0) * 100.0

    perf_string = f"Performance: Agent: {agent_return_pct:+.2f}% vs Buy & Hold: {btc_return_pct:+.2f}%"

    # --------------------------------------------------------
    # Extract LONG / SHORT segments
    # --------------------------------------------------------
    segments = _extract_position_segments(
        timestamps=time,
        holdings=holdings,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6)) #layout="constrained"

    agent_color = "#1f77b4"
    btc_color = "#555555"

    agent_line, = ax.plot(
        time,
        agent_equity,
        color=agent_color,
        linewidth=1.0,
        label="Agent",
        zorder=3,
    )

    btc_line, = ax.plot(
        time,
        btc_equity,
        color=btc_color,
        linewidth=1.0,
        linestyle="-",
        label="Buy & Hold",
        zorder=2,
    )

    # Regime shading
    for seg in segments:
        color = "green" if seg["side"] == "LONG" else "red"
        ax.axvspan(seg["start"], seg["end"], color=color, alpha=0.12, zorder=1)

    # Title & subtitle
    title = "Agent vs Buy & Hold Performance"
    full_subtitle = f"{subtitle} {perf_string}"
    fig.suptitle(title, fontsize=14)  # Main title
    ax.set_title(full_subtitle, fontsize=9)  # Subtitle with performance

    # Labels
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")

    # Grid
    ax.grid(True, alpha=0.3)

    # Smart date axis (dynamic)
    date_range_days = (time.iloc[-1] - time.iloc[0]).days

    if date_range_days <= 2:
        # Very short range (Intraday): Show hours and minutes
        locator = mdates.HourLocator(interval=4)
        formatter = mdates.DateFormatter("%H:%M")

    elif date_range_days <= 14:
        # Short range (Up to 2 weeks): Show every day (Day/Month)
        locator = mdates.DayLocator(interval=1)
        formatter = mdates.DateFormatter("%d/%m")

    elif date_range_days <= 31:
        # Medium range (Up to 1 month): Show every 3 days
        locator = mdates.DayLocator(interval=3)
        formatter = mdates.DateFormatter("%d/%m")

    elif date_range_days <= 120:
        # Standard range (1-4 months): Show start of each month
        locator = mdates.MonthLocator()
        formatter = mdates.DateFormatter("%d/%m")

    elif date_range_days <= 365 * 2:
        # Long range (4 months to 2 years): Every 2 months with year
        locator = mdates.MonthLocator(interval=2)
        formatter = mdates.DateFormatter("%m/%y")

    else:
        # Very long range: Show only years
        locator = mdates.YearLocator()
        formatter = mdates.DateFormatter("%Y")

    # Apply the logic to the axis
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Ensure date labels are rotated for better legibility
    fig.autofmt_xdate()

    # Legend
    long_patch = mpatches.Patch(color="green", alpha=0.12, label="Agent is Long")
    short_patch = mpatches.Patch(color="red", alpha=0.12, label="Agent is Short")
    ax.legend(handles=[agent_line, btc_line, long_patch, short_patch], loc="upper left")

    # Save
    out_path = plots_dir / "agent_vs_btc_benchmark.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ============================================================
# Helpers
# ============================================================

def _extract_position_segments(
    *,
    timestamps: pd.Series,
    holdings: pd.Series,
) -> List[Dict[str, Any]]:
    """
    Detect contiguous LONG / SHORT segments.
    """

    segments: List[Dict[str, Any]] = []

    current_side = None
    start_time = None

    for ts, h in zip(timestamps, holdings):
        side = "LONG" if h > 0 else "SHORT" if h < 0 else None

        if side != current_side:
            if current_side is not None:
                segments.append({"start": start_time, "end": ts, "side": current_side})
            current_side = side
            start_time = ts if side is not None else None

    if current_side is not None:
        segments.append({"start": start_time, "end": timestamps.iloc[-1], "side": current_side})

    return segments
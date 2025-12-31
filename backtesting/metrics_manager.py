"""
Backtest metrics manager.

Computes post-run metrics from backtest CSV artifacts:
- steps.csv
- trades.csv
- summary.json

Writes metrics.json into the same output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json
import math
import numpy as np
import pandas as pd

import config


# ============================================================
# Constants
# ============================================================

TIMEFRAME_TO_STEPS_PER_YEAR = {
    "1m": 60 * 24 * 365,
    "5m": 12 * 24 * 365,
    "15m": 4 * 24 * 365,
    "1h": 24 * 365,
    "4h": 6 * 365,
    "1d": 365,
}


# ============================================================
# Public API
# ============================================================

def compute_and_write_metrics(
    *,
    out_dir: Path,
    model_metadata: Dict[str, Any],
    backtest_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute all backtest metrics and write metrics.json.

    Returns:
        metrics dict
    """

    out_dir = Path(out_dir)

    steps_df = _load_steps(out_dir)
    trades_df = _load_trades(out_dir)
    summary = _load_summary(out_dir)

    timeframe = model_metadata["data"]["timeframe"]
    steps_per_year = _resolve_steps_per_year(timeframe)

    metrics: Dict[str, Any] = {}

    # --------------------------------------------------------
    # Run metadata
    # --------------------------------------------------------
    metrics["run"] = _compute_run_metadata(
        steps_df=steps_df,
        model_metadata=model_metadata,
        summary=summary,
        backtest_config=backtest_config,
    )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------
    metrics["performance"] = _compute_performance_metrics(
        steps_df=steps_df,
        summary=summary,
        steps_per_year=steps_per_year,
    )

    # --------------------------------------------------------
    # Risk & behavior
    # --------------------------------------------------------
    metrics["risk_behavior"] = _compute_risk_behavior_metrics(
        steps_df=steps_df,
        trades_df=trades_df,
    )

    # --------------------------------------------------------
    # Trade-level
    # --------------------------------------------------------
    metrics["trades"] = _compute_trade_metrics(trades_df)

    # --------------------------------------------------------
    # Reward diagnostics
    # --------------------------------------------------------
    metrics["reward"] = _compute_reward_metrics(steps_df)

    # --------------------------------------------------------
    # Data integrity
    # --------------------------------------------------------
    metrics["data_quality"] = _compute_data_quality_metrics(
        steps_df=steps_df,
        summary=summary,
    )

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------
    _write_json(out_dir / "metrics.json", metrics)

    return metrics


# ============================================================
# Loaders
# ============================================================

def _load_steps(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "steps.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing steps.csv: {path}")
    return pd.read_csv(path)


def _load_trades(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "trades.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_summary(out_dir: Path) -> Dict[str, Any]:
    path = out_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary.json: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Metric groups
# ============================================================

def _compute_run_metadata(*, steps_df, model_metadata, summary, backtest_config):
    """
    Compute basic run metadata from model metadata and steps.csv.

    Returns:
        dict with run-level descriptive fields.
    """

    if steps_df.empty:
        raise ValueError("steps.csv is empty; cannot compute run metadata.")

    run_id = backtest_config.get("id")
    timeframe = model_metadata["data"]["timeframe"]
    initial_equity = float(summary["initial_equity"])

    run_metadata = {
        "run_id": run_id,
        "timeframe": timeframe,

        # Bounds from actual backtest data
        "start_timestamp": steps_df["timestamp"].iloc[0],
        "end_timestamp": steps_df["timestamp"].iloc[-1],

        # Size
        "n_steps": int(len(steps_df)),

        # Capital
        "initial_equity": float(initial_equity),
    }

    return run_metadata


def _compute_performance_metrics(*, steps_df, summary, steps_per_year):
    """
    Compute performance-related metrics from equity curve.

    Metrics:
    - final equity (mtm / realized)
    - returns
    - CAGR
    - annualized volatility
    - Sharpe ratio
    - Sortino ratio
    - max drawdown
    - Calmar ratio
    """

    if steps_df.empty:
        raise ValueError("steps.csv is empty; cannot compute performance metrics.")

    equity = steps_df["equity"].astype(float).values

    if len(equity) < 2:
        raise ValueError("Not enough steps to compute performance metrics.")

    initial_equity = float(summary["initial_equity"])
    final_equity_mtm = float(summary["final_equity_mtm"])
    final_equity_realized = summary.get("final_equity_realized")

    # ----------------------------------------------------
    # Returns (per step)
    # ----------------------------------------------------
    valid_mask = equity[:-1] > 0.0
    returns = np.zeros(len(equity) - 1)

    returns[valid_mask] = (
            np.diff(equity)[valid_mask] / equity[:-1][valid_mask]
    )

    returns = returns[np.isfinite(returns)]

    # ----------------------------------------------------
    # Basic returns
    # ----------------------------------------------------
    return_mtm = final_equity_mtm / initial_equity
    return_realized = (
        None if final_equity_realized is None
        else float(final_equity_realized) / initial_equity
    )

    # ----------------------------------------------------
    # CAGR
    # ----------------------------------------------------
    n_years = len(equity) / float(steps_per_year)
    cagr = (
        (final_equity_mtm / initial_equity) ** (1.0 / n_years) - 1.0
        if n_years > 0 and final_equity_mtm > 0
        else 0.0
    )

    # ----------------------------------------------------
    # Volatility (annualized)
    # ----------------------------------------------------
    vol_annual = np.std(returns) * math.sqrt(steps_per_year) if len(returns) > 0 else 0.0

    # ----------------------------------------------------
    # Sharpe & Sortino (standard, step-based)
    # ----------------------------------------------------
    rf_annual = float(getattr(config, "RISK_FREE_RATE", 0.0))
    rf_step = (1.0 + rf_annual) ** (1.0 / steps_per_year) - 1.0

    mu = np.mean(returns) if len(returns) > 0 else 0.0
    sigma = np.std(returns) if len(returns) > 0 else 0.0

    sharpe = (
        ((mu - rf_step) / sigma) * math.sqrt(steps_per_year)
        if sigma > 0
        else 0.0
    )

    downside_returns = returns[returns < 0.0]
    downside_sigma = np.std(downside_returns) if len(downside_returns) > 0 else 0.0

    sortino = (
        ((mu - rf_step) / downside_sigma) * math.sqrt(steps_per_year)
        if downside_sigma > 0
        else 0.0
    )

    # ----------------------------------------------------
    # Max Drawdown
    # ----------------------------------------------------
    cumulative_max = np.maximum.accumulate(equity)
    drawdowns = (equity - cumulative_max) / cumulative_max
    max_drawdown = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    # ----------------------------------------------------
    # Calmar ratio
    # ----------------------------------------------------
    calmar = cagr / abs(max_drawdown) if abs(max_drawdown) > 1e-8 else 0.0

    return {
        "final_equity_mtm": float(final_equity_mtm),
        "final_equity_realized": (
            None if final_equity_realized is None else float(final_equity_realized)
        ),
        "return_mtm": float(return_mtm),
        "return_realized": return_realized,
        "cagr": float(cagr),
        "volatility_annual": float(vol_annual),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar),
    }


def _compute_risk_behavior_metrics(*, steps_df, trades_df):
    """
    Compute risk and behavior metrics.

    Metrics:
    - exposure_time_pct
    - avg_abs_exposure_pct
    - max_abs_exposure_pct
    - turnover_btc
    - stop_trigger_count
    - stop_trigger_rate
    """

    if steps_df.empty:
        return {
            "exposure_time_pct": 0.0,
            "avg_abs_exposure_pct": 0.0,
            "max_abs_exposure_pct": 0.0,
            "turnover_btc": 0.0,
            "stop_trigger_count": 0,
            "stop_trigger_rate": 0.0,
        }

    holdings = steps_df["holdings"].astype(float).values

    # ----------------------------------------------------
    # Exposure
    # ----------------------------------------------------
    in_position = np.abs(holdings) > 0.0
    exposure_time_pct = float(np.mean(in_position))

    abs_holdings = np.abs(holdings)

    avg_abs_exposure = float(np.mean(abs_holdings))
    max_abs_exposure = float(np.max(abs_holdings))

    max_position = float(getattr(config, "MAX_POSITION_BTC", 1.0))

    avg_abs_exposure_pct = (avg_abs_exposure / max_position) * 100.0
    max_abs_exposure_pct = (max_abs_exposure / max_position) * 100.0

    # ----------------------------------------------------
    # Turnover
    # ----------------------------------------------------
    if "effective_delta_btc" in steps_df.columns:
        turnover_btc = float(np.sum(np.abs(steps_df["effective_delta_btc"].astype(float).values)))
    else:
        turnover_btc = 0.0

    # ----------------------------------------------------
    # Stop statistics
    # ----------------------------------------------------
    if "stop_triggered" in steps_df.columns:
        stop_trigger_count = int(steps_df["stop_triggered"].astype(bool).sum())
        stop_trigger_rate = float(stop_trigger_count / len(steps_df))
    else:
        stop_trigger_count = 0
        stop_trigger_rate = 0.0

    return {
        "exposure_time_pct": exposure_time_pct,
        "avg_abs_exposure_pct": avg_abs_exposure_pct,
        "max_abs_exposure_pct": max_abs_exposure_pct,
        "turnover_btc": turnover_btc,
        "stop_trigger_count": stop_trigger_count,
        "stop_trigger_rate": stop_trigger_rate,
    }


def _compute_trade_metrics(trades_df):
    """
    Compute trade-level performance metrics from trades.csv.
    """

    if trades_df is None or trades_df.empty:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl_usd": 0.0,
            "median_trade_pnl_usd": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "max_win_usd": 0.0,
            "max_loss_usd": 0.0,
            "avg_trade_duration_steps": None,
            "avg_trade_duration_time": None,
        }

    pnl = trades_df["pnl_usd"].astype(float)

    num_trades = int(len(pnl))

    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]

    # ----------------------------------------------------
    # Win / loss stats
    # ----------------------------------------------------
    win_rate = float(len(wins) / num_trades) if num_trades > 0 else 0.0

    sum_profits = float(wins.sum())
    sum_losses = float(abs(losses.sum()))

    if sum_losses > 0:
        profit_factor = sum_profits / sum_losses
    elif sum_profits > 0:
        profit_factor = None  # no losses -> undefined/infinite
    else:
        profit_factor = 0.0

    avg_trade_pnl = float(pnl.mean())
    median_trade_pnl = float(pnl.median())

    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0  # negative

    max_win = float(wins.max()) if len(wins) > 0 else 0.0
    max_loss = float(losses.min()) if len(losses) > 0 else 0.0  # negative

    # ----------------------------------------------------
    # Duration stats
    # ----------------------------------------------------
    if "open_timestamp" in trades_df.columns and "close_timestamp" in trades_df.columns:
        try:
            #Add format specification to avoid ambiguous parsing
            # Timestamps in trades.csv are ISO format (YYYY-MM-DD HH:MM:SS)
            open_ts = pd.to_datetime(trades_df["open_timestamp"], format="ISO8601")
            close_ts = pd.to_datetime(trades_df["close_timestamp"], format="ISO8601")
            durations_time = (close_ts - open_ts)
            avg_trade_duration_time = durations_time.mean().total_seconds()
        except Exception:
            avg_trade_duration_time = None
    else:
        avg_trade_duration_time = None

    return {
        "num_trades": num_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_pnl_usd": avg_trade_pnl,
        "median_trade_pnl_usd": median_trade_pnl,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "max_win_usd": max_win,
        "max_loss_usd": max_loss,
        "avg_trade_duration_seconds": avg_trade_duration_time
    }


def _compute_reward_metrics(steps_df):
    """
    Compute reward diagnostics metrics.

    Metrics:
    - sum_reward
    - mean_reward
    - std_reward
    - reward_return_correlation
    """

    if steps_df.empty or "reward" not in steps_df.columns or "equity" not in steps_df.columns:
        return {
            "sum_reward": 0.0,
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "reward_return_correlation": 0.0,
        }

    reward = steps_df["reward"].astype(float).values
    equity = steps_df["equity"].astype(float).values

    # ----------------------------------------------------
    # Basic reward stats
    # ----------------------------------------------------
    sum_reward = float(np.nansum(reward))
    mean_reward = float(np.nanmean(reward))
    std_reward = float(np.nanstd(reward))

    # ----------------------------------------------------
    # Reward <-> return correlation
    # ----------------------------------------------------
    if len(equity) >= 2:
        returns = np.diff(equity) / equity[:-1]
        reward_aligned = reward[1:]  # align with returns

        mask = np.isfinite(returns) & np.isfinite(reward_aligned)

        if np.sum(mask) >= 2:
            corr = float(np.corrcoef(returns[mask], reward_aligned[mask])[0, 1])
        else:
            corr = 0.0
    else:
        corr = 0.0

    return {
        "sum_reward": sum_reward,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "reward_return_correlation": corr,
    }


def _compute_data_quality_metrics(*, steps_df, summary):
    """
    Compute data integrity and sanity-check metrics.
    """

    if steps_df.empty:
        return {
            "equity_min": None,
            "equity_max": None,
            "reward_nan_count": 0,
            "reward_inf_count": 0,
            "equity_nan_count": 0,
            "equity_inf_count": 0,
            "bankrupt": bool(summary.get("bankrupt", False)),
        }

    equity = steps_df["equity"].astype(float).values if "equity" in steps_df.columns else np.array([])
    reward = steps_df["reward"].astype(float).values if "reward" in steps_df.columns else np.array([])

    data_quality = {
        "equity_min": float(np.nanmin(equity)) if equity.size > 0 else None,
        "equity_max": float(np.nanmax(equity)) if equity.size > 0 else None,

        "reward_nan_count": int(np.isnan(reward).sum()),
        "reward_inf_count": int(np.isinf(reward).sum()),

        "equity_nan_count": int(np.isnan(equity).sum()),
        "equity_inf_count": int(np.isinf(equity).sum()),

        "bankrupt": bool(summary.get("bankrupt", False)),
    }

    return data_quality


# ============================================================
# Helpers
# ============================================================

def _resolve_steps_per_year(timeframe: str) -> int:
    if timeframe not in TIMEFRAME_TO_STEPS_PER_YEAR:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TIMEFRAME_TO_STEPS_PER_YEAR[timeframe]


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    def sanitize(x):
        if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
            return None
        if isinstance(x, dict):
            return {k: sanitize(v) for k, v in x.items()}
        if isinstance(x, list):
            return [sanitize(v) for v in x]
        return x

    clean_obj = sanitize(obj)

    with path.open("w", encoding="utf-8") as f:
        json.dump(clean_obj, f, indent=4, allow_nan=False)
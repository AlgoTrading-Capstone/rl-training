"""
Backtest runner.

Runs a full backtest episode using a trained ElegantRL actor (act.pth)
and logs raw step-level data for metrics & plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

import numpy as np
import torch

import config
from bitcoin_env import BitcoinTradingEnv
from trade_engine import compute_equity, close_position
from backtesting.step_logger import StepLogger
from backtesting.state_debug_logger import StateDebugLogger
from backtesting.trade_tracker import TradeTracker
from backtesting.metrics_manager import compute_and_write_metrics
from backtesting.plots.plot_runner import generate_backtest_plots
from utils.logger import RLLogger, LogComponent


# ============================================================
# Public API
# ============================================================

def run_backtest(
    *,
    model_metadata: Dict[str, Any],
    act_path: str | Path,
    price_array: np.ndarray,
    tech_array: np.ndarray,
    turbulence_array: np.ndarray,
    signal_array: np.ndarray,
    datetime_array: np.ndarray,
    out_dir: str | Path,
    backtest_config: Dict[str, Any],
    logger: RLLogger,
) -> None:
    """
    Execute a single backtest run and persist raw logs.

    Args:
        model_metadata: Training run metadata from metadata.json
        act_path: Path to trained actor checkpoint (act.pth)
        price_array: OHLCV price data
        tech_array: Technical indicators
        turbulence_array: Market stress indicators
        signal_array: Strategy signals (One-Hot encoded)
        datetime_array: Timestamps for each candle
        out_dir: Output directory for backtest artifacts
        backtest_config: Backtest configuration (date range, backtest_id)
        logger: RLLogger instance for logging

    Artifacts written:
        - steps.csv   : full step-by-step log (includes full state vector)
        - summary.json: minimal run summary
        - metrics.json: comprehensive performance metrics
        - plots/*.png: visualization plots

    Raises:
        ValueError: On environment / model incompatibility
        FileNotFoundError: If actor checkpoint not found
    """
    backtest_logger = logger.for_component(LogComponent.BACKTEST)
    # --------------------------------------------------------
    # STEP 0: Validate inputs & prepare output dir
    # --------------------------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    act_path = Path(act_path)
    if not act_path.is_file():
        raise FileNotFoundError(f"Actor checkpoint not found: {act_path}")

    # --------------------------------------------------------
    # STEP 1: Build backtest environment
    # --------------------------------------------------------
    env = BitcoinTradingEnv(
        price_ary=price_array,
        tech_ary=tech_array,
        turbulence_array=turbulence_array,
        signal_ary=signal_array,
        datetime_ary=datetime_array,
        mode="backtest",
    )

    # --------------------------------------------------------
    # STEP 2: Validate backtest compatibility against training
    # --------------------------------------------------------

    env_spec = model_metadata["env_spec"]
    train_data = model_metadata["data"]
    train_strategies = model_metadata["strategies"]

    # --------------------------------------------------------
    # (A) Indicator set validation
    # --------------------------------------------------------
    trained_indicators = set(train_data.get("indicators", []))
    current_indicators = set(config.INDICATORS)

    if trained_indicators != current_indicators:
        added = current_indicators - trained_indicators
        removed = trained_indicators - current_indicators

        raise ValueError(
            "Indicator set mismatch between training and backtest:\n"
            f"  Added since training   : {sorted(added)}\n"
            f"  Removed since training : {sorted(removed)}\n"
            "Ensure config.INDICATORS matches the training configuration."
        )

    # --------------------------------------------------------
    # (B) Turbulence / VIX flags validation
    # --------------------------------------------------------
    turbulence_mismatches = []

    if config.ENABLE_TURBULENCE != train_data.get("enable_turbulence"):
        turbulence_mismatches.append(
            f"ENABLE_TURBULENCE (backtest={config.ENABLE_TURBULENCE}, "
            f"training={train_data.get('enable_turbulence')})"
        )

    # Check VIX enabled status from EXTERNAL_ASSETS
    backtest_vix_enabled = any(asset.get('enabled', False) and asset.get('col_name') == 'vix' for asset in config.EXTERNAL_ASSETS)
    if backtest_vix_enabled != train_data.get("enable_vix"):
        turbulence_mismatches.append(
            f"VIX enabled (backtest={backtest_vix_enabled}, "
            f"training={train_data.get('enable_vix')})"
        )

    if turbulence_mismatches:
        raise ValueError(
            "Turbulence / VIX configuration mismatch between training and backtest:\n"
            "  - " + "\n  - ".join(turbulence_mismatches) + "\n"
            "Ensure ENABLE_TURBULENCE and VIX settings in EXTERNAL_ASSETS match the training configuration."
        )

    # --------------------------------------------------------
    # (C) Strategy set + order validation
    # --------------------------------------------------------
    trained_enabled = bool(train_strategies.get("enabled", False))
    trained_list = list(train_strategies.get("strategy_list", []))

    current_enabled = bool(getattr(config, "ENABLE_STRATEGIES", False))
    current_list = list(config.STRATEGY_LIST) if current_enabled else []

    # 1) enabled flag must match
    if trained_enabled != current_enabled:
        raise ValueError(
            "Strategy enable flag mismatch between training and backtest:\n"
            f"  training enabled = {trained_enabled}\n"
            f"  backtest enabled = {current_enabled}\n"
            "Fix: set ENABLE_STRATEGIES to match the training configuration."
        )

    # 2) list content must match (set)
    trained_set = set(trained_list)
    current_set = set(current_list)

    if trained_set != current_set:
        added = sorted(current_set - trained_set)
        removed = sorted(trained_set - current_set)
        raise ValueError(
            "Strategy list mismatch between training and backtest:\n"
            f"  Added since training   : {added}\n"
            f"  Removed since training : {removed}\n"
            "Backtest must use the exact same STRATEGY_LIST as training."
        )

    # 3) order must match (critical for signal_vec layout)
    if trained_list != current_list:
        diffs = []
        for i, (t, c) in enumerate(zip(trained_list, current_list)):
            if t != c:
                diffs.append({"index": i, "training": t, "backtest": c})
                if len(diffs) >= 10:
                    break

        raise ValueError(
            "Strategy ORDER mismatch (same strategies but different order).\n"
            "This is critical because each strategy occupies 4 signal columns in signal_vec.\n"
            f"  First diffs (up to 10): {diffs}\n"
            f"  training order = {trained_list}\n"
            f"  backtest  order = {current_list}\n"
            "Fix: reorder config.STRATEGY_LIST to exactly match training."
        )

    # --------------------------------------------------------
    # (D) Final hard check: state/action dimensions
    # --------------------------------------------------------
    if env.state_dim != int(env_spec["state_dim"]):
        raise ValueError(
            "State dimension mismatch despite identical configuration:\n"
            f"  training state_dim = {env_spec['state_dim']}\n"
            f"  backtest state_dim = {env.state_dim}\n"
            "This indicates a bug in feature construction or normalization."
        )

    if env.action_dim != int(env_spec["action_dim"]):
        raise ValueError(
            "Action dimension mismatch:\n"
            f"  training action_dim = {env_spec['action_dim']}\n"
            f"  backtest action_dim = {env.action_dim}\n"
        )

    # --------------------------------------------------------
    # STEP 3: Load actor policy
    # --------------------------------------------------------
    policy = _load_actor(
        rl_model=model_metadata["rl"]["model"],
        net_dims=model_metadata["rl"]["net_dims"],
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        act_path=act_path,
        device=config.BACKTEST_DEVICE,
    )

    # --------------------------------------------------------
    # STEP 4: Run episode
    # --------------------------------------------------------
    step_logger = StepLogger(out_dir)
    debug_logger = StateDebugLogger(out_dir)
    trade_tracker = TradeTracker(out_dir)

    step_logger.open()
    debug_logger.open()
    trade_tracker.open()

    try:
        state = env.reset()
        done = False

        while not done:
            # ----------------------------------------------------
            # Policy inference
            # ----------------------------------------------------
            a_pos, a_sl = _policy_action(policy, state)

            # ----------------------------------------------------
            # Snapshot BEFORE step (correct candle alignment)
            # ----------------------------------------------------
            price_vec = env.current_price.copy()
            tech_vec = env.current_tech.copy()
            turb_vec = env.current_turbulence.copy()
            sig_vec = env.current_signal.copy()
            timestamp = env.current_datetime

            holdings_before = float(env.position.holdings)
            stop_price_before = env.position.stop_price

            # ----------------------------------------------------
            # Environment step
            # ----------------------------------------------------
            next_state, reward, done, info = env.step(
                np.array([a_pos, a_sl], dtype=np.float32)
            )

            equity = float(info["equity"])
            holdings_after = float(env.position.holdings)
            trade_price = float(info.get("equity_price", price_vec[3]))  # stop_exec_price if triggered, else close
            stop_price_for_tracker = stop_price_before if bool(info["stop_triggered"]) else env.position.stop_price

            # ----------------------------------------------------
            # steps.csv - full step-level logging
            # ----------------------------------------------------
            step_logger.log_step(
                step_idx=int(info["step_idx"]),
                timestamp=timestamp,
                state_norm=state,
                price_vec=price_vec,
                tech_vec=tech_vec,
                turb_vec=turb_vec,
                sig_vec=sig_vec,
                a_pos=float(a_pos),
                a_sl=float(a_sl),
                trade_executed=bool(info["trade_executed"]),
                effective_delta_btc=float(info["effective_delta_btc"]),
                balance=float(env.position.balance),
                holdings=float(env.position.holdings),
                equity=float(equity),
                reward=float(reward),
                stop_triggered=bool(info["stop_triggered"]),
                done=bool(done),
            )

            # ----------------------------------------------------
            # state_debug.csv - raw data logging for debugging
            # ----------------------------------------------------
            debug_logger.log_step(
                step_idx=int(info["step_idx"]),
                timestamp=timestamp,
                state_norm=state,
                price_vec=price_vec,
                tech_vec=tech_vec,
                turb_vec=turb_vec,
                signal_vec=sig_vec,
                action_a_pos=float(a_pos),
                action_a_sl=float(a_sl),
                reward=float(reward),
            )

            # ----------------------------------------------------
            # trades.csv - trade-level tracking
            # Uses close price of the candle that just closed
            # ----------------------------------------------------
            trade_tracker.on_step(
                timestamp=timestamp,
                trade_price=trade_price,
                equity_after=float(info["equity"]),
                holdings_before=holdings_before,
                holdings_after=holdings_after,
                stop_price=stop_price_for_tracker,
                stop_triggered=bool(info["stop_triggered"]),
                forced_close=False,
            )

            # Advance state
            state = next_state

        # --------------------------------------------------------
        # STEP 5: End-of-episode finalization
        # --------------------------------------------------------
        force_close = bool(getattr(config, "BACKTEST_FORCE_CLOSE", True))

        last_close = float(env.current_price[3])

        # Mark-to-market final equity (true portfolio value at last close)
        final_equity_mtm = compute_equity(
            balance=env.position.balance,
            holdings=env.position.holdings,
            price=last_close,
        )

        bankrupt = final_equity_mtm <= 0.0

        # Realized equity is only meaningful if we're flat (either already flat, or after force-close)
        final_equity_realized = float(final_equity_mtm) if np.isclose(env.position.holdings, 0.0) else None

        if not np.isclose(env.position.holdings, 0.0):
            if force_close:
                # Force close to realize PnL (and close any open trade in trades.csv)
                holdings_before_fc = float(env.position.holdings)

                result = close_position(
                    price=last_close,
                    state=env.position,
                    cfg=env.trade_cfg,
                )
                env.position = result.new_state  # holdings should become 0, stop cleared

                equity_after_fc = compute_equity(
                    balance=env.position.balance,
                    holdings=env.position.holdings,  # should be 0
                    price=last_close,
                )
                final_equity_realized = float(equity_after_fc)

                trade_tracker.on_step(
                    timestamp=env.current_datetime,
                    trade_price=last_close,
                    equity_after=float(equity_after_fc),
                    holdings_before=holdings_before_fc,
                    holdings_after=float(env.position.holdings),
                    stop_price=env.position.stop_price,
                    stop_triggered=False,
                    forced_close=True,
                )
            else:
                # No force-close -> ensure trades.csv won't end with an open trade
                trade_tracker.finalize_end(
                    timestamp=env.current_datetime,
                    trade_price=last_close,
                    equity_mtm=float(final_equity_mtm),
                    holdings_after=float(env.position.holdings),
                    stop_price=env.position.stop_price,
                )
    finally:
        step_logger.close()
        debug_logger.close()
        trade_tracker.close()

    summary = {
        "initial_equity": float(env.initial_balance),
        "final_equity_mtm": float(final_equity_mtm),
        "final_equity_realized": final_equity_realized,
        "episode_return_mtm": float(final_equity_mtm / env.initial_balance),
        "episode_return_realized": None if final_equity_realized is None else float(final_equity_realized / env.initial_balance),
        "bankrupt": bool(bankrupt),
        "force_close": bool(force_close),
    }

    # --------------------------------------------------------
    # STEP 6: Persist artifacts
    # --------------------------------------------------------
    # steps.csv/state_debug.csv/trades.csv are already written by their loggers during STEP 4.

    _write_json(out_dir / "summary.json", summary)

    backtest_logger.info(f"Backtest completed. Results saved to: {out_dir}")
    backtest_logger.info("Artifacts: steps.csv, state_debug.csv, trades.csv, summary.json")

    # --------------------------------------------------------
    # STEP 7: Compute metrics & generate plots
    # --------------------------------------------------------

    compute_and_write_metrics(
        out_dir=out_dir,
        model_metadata=model_metadata,
        backtest_config=backtest_config,
    )

    backtest_logger.success("Backtest metrics computed and saved")

    generate_backtest_plots(
        out_dir=out_dir,
        model_metadata=model_metadata,
        backtest_config=backtest_config,
    )

    backtest_logger.success("Backtest plots generated and saved")

# ============================================================
# Internal helpers
# ============================================================

def _load_actor(
    *,
    rl_model: str,
    net_dims: List[int],
    state_dim: int,
    action_dim: int,
    act_path: Path,
    device: str,
):
    """Load ElegantRL actor network in eval mode."""
    from elegantrl.agents import AgentPPO, AgentSAC

    model = rl_model.upper()
    agent_cls = AgentPPO if model == "PPO" else AgentSAC

    agent_cls(
        net_dims=net_dims,
        state_dim=state_dim,
        action_dim=action_dim,
        gpu_id=-1,
    )

    ckpt = torch.load(
        act_path,
        map_location=torch.device(device),
        weights_only=False,
    )

    policy = ckpt.to(device)  # ckpt si the Actor network itself
    policy.eval()
    return policy


def _policy_action(policy, state: np.ndarray) -> Tuple[float, float]:
    """Run policy forward pass and clip actions."""
    device = next(policy.parameters()).device
    s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        a = policy(s).squeeze(0).detach().cpu().numpy()

    return float(np.clip(a[0], -1.0, 1.0)), float(np.clip(a[1], -1.0, 1.0))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)
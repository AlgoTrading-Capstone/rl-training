"""
Metadata utilities.

Responsible for:
- Creating metadata.json skeleton from user_input metadata
- Enriching metadata.json with runtime, RL, environment and system information
- Reading existing metadata.json for backtest usage
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


import config
from utils.logger import RLLogger, LogComponent
from utils.formatting import Formatter

# ============================================================================
# Public API
# ============================================================================

def create_metadata_file(
        metadata: Dict[str, Any],
        run_path: Path,
        logger: RLLogger,
) -> Path:
    """
    Create metadata.json from user-provided metadata.
    This function is called ONCE from main, immediately after user_input.

    Args:
        metadata: Metadata dict returned from user_input
        run_path: Root run directory
        logger: RLLogger instance for logging

    Returns:
        Path to created metadata.json

    Raises:
        RuntimeError: If file cannot be written
    """
    main_logger = logger.for_component(LogComponent.MAIN)
    metadata_file = run_path / "metadata.json"

    try:
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        main_logger.info(f"Metadata saved to: {metadata_file}")
        return metadata_file

    except Exception as e:
        msg = Formatter.error_context(
            f"Failed to save metadata.json at {metadata_file}",
            "Check disk space and file permissions"
        )
        main_logger.error(msg)
        main_logger.debug(f"Exception details: {e}")
        raise RuntimeError("Metadata persistence failed") from e


def enrich_metadata_with_training_config(
    run_path: Path,
    algorithm_config: Dict[str, Any],
    *,
    state_dim: int,
    action_dim: int,
    logger: RLLogger,
) -> None:
    """
    Enrich existing metadata.json with a full training configuration snapshot.
    This function is called AFTER the training configuration is fully resolved.

    Args:
        run_path: Root run directory
        algorithm_config: Algorithm-specific config (from rl_configs.py)
        state_dim: Environment state dimension
        action_dim: Environment action dimension
        logger: RLLogger instance for logging

    Raises:
        RuntimeError: If metadata.json cannot be updated
    """
    metadata_file = run_path / "metadata.json"
    main_logger = logger.for_component(LogComponent.MAIN)
    metadata = _load_metadata(metadata_file)

    # ------------------------------------------------------------
    # RL section
    # ------------------------------------------------------------
    metadata["rl"] = {
        "model": config.RL_MODEL,
        "gamma": config.GAMMA,
        "learning_rate": config.LEARNING_RATE,
        "net_dims": config.NET_DIMS,
        "total_training_steps": config.TOTAL_TRAINING_STEPS,
        "seed": config.SEED,
        "algorithm_params": algorithm_config,
    }

    # ------------------------------------------------------------
    # Environment interface (model contract)
    # ------------------------------------------------------------
    metadata["env_spec"] = {
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
    }

    # ------------------------------------------------------------
    # Environment section
    # ------------------------------------------------------------
    metadata["environment"] = {
        "initial_balance": config.INITIAL_BALANCE,
        "leverage_limit": config.LEVERAGE_LIMIT,
        "min_stop_loss_pct": config.MIN_STOP_LOSS_PCT,
        "max_stop_loss_pct": config.MAX_STOP_LOSS_PCT,
        "exposure_deadzone": config.EXPOSURE_DEADZONE,
        "max_position_btc": config.MAX_POSITION_BTC,
        "transaction_fee": config.TRANSACTION_FEE,
        "slippage_mean": getattr(config, "SLIPPAGE_MEAN", None),
        "reward_function": config.REWARD_FUNCTION,
        "downside_weight": getattr(config, "DOWNSIDE_WEIGHT", None),
    }

    # ------------------------------------------------------------
    # Data section
    # ------------------------------------------------------------
    metadata["data"] = {
        "exchange": config.EXCHANGE_NAME,
        "trading_pair": config.TRADING_PAIR,
        "timeframe": config.DATA_TIMEFRAME,
        "indicators": config.INDICATORS,
        "enable_turbulence": config.ENABLE_TURBULENCE,
        "enable_vix": any(asset.get('enabled', False) and asset.get('col_name') == 'vix' for asset in config.EXTERNAL_ASSETS),
        "vix_symbol": next((asset.get('ticker') for asset in config.EXTERNAL_ASSETS if asset.get('col_name') == 'vix'), None),
        "external_assets": [asset for asset in config.EXTERNAL_ASSETS if asset.get('enabled', False)],
    }

    # ------------------------------------------------------------
    # Strategies section
    # ------------------------------------------------------------
    metadata["strategies"] = {
        "enabled": config.ENABLE_STRATEGIES,
        "strategy_list": config.STRATEGY_LIST if config.ENABLE_STRATEGIES else [],
    }

    _save_metadata(metadata_file, metadata, logger)
    main_logger.info("Metadata enriched with training configuration")


def load_metadata(run_path: Path) -> Dict[str, Any]:
    """
    Load metadata.json from a run directory.

    Args:
        run_path: Root run directory

    Returns:
        Parsed metadata dictionary
    """
    metadata_file = Path(run_path) / "metadata.json"
    return _load_metadata(metadata_file)


def append_backtest_metadata(run_path: Path, backtest_entry: dict) -> None:
    """
    Append a backtest entry to metadata.json.

    Args:
        run_path: Root run directory
        backtest_entry: Backtest metadata entry to append
    """
    metadata = load_metadata(run_path)

    metadata.setdefault("backtests", [])
    metadata["backtests"].append(backtest_entry)

    # Note: append_backtest_metadata doesn't have logger parameter, so _save_metadata creates its own
    _save_metadata(run_path / "metadata.json", metadata, logger=None)


# ============================================================================
# Internal helpers
# ============================================================================

def _load_metadata(metadata_file: Path) -> Dict[str, Any]:
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.json not found at {metadata_file}")

    with metadata_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_metadata(metadata_file: Path, metadata: Dict[str, Any], logger: RLLogger | None = None) -> None:
    """
    Internal helper to persist metadata dictionary to JSON file.

    Args:
        metadata_file: Path to metadata.json file
        metadata: Metadata dictionary to save
        logger: Optional RLLogger instance. If None, creates a new instance (use sparingly).
    """
    if logger is None:
        # Fallback: create logger instance (only for append_backtest_metadata)
        main_logger = RLLogger().for_component(LogComponent.MAIN)
    else:
        main_logger = logger.for_component(LogComponent.MAIN)

    try:
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        msg = Formatter.error_context(
            f"Failed to write metadata.json at {metadata_file}",
            "Check disk space and file permissions"
        )
        main_logger.error(msg)
        main_logger.debug(f"Exception details: {e}")
        raise RuntimeError("Metadata update failed") from e
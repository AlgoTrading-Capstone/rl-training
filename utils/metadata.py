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

# ============================================================================
# Public API
# ============================================================================

def create_metadata_file(
        metadata: Dict[str, Any],
        run_path: str | Path,
) -> Path:
    """
    Create metadata.json from user-provided metadata.
    This function is called ONCE from main, immediately after user_input.

    Args:
        metadata: Metadata dict returned from user_input
        run_path: Root run directory

    Returns:
        Path to created metadata.json

    Raises:
        RuntimeError: If file cannot be written
    """
    run_path = Path(run_path)
    metadata_file = run_path / "metadata.json"

    try:
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"[INFO] Metadata saved to: {metadata_file}")
        return metadata_file

    except Exception as e:
        print(f"[ERROR] Failed to save metadata.json at {metadata_file}")
        raise RuntimeError("Metadata persistence failed") from e


def enrich_metadata_with_training_config(
    run_path: str | Path,
    algorithm_config: Dict[str, Any],
) -> None:
    """
    Enrich existing metadata.json with a full training configuration snapshot.
    This function is called AFTER the training configuration is fully resolved.

    Args:
        run_path: Root run directory
        algorithm_config: Algorithm-specific config (from rl_configs.py)

    Raises:
        RuntimeError: If metadata.json cannot be updated
    """
    run_path = Path(run_path)
    metadata_file = run_path / "metadata.json"

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
        "enable_vix": config.ENABLE_VIX,
        "vix_symbol": getattr(config, "VIX_SYMBOL", None),
        "train_test_split": config.TRAIN_TEST_SPLIT,
    }

    # ------------------------------------------------------------
    # Strategies section
    # ------------------------------------------------------------
    metadata["strategies"] = {
        "enabled": config.ENABLE_STRATEGIES,
        "strategy_list": config.STRATEGY_LIST if config.ENABLE_STRATEGIES else [],
    }

    _save_metadata(metadata_file, metadata)
    print("[INFO] Metadata enriched with training configuration")


def load_metadata(run_path: str | Path) -> Dict[str, Any]:
    """
    Load metadata.json from a run directory.

    Args:
        run_path: Root run directory

    Returns:
        Parsed metadata dictionary
    """
    metadata_file = Path(run_path) / "metadata.json"
    return _load_metadata(metadata_file)


# ============================================================================
# Internal helpers
# ============================================================================

def _load_metadata(metadata_file: Path) -> Dict[str, Any]:
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.json not found at {metadata_file}")

    with metadata_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_metadata(metadata_file: Path, metadata: Dict[str, Any]) -> None:
    try:
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        print(f"[ERROR] Failed to write metadata.json at {metadata_file}")
        raise RuntimeError("Metadata update failed") from e
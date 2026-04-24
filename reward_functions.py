"""
Reward functions for trading environments.

Each function returns `(total_reward, components_dict)` so downstream loggers can
record the decomposition. Component keys are stable across branches (including
bankruptcy and no-context fallbacks) so CSV headers don't shift mid-run.
"""

import numpy as np
from config import (
    DOWNSIDE_WEIGHT,
    STOP_CLUSTER_PENALTY,
    SAME_SIDE_REENTRY_PENALTY,
    DRAWDOWN_PENALTY_THRESHOLD,
    DRAWDOWN_PENALTY_WEIGHT,
)


def reward_log_return(old_equity: float, new_equity: float, *, context=None) -> tuple[float, dict]:
    """
    Logarithmic return reward.
    r = ln(new / old)
    """
    components = {"base_log_return": 0.0, "bankruptcy": 0.0}

    if old_equity <= 0 or new_equity <= 0:
        components["bankruptcy"] = 1.0
        return -1.0, components

    r = float(np.log(new_equity / old_equity))
    components["base_log_return"] = r
    return r, components


def reward_asymmetric_drawdown_penalty(old_equity: float, new_equity: float, *, context=None) -> tuple[float, dict]:
    """
    Asymmetric Log-Return reward:
    - Positive log-returns are rewarded as-is.
    - Negative log-returns are penalized linearly by DOWNSIDE_WEIGHT.
    """
    components = {"base_log_return": 0.0, "downside_extra_penalty": 0.0, "bankruptcy": 0.0}

    if old_equity <= 0 or new_equity <= 0:
        components["bankruptcy"] = 1.0
        return -1.0 * DOWNSIDE_WEIGHT, components

    r = float(np.log(new_equity / old_equity))
    components["base_log_return"] = r

    if r >= 0:
        return r, components

    # Extra penalty = amount beyond the plain log-return.
    extra = r * (DOWNSIDE_WEIGHT - 1.0)
    components["downside_extra_penalty"] = extra
    return r * DOWNSIDE_WEIGHT, components


def reward_stop_aware_drawdown(old_equity: float, new_equity: float, *, context=None) -> tuple[float, dict]:
    """
    Log-return reward with explicit penalties for:
    1. Repeated / clustered stop-loss events
    2. Same-side re-entry shortly after a stop
    3. Sustained drawdown beyond a configurable threshold
    """
    components = {
        "base_log_return": 0.0,
        "stop_penalty": 0.0,
        "reentry_penalty": 0.0,
        "drawdown_penalty": 0.0,
        "bankruptcy": 0.0,
    }

    if old_equity <= 0 or new_equity <= 0:
        components["bankruptcy"] = 1.0
        return -1.0, components

    base = float(np.log(new_equity / old_equity))
    components["base_log_return"] = base

    # Graceful fallback: without context, behave like plain log-return.
    if context is None:
        return base, components

    if context["stop_triggered"]:
        cluster_count = max(context["recent_stop_count"], 1)
        stop_penalty = STOP_CLUSTER_PENALTY * cluster_count
    else:
        stop_penalty = 0.0

    reentry_penalty = SAME_SIDE_REENTRY_PENALTY if context["same_side_reentry"] else 0.0

    dd = context["current_drawdown"]
    drawdown_penalty = DRAWDOWN_PENALTY_WEIGHT * max(0.0, dd - DRAWDOWN_PENALTY_THRESHOLD)

    components["stop_penalty"] = float(stop_penalty)
    components["reentry_penalty"] = float(reentry_penalty)
    components["drawdown_penalty"] = float(drawdown_penalty)

    reward = base - stop_penalty - reentry_penalty - drawdown_penalty
    return float(reward), components


# Registry for easy lookup from config string
REWARD_REGISTRY = {
    "log_return": reward_log_return,
    "asymmetric": reward_asymmetric_drawdown_penalty,
    "stop_aware_drawdown": reward_stop_aware_drawdown,
}
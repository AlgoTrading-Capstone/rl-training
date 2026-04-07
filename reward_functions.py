"""
Reward functions for trading environments.
"""

import numpy as np
from config import (
    DOWNSIDE_WEIGHT,
    STOP_CLUSTER_PENALTY,
    SAME_SIDE_REENTRY_PENALTY,
    DRAWDOWN_PENALTY_THRESHOLD,
    DRAWDOWN_PENALTY_WEIGHT,
)


def reward_log_return(old_equity: float, new_equity: float, *, context=None) -> float:
    """
    Logarithmic return reward.
    r = ln(new / old)
    """
    # Safety check for bankruptcy or invalid values to prevent np.log crash
    if old_equity <= 0 or new_equity <= 0:
        return -1.0  # Significant penalty for blowing up the account

    return np.log(new_equity / old_equity)


def reward_asymmetric_drawdown_penalty(old_equity: float, new_equity: float, *, context=None) -> float:
    """
    Asymmetric Log-Return reward:
    - Positive log-returns are rewarded as-is.
    - Negative log-returns are penalized linearly by DOWNSIDE_WEIGHT.
    """
    # Safety check for bankruptcy
    if old_equity <= 0 or new_equity <= 0:
        # If we just hit <= 0, return a large penalty scaled by the weight
        return -1.0 * DOWNSIDE_WEIGHT

    # Calculate Log Return
    r = np.log(new_equity / old_equity)

    if r >= 0:
        return r
    else:
        # Penalize downside linearly
        return r * DOWNSIDE_WEIGHT


def reward_stop_aware_drawdown(old_equity: float, new_equity: float, *, context=None) -> float:
    """
    Log-return reward with explicit penalties for:
    1. Repeated / clustered stop-loss events
    2. Same-side re-entry shortly after a stop
    3. Sustained drawdown beyond a configurable threshold

    Parameters
    ----------
    old_equity : float
    new_equity : float
    context : dict or None
        Required keys when provided:
            stop_triggered       : bool
            recent_stop_count    : int   (stops within cluster window, including current)
            same_side_reentry    : bool  (entered same side within window after stop)
            current_drawdown     : float (0.0 = at peak, 1.0 = total wipeout)

    Returns
    -------
    float
        Shaped reward value.
    """
    # Safety check for bankruptcy or invalid values
    if old_equity <= 0 or new_equity <= 0:
        return -1.0

    base = np.log(new_equity / old_equity)

    # Graceful fallback: without context, behave like plain log-return
    if context is None:
        return float(base)

    # --- Penalty 1: stop cluster ---
    if context["stop_triggered"]:
        cluster_count = max(context["recent_stop_count"], 1)
        stop_penalty = STOP_CLUSTER_PENALTY * cluster_count
    else:
        stop_penalty = 0.0

    # --- Penalty 2: same-side re-entry after stop ---
    reentry_penalty = SAME_SIDE_REENTRY_PENALTY if context["same_side_reentry"] else 0.0

    # --- Penalty 3: smooth drawdown beyond threshold ---
    dd = context["current_drawdown"]
    drawdown_penalty = DRAWDOWN_PENALTY_WEIGHT * max(0.0, dd - DRAWDOWN_PENALTY_THRESHOLD)

    reward = base - stop_penalty - reentry_penalty - drawdown_penalty
    return float(reward)


# Registry for easy lookup from config string
REWARD_REGISTRY = {
    "log_return": reward_log_return,
    "asymmetric": reward_asymmetric_drawdown_penalty,
    "stop_aware_drawdown": reward_stop_aware_drawdown,
}
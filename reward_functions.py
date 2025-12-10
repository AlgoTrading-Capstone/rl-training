"""
Reward functions for trading environments.
"""

import numpy as np
from config import DOWNSIDE_WEIGHT


def reward_log_return(old_equity: float, new_equity: float) -> float:
    """
    Logarithmic return reward.
    r = ln(new / old)
    """
    # Safety check for bankruptcy or invalid values to prevent np.log crash
    if old_equity <= 0 or new_equity <= 0:
        return -1.0  # Significant penalty for blowing up the account

    return np.log(new_equity / old_equity)


def reward_asymmetric_drawdown_penalty(old_equity: float, new_equity: float) -> float:
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


# Registry for easy lookup from config string
REWARD_REGISTRY = {
    "log_return": reward_log_return,
    "asymmetric": reward_asymmetric_drawdown_penalty,
}
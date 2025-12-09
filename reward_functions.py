"""
Reward functions for trading environments.
"""

from config import DOWNSIDE_WEIGHT


def reward_return(old_equity: float, new_equity: float) -> float:
    """
    Percent return reward.

    r = (new - old) / old
    """
    if old_equity <= 0:
        return 0.0
    return (new_equity - old_equity) / old_equity


def reward_asymmetric_drawdown_penalty(old_equity: float, new_equity: float) -> float:
    """
    Asymmetric reward:
    - Positive returns rewarded as-is.
    - Negative returns punished non-linearly (quadratically).

    The strength of the downside penalty is controlled by DOWNSIDE_WEIGHT
    defined in config.py.
    """
    if old_equity <= 0:
        return 0.0

    r = (new_equity - old_equity) / old_equity

    if r >= 0:
        return r
    else:
        # downside penalty grows quadratically
        return DOWNSIDE_WEIGHT * r * abs(r)


# Registry for easy lookup from config string
REWARD_REGISTRY = {
    "return": reward_return,
    "asymmetric": reward_asymmetric_drawdown_penalty,
}
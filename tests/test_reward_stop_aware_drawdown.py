"""
Tests for reward_stop_aware_drawdown reward function.

Validates:
1. Base return with no penalties matches log return
2. Stop trigger adds penalty
3. Clustered stops produce stronger penalty
4. Same-side re-entry adds penalty
5. No re-entry penalty without flag
6. Drawdown below threshold = no penalty
7. Drawdown above threshold = smooth penalty
8. Reward is always finite
9. Bankruptcy returns -1.0
10. Existing reward functions still work with context kwarg
11. No-context fallback to plain log return
12. Reward functions return (scalar, components_dict) tuple with stable keys
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from reward_functions import (
    reward_log_return as _reward_log_return,
    reward_asymmetric_drawdown_penalty as _reward_asymmetric_drawdown_penalty,
    reward_stop_aware_drawdown as _reward_stop_aware_drawdown,
)


# Scalar-only wrappers so the existing assertions keep their shape.
# The decomposition itself is tested separately in `test_reward_components_shape`.
def reward_log_return(*args, **kwargs):
    return _reward_log_return(*args, **kwargs)[0]


def reward_asymmetric_drawdown_penalty(*args, **kwargs):
    return _reward_asymmetric_drawdown_penalty(*args, **kwargs)[0]


def reward_stop_aware_drawdown(*args, **kwargs):
    return _reward_stop_aware_drawdown(*args, **kwargs)[0]


def _ctx(stop_triggered=False, recent_stop_count=0,
         same_side_reentry=False, current_drawdown=0.0):
    """Helper to build a reward context dict."""
    return {
        "stop_triggered": stop_triggered,
        "recent_stop_count": recent_stop_count,
        "same_side_reentry": same_side_reentry,
        "current_drawdown": current_drawdown,
    }


# ------------------------------------------------------------------ #
# 1. Base positive return, no stop, low drawdown
# ------------------------------------------------------------------ #
def test_base_return_no_penalties():
    old_eq, new_eq = 100_000.0, 100_100.0
    ctx = _ctx()
    r = reward_stop_aware_drawdown(old_eq, new_eq, context=ctx)
    expected = np.log(new_eq / old_eq)
    assert abs(r - expected) < 1e-10, f"Expected ~{expected}, got {r}"
    print("PASS: base return no penalties")


# ------------------------------------------------------------------ #
# 2. Stop triggered (single)
# ------------------------------------------------------------------ #
def test_stop_triggered_single():
    old_eq, new_eq = 100_000.0, 99_500.0
    base = np.log(new_eq / old_eq)
    ctx = _ctx(stop_triggered=True, recent_stop_count=1)
    r = reward_stop_aware_drawdown(old_eq, new_eq, context=ctx)
    assert r < base, f"With stop, reward ({r}) should be < base ({base})"
    print("PASS: single stop penalty")


# ------------------------------------------------------------------ #
# 3. Multiple recent stops -> stronger penalty
# ------------------------------------------------------------------ #
def test_stop_cluster_stronger():
    old_eq, new_eq = 100_000.0, 99_500.0
    r1 = reward_stop_aware_drawdown(
        old_eq, new_eq, context=_ctx(stop_triggered=True, recent_stop_count=1))
    r3 = reward_stop_aware_drawdown(
        old_eq, new_eq, context=_ctx(stop_triggered=True, recent_stop_count=3))
    assert r3 < r1, f"3 stops ({r3}) should give lower reward than 1 stop ({r1})"
    print("PASS: stop cluster stronger")


# ------------------------------------------------------------------ #
# 4. Same-side re-entry penalty
# ------------------------------------------------------------------ #
def test_same_side_reentry_penalty():
    old_eq, new_eq = 100_000.0, 100_050.0
    r_no = reward_stop_aware_drawdown(old_eq, new_eq, context=_ctx())
    r_re = reward_stop_aware_drawdown(
        old_eq, new_eq, context=_ctx(same_side_reentry=True))
    assert r_re < r_no, f"Re-entry ({r_re}) should be < no re-entry ({r_no})"
    print("PASS: same-side re-entry penalty")


# ------------------------------------------------------------------ #
# 5. No re-entry penalty without flag
# ------------------------------------------------------------------ #
def test_reentry_not_penalized_without_flag():
    old_eq, new_eq = 100_000.0, 100_050.0
    r = reward_stop_aware_drawdown(
        old_eq, new_eq, context=_ctx(same_side_reentry=False))
    expected = np.log(new_eq / old_eq)
    assert abs(r - expected) < 1e-10
    print("PASS: no re-entry penalty without flag")


# ------------------------------------------------------------------ #
# 6. Drawdown below threshold -> no drawdown penalty
# ------------------------------------------------------------------ #
def test_drawdown_below_threshold():
    old_eq, new_eq = 100_000.0, 100_050.0
    ctx = _ctx(current_drawdown=0.03)  # Below default 0.05 threshold
    r = reward_stop_aware_drawdown(old_eq, new_eq, context=ctx)
    expected = np.log(new_eq / old_eq)
    assert abs(r - expected) < 1e-10
    print("PASS: drawdown below threshold")


# ------------------------------------------------------------------ #
# 7. Drawdown above threshold -> smooth penalty
# ------------------------------------------------------------------ #
def test_drawdown_above_threshold():
    old_eq, new_eq = 100_000.0, 100_050.0
    ctx = _ctx(current_drawdown=0.10)  # Above 0.05 threshold
    r = reward_stop_aware_drawdown(old_eq, new_eq, context=ctx)
    base = np.log(new_eq / old_eq)
    assert r < base, f"With 10% drawdown, reward ({r}) should be < base ({base})"
    # Verify the penalty magnitude is proportional
    r_20 = reward_stop_aware_drawdown(
        old_eq, new_eq, context=_ctx(current_drawdown=0.20))
    assert r_20 < r, f"20% dd ({r_20}) should be < 10% dd ({r})"
    print("PASS: drawdown above threshold (smooth & proportional)")


# ------------------------------------------------------------------ #
# 8. Reward is finite for all tested scenarios
# ------------------------------------------------------------------ #
def test_reward_finite():
    cases = [
        (100_000, 100_000, _ctx()),
        (100_000, 50_000, _ctx(
            stop_triggered=True, recent_stop_count=5,
            same_side_reentry=True, current_drawdown=0.50)),
        (100_000, 150_000, _ctx()),
        (100_000, 99_999, _ctx(current_drawdown=0.01)),
    ]
    for old_eq, new_eq, ctx in cases:
        r = reward_stop_aware_drawdown(old_eq, new_eq, context=ctx)
        assert np.isfinite(r), f"Non-finite reward for ({old_eq}, {new_eq}): {r}"
    print("PASS: all rewards finite")


# ------------------------------------------------------------------ #
# 9. Bankruptcy
# ------------------------------------------------------------------ #
def test_bankruptcy():
    assert reward_stop_aware_drawdown(100_000, 0, context=_ctx()) == -1.0
    assert reward_stop_aware_drawdown(100_000, -100, context=_ctx()) == -1.0
    assert reward_stop_aware_drawdown(0, 100_000, context=_ctx()) == -1.0
    print("PASS: bankruptcy returns -1.0")


# ------------------------------------------------------------------ #
# 10. Existing functions still work with context kwarg
# ------------------------------------------------------------------ #
def test_existing_functions_accept_context():
    ctx = _ctx(stop_triggered=True, recent_stop_count=2)
    r1 = reward_log_return(100_000, 100_100, context=ctx)
    r2 = reward_asymmetric_drawdown_penalty(100_000, 100_100, context=ctx)
    assert np.isfinite(r1)
    assert np.isfinite(r2)
    # They should produce the same result as without context
    r1b = reward_log_return(100_000, 100_100)
    r2b = reward_asymmetric_drawdown_penalty(100_000, 100_100)
    assert abs(r1 - r1b) < 1e-10
    assert abs(r2 - r2b) < 1e-10
    print("PASS: existing functions accept context kwarg")


# ------------------------------------------------------------------ #
# 11. No-context fallback to plain log return
# ------------------------------------------------------------------ #
def test_no_context_fallback():
    old_eq, new_eq = 100_000.0, 100_100.0
    r = reward_stop_aware_drawdown(old_eq, new_eq)
    expected = np.log(new_eq / old_eq)
    assert abs(r - expected) < 1e-10
    print("PASS: no-context fallback to log return")


# ------------------------------------------------------------------ #
# 12. Combined penalties stack correctly
# ------------------------------------------------------------------ #
def test_combined_penalties_stack():
    old_eq, new_eq = 100_000.0, 99_800.0
    base = np.log(new_eq / old_eq)

    # Only stop
    r_stop = reward_stop_aware_drawdown(
        old_eq, new_eq,
        context=_ctx(stop_triggered=True, recent_stop_count=2))

    # Stop + re-entry
    r_both = reward_stop_aware_drawdown(
        old_eq, new_eq,
        context=_ctx(stop_triggered=True, recent_stop_count=2,
                     same_side_reentry=True))

    # Stop + re-entry + drawdown
    r_all = reward_stop_aware_drawdown(
        old_eq, new_eq,
        context=_ctx(stop_triggered=True, recent_stop_count=2,
                     same_side_reentry=True, current_drawdown=0.15))

    assert r_stop < base, "stop penalty should reduce reward"
    assert r_both < r_stop, "re-entry penalty should stack on top of stop"
    assert r_all < r_both, "drawdown penalty should stack further"
    print("PASS: penalties stack correctly")


# ------------------------------------------------------------------ #
# 13. Decomposition contract: all three fns return (scalar, dict) with
#     stable keys. Checked here so CSV headers downstream don't shift.
# ------------------------------------------------------------------ #
def test_reward_components_shape():
    old_eq, new_eq = 100_000.0, 100_100.0

    r1, c1 = _reward_log_return(old_eq, new_eq)
    assert isinstance(r1, float)
    assert set(c1.keys()) == {"base_log_return", "bankruptcy"}

    r2, c2 = _reward_asymmetric_drawdown_penalty(old_eq, new_eq)
    assert isinstance(r2, float)
    assert set(c2.keys()) == {"base_log_return", "downside_extra_penalty", "bankruptcy"}

    r3, c3 = _reward_stop_aware_drawdown(old_eq, new_eq, context=_ctx(
        stop_triggered=True, recent_stop_count=2,
        same_side_reentry=True, current_drawdown=0.15))
    assert isinstance(r3, float)
    assert set(c3.keys()) == {
        "base_log_return", "stop_penalty", "reentry_penalty",
        "drawdown_penalty", "bankruptcy",
    }
    assert c3["stop_penalty"] > 0
    assert c3["reentry_penalty"] > 0
    assert c3["drawdown_penalty"] > 0

    # Bankruptcy branches must still return a tuple with all keys present.
    r_b, c_b = _reward_stop_aware_drawdown(100_000, 0, context=_ctx())
    assert r_b == -1.0
    assert set(c_b.keys()) == {
        "base_log_return", "stop_penalty", "reentry_penalty",
        "drawdown_penalty", "bankruptcy",
    }
    assert c_b["bankruptcy"] == 1.0

    # No-context fallback for stop_aware still returns full key set.
    r_nc, c_nc = _reward_stop_aware_drawdown(old_eq, new_eq)
    assert set(c_nc.keys()) == {
        "base_log_return", "stop_penalty", "reentry_penalty",
        "drawdown_penalty", "bankruptcy",
    }
    print("PASS: reward functions return (scalar, components) with stable keys")


if __name__ == "__main__":
    test_base_return_no_penalties()
    test_stop_triggered_single()
    test_stop_cluster_stronger()
    test_same_side_reentry_penalty()
    test_reentry_not_penalized_without_flag()
    test_drawdown_below_threshold()
    test_drawdown_above_threshold()
    test_reward_finite()
    test_bankruptcy()
    test_existing_functions_accept_context()
    test_no_context_fallback()
    test_combined_penalties_stack()
    test_reward_components_shape()
    print("\n=== ALL reward_stop_aware_drawdown TESTS PASSED ===")
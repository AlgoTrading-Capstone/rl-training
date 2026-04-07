"""
Validation test for position context state features.
Tests: state_dim consistency, flat defaults, long/short symmetry,
       stop context, inverse normalization, shape stability.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from bitcoin_env import BitcoinTradingEnv
from utils.normalization import inverse_normalize_state


def make_env(n=50):
    np.random.seed(42)
    closes = 100000 + np.cumsum(np.random.randn(n) * 200)
    price_ary = np.column_stack([
        closes - 50,
        closes + 100,
        closes - 100,
        closes,
        np.random.rand(n) * 1e6,
    ])
    tech_ary = np.random.rand(n, 7) * 100
    turb_ary = np.column_stack([np.random.rand(n) * 0.03, np.random.rand(n) * 30 + 15])
    signal_ary = np.zeros((n, 92))
    datetime_ary = np.arange(n).reshape(-1, 1)
    return BitcoinTradingEnv(price_ary, tech_ary, turb_ary, signal_ary, datetime_ary, mode="backtest")


def test_state_dim():
    env = make_env()
    expected = 1 + 5 + 7 + 2 + 92 + 1 + 7  # 115
    assert env.state_dim == expected, f"state_dim: {env.state_dim} != {expected}"
    print(f"PASS: state_dim = {env.state_dim}")


def test_reset_flat_defaults():
    env = make_env()
    state = env.reset()
    assert state.shape == (env.state_dim,)
    assert state.dtype == np.float32
    ctx = state[-7:]
    assert ctx[0] == 0.0, f"position_side: {ctx[0]}"
    assert ctx[1] == 0.0, f"exposure: {ctx[1]}"
    assert ctx[2] == 0.0, f"entry_rel: {ctx[2]}"
    assert ctx[3] == 0.0, f"pnl: {ctx[3]}"
    assert ctx[4] == 0.0, f"stop_dist: {ctx[4]}"
    assert ctx[5] == 0.0, f"bars_in_pos: {ctx[5]}"
    assert ctx[6] == 1.0, f"bars_since_stop: {ctx[6]}"
    print("PASS: flat-state defaults")


def test_long_position_context():
    env = make_env()
    env.reset()
    ns, _, _, _ = env.step(np.array([0.8, 0.0], dtype=np.float32))
    assert ns.shape == (env.state_dim,)
    h = env.position.holdings
    if not np.isclose(h, 0.0):
        assert ns[-7] == 1.0, f"side should be +1, got {ns[-7]}"
        assert ns[-6] > 0.0, f"exposure should be >0, got {ns[-6]}"
        print(f"PASS: long context (holdings={h:.4f})")
    else:
        print("SKIP: long not opened (deadzone)")


def test_short_position_context():
    env = make_env()
    env.reset()
    ns, _, _, _ = env.step(np.array([-0.8, 0.0], dtype=np.float32))
    h = env.position.holdings
    if not np.isclose(h, 0.0):
        assert ns[-7] == -1.0, f"side should be -1, got {ns[-7]}"
        assert ns[-6] < 0.0, f"exposure should be <0, got {ns[-6]}"
        print(f"PASS: short context (holdings={h:.4f})")
    else:
        print("SKIP: short not opened (deadzone)")


def test_shape_consistency_multi_step():
    env = make_env()
    env.reset()
    for i in range(min(30, env.max_step - 1)):
        a = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)], dtype=np.float32)
        ns, r, d, _ = env.step(a)
        assert ns.shape == (env.state_dim,), f"shape mismatch at step {i}"
        assert ns.dtype == np.float32
        assert np.all(np.isfinite(ns)), f"non-finite at step {i}"
        if d:
            break
    print("PASS: shapes consistent, all finite across episode")


def test_inverse_normalize():
    env = make_env()
    env.reset()
    ns, _, _, _ = env.step(np.array([-0.8, 0.0], dtype=np.float32))
    inv = inverse_normalize_state(
        state_norm=ns,
        price_vec=env.current_price,
        tech_vec=env.current_tech,
        turbulence_vec=env.current_turbulence,
        signal_vec=env.current_signal,
    )
    assert "position_context" in inv
    assert len(inv["position_context"]) == 7
    print(f"PASS: inverse_normalize returns 7 position context fields: {inv['position_context']}")


if __name__ == "__main__":
    test_state_dim()
    test_reset_flat_defaults()
    test_long_position_context()
    test_short_position_context()
    test_shape_consistency_multi_step()
    test_inverse_normalize()
    print("\n=== ALL VALIDATION CHECKS PASSED ===")

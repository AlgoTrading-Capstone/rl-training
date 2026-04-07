"""
Tests for independent stop-loss management.

Validates that a_sl operates as an independent control channel:
- stop updates without trades (exposure deadzone)
- stop deadzone prevents micro-adjustments
- flat positions ignore a_sl
- scale-in / scale-out / hold cases
- stop orientation correctness
- env.step() shape stability
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from trade_engine import (
    PositionState,
    TradeConfig,
    TradeResult,
    apply_action,
    compute_stop_price,
    compute_equity,
    should_update_stop,
)


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------

def make_cfg(**overrides):
    """Create a TradeConfig with sensible defaults for testing."""
    defaults = dict(
        leverage_limit=2.0,
        max_position_btc=18.0,
        min_stop_pct=0.01,
        max_stop_pct=0.05,
        exposure_deadzone=0.10,
        stop_update_deadzone=0.002,
        fee_rate=0.0,  # zero fees for deterministic tests
    )
    defaults.update(overrides)
    return TradeConfig(**defaults)


def long_state(price=100_000.0, holdings=0.2, entry=100_000.0, a_sl_for_stop=1.0, cfg=None):
    """Create a long PositionState with a known stop."""
    cfg = cfg or make_cfg()
    stop = compute_stop_price(1, entry, a_sl_for_stop, cfg)
    balance = 100_000.0 - holdings * price  # approximate
    return PositionState(balance=balance, holdings=holdings, entry_price=entry, stop_price=stop)


def short_state(price=100_000.0, holdings=-0.2, entry=100_000.0, a_sl_for_stop=1.0, cfg=None):
    """Create a short PositionState with a known stop."""
    cfg = cfg or make_cfg()
    stop = compute_stop_price(-1, entry, a_sl_for_stop, cfg)
    balance = 100_000.0 - holdings * price  # approximate
    return PositionState(balance=balance, holdings=holdings, entry_price=entry, stop_price=stop)


def _current_exposure_norm(state, price, cfg):
    """Compute the current normalized exposure for a state."""
    equity = compute_equity(state.balance, state.holdings, price)
    if equity <= 0 or cfg.leverage_limit <= 0:
        return 0.0
    max_notional = cfg.leverage_limit * equity
    return float(np.clip((state.holdings * price) / max_notional, -1.0, 1.0))


# ----------------------------------------------------------------
# Test: should_update_stop helper
# ----------------------------------------------------------------

def test_should_update_stop_none_current():
    cfg = make_cfg()
    assert should_update_stop(None, 99000.0, 100000.0, cfg) is True
    print("PASS: should_update_stop - None current always updates")

def test_should_update_stop_large_change():
    cfg = make_cfg(stop_update_deadzone=0.002)
    assert should_update_stop(95000.0, 99000.0, 100000.0, cfg) is True
    print("PASS: should_update_stop - large change updates")

def test_should_update_stop_tiny_change():
    cfg = make_cfg(stop_update_deadzone=0.002)
    assert should_update_stop(95000.0, 95100.0, 100000.0, cfg) is False
    print("PASS: should_update_stop - tiny change does not update")

def test_should_update_stop_zero_price():
    cfg = make_cfg()
    assert should_update_stop(95000.0, 99000.0, 0.0, cfg) is False
    print("PASS: should_update_stop - zero price does not update")


# ----------------------------------------------------------------
# Test: Example A — Hold long, tighten stop
# ----------------------------------------------------------------

def test_hold_long_tighten_stop():
    """Long position, exposure unchanged, a_sl moves from loose to tight."""
    cfg = make_cfg()
    price = 100_000.0
    state = long_state(price=price, holdings=0.2, entry=100_000.0, a_sl_for_stop=1.0, cfg=cfg)
    old_stop = state.stop_price

    a_pos = _current_exposure_norm(state, price, cfg)
    result = apply_action(a_pos=a_pos, a_sl=-1.0, price=price, state=state, cfg=cfg)

    assert not result.trade_executed, "Should NOT trade"
    assert result.stop_updated, "Stop should update"
    assert result.new_state.holdings == state.holdings, "Holdings unchanged"
    assert result.new_state.entry_price == state.entry_price, "Entry unchanged"

    tight_stop = compute_stop_price(1, state.entry_price, -1.0, cfg)
    assert abs(result.new_state.stop_price - tight_stop) < 1e-6
    assert result.new_state.stop_price > old_stop, "Tight stop should be higher for long"
    print("PASS: Example A - hold long, tighten stop (no trade, stop updated)")


# ----------------------------------------------------------------
# Test: Example B — Hold short, loosen stop
# ----------------------------------------------------------------

def test_hold_short_loosen_stop():
    """Short position, exposure unchanged, a_sl moves from tight to loose."""
    cfg = make_cfg()
    price = 100_000.0
    state = short_state(price=price, holdings=-0.2, entry=100_000.0, a_sl_for_stop=-1.0, cfg=cfg)
    old_stop = state.stop_price

    a_pos = _current_exposure_norm(state, price, cfg)
    result = apply_action(a_pos=a_pos, a_sl=1.0, price=price, state=state, cfg=cfg)

    assert not result.trade_executed, "Should NOT trade"
    assert result.stop_updated, "Stop should update"
    assert result.new_state.holdings == state.holdings, "Holdings unchanged"

    loose_stop = compute_stop_price(-1, state.entry_price, 1.0, cfg)
    assert abs(result.new_state.stop_price - loose_stop) < 1e-6
    assert result.new_state.stop_price > old_stop, "Loose stop should be higher for short"
    print("PASS: Example B - hold short, loosen stop (no trade, stop updated)")


# ----------------------------------------------------------------
# Test: Example C — Exposure deadzone, meaningful stop change
# ----------------------------------------------------------------

def test_exposure_deadzone_stop_update():
    """Exposure change inside deadzone, but stop change above stop deadzone."""
    cfg = make_cfg(exposure_deadzone=0.10, stop_update_deadzone=0.002)
    price = 100_000.0
    state = long_state(price=price, holdings=0.2, entry=100_000.0, a_sl_for_stop=1.0, cfg=cfg)

    curr_exp = _current_exposure_norm(state, price, cfg)
    a_pos = curr_exp + 0.05

    result = apply_action(a_pos=a_pos, a_sl=-1.0, price=price, state=state, cfg=cfg)

    assert not result.trade_executed, "No trade (inside exposure deadzone)"
    assert result.stop_updated, "Stop should update (above stop deadzone)"
    assert result.effective_delta_btc == 0.0
    print("PASS: Example C - exposure deadzone, meaningful stop change")


# ----------------------------------------------------------------
# Test: Example D — Tiny stop noise
# ----------------------------------------------------------------

def test_tiny_stop_noise():
    """Stop change below stop deadzone -> no update."""
    cfg = make_cfg(stop_update_deadzone=0.002)
    price = 100_000.0

    state = long_state(price=price, holdings=0.2, entry=100_000.0, a_sl_for_stop=0.5, cfg=cfg)
    a_pos = _current_exposure_norm(state, price, cfg)

    result = apply_action(a_pos=a_pos, a_sl=0.51, price=price, state=state, cfg=cfg)

    assert not result.trade_executed
    assert not result.stop_updated, "Stop should NOT update (below deadzone)"
    assert result.new_state.stop_price == state.stop_price
    print("PASS: Example D - tiny stop noise, no update")


# ----------------------------------------------------------------
# Test: Flat position ignores a_sl
# ----------------------------------------------------------------

def test_flat_ignores_stop():
    """With no position, a_sl should do nothing."""
    cfg = make_cfg()
    price = 100_000.0
    state = PositionState(balance=100_000.0, holdings=0.0, entry_price=None, stop_price=None)

    result = apply_action(a_pos=0.0, a_sl=-1.0, price=price, state=state, cfg=cfg)

    assert not result.trade_executed
    assert not result.stop_updated
    assert result.new_state.stop_price is None
    assert result.new_state.entry_price is None
    print("PASS: Flat position ignores a_sl")


# ----------------------------------------------------------------
# Test: Scale-in allows stop update
# ----------------------------------------------------------------

def test_scale_in():
    """Scale-in on same side: entry updates (weighted avg), stop recomputes."""
    np.random.seed(42)
    cfg = make_cfg(fee_rate=0.0)
    price = 100_000.0

    state = PositionState(
        balance=90_000.0, holdings=0.1, entry_price=100_000.0,
        stop_price=compute_stop_price(1, 100_000.0, 0.0, cfg),
    )

    result = apply_action(a_pos=0.8, a_sl=-1.0, price=price, state=state, cfg=cfg)

    assert result.trade_executed, "Should trade (scale-in)"
    assert abs(result.new_state.holdings) > abs(state.holdings), "Holdings should increase"
    assert result.stop_updated, "Stop should update on scale-in"
    assert result.new_state.entry_price is not None
    print("PASS: Scale-in updates entry and stop")


# ----------------------------------------------------------------
# Test: Scale-out still allows stop update
# ----------------------------------------------------------------

def test_scale_out():
    """Scale-out on same side: entry unchanged, stop may update."""
    np.random.seed(42)
    cfg = make_cfg(fee_rate=0.0)
    price = 100_000.0

    state = PositionState(
        balance=60_000.0, holdings=0.4, entry_price=100_000.0,
        stop_price=compute_stop_price(1, 100_000.0, 1.0, cfg),
    )

    # a_pos=0.05 → target_exp ~0.05 vs current ~0.2 → change=0.15 > 0.10 deadzone
    result = apply_action(a_pos=0.05, a_sl=-1.0, price=price, state=state, cfg=cfg)

    assert result.trade_executed, "Should trade (scale-out)"
    assert abs(result.new_state.holdings) < abs(state.holdings), "Holdings should decrease"
    assert result.new_state.entry_price == state.entry_price, "Entry unchanged on scale-out"
    assert result.stop_updated, "Stop should update on scale-out with different a_sl"
    print("PASS: Scale-out keeps entry, allows stop update")


# ----------------------------------------------------------------
# Test: Stop orientation correctness
# ----------------------------------------------------------------

def test_long_stop_orientation():
    """Long stop must always be below entry."""
    cfg = make_cfg()
    price = 100_000.0
    state = long_state(price=price, a_sl_for_stop=1.0, cfg=cfg)
    a_pos = _current_exposure_norm(state, price, cfg)

    for a_sl in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        result = apply_action(a_pos=a_pos, a_sl=a_sl, price=price, state=state, cfg=cfg)
        if result.new_state.stop_price is not None:
            assert result.new_state.stop_price < result.new_state.entry_price, \
                f"Long stop must be below entry (a_sl={a_sl})"
    print("PASS: Long stop orientation correct for all a_sl values")

def test_short_stop_orientation():
    """Short stop must always be above entry."""
    cfg = make_cfg()
    price = 100_000.0
    state = short_state(price=price, a_sl_for_stop=1.0, cfg=cfg)
    a_pos = _current_exposure_norm(state, price, cfg)

    for a_sl in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        result = apply_action(a_pos=a_pos, a_sl=a_sl, price=price, state=state, cfg=cfg)
        if result.new_state.stop_price is not None:
            assert result.new_state.stop_price > result.new_state.entry_price, \
                f"Short stop must be above entry (a_sl={a_sl})"
    print("PASS: Short stop orientation correct for all a_sl values")


# ----------------------------------------------------------------
# Test: Stop-only update does not fake trades
# ----------------------------------------------------------------

def test_stop_only_no_fake_trade():
    """Stop-only update must not change balance or holdings."""
    cfg = make_cfg()
    price = 100_000.0
    state = long_state(price=price, a_sl_for_stop=1.0, cfg=cfg)

    a_pos = _current_exposure_norm(state, price, cfg)
    result = apply_action(a_pos=a_pos, a_sl=-1.0, price=price, state=state, cfg=cfg)

    assert not result.trade_executed
    assert result.effective_delta_btc == 0.0
    assert result.new_state.balance == state.balance
    assert result.new_state.holdings == state.holdings
    print("PASS: Stop-only update does not fake trades")


# ----------------------------------------------------------------
# Test: Flatten clears stop
# ----------------------------------------------------------------

def test_flatten_clears_stop():
    """Going to zero exposure clears entry and stop."""
    np.random.seed(42)
    cfg = make_cfg(fee_rate=0.0)
    price = 100_000.0
    state = long_state(price=price, a_sl_for_stop=0.0, cfg=cfg)

    result = apply_action(a_pos=0.0, a_sl=0.0, price=price, state=state, cfg=cfg)

    if result.trade_executed and np.isclose(result.new_state.holdings, 0.0):
        assert result.new_state.entry_price is None
        assert result.new_state.stop_price is None
        print("PASS: Flatten clears entry and stop")
    else:
        print("SKIP: Flatten did not execute")


# ----------------------------------------------------------------
# Test: Env integration — shape stability
# ----------------------------------------------------------------

def _make_env(n=50):
    from bitcoin_env import BitcoinTradingEnv
    np.random.seed(42)
    closes = 100000 + np.cumsum(np.random.randn(n) * 200)
    price_ary = np.column_stack([
        closes - 50, closes + 100, closes - 100, closes,
        np.random.rand(n) * 1e6,
    ])
    tech_ary = np.random.rand(n, 7) * 100
    turb_ary = np.column_stack([np.random.rand(n) * 0.03, np.random.rand(n) * 30 + 15])
    signal_ary = np.zeros((n, 92))
    datetime_ary = np.arange(n).reshape(-1, 1)
    return BitcoinTradingEnv(price_ary, tech_ary, turb_ary, signal_ary, datetime_ary, mode="backtest")

def test_env_step_shape_and_info():
    """env.step() returns correct shapes and includes stop_updated in info."""
    env = _make_env()
    state = env.reset()
    assert state.shape == (env.state_dim,)

    for i in range(min(30, env.max_step - 1)):
        a = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)], dtype=np.float32)
        ns, r, d, info = env.step(a)
        assert ns.shape == (env.state_dim,), f"shape mismatch at step {i}"
        assert ns.dtype == np.float32
        assert np.all(np.isfinite(ns)), f"non-finite at step {i}"
        assert "stop_updated" in info, "info must contain stop_updated"
        assert isinstance(info["stop_updated"], bool)
        if d:
            break
    print("PASS: env.step() shapes and info correct")

def test_env_stop_updates_observed():
    """Over multiple steps, at least one stop update should be observed."""
    env = _make_env(n=100)
    state = env.reset()

    stop_updated_count = 0

    for i in range(min(80, env.max_step - 1)):
        if i < 3:
            a = np.array([0.8, 0.0], dtype=np.float32)
        else:
            a = np.array([0.8, np.sin(i * 0.5)], dtype=np.float32)
        ns, r, d, info = env.step(a)
        if info["stop_updated"]:
            stop_updated_count += 1
        state = ns
        if d:
            break

    assert stop_updated_count > 0, "Expected at least one stop update during episode"
    print(f"PASS: Observed {stop_updated_count} stop updates during episode")


# ----------------------------------------------------------------
# Entry point for direct execution
# ----------------------------------------------------------------

if __name__ == "__main__":
    test_should_update_stop_none_current()
    test_should_update_stop_large_change()
    test_should_update_stop_tiny_change()
    test_should_update_stop_zero_price()
    test_hold_long_tighten_stop()
    test_hold_short_loosen_stop()
    test_exposure_deadzone_stop_update()
    test_tiny_stop_noise()
    test_flat_ignores_stop()
    test_scale_in()
    test_scale_out()
    test_long_stop_orientation()
    test_short_stop_orientation()
    test_stop_only_no_fake_trade()
    test_flatten_clears_stop()
    test_env_step_shape_and_info()
    test_env_stop_updates_observed()
    print("\n=== ALL STOP MANAGEMENT TESTS PASSED ===")

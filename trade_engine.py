"""
Trade engine for the Bitcoin RL environment.

This module encapsulates all trading logic:
- mapping actions to target exposure (long/short/flat)
- enforcing leverage and debt constraints
- computing position changes (delta BTC)
- applying transaction fees
- managing stop-loss levels
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ------------------------------------------------------------
# Core state & configuration dataclasses
# ------------------------------------------------------------

@dataclass
class PositionState:
    """
    Holds the current state of the trading position.
    """
    balance: float                       # USD balance (can be negative within MAX_DEBT limit)
    holdings: float                      # BTC position (positive = long, negative = short)
    entry_price: Optional[float] = None  # Entry price for the current (non-flat) position
    stop_price: Optional[float] = None   # Active stop-loss level


@dataclass
class TradeConfig:
    """
    Configuration parameters for the trade engine.
    All fields are meant to be populated from the global config.py.
    """
    leverage_limit: float
    max_debt: float
    max_position_btc: Optional[float]
    min_stop_pct: float
    max_stop_pct: float
    exposure_deadzone: float
    fee_rate: float


@dataclass
class TradeResult:
    """
    Result of applying a trading action.
    """
    new_state: PositionState
    effective_delta_btc: float  # how many BTC were actually traded (can be 0.0)
    trade_executed: bool        # True if any trade (market buy/sell) occurred


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def compute_equity(balance: float, holdings: float, price: float) -> float:
    """
    Compute total equity: cash + position value (Net Asset Value).
    """
    return balance + holdings * price


def clip_exposure(target_exposure_norm: float) -> float:
    """
    Ensure the requested normalized exposure stays within [-1, +1].

    RL agents may occasionally output values slightly outside this range
    due to noise or network instability. Clipping prevents invalid or
    unsafe exposure requests and keeps the action space well-defined.
    """
    return float(np.clip(target_exposure_norm, -1.0, 1.0))


def target_notional_from_action(a_pos: float,
                                equity: float,
                                cfg: TradeConfig) -> float:
    """
    Map a_pos ∈ [-1, +1] to a target notional exposure in USD.
    """
    if equity <= 0.0:
        return 0.0

    max_notional = cfg.leverage_limit * equity
    clipped_a_pos = clip_exposure(a_pos)
    target_notional = clipped_a_pos * max_notional
    return target_notional


def limit_btc_position(target_notional: float,
                       price: float,
                       cfg: TradeConfig) -> float:
    """
    Enforce a hard cap on the BTC position size (in units of BTC).

    Converts the desired notional exposure (in USD) into a target
    BTC quantity, clips it to the allowed range
    [-max_position_btc, +max_position_btc], and converts it back
    to notional form.

    This ensures the agent cannot open oversized positions even
    if leverage would allow it.
    """
    if cfg.max_position_btc is None or price <= 0.0:
        return target_notional

    target_btc = target_notional / price
    capped_btc = float(np.clip(target_btc, -cfg.max_position_btc, cfg.max_position_btc))
    return capped_btc * price


def compute_delta_btc(current_holdings: float,
                      target_notional: float,
                      price: float,
                      cfg: TradeConfig) -> float:
    """
    Compute how many BTC we need to buy/sell to move from current_holdings
    to the target position implied by target_notional (after caps).
    """
    if price <= 0.0:
        return 0.0

    # Convert target notional exposure to BTC quantity
    target_btc = target_notional / price

    # Delta = how many BTC must be executed
    delta_btc = target_btc - current_holdings
    return float(delta_btc)


# ------------------------------------------------------------
# Trade execution with fees and debt constraint
# ------------------------------------------------------------

def apply_trade(balance: float,
                holdings: float,
                price: float,
                delta_btc: float,
                cfg: TradeConfig) -> Tuple[float, float, float]:
    """
    Apply a market trade (BUY/SELL) to balance and holdings with fees
    and debt constraints.

    Returns:
        new_balance, new_holdings, effective_delta_btc
    """
    if np.isclose(delta_btc, 0.0) or price <= 0.0:
        # No trade needed or invalid price
        return balance, holdings, 0.0

    # Proposed trade notional (USD)
    trade_notional = delta_btc * price
    fee = abs(trade_notional) * cfg.fee_rate

    # Tentative new balance
    new_balance = balance - trade_notional - fee

    # Enforce max debt: only relevant when trade_notional > 0 (buying reduces balance)
    if cfg.max_debt is not None and new_balance < -cfg.max_debt and trade_notional > 0.0:
        # Compute maximum affordable notional given debt constraint:
        # balance - N * (1 + fee_rate) >= -max_debt
        # N <= (balance + max_debt) / (1 + fee_rate)
        max_affordable = (balance + cfg.max_debt) / (1.0 + cfg.fee_rate)

        if max_affordable <= 0.0:
            # Cannot buy anything without violating debt constraint
            return balance, holdings, 0.0

        # Scale down the trade
        scale = max_affordable / trade_notional  # trade_notional > 0 here
        trade_notional = max_affordable
        delta_btc = delta_btc * scale
        fee = abs(trade_notional) * cfg.fee_rate
        new_balance = balance - trade_notional - fee

    # Update holdings
    new_holdings = holdings + delta_btc

    return float(new_balance), float(new_holdings), float(delta_btc)


# ------------------------------------------------------------
# Stop-loss helper
# ------------------------------------------------------------

def compute_stop_price(side: int,
                       entry_price: float,
                       a_sl: float,
                       cfg: TradeConfig) -> float:
    """
    Compute a stop-loss price based on:
    - position side (+1 for long, -1 for short)
    - entry price
    - a_sl ∈ [0, 1] controlling how tight/loose the stop is
    """
    # Safety clamp on a_sl
    a_sl = float(np.clip(a_sl, 0.0, 1.0))

    stop_pct = cfg.min_stop_pct + a_sl * (cfg.max_stop_pct - cfg.min_stop_pct)

    if side > 0:
        # Long position: stop below entry
        return entry_price * (1.0 - stop_pct)
    elif side < 0:
        # Short position: stop above entry
        return entry_price * (1.0 + stop_pct)
    else:
        # No position
        return entry_price


# ------------------------------------------------------------
# Main entry point: apply_action
# ------------------------------------------------------------

def apply_action(a_pos: float,
                 a_sl: float,
                 price: float,
                 state: PositionState,
                 cfg: TradeConfig) -> TradeResult:
    """
    High-level entry point for the environment.

    Given the current PositionState and an action [a_pos, a_sl],
    compute and apply the trade, update stop-loss, and return the new state.

    Steps:
    - compute equity
    - derive target notional from a_pos (respect leverage)
    - apply BTC position cap
    - apply exposure deadzone
    - compute delta_btc and execute trade with fees & debt constraint
    - update entry_price & stop_price (on new / flipped positions)
    """

    balance = state.balance
    holdings = state.holdings

    # If price or equity are invalid, freeze trading
    if price <= 0.0:
        return TradeResult(
            new_state=state,
            effective_delta_btc=0.0,
            trade_executed=False,
        )

    equity = compute_equity(balance, holdings, price)

    if equity <= 0.0:
        # Account effectively blown up → no more trading
        return TradeResult(
            new_state=state,
            effective_delta_btc=0.0,
            trade_executed=False,
        )

    # --------------------------------------------------------
    # 1. Map action to target notional exposure (USD)
    # --------------------------------------------------------
    max_notional = cfg.leverage_limit * equity

    # Current normalized exposure (clipped to [-1, 1])
    current_notional = holdings * price
    current_exposure_norm = 0.0 if max_notional <= 0.0 else current_notional / max_notional
    current_exposure_norm = float(np.clip(current_exposure_norm, -1.0, 1.0))

    # Target notional from action (respect leverage and BTC cap)
    target_notional = target_notional_from_action(a_pos, equity, cfg)
    target_notional = limit_btc_position(target_notional, price, cfg)

    target_exposure_norm = 0.0 if max_notional <= 0.0 else target_notional / max_notional
    target_exposure_norm = float(np.clip(target_exposure_norm, -1.0, 1.0))

    # --------------------------------------------------------
    # 2. Deadzone: small exposure changes - HOLD
    # --------------------------------------------------------
    exposure_change = abs(target_exposure_norm - current_exposure_norm)
    if exposure_change < cfg.exposure_deadzone:
        # Treat as HOLD (no trade, keep existing stop/entry)
        return TradeResult(
            new_state=state,
            effective_delta_btc=0.0,
            trade_executed=False,
        )

    # --------------------------------------------------------
    # 3. Compute BTC delta and execute trade
    # --------------------------------------------------------
    delta_btc = compute_delta_btc(
        current_holdings=holdings,
        target_notional=target_notional,
        price=price,
        cfg=cfg,
    )

    new_balance, new_holdings, effective_delta_btc = apply_trade(
        balance=balance,
        holdings=holdings,
        price=price,
        delta_btc=delta_btc,
        cfg=cfg,
    )

    trade_executed = not np.isclose(effective_delta_btc, 0.0)

    # --------------------------------------------------------
    # 4. Update entry price & stop-loss
    # --------------------------------------------------------
    old_entry_price = state.entry_price
    old_stop_price = state.stop_price

    new_entry_price = old_entry_price
    new_stop_price = old_stop_price

    # If we flattened the position - clear entry & stop
    if np.isclose(new_holdings, 0.0):
        new_entry_price = None
        new_stop_price = None
    else:
        # Determine old/new side (+1 long, -1 short, 0 flat)
        old_side = 0
        if not np.isclose(holdings, 0.0):
            old_side = 1 if holdings > 0.0 else -1

        new_side = 1 if new_holdings > 0.0 else -1

        # New position opened from flat
        if np.isclose(holdings, 0.0) and not np.isclose(new_holdings, 0.0):
            new_entry_price = price
            new_stop_price = compute_stop_price(new_side, new_entry_price, a_sl, cfg)

        # Side flipped (long to short or short to long)
        elif old_side != 0 and old_side != new_side:
            new_entry_price = price
            new_stop_price = compute_stop_price(new_side, new_entry_price, a_sl, cfg)

    new_state = PositionState(
        balance=new_balance,
        holdings=new_holdings,
        entry_price=new_entry_price,
        stop_price=new_stop_price,
    )

    return TradeResult(
        new_state=new_state,
        effective_delta_btc=effective_delta_btc,
        trade_executed=trade_executed,
    )
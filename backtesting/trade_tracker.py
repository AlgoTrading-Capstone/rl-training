"""
Trade tracker for backtesting.

Tracks position lifecycle trades (open -> close) and writes trades.csv.

Supports:
- STOP exits with correct execution price (equity_price)
- FLIP handling (LONG->SHORT / SHORT->LONG)
- END finalization (mark-to-market) when force_close is disabled

PnL methods:
- EQUITY: close_equity - open_equity (includes fees/slippage reflected in env equity)
- PRICE : (close_price - open_price) * size (approx; used for FLIP)
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np


class TradeTracker:
    """
    Tracks a single active position-trade and logs completed trades.
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.file_path = self.out_dir / "trades.csv"
        self._file = None
        self._writer = None

        self._trade_id_gen = itertools.count(1)

        # Active trade state
        self.active_trade: Optional[Dict[str, Any]] = None

    # ----------------------------------------------------
    # File lifecycle
    # ----------------------------------------------------
    def open(self) -> None:
        self._file = self.file_path.open("w", newline="", encoding="utf-8")
        self._writer = None

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

    # ----------------------------------------------------
    # Trade lifecycle hooks
    # ----------------------------------------------------
    def on_step(
        self,
        *,
        timestamp,
        trade_price: float,
        equity_after: float,
        holdings_before: float,
        holdings_after: float,
        stop_price: Optional[float],
        stop_triggered: bool,
        forced_close: bool = False,
    ) -> None:
        """
        Call once per environment step AFTER env.step().

        Parameters
        ----------
        trade_price:
            The effective price for this step (use env info equity_price).
            If stop_triggered=True -> stop execution price.
            Else -> candle close.

        equity_after:
            env.info["equity"] after this step (valued at trade_price).

        holdings_before / holdings_after:
            Holdings before and after env.step(), used to detect FLIP and size at close.

        forced_close:
            If True and holdings_after==0, close_reason will be FORCE_CLOSE.
        """

        hb = float(holdings_before)
        ha = float(holdings_after)

        # Normalize near-zero holdings
        if np.isclose(hb, 0.0):
            hb = 0.0
        if np.isclose(ha, 0.0):
            ha = 0.0

        # ------------------------------------------------
        # No active trade -> maybe open one
        # ------------------------------------------------
        if self.active_trade is None:
            if ha != 0.0:
                self._open_trade(
                    timestamp=timestamp,
                    price=float(trade_price),
                    equity=float(equity_after),
                    holdings=float(ha),
                    stop_price=stop_price,
                )
            return

        # Update latest stop price (for reporting)
        self.active_trade["stop_price"] = stop_price

        # ------------------------------------------------
        # Detect FLIP (side changed without going flat)
        # ------------------------------------------------
        if hb != 0.0 and ha != 0.0 and np.sign(hb) != np.sign(ha):
            # Close old trade as FLIP using PRICE method (approx)
            self._close_trade(
                timestamp=timestamp,
                price=float(trade_price),
                close_reason="FLIP",
                close_size_btc=abs(float(hb)),
                close_equity=None,  # cannot isolate cleanly on flip
            )

            # Open new trade immediately (same timestamp/price)
            self._open_trade(
                timestamp=timestamp,
                price=float(trade_price),
                equity=float(equity_after),
                holdings=float(ha),
                stop_price=stop_price,
            )
            return

        # ------------------------------------------------
        # STOP close (preferred over holdings==0)
        # ------------------------------------------------
        if stop_triggered:
            self._close_trade(
                timestamp=timestamp,
                price=float(trade_price),
                close_reason="STOP",
                close_size_btc=abs(float(hb)) if hb != 0.0 else abs(float(ha)),
                close_equity=float(equity_after),
            )
            return

        # ------------------------------------------------
        # Close when position becomes flat (agent close or forced close)
        # ------------------------------------------------
        if ha == 0.0 and hb != 0.0:
            close_reason = "FORCE_CLOSE" if forced_close else "AGENT"
            self._close_trade(
                timestamp=timestamp,
                price=float(trade_price),
                close_reason=close_reason,
                close_size_btc=abs(float(hb)),
                close_equity=float(equity_after),
            )
            return

        # Otherwise: still in the same trade (may have resized)

    def finalize_end(
        self,
        *,
        timestamp,
        trade_price: float,
        equity_mtm: float,
        holdings_after: float,
        stop_price: Optional[float],
    ) -> None:
        """
        Call at end of backtest if force_close is disabled.

        Closes any still-open trade using mark-to-market equity at last price.
        """
        if self.active_trade is None:
            return

        ha = float(holdings_after)
        if np.isclose(ha, 0.0):
            ha = 0.0

        if ha == 0.0:
            # Nothing to finalize
            self.active_trade = None
            return

        self.active_trade["stop_price"] = stop_price

        self._close_trade(
            timestamp=timestamp,
            price=float(trade_price),
            close_reason="END",
            close_size_btc=abs(float(ha)),
            close_equity=float(equity_mtm),
        )

    # ----------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------
    def _open_trade(
        self,
        *,
        timestamp,
        price: float,
        equity: float,
        holdings: float,
        stop_price: Optional[float],
    ) -> None:
        trade_id = next(self._trade_id_gen)

        self.active_trade = {
            "trade_id": trade_id,
            "side": "LONG" if holdings > 0 else "SHORT",
            "open_timestamp": timestamp,
            "open_price": float(price),
            "open_size_btc": abs(float(holdings)),
            "open_equity": float(equity),
            "stop_price": stop_price,
        }

    def _close_trade(
        self,
        *,
        timestamp,
        price: float,
        close_reason: str,
        close_size_btc: float,
        close_equity: Optional[float],
    ) -> None:
        trade = self.active_trade
        self.active_trade = None

        open_price = float(trade["open_price"])
        open_equity = float(trade["open_equity"])
        side = trade["side"]

        pnl_method = None
        pnl_usd = None
        pnl_pct = None

        if close_reason == "FLIP":
            # PRICE-based approx (fees/slippage not isolated)
            direction = 1.0 if side == "LONG" else -1.0
            pnl_usd = (float(price) - open_price) * float(close_size_btc) * direction
            denom = open_price * float(close_size_btc)
            pnl_pct = (pnl_usd / denom) * 100.0 if denom > 0 else 0.0
            pnl_method = "PRICE"
        else:
            # EQUITY-based (includes costs reflected in equity)
            if close_equity is None:
                close_equity = open_equity  # fallback
            pnl_usd = float(close_equity) - open_equity
            pnl_pct = (pnl_usd / open_equity) * 100.0 if open_equity > 0 else 0.0
            pnl_method = "EQUITY"

        row = {
            **trade,
            "close_timestamp": timestamp,
            "close_price": float(price),
            "close_reason": close_reason,
            "close_size_btc": float(close_size_btc),
            "close_equity": None if close_equity is None else float(close_equity),
            "pnl_method": pnl_method,
            "pnl_usd": float(pnl_usd),
            "pnl_pct": float(pnl_pct),
            "is_stop": bool(close_reason == "STOP"),
        }

        self._write_trade(row)

    def _write_trade(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=row.keys())
            self._writer.writeheader()

        self._writer.writerow(row)
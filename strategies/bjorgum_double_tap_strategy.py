"""Bjorgum Double Tap Strategy

Pine Script source: input/Bjorgum-Double-Tap.pine
Timeframe: 4h  |  Lookback: 300 bars (~1200 h)

Entry logic:
  Long  — Double Bottom detected: 5 successive pivots form a W-shape where
          the two bottoms are within tolerance of each other, and the close
          crosses ABOVE the neckline (trough between the two bottoms) for
          the first time on this bar.

  Short — Double Top detected: 5 successive pivots form an M-shape where
          the two tops are within tolerance of each other, and the close
          crosses BELOW the neckline (peak between the two tops) for the
          first time on this bar.

Exit / position management (stop/limit placement) is NOT handled here.
Configure that in the external execution layer.

Key parameters (defaults match Pine Script inputs):
    len      : 50   — pivot lookback length (bars since highest/lowest high/low)
    tol      : 15.0 — tolerance % of pattern height for top/bottom matching
    fib      : 100  — target extension % (informational only; not used for signal)
    atr_len  : 14   — ATR period
    lookback : 5    — swing lookback for ATR trailing stop computation
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import talib

from src.base_strategy import BaseStrategy, SignalType, StrategyRecommendation


class BjorgumDoubleTapStrategy(BaseStrategy):
    """Double top/bottom pattern detection with pivot-based entries.

    The strategy mirrors the PineScript logic:
    1. Track pivot highs and lows using rolling highest/lowest over `len` bars.
    2. Identify direction changes: direction flips to 1 (bearish swing high set)
       when hbar == 0 (current bar is the highest bar in the window), and to -1
       (bullish swing low set) when lbar == 0.
    3. On each direction change, log the pivot price into a rolling list of up to 5
       recent pivots.  If the direction continues (no flip) but a new extreme is
       reached, update the most recent pivot in place.
    4. Double Top (SHORT signal):
         pivots[-5...-1] = [low1, high1, neckline, high2, ...] where
           - high2 is within tol% of high1 (relative to pattern height)
           - close crosses below neckline on this bar (was above on prior bar)
         → emit SHORT
    5. Double Bottom (LONG signal):
         pivots[-5...-1] = [high1, low1, neckline, low2, ...] where
           - low2 is within tol% of low1 (relative to pattern height)
           - close crosses above neckline on this bar (was below on prior bar)
         → emit LONG
    6. Otherwise emit HOLD.
    """

    # Minimum candles before signals can be evaluated.
    # Needs len (50) for pivot detection + ATR (14) warm-up + several extra pivots.
    MIN_BARS: int = 120

    def __init__(
        self,
        len_: int = 50,
        tol: float = 15.0,
        fib: float = 100.0,
        atr_len: int = 14,
        swing_lookback: int = 5,
    ):
        super().__init__(
            name="Bjorgum Double Tap",
            description=(
                "Double top/bottom pattern detection with pivot-based entries. "
                "Enters long on double bottom neckline breakout, short on double "
                "top neckline breakdown."
            ),
            timeframe="4h",
            lookback_hours=300 * 4,  # 300 bars × 4 h
        )
        self.len_ = len_
        self.tol = tol
        self.fib = fib
        self.atr_len = atr_len
        self.swing_lookback = swing_lookback
        # Tracks the bar index of p3 (neckline pivot) for the last traded pattern.
        # Mirrors Pine Script's `traded` flag to prevent re-signaling the same structure.
        self._last_traded_p3_idx: int = -1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hbar(high: np.ndarray, length: int) -> np.ndarray:
        """Return bars-since-highest-high for each bar (vectorized).

        Pine Script: ta.highestbars(len) — 0 means *this* bar is the highest.
        Here we return the index offset (0 = current bar is the pivot high).
        Uses a forward-filled rolling argmax to avoid lookahead.
        """
        n = len(high)
        result = np.full(n, np.nan)
        for i in range(length - 1, n):
            window = high[i - length + 1 : i + 1]
            # argmax gives position within window; 0=oldest, length-1=current
            argmax = int(np.argmax(window))
            result[i] = (length - 1) - argmax  # bars since the highest bar
        return result

    @staticmethod
    def _compute_lbar(low: np.ndarray, length: int) -> np.ndarray:
        """Return bars-since-lowest-low for each bar (vectorized)."""
        n = len(low)
        result = np.full(n, np.nan)
        for i in range(length - 1, n):
            window = low[i - length + 1 : i + 1]
            argmin = int(np.argmin(window))
            result[i] = (length - 1) - argmin
        return result

    @staticmethod
    def _build_pivot_list(
        hbar: np.ndarray,
        lbar: np.ndarray,
        piv_high: np.ndarray,
        piv_low: np.ndarray,
    ) -> List[Tuple[int, float, int]]:
        """Build a list of pivots by iterating through the bar series.

        Each entry is (bar_index, price, direction) where direction=1 means a
        bearish swing high pivot, direction=-1 means a bullish swing low pivot.

        Mirrors the Pine Script _mxLog / _mxUpdate logic:
        - On direction change (dirUp / dirDn): append a new pivot row.
        - While same direction continues but a new extreme is reached: update last
          pivot in place (higher high stays the high; lower low stays the low).

        Returns a list of all pivots in chronological order (max tracked = all,
        but callers only use the last 5).
        """
        n = len(hbar)
        pivots: List[Tuple[int, float, int]] = []
        current_dir: int | None = None  # 1 = up (bearish swing high), 0 = down (bullish swing low)

        for i in range(n):
            if np.isnan(hbar[i]) or np.isnan(lbar[i]):
                continue

            hb = hbar[i]
            lb = lbar[i]

            # Direction update: Pine — dir := not hbar ? 1 : not lbar ? 0 : dir
            # "not hbar" means hbar == 0 (truthy → false; 0 is falsy in Pine)
            new_dir: int | None = None
            if hb == 0:
                new_dir = 1   # swing high pivot set
            elif lb == 0:
                new_dir = 0   # swing low pivot set

            if new_dir is None:
                new_dir = current_dir  # keep previous direction

            if new_dir is None:
                continue  # not enough direction info yet

            dir_changed = new_dir != current_dir
            current_dir = new_dir

            if dir_changed:
                # Log new pivot
                price = piv_high[i] if new_dir == 1 else piv_low[i]
                pivots.append((i, price, new_dir))
            else:
                # Update last pivot if new extreme reached
                if pivots:
                    last_idx, last_price, last_dir = pivots[-1]
                    if new_dir == 1:
                        # Bearish high — update if new high is higher
                        cur_price = piv_high[i]
                        if cur_price > last_price:
                            pivots[-1] = (i, cur_price, last_dir)
                    else:
                        # Bullish low — update if new low is lower
                        cur_price = piv_low[i]
                        if cur_price < last_price:
                            pivots[-1] = (i, cur_price, last_dir)

        return pivots

    # ------------------------------------------------------------------
    # Public run() method
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        """Evaluate the Bjorgum Double Tap strategy for the current bar.

        Parameters
        ----------
        df        : OHLCV DataFrame (columns: open, high, low, close, volume, date).
                    All rolling/shift operations are backward-looking only.
        timestamp : UTC datetime for the evaluation bar.

        Returns
        -------
        StrategyRecommendation with LONG, SHORT, or HOLD signal.
        """
        if len(df) < self.MIN_BARS:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        df = df.reset_index(drop=True)

        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        length = self.len_
        n = len(df)
        idx = n - 1  # current (last) bar index

        # ---- Rolling indicators -----------------------------------------

        # Pivot high/low: rolling max / min over `length` bars (backward-looking)
        piv_high = pd.Series(high).rolling(length).max().values
        piv_low = pd.Series(low).rolling(length).min().values

        # Bars-since-highest / lowest (0 = current bar is the pivot)
        hbar = self._compute_hbar(high, length)
        lbar = self._compute_lbar(low, length)

        # ATR (used conceptually; computed for completeness but not gating signal)
        atr_arr = talib.ATR(high, low, close, timeperiod=self.atr_len)

        # Guard: ensure ATR and pivot arrays are available at current bar
        if np.isnan(atr_arr[idx]) or np.isnan(piv_high[idx]) or np.isnan(piv_low[idx]):
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # ---- Build pivot log up to and including current bar ---------------

        # We only need pivots up to bar idx (no future data)
        pivots = self._build_pivot_list(
            hbar[: idx + 1],
            lbar[: idx + 1],
            piv_high[: idx + 1],
            piv_low[: idx + 1],
        )

        # Need at least 5 pivots for a double pattern
        if len(pivots) < 5:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        close_curr = close[idx]
        close_prev = close[idx - 1] if idx >= 1 else np.nan

        if np.isnan(close_prev):
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # Take the last 5 pivots
        p1, p2, p3, p4, p5 = pivots[-5], pivots[-4], pivots[-3], pivots[-2], pivots[-1]

        # ---- Double Top detection (SHORT signal) ---------------------------
        # Pattern shape: low1 → high1 → neckline_low → high2 → (current)
        # Pivot directions for DT: p1=low(-1), p2=high(1), p3=low(-1), p4=high(1)
        # The "current" observation is that close just crossed below p3 (neckline)
        #
        # In Pine: m=1 means bearish (double top) detection
        #   y1=p1_price, y2=p2_price (high1), y3=p3_price (neck), y4=p4_price (high2)
        #   height = avg(y2, y4) - y3
        #   condition: y1 < y3 AND y4 within tol AND close < y3 AND close[1] >= y3

        short_signal = False
        long_signal = False

        # Double Top: pivots alternate low-high-low-high (last direction = high=1)
        if (p1[2] == 0 and p2[2] == 1 and p3[2] == 0 and p4[2] == 1):
            y1 = p1[1]  # first low (prior to first top)
            y2 = p2[1]  # first top (high1)
            y3 = p3[1]  # neckline (trough between tops)
            y4 = p4[1]  # second top (high2)

            # Pattern validity: y1 must be below neckline, and both tops near equal
            if y1 < y3 and y2 > y3 and y4 > y3:
                height = (y2 + y4) / 2.0 - y3
                if height > 0:
                    tol_band = height * (self.tol / 100.0)
                    top_match = abs(y4 - y2) <= tol_band
                    # Neckline crossdown: close crosses below y3 on this bar
                    neck_cross_dn = close_curr < y3 and close_prev >= y3
                    if top_match and neck_cross_dn and p3[0] != self._last_traded_p3_idx:
                        short_signal = True

        # Double Bottom: pivots alternate high-low-high-low (last direction = low=-1...
        # but in our encoding 0=low-pivot, 1=high-pivot)
        # Double Bottom: p1=high(1), p2=low(0), p3=high(1), p4=low(0)
        if (p1[2] == 1 and p2[2] == 0 and p3[2] == 1 and p4[2] == 0):
            y1 = p1[1]  # first high (prior to first bottom)
            y2 = p2[1]  # first bottom (low1)
            y3 = p3[1]  # neckline (peak between bottoms)
            y4 = p4[1]  # second bottom (low2)

            # Pattern validity: y1 must be above neckline, both bottoms near equal
            if y1 > y3 and y2 < y3 and y4 < y3:
                height = y3 - (y2 + y4) / 2.0
                if height > 0:
                    tol_band = height * (self.tol / 100.0)
                    bottom_match = abs(y4 - y2) <= tol_band
                    # Neckline crossup: close crosses above y3 on this bar
                    neck_cross_up = close_curr > y3 and close_prev <= y3
                    if bottom_match and neck_cross_up and p3[0] != self._last_traded_p3_idx:
                        long_signal = True

        # ---- Emit signal (SHORT takes priority if both fire simultaneously) ---
        if short_signal:
            self._last_traded_p3_idx = p3[0]
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)
        if long_signal:
            self._last_traded_p3_idx = p3[0]
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)

        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

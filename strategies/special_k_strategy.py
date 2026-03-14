"""
Special K Strategy

Pine Script source: input/Special-K-Strategy.pine
Timeframe: 1D  |  Lookback: 824 bars (~19776 h)

Entry logic:
  Long  — Special K crosses above its signal line AND (above zero if zero filter on)
           AND (Special K rising if slope filter on).
  Short — Special K crosses below its signal line AND (below zero if zero filter on)
           AND (Special K falling if slope filter on).

Exit logic:
  Long exit  — Special K crosses under zero.
  Short exit — Special K crosses over zero.

Special K is Martin Pring's momentum oscillator: a weighted sum of smoothed
Rate-of-Change (ROC) components across multiple time horizons. It does NOT
depend on any TradingView proprietary library.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.base_strategy import BaseStrategy, StrategyRecommendation, SignalType

# ---------------------------------------------------------------------------
# Special K component definition
# (period, weight) pairs following Martin Pring's published formula
# ---------------------------------------------------------------------------
_SK_COMPONENTS = [
    (10,  10,  1),
    (13,  13,  2),
    (14,  14,  3),
    (15,  15,  4),
    (20,  20,  1),
    (25,  25,  2),
    (35,  35,  3),
    (40,  40,  4),
    (65,  65,  1),
    (75,  75,  2),
    (100, 100, 3),
    (195, 195, 4),
    (265, 265, 1),
    (390, 390, 2),
    (530, 530, 3),
]
# Each tuple is (roc_period, sma_period, weight)


def _compute_special_k(src: pd.Series, len1: int = 100) -> tuple[pd.Series, pd.Series]:
    """Compute Martin Pring's Special K oscillator and its signal line.

    Parameters
    ----------
    src  : Price series (typically close).
    len1 : Signal line SMA length (default 100).

    Returns
    -------
    (special_k, signal) as pandas Series aligned to the input index.

    Formula
    -------
    ROC(src, n)  = (src / src.shift(n) - 1) * 100
    specialK     = sum(weight * SMA(ROC(src, roc_p), sma_p)
                       for roc_p, sma_p, weight in components)
    signal       = SMA(specialK, len1)

    All shift() calls use positive integers — strictly backward-looking.
    """
    special_k = pd.Series(0.0, index=src.index, dtype=float)

    for roc_p, sma_p, weight in _SK_COMPONENTS:
        roc = (src / src.shift(roc_p) - 1.0) * 100.0   # shift(+n) — backward only
        smoothed = roc.rolling(sma_p, min_periods=sma_p).mean()
        special_k = special_k + weight * smoothed

    signal = special_k.rolling(len1, min_periods=len1).mean()

    return special_k, signal


class SpecialKStrategy(BaseStrategy):
    """Special K Strategy.

    Implements Martin Pring's Special K momentum oscillator with configurable
    zero-line and slope filters. Entries are triggered when Special K crosses
    its signal line with optional confirmation filters; exits occur when
    Special K crosses the zero line.

    Default parameters match the Pine Script inputs:
        len1            : 100   (signal line SMA length)
        len2            : 100   (unused — kept for parity; single smoothing used)
        use_zero_filter : True  (require Special K to be above/below zero)
        use_slope_filter: True  (require Special K to be rising/falling)
        slope_len       : 3     (bars back for slope comparison)
        atr_len         : 14    (ATR length — computed but exits handled externally)

    Minimum bars required: 530 (longest ROC period) + 530 (longest SMA period)
    + 100 (signal SMA) = 1160. We use 1160 as the conservative min_bars guard.
    """

    MIN_BARS: int = 1160

    def __init__(
        self,
        len1: int = 100,
        use_zero_filter: bool = True,
        use_slope_filter: bool = True,
        slope_len: int = 3,
    ):
        super().__init__(
            name="Special K Strategy",
            description=(
                "Martin Pring's Special K momentum strategy using weighted smoothed "
                "ROC components across 15 time horizons with zero-line and slope filters"
            ),
            timeframe="1D",
            lookback_hours=19776,
        )
        self.len1 = len1
        self.use_zero_filter = use_zero_filter
        self.use_slope_filter = use_slope_filter
        self.slope_len = slope_len

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        """Evaluate the Special K Strategy for the current bar.

        Parameters
        ----------
        df        : OHLCV DataFrame with columns: open, high, low, close, volume, date.
                    All shift() operations use positive integers (backward-looking only).
        timestamp : UTC datetime for the current evaluation bar.

        Returns
        -------
        StrategyRecommendation with LONG, SHORT, FLAT, or HOLD signal.
        """
        if len(df) < self.MIN_BARS:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        src = df["close"].reset_index(drop=True)

        # ---- Compute Special K and signal line -------------------------
        special_k, signal = _compute_special_k(src, self.len1)

        # ---- Evaluate on the last bar ----------------------------------
        idx = len(special_k) - 1

        # Need at least slope_len + 1 valid Special K values
        if idx < self.slope_len:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        sk_curr = special_k.iloc[idx]
        sk_prev = special_k.iloc[idx - 1]           # shift(1) — backward only
        sk_slope_ref = special_k.iloc[idx - self.slope_len]  # shift(slope_len) — backward only

        sig_curr = signal.iloc[idx]
        sig_prev = signal.iloc[idx - 1]             # shift(1) — backward only

        # Guard against NaN — not enough data for indicator convergence
        if pd.isna(sk_curr) or pd.isna(sig_curr) or pd.isna(sig_prev):
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        # ---- Crossover / Crossunder detection -------------------------
        # ta.crossover(specialK, signal):  curr > sig AND prev <= sig
        cross_up   = (sk_curr > sig_curr) and (sk_prev <= sig_prev)
        # ta.crossunder(specialK, signal): curr < sig AND prev >= sig
        cross_down = (sk_curr < sig_curr) and (sk_prev >= sig_prev)

        # ---- Filters --------------------------------------------------
        above_zero = sk_curr > 0.0
        below_zero = sk_curr < 0.0

        k_rising  = sk_curr > sk_slope_ref   # Special K higher than slope_len bars ago
        k_falling = sk_curr < sk_slope_ref

        # ---- Entry conditions -----------------------------------------
        long_entry = (
            cross_up
            and (not self.use_zero_filter or above_zero)
            and (not self.use_slope_filter or k_rising)
        )
        short_entry = (
            cross_down
            and (not self.use_zero_filter or below_zero)
            and (not self.use_slope_filter or k_falling)
        )

        # ---- Zero-line exit conditions --------------------------------
        # long_exit:  Special K crosses under zero
        long_exit  = (sk_prev >= 0.0) and (sk_curr < 0.0)
        # short_exit: Special K crosses over zero
        short_exit = (sk_prev <= 0.0) and (sk_curr > 0.0)

        # ---- Signal priority: entries > exits > hold ------------------
        if long_entry:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        if short_entry:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)
        if long_exit or short_exit:
            return StrategyRecommendation(signal=SignalType.FLAT, timestamp=timestamp)

        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

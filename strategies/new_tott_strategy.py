"""
NEW TOTT Strategy

Pine Script source: input/NEW-TOTT-Strategy.pine
Timeframe: 15m  |  Lookback: 49 bars (~13 h)

Entry logic:
  Long  — MAvg (VAR MA) crosses over OTTup (upper Twin OTT band).
  Short — MAvg crosses under OTTdn (lower Twin OTT band).

Exit management (SL, TP, break-even) is NOT implemented here.
It must be configured in the external execution layer.

OTT (Optimized Trend Tracker) uses a Variable Index Dynamic Average (VAR MA)
as its base moving average, then applies a ratcheting stop mechanism to produce
the OTT line and twin bands.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.base_strategy import BaseStrategy, StrategyRecommendation, SignalType


def _compute_var_ma(src: pd.Series, length: int) -> np.ndarray:
    """Variable Index Dynamic Average (VAR / Volatility-Adjusted Recursive MA).

    Matches Pine Script's Var_Func(src, length).

    Parameters
    ----------
    src    : Close price series (pandas Series).
    length : Smoothing period (used for valpha).
    """
    valpha = 2.0 / (length + 1)
    n = len(src)
    src_arr = src.to_numpy(dtype=float)

    diff = pd.Series(src_arr).diff()        # NaN at index 0, backward-looking elsewhere
    vud1 = diff.clip(lower=0).fillna(0.0).to_numpy()
    vdd1 = (-diff).clip(lower=0).fillna(0.0).to_numpy()

    # Rolling 9-bar sums (forward-safe; only looks back)
    vud_series = pd.Series(vud1).rolling(9).sum().to_numpy()
    vdd_series = pd.Series(vdd1).rolling(9).sum().to_numpy()

    denom = vud_series + vdd_series
    with np.errstate(invalid="ignore", divide="ignore"):
        vcmo = np.where(denom != 0.0, (vud_series - vdd_series) / denom, 0.0)
    vcmo = np.nan_to_num(vcmo, nan=0.0)

    # Iterative VAR (recursive; cannot be vectorized)
    var_vals = np.zeros(n)
    for i in range(1, n):
        alpha_i = valpha * abs(vcmo[i])
        var_vals[i] = alpha_i * src_arr[i] + (1.0 - alpha_i) * var_vals[i - 1]

    return var_vals


def _compute_ott(
    mavg: np.ndarray, percent: float, coeff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Optimized Trend Tracker core with ratcheting stops.

    Returns (OTTup, OTTdn) — the twin OTT bands.

    Parameters
    ----------
    mavg    : Moving average values (1-D float array).
    percent : OTT optimisation constant (default 1.0).
    coeff   : Twin OTT coefficient (default 0.001).
    """
    n = len(mavg)
    long_stop = np.zeros(n)
    short_stop = np.zeros(n)
    direction = np.ones(n, dtype=int)

    for i in range(1, n):
        fark_i = mavg[i] * percent * 0.01
        ls = mavg[i] - fark_i
        ss = mavg[i] + fark_i

        # Ratchet: only tighten in the direction of the trend
        long_stop[i] = max(ls, long_stop[i - 1]) if mavg[i] > long_stop[i - 1] else ls
        short_stop[i] = min(ss, short_stop[i - 1]) if mavg[i] < short_stop[i - 1] else ss

        # Direction flip logic (matches Pine dir variable)
        if direction[i - 1] == -1 and mavg[i] > short_stop[i - 1]:
            direction[i] = 1
        elif direction[i - 1] == 1 and mavg[i] < long_stop[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    # MT: trailing stop in current direction
    mt = np.where(direction == 1, long_stop, short_stop)

    # OTT line
    ott = np.where(mavg > mt, mt * (200.0 + percent) / 200.0, mt * (200.0 - percent) / 200.0)

    ott_up = ott * (1.0 + coeff)
    ott_dn = ott * (1.0 - coeff)

    return ott_up, ott_dn


class NewTottStrategy(BaseStrategy):
    """NEW TOTT Strategy.

    Implements the OTT (Optimized Trend Tracker) using a Variable Index Dynamic
    Average (VAR MA) as its base. Entries are triggered when the VAR MA crosses
    the Twin OTT bands (OTTup / OTTdn).

    Default parameters match the Pine Script inputs:
        length  : 40    (OTT period)
        percent : 1.0   (optimisation constant)
        coeff   : 0.001 (twin OTT coefficient)
        mav     : "VAR" (moving average type — VAR MA is the default)

    Exit logic (SL, TP, break-even) is NOT implemented. Configure exits in the
    execution layer.
    """

    # Minimum candles required before the first meaningful signal can be emitted.
    # VAR MA needs ~40 bars to stabilise; the rolling-9 sum needs 9 more.
    MIN_BARS: int = 50

    def __init__(
        self,
        length: int = 40,
        percent: float = 1.0,
        coeff: float = 0.001,
    ):
        super().__init__(
            name="NEW TOTT Strategy",
            description=(
                "OTT-based trend following strategy using VAR moving average with "
                "Twin OTT bands for signal generation"
            ),
            timeframe="15m",
            lookback_hours=13,
        )
        self.length = length
        self.percent = percent
        self.coeff = coeff

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        """Evaluate the strategy for the current bar.

        Parameters
        ----------
        df        : OHLCV DataFrame with columns: open, high, low, close, volume, date.
                    All shift() operations use positive integers (backward-looking only).
        timestamp : UTC datetime for the current evaluation bar.

        Returns
        -------
        StrategyRecommendation with LONG, SHORT, or HOLD signal.
        """
        if len(df) < self.MIN_BARS:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        src = df["close"].reset_index(drop=True)

        # ---- VAR Moving Average -----------------------------------------
        mavg = _compute_var_ma(src, self.length)

        # ---- OTT Twin Bands --------------------------------------------
        ott_up, ott_dn = _compute_ott(mavg, self.percent, self.coeff)

        # ---- Crossover / Crossunder detection on last two bars ----------
        # Uses shift(1) equivalent: index[-2] vs index[-1] — fully backward-looking
        idx = len(mavg) - 1
        if idx < 1:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        mavg_curr = mavg[idx]
        mavg_prev = mavg[idx - 1]
        ott_up_curr = ott_up[idx]
        ott_up_prev = ott_up[idx - 1]
        ott_dn_curr = ott_dn[idx]
        ott_dn_prev = ott_dn[idx - 1]

        # buySignal  = ta.crossover(MAvg, OTTup)
        buy_signal = mavg_curr > ott_up_curr and mavg_prev <= ott_up_prev

        # sellSignal = ta.crossunder(MAvg, OTTdn)
        sell_signal = mavg_curr < ott_dn_curr and mavg_prev >= ott_dn_prev

        if buy_signal:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        if sell_signal:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)
        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

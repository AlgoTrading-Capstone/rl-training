"""
EMA5 Breakout with Target Shifting (TF & Buffer Options) — Python transpilation
of the PineScript v6 strategy of the same name.

Original logic
--------------
- Compute an EMA (length `ema_length`, default 5) of close. The EMA can be
  taken on the base timeframe (`ema_timeframe="current"`) or on a higher
  timeframe resolved via src.utils.resampling (strictly causal merge).
- Signal detection (on the EMA selected above):
    * Long setup   (`isLongSignal`):  high  < ema AND open < ema AND close < ema
    * Short setup  (`isShortSignal`): low   > ema AND open > ema AND close > ema
- On a Long setup, arm a long target = high + buffer. On a Short setup, arm a
  short target = low - buffer. Only one target may be armed at a time; arming
  one clears the other.
- Buffer is either a fixed point value (`buffer_type="points"`, default 1.0)
  or a percentage of the signal candle's high/low
  (`buffer_type="percentage"`).
- Breakout (entries):
    * long breakout  when high crosses OVER the armed long target
      (prior bar high <= target AND current high > target). Emit LONG, clear
      both targets.
    * short breakout when low crosses UNDER the armed short target
      (prior bar low >= target AND current low < target). Emit SHORT, clear
      both targets.
- If neither breakout fired this bar and a target is still armed, shift the
  target to track the new bar: long target = high + buffer, short
  target = low - buffer.

Exit logic (intentionally dropped)
----------------------------------
The Pine script uses `strategy.entry` which relies on the TradingView engine's
default position management (reverse on opposite entry, pyramid settings).
This Python implementation only emits direction signals (LONG / SHORT / FLAT);
position sizing and exit orchestration belong to the execution / RL layer.

Anti-lookahead
--------------
- No `np.roll`, no negative `.shift()`.
- Higher-timeframe EMA is merged back via `src.utils.resampling.resampled_merge`
  which is strictly causal (HTF bar only becomes visible AFTER it closes).
- The target / breakout update is an O(n) numpy loop over the base-TF index,
  updating state using only current and past bar values.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, SignalType
from utils.resampling import (
    compute_interval_minutes,
    resample_to_interval,
    resampled_merge,
)
from utils.timeframes import timeframe_to_minutes


# Map Pine's "EMA Timeframe" dropdown to lowercase timeframes understood by
# src.utils.timeframes / src.utils.resampling. "current" means "use the base
# timeframe of the input DataFrame" — no higher-TF merge needed.
_TF_OPTION_MAP = {
    "current": None,
    "1 min": "1m",
    "5 min": "5m",
    "15 min": "15m",
    "30 min": "30m",
    "1 hour": "1h",
    "4 hour": "4h",
    "1 day": "1d",
}


class Ema5BreakoutTargetShiftingMtfStrategy(BaseStrategy):
    """EMA5 breakout with target-shifting (multi-TF + buffer options)."""

    def __init__(
        self,
        ema_length: int = 5,
        ema_timeframe: str = "current",
        buffer_type: str = "points",
        buffer_value: float = 1.0,
        timeframe: str = "15m",
    ):
        super().__init__(
            name="EMA5 Breakout with Target Shifting (TF & Buffer Options)",
            description=(
                "Arm a target above/below a candle that closes entirely on one "
                "side of the EMA; enter on target breakout. Target shifts with "
                "the latest bar until broken."
            ),
            timeframe=timeframe,
            lookback_hours=1,
        )

        # --- inputs ----------------------------------------------------------
        self.ema_length = int(ema_length)
        if self.ema_length <= 0:
            raise ValueError("ema_length must be positive")

        key = str(ema_timeframe).strip().lower()
        if key not in _TF_OPTION_MAP:
            raise ValueError(
                f"ema_timeframe must be one of {list(_TF_OPTION_MAP)}, got {ema_timeframe!r}"
            )
        self.ema_timeframe: str = key
        self._htf: Optional[str] = _TF_OPTION_MAP[key]

        bt = str(buffer_type).strip().lower()
        if bt not in ("points", "percentage"):
            raise ValueError("buffer_type must be 'points' or 'percentage'")
        self.buffer_type: str = bt

        self.buffer_value = float(buffer_value)
        if self.buffer_value <= 0:
            raise ValueError("buffer_value must be positive")

        # --- dynamic warmup --------------------------------------------------
        # 3x the EMA length covers recursive EMA convergence. If we resample
        # to a higher timeframe, the effective warmup on the base TF is
        # multiplied by the TF ratio (the HTF EMA needs 3*ema_length HTF bars,
        # each worth `ratio` base bars).
        htf_ratio = self._htf_ratio_against_base_fallback()
        self.MIN_CANDLES_REQUIRED = 3 * self.ema_length * htf_ratio

        # --- streaming state -------------------------------------------------
        self._observed: int = 0
        self._buffer_candles: list[dict] = []

    # ---- batch / vectorized mode -------------------------------------------

    def generate_all_signals(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        signals = pd.Series(["FLAT"] * n, index=df.index, dtype=object)
        if n == 0:
            return signals
        if n < self.MIN_CANDLES_REQUIRED:
            return signals

        work = self._ensure_date_column(df)
        ema = self._compute_ema(work)

        high = work["high"].astype(float).to_numpy()
        low = work["low"].astype(float).to_numpy()
        open_ = work["open"].astype(float).to_numpy()
        close = work["close"].astype(float).to_numpy()
        ema_arr = np.asarray(ema, dtype=float)

        # Degenerate input (all-NaN closes): the EMA is NaN → no signals fire.
        if not np.isfinite(ema_arr).any():
            return signals

        out = self._run_target_state_machine(
            high=high,
            low=low,
            open_=open_,
            close=close,
            ema=ema_arr,
        )

        # Enforce warmup contract.
        out[: self.MIN_CANDLES_REQUIRED] = "FLAT"
        return pd.Series(out, index=df.index, dtype=object)

    # ---- live / streaming mode ---------------------------------------------

    def step(self, candle: pd.Series) -> SignalType:
        self._observed += 1
        self._buffer_candles.append(
            {
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": float(candle.get("volume", 0.0)),
            }
        )

        if self._observed < self.MIN_CANDLES_REQUIRED:
            return SignalType.FLAT

        # Keep the rolling buffer bounded — 3x warmup is more than enough.
        cap = self.MIN_CANDLES_REQUIRED * 3
        if len(self._buffer_candles) > cap:
            self._buffer_candles = self._buffer_candles[-cap:]

        buf_df = pd.DataFrame(self._buffer_candles)
        base_min = timeframe_to_minutes(self._timeframe)
        buf_df.insert(
            0,
            "date",
            pd.date_range(
                start="2020-01-01",
                periods=len(buf_df),
                freq=f"{base_min}min",
                tz="UTC",
            ),
        )

        sig_series = self.generate_all_signals(buf_df)
        try:
            return SignalType(sig_series.iloc[-1])
        except (ValueError, KeyError):
            return SignalType.FLAT

    # ---- helpers ------------------------------------------------------------

    def _ensure_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Guarantee a UTC 'date' column so resample utilities can consume it."""
        work = df.copy()
        if "date" in work.columns:
            work["date"] = pd.to_datetime(work["date"], utc=True)
            return work
        if isinstance(work.index, pd.DatetimeIndex):
            work = work.reset_index()
            first_col = work.columns[0]
            if first_col != "date":
                work = work.rename(columns={first_col: "date"})
            work["date"] = pd.to_datetime(work["date"], utc=True)
            return work
        # Fallback — fabricate a synthetic date axis at the declared timeframe.
        base_min = timeframe_to_minutes(self._timeframe)
        work.insert(
            0,
            "date",
            pd.date_range(
                start="2020-01-01",
                periods=len(work),
                freq=f"{base_min}min",
                tz="UTC",
            ),
        )
        return work

    def _compute_ema(self, work: pd.DataFrame) -> np.ndarray:
        """
        Compute the EMA series aligned to `work`. If ema_timeframe selects a
        higher TF, resample-and-merge causally via src.utils.resampling.
        """
        close = work["close"].astype(float)
        if self._htf is None:
            return close.ewm(span=self.ema_length, adjust=False).mean().to_numpy()

        # Higher-TF EMA via causal merge.
        base_min = compute_interval_minutes(work)
        htf_min = timeframe_to_minutes(self._htf)

        if htf_min <= base_min:
            # Requested HTF is not actually higher — fall back to base-TF EMA.
            return close.ewm(span=self.ema_length, adjust=False).mean().to_numpy()

        resampled = resample_to_interval(work, htf_min)
        if len(resampled) < 2:
            return close.ewm(span=self.ema_length, adjust=False).mean().to_numpy()

        htf_ema = (
            resampled["close"]
            .astype(float)
            .ewm(span=self.ema_length, adjust=False)
            .mean()
        )
        htf_frame = pd.DataFrame(
            {
                "date": resampled["date"],
                "open": resampled["open"],
                "high": resampled["high"],
                "low": resampled["low"],
                "close": resampled["close"],
                "volume": resampled["volume"],
                "ema": htf_ema,
            }
        )
        merged = resampled_merge(work, htf_frame, fill_na=True)
        col = f"resample_{htf_min}_ema"
        return merged[col].to_numpy()

    def _run_target_state_machine(
        self,
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        close: np.ndarray,
        ema: np.ndarray,
    ) -> np.ndarray:
        """
        Replicate Pine's `var float longTarget / shortTarget` state machine.

        Strictly causal — each bar's decision depends only on this bar's OHLC +
        EMA (both computed causally) and the target carried over from the
        previous bar.

        Returns an object array of "LONG" / "SHORT" / "FLAT" strings.
        """
        n = len(high)
        out = np.full(n, "FLAT", dtype=object)

        long_target = np.nan   # currently-armed long target (NaN if disarmed)
        short_target = np.nan  # currently-armed short target (NaN if disarmed)

        use_pct = self.buffer_type == "percentage"
        bv = self.buffer_value

        for i in range(n):
            h = high[i]
            l = low[i]
            o = open_[i]
            c = close[i]
            e = ema[i]

            # NaN-guard — if EMA or OHLC is NaN, just carry targets forward.
            if not (np.isfinite(e) and np.isfinite(h) and np.isfinite(l)
                    and np.isfinite(o) and np.isfinite(c)):
                continue

            # 1) Arm fresh targets on new setup bars.
            is_long_setup = h < e and o < e and c < e
            is_short_setup = l > e and o > e and c > e

            if is_long_setup:
                buf = (h * bv / 100.0) if use_pct else bv
                long_target = h + buf
                short_target = np.nan
            if is_short_setup:
                buf = (l * bv / 100.0) if use_pct else bv
                short_target = l - buf
                long_target = np.nan

            # 2) Check breakouts against the armed target.
            long_breakout = False
            short_breakout = False

            if np.isfinite(long_target) and h > long_target:
                long_breakout = True
                out[i] = "LONG"
                long_target = np.nan
                short_target = np.nan
            elif np.isfinite(short_target) and l < short_target:
                short_breakout = True
                out[i] = "SHORT"
                short_target = np.nan
                long_target = np.nan

            # 3) If no breakout fired, shift the armed target with this bar.
            if not long_breakout and not short_breakout:
                if np.isfinite(long_target):
                    buf = (h * bv / 100.0) if use_pct else bv
                    long_target = h + buf
                if np.isfinite(short_target):
                    buf = (l * bv / 100.0) if use_pct else bv
                    short_target = l - buf

        return out

    def _htf_ratio_against_base_fallback(self) -> int:
        """
        Best-effort HTF-to-base ratio for dynamic warmup sizing.

        Uses the strategy's declared base `timeframe` (set in __init__). If the
        selected HTF is not strictly higher than the base, ratio is 1.
        """
        if self._htf is None:
            return 1
        try:
            base_min = timeframe_to_minutes(self._timeframe)
            htf_min = timeframe_to_minutes(self._htf)
        except ValueError:
            return 1
        if htf_min <= base_min:
            return 1
        # Integer ratio, floor division — good enough for warmup sizing.
        return max(1, htf_min // base_min)

from datetime import datetime

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType


class Sensex500PointInstitutionalBreakoutStrategy(BaseStrategy):
    """
    Sensex 9:45 3-Trade Limit Breakout (converted from Pine Script v5).

    At 09:45 each session the strategy snaps two round-number levels:
        lower_level = floor(open / 500) * 500
        upper_level = ceil(open  / 500) * 500

    It then watches for price to break out of that range:
        - close >= upper_level  →  LONG  (upside breakout)
        - close <= lower_level  →  SHORT (downside breakout)

    Safety rules:
        - Entries only within 09:45–15:15 session window.
        - Maximum 3 trades per calendar day.
        - EOD forced-flat at 15:25–15:30 → FLAT signal.

    Pine Script parameters (not user-tunable at runtime):
        increment        = 500   (round-number step)
        sl_points        = 100   (initial SL offset — managed externally)
        be_trigger       = 200   (breakeven trigger   — managed externally)
        max_trades       = 3     (max entries per day)
        session_start    = 09:45
        trading_session  = 09:45 – 15:15
        eod_window       = 15:25 – 15:30

    strategy.exit() / strategy.position_size / strategy.position_avg_price are
    all managed by the external RL execution engine and are intentionally omitted.
    """

    def __init__(self) -> None:
        super().__init__(
            name="Sensex500PointInstitutionalBreakoutStrategy",
            description=(
                "At 09:45 computes floor/ceil round-number breakout levels "
                "(500-point increment). Enters long on upside breakout, short "
                "on downside breakout. Max 3 trades/day; EOD flat at 15:25."
            ),
            timeframe="5m",
            lookback_hours=8,
        )

        # --- Strategy parameters ---
        self.increment: int = 500
        self.max_trades: int = 3
        self.session_start_hour: int = 9
        self.session_start_min: int = 45

        # --- RL dynamic warmup guard ---
        # The strategy needs at least one full session lookback so the 09:45
        # level-reset bar is guaranteed to be present in the window.
        self.lookback_bars: int = 80
        self.MIN_CANDLES_REQUIRED: int = self.lookback_bars + 1

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        # --- RL warmup guard ---
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        df = df.copy()

        # --- Time decomposition ---
        df["_hour"] = df["date"].dt.hour
        df["_min"] = df["date"].dt.minute
        df["_hm"] = df["_hour"] * 100 + df["_min"]

        # --- 09:45 session-open bar ---
        df["_is_start"] = (
            (df["_hour"] == self.session_start_hour)
            & (df["_min"] == self.session_start_min)
        )

        # Compute floor/ceil levels only at 09:45; forward-fill to remaining bars.
        # np.where avoids NaN propagation: non-start bars get NaN, ffill carries
        # the last computed level forward (same as Pine's `var` semantics).
        start_open = df["open"].where(df["_is_start"])
        df["_lower_level"] = (
            np.floor(start_open / self.increment) * self.increment
        ).ffill()
        df["_upper_level"] = (
            np.ceil(start_open / self.increment) * self.increment
        ).ffill()

        # --- Session and EOD masks ---
        df["_can_trade"] = (df["_hm"] >= 945) & (df["_hm"] <= 1515)
        df["_is_eod"] = (df["_hm"] >= 1525) & (df["_hm"] <= 1530)

        # --- Raw entry conditions (pre-trade-count filter) ---
        levels_valid = df["_upper_level"].notna() & df["_lower_level"].notna()
        not_start = ~df["_is_start"]

        df["_long_raw"] = (
            not_start
            & (df["close"] >= df["_upper_level"])
            & df["_can_trade"]
            & levels_valid
        )
        df["_short_raw"] = (
            not_start
            & (df["close"] <= df["_lower_level"])
            & df["_can_trade"]
            & levels_valid
        )

        # --- Per-day trade-count cap (vectorized) ---
        # count how many entry signals have fired *before* the current bar today.
        df["_date_only"] = df["date"].dt.date
        df["_entry"] = (df["_long_raw"] | df["_short_raw"]).astype(int)
        cum = df.groupby("_date_only")["_entry"].cumsum()
        df["_prior_count"] = cum - df["_entry"]  # exclude current bar
        df["_can_enter"] = df["_prior_count"] < self.max_trades

        # --- Final conditions ---
        df["_long_cond"] = df["_long_raw"] & df["_can_enter"]
        df["_short_cond"] = df["_short_raw"] & df["_can_enter"]

        # --- Evaluate last bar ---
        last = df.iloc[-1]

        # Priority: EOD flat > long entry > short entry > hold
        if last["_is_eod"]:
            return StrategyRecommendation(signal=SignalType.FLAT, timestamp=timestamp)

        if last["_long_cond"]:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)

        if last["_short_cond"]:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)

        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

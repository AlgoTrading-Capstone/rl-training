"""
Enhanced OR Strategy (Opening Range Breakout) - NY 15m ORB with Fixed SL/TP

Converted from Pine Script v5:
  "Enhanced OR Strategy (Futures + Breakeven + TV Position Box - History)"

Signal logic:
  - Defines an Opening Range (OR) during the first 15 minutes of the NY session
    (default: 9:30-9:45 ET)
  - After OR is established, counts consecutive candles closing above/below the range
  - LONG  when 2+ consecutive candles close above OR high (bullish breakout)
  - SHORT when 2+ consecutive candles close below OR low (bearish breakout)
  - Only one signal per trading day
  - Day-of-week filtering (default: all weekdays enabled)
  - HOLD otherwise

Exit logic NOT converted (architecture constraint - StrategyRecommendation carries
signal direction only):
  - Stop-loss: ATR Multiple / Range % / Fixed Points / Opposite Range
  - Take-profit: Risk:Reward / ATR Multiple / Fixed Points
  - End-of-day close at 22:30 GMT+2
  - Position box drawing / breakeven logic
  These must be configured in the execution layer.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import talib

from strategies.base_strategy import BaseStrategy, SignalType, StrategyRecommendation


class Ny15mOrbWithAFixedSlTpNasdaqStrategy(BaseStrategy):
    """
    NY Session Opening Range Breakout Strategy.

    Tracks the high/low of the first 15 minutes of the NY session,
    then signals on consecutive breakout candles above/below the range.

    Parameters (matching Pine Script defaults):
      - OR duration: 15 minutes (9:30-9:45 ET)
      - Breakout candles: 2 consecutive closes required
      - ATR period: 14 (used in original for stop calculation, not for signal)
      - Trading days: Monday through Friday

    Limitations:
      - Stop-loss, take-profit, and EOD close logic from the original Pine Script
        are NOT implemented. StrategyRecommendation only carries signal direction.
        Configure SL/TP in the execution layer.
      - Pyramiding (original: 10) is not applicable in signal-only mode.
    """

    OR_DURATION_MINUTES: int = 15
    BREAKOUT_CANDLES: int = 2
    ATR_PERIOD: int = 14
    TRADE_DAYS: tuple = (0, 1, 2, 3, 4)  # Monday=0 through Friday=4
    OR_START_HOUR: int = 9
    OR_START_MINUTE: int = 30

    def __init__(self):
        super().__init__(
            name="Enhanced OR Strategy",
            description=(
                "NY session Opening Range breakout strategy. Signals LONG/SHORT on "
                "consecutive candle closes above/below the opening range high/low."
            ),
            timeframe="15m",
            lookback_hours=13,
        )
        # 3x ATR period for indicator convergence (ATR used in original for risk calc)
        self.MIN_CANDLES_REQUIRED = 3 * max(self.ATR_PERIOD, self.BREAKOUT_CANDLES)

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

        work = df.copy()

        # Convert to NY timezone for session detection
        ny = pd.to_datetime(work["date"], utc=True).dt.tz_convert("America/New_York")
        work["ny_date"] = ny.dt.date
        work["time_mins"] = ny.dt.hour * 60 + ny.dt.minute
        work["dow"] = ny.dt.dayofweek

        # Day-of-week filter
        day_ok = work["dow"].isin(self.TRADE_DAYS)

        # Opening range time window
        or_start = self.OR_START_HOUR * 60 + self.OR_START_MINUTE  # 570
        or_end = or_start + self.OR_DURATION_MINUTES  # 585

        in_or = (
            (work["time_mins"] >= or_start)
            & (work["time_mins"] < or_end)
            & day_ok
        )

        # Compute OR high/low per trading day
        or_data = work.loc[in_or].groupby("ny_date").agg(
            or_high=("high", "max"),
            or_low=("low", "min"),
        )
        work["or_high"] = work["ny_date"].map(or_data["or_high"])
        work["or_low"] = work["ny_date"].map(or_data["or_low"])

        # Session active: after OR on a valid day with valid OR levels
        session_done = (
            (work["time_mins"] >= or_end) & day_ok & work["or_high"].notna()
        )

        # Breakout direction per bar
        bull_bar = session_done & (work["close"] > work["or_high"])
        bear_bar = session_done & (work["close"] < work["or_low"])

        direction = pd.Series(0, index=work.index, dtype=int)
        direction[bull_bar] = 1
        direction[bear_bar] = -1

        # Consecutive breakout counting (reset on direction change or new day)
        new_day = work["ny_date"] != work["ny_date"].shift(1)
        direction_change = (direction != direction.shift(1)) | new_day
        groups = direction_change.cumsum()
        cum_count = work.groupby(groups).cumcount() + 1

        bull_count = (direction == 1).astype(int) * cum_count
        bear_count = (direction == -1).astype(int) * cum_count

        # Breakout trigger
        long_trigger = bull_count >= self.BREAKOUT_CANDLES
        short_trigger = bear_count >= self.BREAKOUT_CANDLES
        any_trigger = long_trigger | short_trigger

        # Only first signal per day (traded_today equivalent from Pine Script)
        prior_triggered = any_trigger.groupby(work["ny_date"]).transform(
            lambda x: x.shift(1, fill_value=False).cummax()
        )
        first_long = long_trigger & ~prior_triggered
        first_short = short_trigger & ~prior_triggered

        # Return signal for the last bar
        last = len(work) - 1
        if first_long.iloc[last]:
            return StrategyRecommendation(signal=SignalType.LONG, timestamp=timestamp)
        elif first_short.iloc[last]:
            return StrategyRecommendation(signal=SignalType.SHORT, timestamp=timestamp)

        return StrategyRecommendation(signal=SignalType.HOLD, timestamp=timestamp)

from datetime import datetime

import pandas as pd

from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType


class Tasc202603OnePercentAWeekStrategy(BaseStrategy):
    """
    TASC 2026.03 — One Percent A Week
    Original article: "Trading Snapbacks In A Leveraged ETF" by Dion Kurczek (March 2026).

    Logic:
      1. Record the Monday open as the weekly reference price (wOpen).
      2. Enter LONG when price dips to 1% below wOpen (limit fill: low <= wOpen * 0.99).
      3. Profit target (+1%) and break-even risk management are handled by the RL engine.
      4. Hard close on Friday — return FLAT regardless of position.

    strategy.exit(), strategy.position_size, and strategy.position_avg_price are all
    ignored — the execution engine manages those externally.
    """

    def __init__(self):
        super().__init__(
            name="TASC 2026.03 One Percent A Week",
            description=(
                "Weekly mean-reversion strategy for leveraged ETFs. "
                "Buys on a 1% dip from Monday's open, targeting 1% profit. "
                "Hard exit on Friday close. Converted from Pine Script v6."
            ),
            timeframe="1d",
            lookback_hours=10 * 24,  # 10 daily bars
        )
        # No indicator lengths — warmup based on the strategy's lookback_bars parameter.
        self.lookback_bars = 10
        self.dip_pct = 0.99  # 1% below Monday open triggers entry
        self.MIN_CANDLES_REQUIRED = 3 * self.lookback_bars

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(SignalType.HOLD, timestamp)

        df = df.copy()

        # --- Day of week (0=Monday … 4=Friday) ---
        df['_dow'] = pd.to_datetime(df['date']).dt.dayofweek

        # --- Weekly open: Monday's open price, forward-filled through the week ---
        # Strictly use shift(0) semantics — ffill carries the Monday value forward
        # without ever seeing future bars, satisfying the anti-lookahead constraint.
        monday_open = df['open'].where(df['_dow'] == 0)
        df['_w_open'] = monday_open.ffill()

        # --- 1% dip level ---
        df['_down1'] = df['_w_open'] * self.dip_pct

        last = df.iloc[-1]

        # Priority 1 — hard Friday exit (close all positions at week end)
        if last['_dow'] == 4:
            return StrategyRecommendation(SignalType.FLAT, timestamp)

        # Priority 2 — LONG entry: limit order fills when today's low touches down1
        # Requires a valid weekly open reference (not NaN) and not Friday.
        if not pd.isna(last['_w_open']) and last['low'] <= last['_down1']:
            return StrategyRecommendation(SignalType.LONG, timestamp)

        return StrategyRecommendation(SignalType.HOLD, timestamp)

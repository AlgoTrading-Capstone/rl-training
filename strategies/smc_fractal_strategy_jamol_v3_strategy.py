from datetime import datetime
import numpy as np
import pandas as pd
import talib
from strategies.base_strategy import BaseStrategy, StrategyRecommendation, SignalType
from utils.resampling import resample_to_interval, resampled_merge


class SmcFractalStrategyJamolV3(BaseStrategy):
    """
    Smart Money Concepts (SMC) Fractal Strategy [Jamol] v3.

    Entry sequence: liquidity sweep → break of structure (BOS) → pullback into
    an Order Block (OB), Fair Value Gap (FVG), or Breaker Block.

    Filters:
      - 4H EMA HTF bias (long only above, short only below).
      - NY + London Close session: 07:00–15:00 EST.

    Converted from PineScript v6.
    - SL/TP (stop-loss, take-profit via strategy.exit) is NOT converted;
      it is managed by the external RL execution engine.
    - The `strategy.position_size == 0` entry guard was removed; position
      management is handled externally by the RL engine.
    """

    def __init__(self):
        super().__init__(
            name="SmcFractalStrategyJamolV3",
            description=(
                "SMC fractal: sweep → BOS → OB/FVG/Breaker pullback entry. "
                "4H EMA HTF bias filter. NY+London session filter."
            ),
            timeframe="15m",
            lookback_hours=125,  # 500 bars × 15 min / 60
        )
        self.swing_len = 10
        self.htf_ema_period = 20
        self.sweep_exp = 20
        self.fvg_min = 0.00010
        self.ob_lookback = 30

        # Dominant warmup: HTF EMA(20) on 4H needs 20 × 16 = 320 base-tf bars.
        # 3× safety multiplier for RL hyperparameter tuning.
        htf_bars = self.htf_ema_period * (240 // 15)  # 320
        self.MIN_CANDLES_REQUIRED = 3 * max(htf_bars, 2 * self.swing_len + self.ob_lookback)
        # = 3 * max(320, 50) = 960

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pivot_high(self, high: pd.Series, left: int, right: int) -> pd.Series:
        """
        Vectorized pivot-high detection equivalent to ta.pivothigh(high, left, right).
        Result is shifted by `right` bars so it is only visible after confirmation,
        eliminating lookahead bias.
        """
        left_max = high.rolling(left + 1).max()
        right_max = high.iloc[::-1].rolling(right + 1).max().iloc[::-1]
        is_pivot = (high == left_max) & (high == right_max)
        ph = pd.Series(np.nan, index=high.index)
        ph[is_pivot] = high[is_pivot]
        return ph.shift(right)

    def _pivot_low(self, low: pd.Series, left: int, right: int) -> pd.Series:
        """
        Vectorized pivot-low detection equivalent to ta.pivotlow(low, left, right).
        Result is shifted by `right` bars to avoid lookahead bias.
        """
        left_min = low.rolling(left + 1).min()
        right_min = low.iloc[::-1].rolling(right + 1).min().iloc[::-1]
        is_pivot = (low == left_min) & (low == right_min)
        pl = pd.Series(np.nan, index=low.index)
        pl[is_pivot] = low[is_pivot]
        return pl.shift(right)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        if len(df) < self.MIN_CANDLES_REQUIRED:
            return StrategyRecommendation(SignalType.HOLD, timestamp)

        df = df.copy()

        # --- HTF Bias: 4H EMA via lookahead-safe resampling ---
        resampled = resample_to_interval(df, "4h")
        resampled['ema_20'] = talib.EMA(
            resampled['close'].values, timeperiod=self.htf_ema_period
        )
        df = resampled_merge(original=df, resampled=resampled, fill_na=True)

        htf_close_arr = df['resample_240_close'].values
        htf_ema_arr = df['resample_240_ema_20'].values
        htf_trend_arr = np.where(
            htf_close_arr > htf_ema_arr, 1,
            np.where(htf_close_arr < htf_ema_arr, -1, 0)
        )

        # --- Session filter: NY + London Close 07:00–15:00 EST ---
        if df['date'].dt.tz is None:
            date_est = df['date'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        else:
            date_est = df['date'].dt.tz_convert('America/New_York')
        time_val = date_est.dt.hour * 100 + date_est.dt.minute
        in_ses_arr = ((time_val >= 700) & (time_val < 1500)).values

        # --- Pivot highs / lows (lookahead-safe, forward-filled) ---
        ph = self._pivot_high(df['high'], self.swing_len, self.swing_len)
        pl = self._pivot_low(df['low'], self.swing_len, self.swing_len)
        last_hi = ph.ffill().values
        last_lo = pl.ffill().values

        # --- Raw arrays for the stateful loop ---
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        n = len(df)

        # --- Stateful sweep / BOS / zone tracking ---
        # These variables carry forward-state; vectorization is not possible
        # because each bar's output depends on prior output (flip-flop logic).
        swept_bull_bar = -9999
        swept_bear_bar = -9999
        trend = 0

        bull_ob_hi = np.nan
        bull_ob_lo = np.nan
        bear_ob_hi = np.nan
        bear_ob_lo = np.nan
        bull_brk_hi = np.nan
        bull_brk_lo = np.nan
        bear_brk_hi = np.nan
        bear_brk_lo = np.nan
        bull_fvg_hi = np.nan
        bull_fvg_lo = np.nan
        bear_fvg_hi = np.nan
        bear_fvg_lo = np.nan

        for i in range(n):
            c = closes[i]
            h = highs[i]
            lo = lows[i]
            lhi = last_hi[i]
            llo = last_lo[i]
            c_prev = closes[i - 1] if i > 0 else c

            # ---- Liquidity sweep ----
            bull_sweep = (not np.isnan(llo)) and lo < llo and c > llo
            bear_sweep = (not np.isnan(lhi)) and h > lhi and c < lhi
            if bull_sweep:
                swept_bull_bar = i
            if bear_sweep:
                swept_bear_bar = i

            sweep_bull_ok = (i - swept_bull_bar) <= self.sweep_exp
            sweep_bear_ok = (i - swept_bear_bar) <= self.sweep_exp

            # ---- Break of Structure ----
            bull_bos = sweep_bull_ok and (not np.isnan(lhi)) and c > lhi and c_prev <= lhi
            bear_bos = sweep_bear_ok and (not np.isnan(llo)) and c < llo and c_prev >= llo

            if bull_bos:
                trend = 1
                swept_bull_bar = -9999
                bull_ob_hi = np.nan
                bull_ob_lo = np.nan
                # Find the most recent bearish candle within ob_lookback for the OB
                for j in range(1, min(self.ob_lookback + 1, i + 1)):
                    if closes[i - j] < opens[i - j]:
                        bull_ob_hi = highs[i - j]
                        bull_ob_lo = lows[i - j]
                        break

            if bear_bos:
                trend = -1
                swept_bear_bar = -9999
                bear_ob_hi = np.nan
                bear_ob_lo = np.nan
                # Find the most recent bullish candle within ob_lookback for the OB
                for j in range(1, min(self.ob_lookback + 1, i + 1)):
                    if closes[i - j] > opens[i - j]:
                        bear_ob_hi = highs[i - j]
                        bear_ob_lo = lows[i - j]
                        break

            # ---- OB invalidation (close-through only; wick = potential sweep) ----
            if not np.isnan(bull_ob_lo) and c < bull_ob_lo:
                bull_ob_hi = np.nan
                bull_ob_lo = np.nan
            if not np.isnan(bear_ob_hi) and c > bear_ob_hi:
                bear_ob_hi = np.nan
                bear_ob_lo = np.nan

            # ---- Breaker block formation (requires 3 prior bars) ----
            if i >= 3:
                prior_bear3 = (
                    closes[i - 1] < opens[i - 1]
                    and closes[i - 2] < opens[i - 2]
                    and closes[i - 3] < opens[i - 3]
                )
                prior_bull3 = (
                    closes[i - 1] > opens[i - 1]
                    and closes[i - 2] > opens[i - 2]
                    and closes[i - 3] > opens[i - 3]
                )
                bear_body_top = max(opens[i - 1], opens[i - 2], opens[i - 3])
                bull_body_bot = min(opens[i - 1], opens[i - 2], opens[i - 3])

                if trend == 1 and prior_bear3 and c > bear_body_top:
                    bull_brk_hi = max(highs[i - 1], highs[i - 2], highs[i - 3])
                    bull_brk_lo = min(lows[i - 1], lows[i - 2], lows[i - 3])
                if trend == -1 and prior_bull3 and c < bull_body_bot:
                    bear_brk_hi = max(highs[i - 1], highs[i - 2], highs[i - 3])
                    bear_brk_lo = min(lows[i - 1], lows[i - 2], lows[i - 3])

            # ---- Breaker invalidation (close-through) ----
            if not np.isnan(bull_brk_lo) and c < bull_brk_lo:
                bull_brk_hi = np.nan
                bull_brk_lo = np.nan
            if not np.isnan(bear_brk_hi) and c > bear_brk_hi:
                bear_brk_hi = np.nan
                bear_brk_lo = np.nan

            # ---- Fair Value Gap detection (requires 2 prior bars) ----
            if i >= 2:
                # Bull FVG: gap between high[i-2] and low[i] (no overlap)
                if trend == 1 and highs[i - 2] < lows[i]:
                    gap = lows[i] - highs[i - 2]
                    if gap >= self.fvg_min:
                        bull_fvg_hi = lows[i]
                        bull_fvg_lo = highs[i - 2]
                # Bear FVG: gap between low[i-2] and high[i] (no overlap)
                if trend == -1 and lows[i - 2] > highs[i]:
                    gap = lows[i - 2] - highs[i]
                    if gap >= self.fvg_min:
                        bear_fvg_hi = lows[i - 2]
                        bear_fvg_lo = highs[i]

        # --- Entry conditions evaluated at the last bar ---
        last_c = closes[n - 1]

        bull_ob_tap = (
            not np.isnan(bull_ob_hi) and not np.isnan(bull_ob_lo)
            and last_c <= bull_ob_hi and last_c >= bull_ob_lo
        )
        bull_fvg_tap = (
            not np.isnan(bull_fvg_hi) and not np.isnan(bull_fvg_lo)
            and last_c <= bull_fvg_hi and last_c >= bull_fvg_lo
        )
        bull_brk_tap = (
            not np.isnan(bull_brk_hi) and not np.isnan(bull_brk_lo)
            and last_c <= bull_brk_hi and last_c >= bull_brk_lo
        )

        bear_ob_tap = (
            not np.isnan(bear_ob_hi) and not np.isnan(bear_ob_lo)
            and last_c >= bear_ob_lo and last_c <= bear_ob_hi
        )
        bear_fvg_tap = (
            not np.isnan(bear_fvg_hi) and not np.isnan(bear_fvg_lo)
            and last_c >= bear_fvg_lo and last_c <= bear_fvg_hi
        )
        bear_brk_tap = (
            not np.isnan(bear_brk_hi) and not np.isnan(bear_brk_lo)
            and last_c >= bear_brk_lo and last_c <= bear_brk_hi
        )

        long_zone = bull_ob_tap or bull_fvg_tap or bull_brk_tap
        short_zone = bear_ob_tap or bear_fvg_tap or bear_brk_tap

        htf_trend_last = int(htf_trend_arr[n - 1])
        in_ses_last = bool(in_ses_arr[n - 1])

        long_sig = in_ses_last and trend == 1 and htf_trend_last == 1 and long_zone
        short_sig = in_ses_last and trend == -1 and htf_trend_last == -1 and short_zone

        if long_sig:
            return StrategyRecommendation(SignalType.LONG, timestamp)
        if short_sig:
            return StrategyRecommendation(SignalType.SHORT, timestamp)
        return StrategyRecommendation(SignalType.HOLD, timestamp)

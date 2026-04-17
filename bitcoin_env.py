import csv
import os
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    INITIAL_BALANCE,
    LEVERAGE_LIMIT,
    TRANSACTION_FEE,
    TRAIN_TEST_SPLIT,
    MAX_POSITION_BTC,
    MIN_STOP_LOSS_PCT,
    MAX_STOP_LOSS_PCT,
    EXPOSURE_DEADZONE,
    STOP_UPDATE_DEADZONE_PCT,
    REWARD_FUNCTION,
    STOP_CLUSTER_WINDOW_BARS,
    SAME_SIDE_REENTRY_WINDOW_BARS,
    STRATEGY_LIST,
    ENABLE_STRATEGIES,
)

from utils.normalization import normalize_state
from reward_functions import REWARD_REGISTRY

from trade_engine import (
    TradeConfig,
    PositionState,
    apply_action,
    compute_equity,
    close_position,
)


class BitcoinTradingEnv:

    def __init__(self, price_ary, tech_ary, turbulence_array, signal_ary, datetime_ary,
                 strategy_names=None,
                 mode="train", signal_log_path: Optional[str] = None,
                 signal_log_dir: Optional[str] = None,
                 signal_log_filename: Optional[str] = None,
                 signal_log_flush_every: int = 100,
                 signal_log_worker_id: Optional[str] = None):
        """
        RL Trading Environment for Bitcoin.
        price_ary:        np.ndarray - price features
        tech_ary:         np.ndarray - technical indicators
        turbulence_array: np.ndarray - turbulence & VIX
        signal_ary:       np.ndarray - strategy outputs
        datetime_ary:     np.ndarray - datetime for backtesting
        mode:             "train" | "test" | "backtest"
        signal_log_path:  Optional path to write strategy_signals_log.csv
        """

        assert mode in ["train", "test", "backtest"], "mode must be 'train', 'test', or 'backtest'."
        self.mode = mode
        self.strategy_names = list(strategy_names) if strategy_names is not None else list(STRATEGY_LIST)

        # Signal logging (strategy visibility)
        self._signal_log_flush_every = max(1, int(signal_log_flush_every))
        self._signal_log_worker_id = (
            str(signal_log_worker_id)
            if signal_log_worker_id is not None
            else (str(os.getpid()) if mode == "train" and signal_log_dir and not signal_log_filename else None)
        )
        self._signal_log_path = self._resolve_signal_log_path(
            signal_log_path=signal_log_path,
            signal_log_dir=signal_log_dir,
            signal_log_filename=signal_log_filename,
        )
        self._signal_log_buffer = []
        self._signal_log_fieldnames = None

        # ------------------------------------------------------------
        # Basic sanity checks on input arrays
        # ------------------------------------------------------------
        num_candles = price_ary.shape[0]

        # Ensure all arrays have the same length (number of candles)
        if any(arr.shape[0] != num_candles for arr in (tech_ary, turbulence_array, signal_ary, datetime_ary)):
            raise ValueError("All input arrays must have the same length.")

        # Ensure all arrays are 2D
        if signal_ary.ndim != 2:
            raise ValueError("signal_ary must be 2D (shape: [T, signal_dim]).")
        if price_ary.ndim != 2 or tech_ary.ndim != 2 or turbulence_array.ndim != 2:
            raise ValueError("price_ary / tech_ary / turbulence_array must be 2D (shape: [T, dim]).")

        # Warn if no strategy signals are provided
        if signal_ary.shape[1] == 0:
            print("Warning: No strategy signals (ENABLE_STRATEGIES=False or empty STRATEGY_LIST). Environment will run without strategy features in state space.")
        elif signal_ary.shape[1] != 4 * len(self.strategy_names):
            raise ValueError(
                f"signal_ary width mismatch: got {signal_ary.shape[1]}, "
                f"expected {4 * len(self.strategy_names)} from strategy_names."
            )

        # ------------------------------------------------------------
        # Load configuration values
        # ------------------------------------------------------------
        self.initial_balance = INITIAL_BALANCE
        self.leverage_limit = LEVERAGE_LIMIT
        self.transaction_fee = TRANSACTION_FEE

        self.reward_fn = REWARD_REGISTRY[REWARD_FUNCTION]

        self.trade_cfg = TradeConfig(
            leverage_limit=self.leverage_limit,
            max_position_btc=MAX_POSITION_BTC,
            min_stop_pct=MIN_STOP_LOSS_PCT,
            max_stop_pct=MAX_STOP_LOSS_PCT,
            exposure_deadzone=EXPOSURE_DEADZONE,
            stop_update_deadzone=STOP_UPDATE_DEADZONE_PCT,
            fee_rate=self.transaction_fee,
        )

        # ------------------------------------------------------------
        # Prepare historical data (splitting train/test)
        # ------------------------------------------------------------
        split_idx = int(num_candles * TRAIN_TEST_SPLIT)

        if mode == "train":
            sl = slice(0, split_idx)
        elif mode == "test":
            sl = slice(split_idx, num_candles)
        else:  # backtest
            sl = slice(0, num_candles)

        self.price_ary = price_ary[sl]
        self.tech_ary = tech_ary[sl]
        self.turbulence_ary = turbulence_array[sl]
        self.signal_ary = signal_ary[sl]
        self.datetime_ary = datetime_ary[sl]

        # ------------------------------------------------------------
        # RL Environment internal state
        # ------------------------------------------------------------
        self.step_idx = 0
        self.position = PositionState(
            balance=float(self.initial_balance),
            holdings=0.0,
            entry_price=None,
            stop_price=None,
        )
        self.prev_equity = float(self.initial_balance)
        self.episode_return = 0.0  # Final episode profit ratio (total_asset / initial_balance)

        # Position context bookkeeping (for state features)
        self._bars_in_position = 0
        self._bars_since_stop = -1   # -1 = never stopped this episode
        self._last_position_side = 0 # -1 short, 0 flat, +1 long

        # Stop-aware reward bookkeeping
        self._peak_equity = float(self.initial_balance)
        self._stop_steps = []           # Step indices where stops occurred
        self._last_stopped_side = 0     # Side of most recent stop (+1/-1, 0=never)

        # ------------------------------------------------------------
        # Environment dimensions (state & action)
        # ------------------------------------------------------------
        price_dim = self.price_ary.shape[1]
        tech_dim = self.tech_ary.shape[1]
        turbulence_dim = self.turbulence_ary.shape[1]
        signal_dim = self.signal_ary.shape[1]

        # State includes: balance, price features, indicators, turbulence, strategy signals, position size
        # + 7 position context features (side, exposure, entry_rel, pnl, stop_dist, bars_in_pos, bars_since_stop)
        self.state_dim = 1 + price_dim + tech_dim + turbulence_dim + signal_dim + 1 + 7

        # Action space: action = [a_pos, a_sl]
        # a_pos ∈ [-1, +1] - desired exposure: -1 = max short, 0 = flat, +1 = max long
        # a_sl ∈ [-1, 1] - stop-loss tightness within a predefined range
        # The trade engine converts these targets into actual market buy/sell actions.
        self.action_dim = 2

        # Maximum number of timesteps in a single episode.
        # Equals the number of available candles after splitting and downsampling.
        # When the agent reaches (max_step - 1), the episode ends (done=True).
        self.max_step = self.price_ary.shape[0]

        # Load first observation (step_idx = 0)
        self.current_price = self.price_ary[0]
        self.current_tech = self.tech_ary[0]
        self.current_turbulence = self.turbulence_ary[0]
        self.current_signal = self.signal_ary[0]
        self.current_datetime = self.datetime_ary[0]

    def _resolve_signal_log_path(
        self,
        *,
        signal_log_path: Optional[str],
        signal_log_dir: Optional[str],
        signal_log_filename: Optional[str],
    ) -> Optional[Path]:
        """Resolve the final CSV path for signal logging."""
        if signal_log_path:
            return Path(signal_log_path)
        if not signal_log_dir:
            return None

        log_dir = Path(signal_log_dir)
        if signal_log_filename:
            return log_dir / signal_log_filename

        if self.mode == "train":
            worker_id = self._signal_log_worker_id or str(os.getpid())
            return log_dir / f"strategy_signals_train_worker_{worker_id}.csv"

        return None

    # ------------------------------------------------------------------
    # Signal logging helpers
    # ------------------------------------------------------------------

    def _decode_signal_vec(self, sig_vec: np.ndarray) -> dict:
        """Decode one-hot signal vector into {strategy_name: signal_label} dict."""
        if not ENABLE_STRATEGIES or sig_vec is None or len(sig_vec) == 0:
            return {}
        labels = {0: "FLAT", 1: "LONG", 2: "SHORT", 3: "HOLD"}
        decisions = {}
        idx = 0
        for name in self.strategy_names:
            if idx + 4 > len(sig_vec):
                break
            s = sig_vec[idx: idx + 4]
            idx += 4
            hot = int(np.argmax(s)) if np.any(s) else 3
            decisions[name] = labels.get(hot, "UNKNOWN")
        return decisions

    def _log_signal_row(self, step_idx, timestamp, sig_vec, a_pos, a_sl, reward, close_price):
        """Append a signal log row to the in-memory buffer."""
        if self._signal_log_path is None:
            return
        if hasattr(timestamp, "item"):
            try:
                timestamp = timestamp.item()
            except ValueError:
                pass
        row = {
            "step": step_idx,
            "timestamp": timestamp,
            "close": float(close_price),
            "agent_exposure": float(a_pos),
            "agent_stop_loss": float(a_sl),
            "reward": float(reward),
        }
        if self._signal_log_worker_id is not None:
            row["worker_id"] = self._signal_log_worker_id
        row.update(self._decode_signal_vec(sig_vec))
        self._signal_log_buffer.append(row)

    def _flush_signal_log(self):
        """Write buffered signal rows to CSV and clear the buffer."""
        if self._signal_log_path is None or not self._signal_log_buffer:
            return
        self._signal_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._signal_log_fieldnames is None:
            self._signal_log_fieldnames = list(self._signal_log_buffer[0].keys())

        write_header = not self._signal_log_path.exists() or self._signal_log_path.stat().st_size == 0
        with open(self._signal_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._signal_log_fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(self._signal_log_buffer)
        self._signal_log_buffer.clear()

    def reset(self):
        """
        Reset the environment at the beginning of an episode.
        Returns the initial normalized state.
        """

        # Flush any accumulated signal log from previous episode
        self._flush_signal_log()

        # Reset internal counters
        self.step_idx = 0

        # Reset financial variables (via PositionState)
        self.position = PositionState(
            balance=float(self.initial_balance),
            holdings=0.0,
            entry_price=None,
            stop_price=None,
        )
        self.prev_equity = float(self.initial_balance)
        self.episode_return = 0.0

        # Reset position context bookkeeping
        self._bars_in_position = 0
        self._bars_since_stop = -1
        self._last_position_side = 0

        # Reset stop-aware reward bookkeeping
        self._peak_equity = float(self.initial_balance)
        self._stop_steps = []
        self._last_stopped_side = 0

        # Load first timestep features
        self.current_price = self.price_ary[self.step_idx]
        self.current_tech = self.tech_ary[self.step_idx]
        self.current_turbulence = self.turbulence_ary[self.step_idx]
        self.current_signal = self.signal_ary[self.step_idx]
        self.current_datetime = self.datetime_ary[self.step_idx]

        # Build initial normalized state (flat position defaults)
        state = normalize_state(
            balance=self.position.balance,
            price_vec=self.current_price,
            tech_vec=self.current_tech,
            turbulence_vec=self.current_turbulence,
            signal_vec=self.current_signal,
            holdings=self.position.holdings,
            position_side=0,
            exposure_norm=0.0,
            entry_price_rel=0.0,
            unrealized_pnl_pct=0.0,
            stop_distance_pct=0.0,
            bars_in_position=0,
            bars_since_stop=-1,
        )

        return state


    def step(self, action):
        """
        Execute one environment step given the agent's action.

        Parameters
        ----------
        action : array_like
            Continuous action [a_pos, a_sl]:
                a_pos ∈ [-1, +1] - desired exposure
                a_sl  ∈ [-1, 1] - stop-loss tightness

        Returns
        -------
        next_state : np.ndarray
        reward     : float
        done       : bool
        info       : dict (debugging)
        """

        # -------------------------------
        # Current timestep context (before action)
        # -------------------------------
        t = self.step_idx
        t_datetime = self.current_datetime

        # -------------------------------
        # Unpack action
        # -------------------------------
        a_pos = float(action[0])
        a_sl = float(action[1])

        # -------------------------------
        # Extract OHLC
        # -------------------------------
        open_p, high_p, low_p, close_p = self.current_price[:4]
        exec_price = float(close_p)
        equity_price = exec_price

        # -------------------------------
        # Compute old equity (for reward)
        # -------------------------------
        old_equity = self.prev_equity

        # -------------------------------
        # STOP-LOSS CHECK (intra-candle)
        # -------------------------------
        stop_triggered = False
        stop_exec_price = None
        stopped_side = 0  # Side of position that was stopped (for reward context)

        if self.position.stop_price is not None and not np.isclose(self.position.holdings, 0.0):

            if self.position.holdings > 0:
                # LONG stop
                if low_p <= self.position.stop_price:
                    stop_triggered = True
                    stopped_side = 1
                    # Execute at the worse of open or stop price - handles gap downs
                    stop_exec_price = min(open_p, self.position.stop_price)

            elif self.position.holdings < 0:
                # SHORT stop
                if high_p >= self.position.stop_price:
                    stop_triggered = True
                    stopped_side = -1
                    # Execute at the worse of open or stop price - handles gap ups
                    stop_exec_price = max(open_p, self.position.stop_price)

        # -------------------------------
        # Apply trading logic
        # -------------------------------
        if stop_triggered:
            # Forced exit at stop price (with slippage inside engine)
            result = close_position(
                price=stop_exec_price,
                state=self.position,
                cfg=self.trade_cfg,
            )
            equity_price = stop_exec_price
        else:
            result = apply_action(
                a_pos=a_pos,
                a_sl=a_sl,
                price=exec_price,
                state=self.position,
                cfg=self.trade_cfg,
            )

        self.position = result.new_state

        # -------------------------------
        # Position context bookkeeping
        # -------------------------------
        prev_side = self._last_position_side  # Side before this step (for re-entry detection)

        if stop_triggered:
            self._last_stopped_side = stopped_side
            self._stop_steps.append(t)
            self._bars_since_stop = 0
        elif self._bars_since_stop >= 0:
            self._bars_since_stop += 1

        if np.isclose(self.position.holdings, 0.0):
            current_side = 0
        else:
            current_side = 1 if self.position.holdings > 0 else -1

        if current_side == 0 or current_side != self._last_position_side:
            self._bars_in_position = 0
        else:
            self._bars_in_position += 1
        self._last_position_side = current_side

        # -------------------------------
        # Compute new equity & reward
        # -------------------------------
        new_equity = compute_equity(
            balance=self.position.balance,
            holdings=self.position.holdings,
            price=equity_price,
        )

        # --- Stop-aware reward context ---
        self._peak_equity = max(self._peak_equity, new_equity)
        current_drawdown = (
            (self._peak_equity - new_equity) / self._peak_equity
            if self._peak_equity > 0 else 0.0
        )

        recent_stop_count = sum(
            1 for s in self._stop_steps if (t - s) <= STOP_CLUSTER_WINDOW_BARS
        )

        same_side_reentry = (
            not stop_triggered
            and self._last_stopped_side != 0
            and current_side != 0
            and current_side == self._last_stopped_side
            and prev_side != current_side
            and 0 < self._bars_since_stop <= SAME_SIDE_REENTRY_WINDOW_BARS
        )

        reward_context = {
            "stop_triggered": stop_triggered,
            "recent_stop_count": recent_stop_count,
            "same_side_reentry": same_side_reentry,
            "current_drawdown": current_drawdown,
        }

        reward = self.reward_fn(old_equity, new_equity, context=reward_context)

        # -------------------------------
        # Update equity for next step
        # -------------------------------
        self.prev_equity = new_equity

        # -------------------------------
        # Determine episode termination
        # -------------------------------
        is_bankrupt = (new_equity <= 0)  # Agent has lost all equity
        done = (self.step_idx == self.max_step - 1) or is_bankrupt

        # -------------------------------
        # Advance time if not done
        # -------------------------------
        if not done:
            self.step_idx += 1
            self.current_price = self.price_ary[self.step_idx]
            self.current_tech = self.tech_ary[self.step_idx]
            self.current_turbulence = self.turbulence_ary[self.step_idx]
            self.current_signal = self.signal_ary[self.step_idx]
            self.current_datetime = self.datetime_ary[self.step_idx]

        # -------------------------------
        # Compute position context for next state
        # (relative to the candle the agent will observe)
        # -------------------------------
        next_close = float(self.current_price[3])
        equity_at_obs = compute_equity(
            self.position.balance, self.position.holdings, next_close
        )

        if equity_at_obs > 0.0 and self.leverage_limit > 0.0:
            max_notional = self.leverage_limit * equity_at_obs
            _exposure_norm = float(np.clip(
                (self.position.holdings * next_close) / max_notional, -1.0, 1.0
            ))
        else:
            _exposure_norm = 0.0

        if current_side != 0 and self.position.entry_price is not None and next_close > 0:
            _entry_price_rel = self.position.entry_price / next_close - 1.0
        else:
            _entry_price_rel = 0.0

        if current_side != 0 and self.position.entry_price is not None and self.position.entry_price > 0:
            _unrealized_pnl_pct = current_side * (next_close / self.position.entry_price - 1.0)
        else:
            _unrealized_pnl_pct = 0.0

        if current_side != 0 and self.position.stop_price is not None and next_close > 0:
            _stop_distance_pct = self.position.stop_price / next_close - 1.0
        else:
            _stop_distance_pct = 0.0

        # -------------------------------
        # Build next state
        # -------------------------------
        next_state = normalize_state(
            balance=self.position.balance,
            price_vec=self.current_price,
            tech_vec=self.current_tech,
            turbulence_vec=self.current_turbulence,
            signal_vec=self.current_signal,
            holdings=self.position.holdings,
            position_side=current_side,
            exposure_norm=_exposure_norm,
            entry_price_rel=_entry_price_rel,
            unrealized_pnl_pct=_unrealized_pnl_pct,
            stop_distance_pct=_stop_distance_pct,
            bars_in_position=self._bars_in_position,
            bars_since_stop=self._bars_since_stop,
        )

        # -------------------------------
        # Signal logging (strategy visibility)
        # -------------------------------
        # Use the signal_vec from the candle BEFORE time-advance (the one
        # that was visible when the agent chose its action).
        self._log_signal_row(
            step_idx=t,
            timestamp=t_datetime,
            sig_vec=self.signal_ary[t] if t < len(self.signal_ary) else self.current_signal,
            a_pos=a_pos,
            a_sl=a_sl,
            reward=reward,
            close_price=close_p,
        )
        if len(self._signal_log_buffer) >= self._signal_log_flush_every:
            self._flush_signal_log()

        # -------------------------------
        # Final episode return
        # -------------------------------
        if done:
            self.episode_return = new_equity / self.initial_balance
            self._flush_signal_log()

        # -------------------------------
        # Info dict for debugging/logs
        # -------------------------------
        info = {
            "step_idx": t,
            "timestamp": t_datetime,
            "stop_triggered": stop_triggered,
            "stop_exec_price": None if stop_exec_price is None else float(stop_exec_price),
            "trade_executed": result.trade_executed,
            "effective_delta_btc": result.effective_delta_btc,
            "stop_updated": result.stop_updated,
            "equity": new_equity,                 # Portfolio total value (cash + holdings * equity_price) AFTER this step
            "equity_price": float(equity_price),  # Price used to value holdings this step (close if no stop, else stop execution price)
        }

        return next_state, reward, done, info

    def close(self):
        """Flush any pending signal rows before shutdown."""
        self._flush_signal_log()

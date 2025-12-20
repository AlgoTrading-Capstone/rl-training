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
    REWARD_FUNCTION,
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

    def __init__(self, price_ary, tech_ary, turbulence_array, signal_ary, datetime_ary, mode="train"):
        """
        RL Trading Environment for Bitcoin.
        price_ary:        np.ndarray - price features
        tech_ary:         np.ndarray - technical indicators
        turbulence_array: np.ndarray - turbulence & VIX
        signal_ary:       np.ndarray - strategy outputs
        datetime_ary:     np.ndarray - datetime for backtesting
        mode:             "train" | "test" | "backtest"
        """

        assert mode in ["train", "test", "backtest"], "mode must be 'train', 'test', or 'backtest'."
        self.mode = mode

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

        # ------------------------------------------------------------
        # Environment dimensions (state & action)
        # ------------------------------------------------------------
        price_dim = self.price_ary.shape[1]
        tech_dim = self.tech_ary.shape[1]
        turbulence_dim = self.turbulence_ary.shape[1]
        signal_dim = self.signal_ary.shape[1]

        # State includes: balance, price features, indicators, turbulence, strategy signals, position size
        self.state_dim = 1 + price_dim + tech_dim + turbulence_dim + signal_dim + 1

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


    def reset(self):
        """
        Reset the environment at the beginning of an episode.
        Returns the initial normalized state.
        """

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

        # Load first timestep features
        self.current_price = self.price_ary[self.step_idx]
        self.current_tech = self.tech_ary[self.step_idx]
        self.current_turbulence = self.turbulence_ary[self.step_idx]
        self.current_signal = self.signal_ary[self.step_idx]
        self.current_datetime = self.datetime_ary[self.step_idx]

        # Build initial normalized state
        state = normalize_state(
            balance=self.position.balance,
            price_vec=self.current_price,
            tech_vec=self.current_tech,
            turbulence_vec=self.current_turbulence,
            signal_vec=self.current_signal,
            holdings=self.position.holdings,
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

        if self.position.stop_price is not None and not np.isclose(self.position.holdings, 0.0):

            if self.position.holdings > 0:
                # LONG stop
                if low_p <= self.position.stop_price:
                    stop_triggered = True
                    # Execute at the worse of open or stop price - handles gap downs
                    stop_exec_price = min(open_p, self.position.stop_price)

            elif self.position.holdings < 0:
                # SHORT stop
                if high_p >= self.position.stop_price:
                    stop_triggered = True
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
        # Compute new equity & reward
        # -------------------------------
        new_equity = compute_equity(
            balance=self.position.balance,
            holdings=self.position.holdings,
            price=equity_price,
        )

        reward = self.reward_fn(old_equity, new_equity)

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
        # Build next state
        # -------------------------------
        next_state = normalize_state(
            balance=self.position.balance,
            price_vec=self.current_price,
            tech_vec=self.current_tech,
            turbulence_vec=self.current_turbulence,
            signal_vec=self.current_signal,
            holdings=self.position.holdings,
        )

        # -------------------------------
        # Final episode return
        # -------------------------------
        if done:
            self.episode_return = new_equity / self.initial_balance

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
            "equity": new_equity,                 # Portfolio total value (cash + holdings * equity_price) AFTER this step
            "equity_price": float(equity_price),  # Price used to value holdings this step (close if no stop, else stop execution price)
        }

        return next_state, reward, done, info
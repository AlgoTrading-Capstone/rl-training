import numpy as np

from config import (
    INITIAL_BALANCE,
    LEVERAGE_LIMIT,
    TRANSACTION_FEE,
    GAMMA,
    DECISION_INTERVAL,
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
)


class BitcoinTradingEnv:

    def __init__(self, price_ary, tech_ary, turbulence_array, signal_ary, mode="train"):
        """
        RL Trading Environment for Bitcoin.
        price_ary:        np.ndarray - price features
        tech_ary:         np.ndarray - technical indicators
        turbulence_array: np.ndarray - turbulence & VIX
        signal_ary:       np.ndarray - strategy outputs
        mode:             "train" | "test"
        """

        assert mode in ["train", "test"], "mode must be 'train' or 'test'."
        self.mode = mode

        # Handle empty signal array (backward compatibility)
        if signal_ary.shape[1] == 0:
            print("Warning: No strategy signals (ENABLE_STRATEGIES=False or empty STRATEGY_LIST)")
            print("   Environment will run without strategy features in state space")

        # ------------------------------------------------------------
        # Load configuration values
        # ------------------------------------------------------------
        self.initial_balance = INITIAL_BALANCE
        self.leverage_limit = LEVERAGE_LIMIT
        self.transaction_fee = TRANSACTION_FEE
        self.gamma = GAMMA
        self.decision_interval = DECISION_INTERVAL

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
        num_candles = price_ary.shape[0]
        split_idx = int(num_candles * TRAIN_TEST_SPLIT)

        if mode == "train":
            self.price_ary = price_ary[:split_idx]
            self.tech_ary = tech_ary[:split_idx]
            self.turbulence_ary = turbulence_array[:split_idx]
            self.signal_ary = signal_ary[:split_idx]
        else:  # test mode
            self.price_ary = price_ary[split_idx:]
            self.tech_ary = tech_ary[split_idx:]
            self.turbulence_ary = turbulence_array[split_idx:]
            self.signal_ary = signal_ary[split_idx:]

        # ------------------------------------------------------------
        # Apply decision interval (downsampling)
        # ------------------------------------------------------------
        self.price_ary = self.price_ary[::self.decision_interval]
        self.tech_ary = self.tech_ary[::self.decision_interval]
        self.turbulence_ary = self.turbulence_ary[::self.decision_interval]
        self.signal_ary = self.signal_ary[::self.decision_interval]

        # ------------------------------------------------------------
        # RL Environment internal state
        # ------------------------------------------------------------
        self.day = 0
        self.position = PositionState(
            balance=float(self.initial_balance),
            holdings=0.0,
            entry_price=None,
            stop_price=None,
        )
        self.gamma_return = 0.0  # Discounted return accumulator
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
        # a_sl ∈ [0, 1] - stop-loss tightness within a predefined range
        # The trade engine converts these targets into actual market buy/sell actions.
        self.action_dim = 2

        # Maximum number of timesteps in a single episode.
        # Equals the number of available candles after splitting and downsampling.
        # When the agent reaches (max_step - 1), the episode ends (done=True).
        self.max_step = self.price_ary.shape[0]

        # Load first observation (day = 0)
        self.current_price = self.price_ary[0]
        self.current_tech = self.tech_ary[0]
        self.current_turbulence = self.turbulence_ary[0]
        self.current_signal = self.signal_ary[0]


    def reset(self):
        """
        Reset the environment at the beginning of an episode.
        Returns the initial normalized state.
        """

        # Reset internal counters
        self.day = 0

        # Reset financial variables (via PositionState)
        self.position = PositionState(
            balance=float(self.initial_balance),
            holdings=0.0,
            entry_price=None,
            stop_price=None,
        )
        self.gamma_return = 0.0
        self.episode_return = 0.0

        # Load first timestep features
        self.current_price = self.price_ary[self.day]
        self.current_tech = self.tech_ary[self.day]
        self.current_turbulence = self.turbulence_ary[self.day]
        self.current_signal = self.signal_ary[self.day]

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
                a_sl  ∈ [0, 1] - stop-loss tightness

        Returns
        -------
        next_state : np.ndarray
        reward     : float
        done       : bool
        info       : dict (debugging)
        """

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

        # -------------------------------
        # Compute old equity (for reward)
        # -------------------------------
        old_equity = compute_equity(
            balance=self.position.balance,
            holdings=self.position.holdings,
            price=exec_price,
        )

        # -------------------------------
        # STOP-LOSS CHECK BEFORE ACTION
        # -------------------------------
        stop_triggered = False

        if self.position.stop_price is not None and not np.isclose(self.position.holdings, 0.0):

            if self.position.holdings > 0:
                # LONG: stop triggered if low <= stop_price
                if low_p <= self.position.stop_price:
                    stop_triggered = True

            elif self.position.holdings < 0:
                # SHORT: stop triggered if high >= stop_price
                if high_p >= self.position.stop_price:
                    stop_triggered = True

        # If stop-loss triggers - force close and skip agent action
        if stop_triggered:
            result = apply_action(
                a_pos=0.0,  # force close
                a_sl=a_sl,
                price=exec_price,
                state=self.position,
                cfg=self.trade_cfg,
            )
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
            price=exec_price,
        )

        reward = self.reward_fn(old_equity, new_equity)

        # FinRL discounted accumulator
        self.gamma_return = self.gamma_return * self.gamma + reward

        # -------------------------------
        # Advance timestep
        # -------------------------------
        self.day += 1
        done = (self.day >= self.max_step - 1)

        if not done:
            self.current_price = self.price_ary[self.day]
            self.current_tech = self.tech_ary[self.day]
            self.current_turbulence = self.turbulence_ary[self.day]
            self.current_signal = self.signal_ary[self.day]

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
        # At the final timestep we add the accumulated discounted rewards (gamma_return) to the last reward.
        # This gives the agent a signal about the performance of the entire episode.
        # -------------------------------
        if done:
            reward += self.gamma_return
            self.gamma_return = 0.0
            self.episode_return = new_equity / self.initial_balance

        # -------------------------------
        # Info dict for debugging/logs
        # -------------------------------
        info = {
            "stop_triggered": stop_triggered,
            "trade_executed": result.trade_executed,
            "effective_delta_btc": result.effective_delta_btc,
            "equity": new_equity,
        }

        return next_state, reward, done, info
import numpy as np
from config import (
    INITIAL_BALANCE,
    LEVERAGE_LIMIT,
    MAX_DEBT,
    TRANSACTION_FEE,
    GAMMA,
    DECISION_INTERVAL,
    TRAIN_TEST_SPLIT,
)


class BitcoinTradingEnv:

    def __init__(self, price_ary, tech_ary, signal_ary, mode="train"):
        """
        RL Trading Environment for Bitcoin.
        price_ary:  np.ndarray of shape (T, P) - price features
        tech_ary:   np.ndarray of shape (T, K) - technical indicators
        signal_ary: np.ndarray of shape (T, S) - strategy outputs
        mode:       "train" | "test"
        """

        assert mode in ["train", "test"], "mode must be 'train' or 'test'."
        self.mode = mode

        # ------------------------------------------------------------
        # Load configuration values
        # ------------------------------------------------------------
        self.initial_balance = INITIAL_BALANCE
        self.leverage_limit = LEVERAGE_LIMIT
        self.max_debt = MAX_DEBT
        self.transaction_fee = TRANSACTION_FEE
        self.gamma = GAMMA
        self.decision_interval = DECISION_INTERVAL

        # ------------------------------------------------------------
        # Prepare historical data (splitting train/test)
        # ------------------------------------------------------------
        total_len = price_ary.shape[0]
        split_idx = int(total_len * TRAIN_TEST_SPLIT)

        if mode == "train":
            self.price_ary = price_ary[:split_idx]
            self.tech_ary = tech_ary[:split_idx]
            self.signal_ary = signal_ary[:split_idx]
        else: # test mode
            self.price_ary = price_ary[split_idx:]
            self.tech_ary = tech_ary[split_idx:]
            self.signal_ary = signal_ary[split_idx:]

        # ------------------------------------------------------------
        # Apply decision interval (downsampling)
        # ------------------------------------------------------------
        self.price_ary = self.price_ary[::self.decision_interval]
        self.tech_ary = self.tech_ary[::self.decision_interval]
        self.signal_ary = self.signal_ary[::self.decision_interval]

        # ------------------------------------------------------------
        # RL Environment internal state
        # ------------------------------------------------------------
        self.day = 0
        self.balance = float(self.initial_balance)
        self.holdings = 0.0  # BTC position (can be positive or negative)
        self.total_asset = self.balance
        self.gamma_return = 0.0
        self.episode_return = 0.0

        # ------------------------------------------------------------
        # Environment dimensions (state & action)
        # ------------------------------------------------------------
        price_dim = self.price_ary.shape[1]
        tech_dim = self.tech_ary.shape[1]
        sig_dim = self.signal_ary.shape[1]

        # State includes: balance, price features, indicators, strategy signals, position size
        self.state_dim = 1 + price_dim + tech_dim + sig_dim + 1
        self.action_dim = 1 # continuous: [-1, +1] meaning sell/buy scaled
        self.if_discrete = False

        # Maximum episode length
        self.max_step = self.price_ary.shape[0]

        # Load first observation
        self.current_price = self.price_ary[0]
        self.current_tech = self.tech_ary[0]
        self.current_signal = self.signal_ary[0]

        # Initialize state
        self.state = self.reset()

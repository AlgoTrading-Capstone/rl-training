import numpy as np
from bitcoin_env import BitcoinTradingEnv


class ElegantRLBitcoinEnv:
    """
    Adapter between BitcoinTradingEnv (domain environment)
    and ElegantRL-compatible environment API.
    """
    def __init__(
        self,
        bitcoin_env=None,
        *,
        price_array=None,
        tech_array=None,
        turbulence_array=None,
        signal_array=None,
        mode="train",
        env_name="BitcoinTradingEnv",
        num_envs=1,
        max_step=None,
        state_dim=None,
        action_dim=None,
        if_discrete=False,
        **_,
    ):
        """
        Initialize an ElegantRL-compatible environment.

        Notes
        -----
        - Return types for reset/step are numpy-based to match ElegantRL expectations.
        - If `bitcoin_env` is not provided, the adapter builds BitcoinTradingEnv from arrays.
        - `**_` allows ElegantRL to pass extra kwargs without breaking this adapter.
        """
        if bitcoin_env is None:
            bitcoin_env = BitcoinTradingEnv(
                price_array,
                tech_array,
                turbulence_array,
                signal_array,
                mode=mode,
            )

        self.env = bitcoin_env
        self.env_name = env_name
        self.num_envs = num_envs
        self.if_discrete = if_discrete

        self.state_dim = state_dim if state_dim is not None else bitcoin_env.state_dim
        self.action_dim = action_dim if action_dim is not None else bitcoin_env.action_dim
        self.max_step = max_step if max_step is not None else bitcoin_env.max_step

        self._step_count = 0


    def reset(self):
        """
        ElegantRL expects:
            state, info_dict
        """
        self._step_count = 0

        state = self.env.reset()
        state = np.asarray(state, dtype=np.float32)

        info = {}
        return state, info


    def step(self, action):
        """
        ElegantRL expects:
            next_state, reward, terminated, truncated, info

        Domain env returns:
            next_state, reward, done, info
        """
        action = self._to_numpy(action).astype(np.float32, copy=False)

        next_state, reward, done, info = self.env.step(action)

        self._step_count += 1

        # Termination semantics:
        # - terminated: environment ended "naturally" (end of data / episode logic)
        # - truncated : ended due to an external time limit (not typical in our domain env)
        terminated = bool(done)
        truncated = bool((not done) and (self._step_count >= self.max_step))

        next_state = np.asarray(next_state, dtype=np.float32)
        reward = float(reward)

        return next_state, reward, terminated, truncated, info


    # close (for API completeness)
    def close(self):
        pass


    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(x):
        """
        Convert torch.Tensor (CPU/GPU) to numpy, otherwise np.asarray.
        """
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

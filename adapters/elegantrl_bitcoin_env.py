import numpy as np


class ElegantRLBitcoinEnv:
    """
    Adapter between BitcoinTradingEnv (domain environment)
    and ElegantRL-compatible environment API.

    Responsibilities:
    - Translate reset/step signatures
    - Provide metadata expected by ElegantRL
    - Handle tensor/np conversions
    - Separate 'terminated' vs 'truncated'
    """

    def __init__(self, bitcoin_env, env_name="BitcoinTradingEnv",
                 num_envs=1, max_step=None, state_dim=None, action_dim=None,
                 if_discrete=False, **_):
        """
        Initialize an ElegantRL-compatible wrapper around the domain environment.

        Notes
        -----
        - This adapter keeps BitcoinTradingEnv unchanged and only translates its API
          to what ElegantRL expects (reset/step signatures + required attributes).
        - `**_` allows ElegantRL to pass extra kwargs without breaking the wrapper.
        """
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
        action = self._to_numpy(action)

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
        Convert numpy, lists, and torch tensors to numpy arrays.
        """
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)
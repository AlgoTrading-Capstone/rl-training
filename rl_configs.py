"""
ElegantRL configuration builder.
"""

import os
from elegantrl.train.config import Config
from elegantrl.agents import AgentPPO, AgentSAC
from adapters.elegantrl_bitcoin_env import ElegantRLBitcoinEnv

from config import (
    RL_MODEL,
    GAMMA,
    LEARNING_RATE,
    NET_DIMS,
    TOTAL_TRAINING_STEPS,
    SEED,
)

SUPPORTED_RL_MODELS = ("PPO", "SAC")

PPO_CONFIG = {
    "horizon_len": 2048,       # Steps collected per rollout before update
    "repeat_times": 8,         # Gradient passes over the same rollout batch
    "ratio_clip": 0.2,         # PPO clip threshold (stability)
    "lambda_gae_adv": 0.95,    # GAE lambda (bias/variance tradeoff)
    "lambda_entropy": 0.01,    # Exploration encouragement
    "if_use_vtrace": True,     # Use V-trace off-policy correction
}

SAC_CONFIG = {
    "buffer_size": int(1e6),   # Replay buffer capacity
    "batch_size": 256,         # Batch size per update
    "soft_update_tau": 0.005,  # Target network Polyak factor
    "alpha": None,             # None = automatic entropy tuning
}


def _validate_net_dims(state_dim: int, net_dims: list[int]) -> None:
    """
    Validate network dimensions.

    Rules:
    - NET_DIMS must not be empty
    - The first hidden layer must be >= state_dim (your project policy)
    """
    if not net_dims:
        raise ValueError("NET_DIMS is empty. Provide something like [128, 128].")

    first_layer = int(net_dims[0])
    if first_layer < int(state_dim):
        raise ValueError(
            f"Invalid NET_DIMS: NET_DIMS[0] ({first_layer}) < state_dim ({state_dim}). "
            f"Fix: set NET_DIMS[0] >= {state_dim}."
        )


def build_elegantrl_config(
    *,
    price_array,
    tech_array,
    turbulence_array,
    signal_array,
    state_dim: int,
    action_dim: int,
    train_max_step: int,
    eval_max_step: int,
    run_path: str,
) -> Config:
    if RL_MODEL not in SUPPORTED_RL_MODELS:
        raise ValueError(f"Unsupported RL_MODEL='{RL_MODEL}'. Supported: {SUPPORTED_RL_MODELS}")

    if RL_MODEL == "PPO":
        agent_class = AgentPPO
        algo_cfg = PPO_CONFIG
    else:
        agent_class = AgentSAC
        algo_cfg = SAC_CONFIG

    train_env_args = {
        "env_name": "BitcoinTradingEnv",
        "num_envs": 1,
        "max_step": int(train_max_step),
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "if_discrete": False,

        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "signal_array": signal_array,
        "mode": "train",
    }

    eval_env_args = dict(train_env_args)
    eval_env_args["max_step"] = int(eval_max_step)
    eval_env_args["mode"] = "test"

    # Enforce NET_DIMS[0] >= state_dim
    _validate_net_dims(train_env_args["state_dim"], NET_DIMS)

    erl_config = Config(
        agent_class=agent_class,
        env_class=ElegantRLBitcoinEnv,
        env_args=train_env_args,
    )

    # Put ElegantRL outputs in a dedicated subfolder
    erl_config.cwd = os.path.join(run_path, "elegantrl")
    os.makedirs(erl_config.cwd, exist_ok=True)

    # IMPORTANT: never delete the run folder contents
    erl_config.if_remove = False

    # Reproducibility
    erl_config.random_seed = SEED

    # Training control
    erl_config.break_step = TOTAL_TRAINING_STEPS
    erl_config.gamma = GAMMA
    erl_config.learning_rate = LEARNING_RATE
    erl_config.net_dims = NET_DIMS
    erl_config.if_discrete = train_env_args["if_discrete"]

    # Algorithm-specific parameters (set even if None, e.g. SAC alpha=None)
    for key, value in algo_cfg.items():
        setattr(erl_config, key, value)

    # Evaluation
    erl_config.eval_env_class = ElegantRLBitcoinEnv
    erl_config.eval_env_args = eval_env_args
    erl_config.eval_per_step = int(2e4)
    erl_config.eval_times = 8

    return erl_config
"""
ElegantRL configuration builder.
"""

from pathlib import Path

from elegantrl.train.config import Config
from elegantrl.agents import AgentPPO, AgentSAC
from adapters.elegantrl_bitcoin_env import ElegantRLBitcoinEnv
from utils.metadata import enrich_metadata_with_training_config

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
    "lambda_entropy": 0.01,    # Exploration encouragement (start value)
    "lambda_entropy_end": 0.001,  # End value — linearly decayed over break_step
    "target_kl": 0.015,        # KL early-stop: break inner loop if approx_kl > 1.5 * target_kl
    "if_use_vtrace": True,     # Use V-trace off-policy correction
    "reward_scale": 2 ** 7,    # Scale log-return rewards (~1e-4) to critic-friendly magnitude
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
    datetime_array,
    strategy_names,
    state_dim: int,
    action_dim: int,
    train_max_step: int,
    eval_max_step: int,
    run_path: Path,
    logger,
    use_gpu: bool = True,
) -> Config:
    if RL_MODEL not in SUPPORTED_RL_MODELS:
        raise ValueError(f"Unsupported RL_MODEL='{RL_MODEL}'. Supported: {SUPPORTED_RL_MODELS}")

    if RL_MODEL == "PPO":
        # Safety Check for PPO Horizon
        horizon = PPO_CONFIG["horizon_len"]
        if train_max_step < horizon:
            raise ValueError(
                f"Training data length ({train_max_step}) is smaller than PPO horizon_len ({horizon}). "
                "Collect more data or decrease horizon_len."
            )

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
        "datetime_array": datetime_array,
        "strategy_names": strategy_names,
        "mode": "train",
    }

    train_env_args["signal_log_dir"] = str(run_path)
    train_env_args["signal_log_flush_every"] = 100

    eval_env_args = dict(train_env_args)
    eval_env_args["max_step"] = int(eval_max_step)
    eval_env_args["mode"] = "test"
    eval_env_args["signal_log_dir"] = str(run_path)
    eval_env_args["signal_log_filename"] = "strategy_signals_log.csv"
    eval_env_args["signal_log_flush_every"] = 100

    # Enforce NET_DIMS[0] >= state_dim
    _validate_net_dims(train_env_args["state_dim"], NET_DIMS)

    erl_config = Config(
        agent_class=agent_class,
        env_class=ElegantRLBitcoinEnv,
        env_args=train_env_args,
    )

    # Put ElegantRL outputs in a dedicated subfolder
    erl_output_dir = run_path / "elegantrl"
    erl_output_dir.mkdir(exist_ok=True)
    erl_config.cwd = str(erl_output_dir)

    # IMPORTANT: never delete the run folder contents
    erl_config.if_remove = False

    # GPU selection
    erl_config.gpu_id = 0 if use_gpu else -1

    # Reproducibility
    erl_config.random_seed = SEED

    # Training control
    erl_config.break_step = TOTAL_TRAINING_STEPS
    erl_config.gamma = GAMMA
    erl_config.learning_rate = LEARNING_RATE
    erl_config.net_dims = list(NET_DIMS)  # Make a copy
    erl_config.if_discrete = train_env_args["if_discrete"]

    # Algorithm-specific parameters (set even if None, e.g. SAC alpha=None)
    for key, value in algo_cfg.items():
        setattr(erl_config, key, value)

    # Evaluation
    erl_config.eval_env_class = ElegantRLBitcoinEnv
    erl_config.eval_env_args = eval_env_args
    # Eval / checkpoint cadence: 5K-step eval window × save_gap=5 = periodic checkpoint
    # every 25K training steps, so early collapses (observed ~25K in novix_nostrat runs)
    # are captured before being overwritten by the next best-val save.
    erl_config.eval_per_step = int(5e3)
    erl_config.save_gap = 5
    erl_config.eval_times = 8
    # Enable TensorBoard logging for reward curves, losses, and custom scalars (e.g. strategy signals)
    erl_config.if_use_tb = True
    # Enrich metadata with training + env interface (contract)
    enrich_metadata_with_training_config(
        run_path=run_path,
        algorithm_config=algo_cfg,
        state_dim=state_dim,
        action_dim=action_dim,
        logger=logger,
    )

    return erl_config

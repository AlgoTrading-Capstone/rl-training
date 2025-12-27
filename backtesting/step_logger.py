from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any, Tuple

import config
from utils.normalization import inverse_normalize_state


class StepLogger:
    """
    Writes human-readable step-level backtest logs to steps.csv.

    Responsibilities:
    - Build human-readable, de-normalized state for steps.csv.
    - Decode strategy decisions (4 binary -> single decision).
    - Translate agent actions to business terms (exposure %, stop-loss %).
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.file_path = self.out_dir / "steps.csv"
        self._file = None
        self._writer = None

    def open(self) -> None:
        self._file = self.file_path.open("w", newline="", encoding="utf-8")
        self._writer = None

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log_step(
        self,
        *,
        step_idx: int,
        timestamp,
        state_norm,
        price_vec,
        tech_vec,
        turb_vec,
        sig_vec,
        a_pos: float,
        a_sl: float,
        trade_executed: bool,
        effective_delta_btc: float,
        balance: float,
        holdings: float,
        equity: float,
        reward: float,
        stop_triggered: bool,
        done: bool,
    ) -> None:
        """
        Log a single backtest step in a unified, human-readable format.
        """

        # ----------------------------------------------------
        # Human-readable (de-normalized) state for steps.csv
        # ----------------------------------------------------
        state_human = inverse_normalize_state(
            state_norm=state_norm,
            price_vec=price_vec,
            tech_vec=tech_vec,
            turbulence_vec=turb_vec,
            signal_vec=sig_vec,
        )

        # ----------------------------------------------------
        # Decode strategy decisions (4 binary -> single decision)
        # Each strategy occupies 4 columns: LONG, SHORT, FLAT, HOLD
        # ----------------------------------------------------
        strategy_decisions = self._decode_strategy_decisions(sig_vec)

        # ----------------------------------------------------
        # Agent action in % terms (business interpretation)
        # ----------------------------------------------------
        agent_exposure_pct, agent_stop_loss_pct = self._decode_agent_action(a_pos, a_sl)

        # ----------------------------------------------------
        # Build row
        # ----------------------------------------------------
        row: Dict[str, Any] = {}

        # Time
        row["step_idx"] = int(step_idx)
        row["timestamp"] = timestamp

        # State (human-readable, de-normalized)
        for k in ("open", "high", "low", "close", "volume"):
            if k in state_human:
                row[k] = state_human[k]

        for name, value in state_human.get("indicators", {}).items():
            row[name] = value

        for name, value in state_human.get("turbulence", {}).items():
            row[name] = value

        # Strategies (single decision per strategy)
        for strategy_name, decision in strategy_decisions.items():
            row[strategy_name] = decision

        # Agent decision (business interpretation)
        row["agent_target_exposure_pct"] = float(agent_exposure_pct)
        row["agent_stop_loss_pct"] = float(agent_stop_loss_pct)

        # Execution
        row["trade_executed"] = bool(trade_executed)
        row["effective_delta_btc"] = float(effective_delta_btc)

        # Account state
        row["balance"] = float(balance)
        row["holdings"] = float(holdings)
        row["equity"] = float(equity)
        row["reward"] = float(reward)

        # Control flags
        # NOTE: stop_triggered refers to the candle that just closed.
        row["stop_triggered"] = bool(stop_triggered)
        row["done"] = bool(done)

        # Write row
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=row.keys())
            self._writer.writeheader()

        self._writer.writerow(row)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _decode_strategy_decisions(sig_vec) -> Dict[str, str]:
        # If strategies are disabled (or no signals), nothing to decode
        if not bool(getattr(config, "ENABLE_STRATEGIES", False)) or sig_vec is None or len(sig_vec) == 0:
            return {}

        expected = 4 * len(config.STRATEGY_LIST)
        if len(sig_vec) != expected:
            # Fail loudly - signal vector layout must match strategy list exactly
            raise ValueError(
                f"signal_vec length mismatch: got {len(sig_vec)}, expected {expected} "
                f"(4 per strategy * {len(config.STRATEGY_LIST)} strategies)."
            )

        decisions: Dict[str, str] = {}
        idx = 0

        for strategy_name in config.STRATEGY_LIST:
            s = sig_vec[idx: idx + 4]
            idx += 4

            if s[0] == 1:
                decision = "FLAT"
            elif s[1] == 1:
                decision = "LONG"
            elif s[2] == 1:
                decision = "SHORT"
            elif s[3] == 1:
                decision = "HOLD"
            else:
                decision = "UNKNOWN"

            decisions[strategy_name] = decision

        return decisions

    @staticmethod
    def _decode_agent_action(a_pos: float, a_sl: float) -> Tuple[float, float]:
        agent_exposure_pct = float(a_pos * 100.0)

        # a_sl ∈ [-1, 1] -> map to stop loss pct range -> convert to %
        a_sl01 = 0.5 * (float(a_sl) + 1.0)
        stop_loss_pct = (
            config.MIN_STOP_LOSS_PCT
            + a_sl01 * (config.MAX_STOP_LOSS_PCT - config.MIN_STOP_LOSS_PCT)
        )
        agent_stop_loss_pct = float(stop_loss_pct * 100.0)

        return agent_exposure_pct, agent_stop_loss_pct
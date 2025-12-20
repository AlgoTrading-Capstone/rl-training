"""
State debug logger for backtesting.

Writes low-level, ML-oriented information into state_debug.csv:
- normalized state vector
- raw feature vectors
- raw actions and reward

This file is intended for debugging, research and forensic analysis only.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


class StateDebugLogger:
    """
    Logs normalized state and raw inputs for each backtest step.
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.file_path = self.out_dir / "state_debug.csv"
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

    def log_step(
        self,
        *,
        step_idx: int,
        timestamp,
        state_norm,
        price_vec,
        tech_vec,
        turb_vec,
        signal_vec,
        action_a_pos: float,
        action_a_sl: float,
        reward: float,
    ) -> None:
        """
        Log a single debug step.
        """

        row: Dict[str, Any] = {}

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------
        row["step_idx"] = step_idx
        row["timestamp"] = timestamp

        # ----------------------------------------------------
        # Normalized state (what the model actually saw)
        # ----------------------------------------------------
        for i, v in enumerate(state_norm):
            row[f"state_{i}"] = float(v)

        # ----------------------------------------------------
        # Raw input vectors (pre-normalization)
        # ----------------------------------------------------
        for i, v in enumerate(price_vec):
            row[f"price_raw_{i}"] = float(v)

        for i, v in enumerate(tech_vec):
            row[f"tech_raw_{i}"] = float(v)

        for i, v in enumerate(turb_vec):
            row[f"turb_raw_{i}"] = float(v)

        for i, v in enumerate(signal_vec):
            row[f"signal_raw_{i}"] = float(v)

        # ----------------------------------------------------
        # Action & reward (raw)
        # ----------------------------------------------------
        row["action_a_pos"] = float(action_a_pos)
        row["action_a_sl"] = float(action_a_sl)
        row["reward"] = float(reward)

        # ----------------------------------------------------
        # Write row
        # ----------------------------------------------------
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=row.keys())
            self._writer.writeheader()

        self._writer.writerow(row)
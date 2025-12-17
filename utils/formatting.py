"""
Plain-text formatting helpers.

Important: This module must not emit colors or log levels. The logger is responsible
for presentation; these helpers only return strings.
"""

from __future__ import annotations


class Formatter:
    @staticmethod
    def section_separator(title: str, width: int = 60) -> str:
        line = "=" * width
        return f"{line}\n{title}\n{line}"

    @staticmethod
    def error_context(error_msg: str, context: str | None = None) -> str:
        if context:
            return f"{context}\n{error_msg}"
        return error_msg

    @staticmethod
    def config_table(metadata: dict) -> str:
        """Returns plain text configuration table for training runs."""
        lines = [
            "",
            "=" * 60,
            " Training Configuration",
            "=" * 60,
            "",
            "MODEL DETAILS:",
            f"  Name: {metadata.get('model_name', 'N/A')}",
            f"  Description: {metadata.get('description', 'N/A')}",
            f"  Mode: {metadata.get('mode', 'N/A')}",
            "",
            "DATE RANGES:",
        ]

        if "training" in metadata:
            lines.extend([
                f"  Training: {metadata['training'].get('start_date', 'N/A')} to {metadata['training'].get('end_date', 'N/A')}",
            ])

        if "backtest" in metadata:
            lines.extend([
                f"  Backtest: {metadata['backtest'].get('start_date', 'N/A')} to {metadata['backtest'].get('end_date', 'N/A')}",
            ])

        lines.extend([
            "",
            "RL CONFIGURATION:",
            f"  Algorithm: {metadata.get('rl_model', 'N/A')}",
            f"  Learning Rate: {metadata.get('learning_rate', 'N/A')}",
            f"  Gamma: {metadata.get('gamma', 'N/A')}",
            f"  Network Dims: {metadata.get('net_dims', 'N/A')}",
            "",
            "STRATEGIES:",
            f"  Enabled: {metadata.get('strategies_enabled', 'N/A')}",
            f"  List: {', '.join(metadata.get('strategy_list', [])) if metadata.get('strategy_list') else 'None'}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)



"""
Plain-text formatting helpers.

Important: This module must not emit colors or log levels. The logger is responsible
for presentation; these helpers only return strings.
"""

from __future__ import annotations


class Formatter:
    @staticmethod
    def section_separator(title: str, width: int = None) -> str:
        """Create a section separator with adaptive width."""
        if width is None:
            width = max(60, len(title) + 4)
        line = "=" * width
        return f"{line}\n{title}\n{line}"

    @staticmethod
    def duration_format(seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    @staticmethod
    def list_format(items: list, bullet: str = "•") -> str:
        """Format list with bullets."""
        if not items:
            return "  None"
        return "\n".join([f"  {bullet} {item}" for item in items])

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
            f"  Description: {metadata.get('description', 'N/A') or 'None'}",
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

        # RL config (may be enriched later)
        rl_config = metadata.get('rl', {})
        lines.extend([
            "",
            "RL CONFIGURATION:",
            f"  Algorithm: {rl_config.get('model', 'N/A')}",
            f"  Learning Rate: {rl_config.get('learning_rate', 'N/A')}",
            f"  Gamma: {rl_config.get('gamma', 'N/A')}",
            f"  Network Dims: {rl_config.get('net_dims', 'N/A')}",
        ])

        # Strategies config (may be enriched later)
        strategies_config = metadata.get('strategies', {})
        strategy_list = strategies_config.get('strategy_list', [])
        lines.extend([
            "",
            "STRATEGIES:",
            f"  Enabled: {strategies_config.get('enabled', 'N/A')}",
            f"  List: {', '.join(strategy_list) if strategy_list else 'None'}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)



"""
Plain-text formatting helpers.

Important: This module must not emit colors or log levels. The logger is responsible
for presentation; these helpers only return strings.
"""

from __future__ import annotations # For type hinting of str | None

import config


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
    def display_training_config(metadata: dict, logger) -> None:
        """
        Display training configuration by combining metadata with current config.py values.
        This avoids showing 'N/A' for RL settings that haven't been enriched yet.
        
        Args:
            metadata: User-provided metadata (model name, dates, description)
            logger: Logger instance for output
        """
        lines = [
            "",
            "=" * 60,
            " Training Configuration",
            "=" * 60,
            "",
            "MODEL DETAILS:",
            f"  Name: {metadata.get('model_name', 'N/A')}",
            f"  Description: {metadata.get('description', 'N/A') or 'None'}",
            f"  Mode: {metadata.get('run_mode', 'N/A')}",
            "",
            "DATE RANGES:",
        ]
        
        if "training" in metadata:
            train = metadata["training"]
            lines.append(f"  Training: {train.get('start_date', 'N/A')} to {train.get('end_date', 'N/A')}")
        
        if "backtest" in metadata:
            bt = metadata["backtest"]
            lines.append(f"  Backtest: {bt.get('start_date', 'N/A')} to {bt.get('end_date', 'N/A')}")
        
        # Read from config.py directly (not from metadata which isn't enriched yet)
        lines.extend([
            "",
            "RL CONFIGURATION:",
            f"  Algorithm: {config.RL_MODEL}",
            f"  Learning Rate: {config.LEARNING_RATE}",
            f"  Gamma: {config.GAMMA}",
            f"  Network Dims: {config.NET_DIMS}",
            "",
            "STRATEGIES:",
            f"  Enabled: {config.ENABLE_STRATEGIES}",
            f"  List: {', '.join(config.STRATEGY_LIST) if config.ENABLE_STRATEGIES and config.STRATEGY_LIST else 'None'}",
            "",
            "=" * 60,
        ])
        
        logger.info("\n".join(lines))




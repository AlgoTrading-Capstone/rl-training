"""
Strategies Package

This package contains all trading strategy implementations.
"""

from .base_strategy import (
    BaseStrategy,
    SignalType,
    StrategyRecommendation
)

__all__ = [
    'BaseStrategy',
    'SignalType',
    'StrategyRecommendation'
]
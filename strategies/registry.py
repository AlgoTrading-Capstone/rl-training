"""
Strategy Registry

Provides dynamic loading and instantiation of trading strategies by name.
This allows configuration-driven strategy selection without code changes.
"""

from typing import Dict, Type, List
from strategies.base_strategy import BaseStrategy
from strategies.supertrend_legacy_strategy import SupertrendStrategy
from strategies.pg_qsd_for_nifty_future_strategy import PgQsdForNiftyFutureStrategy
from strategies.monthly_returns_in_pinescript_strategies_strategy import MonthlyReturnsInPinescriptStrategiesStrategy
from strategies.trend_pullback_momentum_side_aware_strategy import TrendPullbackMomentumSideAwareStrategy
from strategies.ema5_breakout_target_shifting_mtf_strategy import Ema5BreakoutTargetShiftingMtfStrategy


class StrategyRegistry:
    """
    Registry for trading strategies.
    Handles dynamic loading and instantiation of strategy classes by name.
    """

    # Map strategy names to classes
    # Only the 4 strategies that passed the 5% signal-activity variance filter.
    # Dead strategies archived in archive_strategies/ — see strategy_post_mortem_analysis.md
    _STRATEGIES: Dict[str, Type[BaseStrategy]] = {
        "SupertrendStrategy": SupertrendStrategy,
        "PgQsdForNiftyFutureStrategy": PgQsdForNiftyFutureStrategy,
        "MonthlyReturnsInPinescriptStrategiesStrategy": MonthlyReturnsInPinescriptStrategiesStrategy,
        "TrendPullbackMomentumSideAwareStrategy": TrendPullbackMomentumSideAwareStrategy,
            "Ema5BreakoutTargetShiftingMtfStrategy": Ema5BreakoutTargetShiftingMtfStrategy,
    }

    @classmethod
    def get_strategy(cls, name: str) -> BaseStrategy:
        """
        Get strategy instance by name.

        Args:
            name: Strategy class name (e.g., "AwesomeMacd")

        Returns:
            Instantiated strategy object

        Raises:
            ValueError: If strategy name not found in registry
        """
        if name not in cls._STRATEGIES:
            available = ', '.join(cls._STRATEGIES.keys())
            raise ValueError(
                f"Strategy '{name}' not found in registry. "
                f"Available strategies: {available}"
            )

        strategy_class = cls._STRATEGIES[name]
        return strategy_class()

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        Return list of available strategy names.

        Returns:
            List of registered strategy names
        """
        return list(cls._STRATEGIES.keys())

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a new strategy dynamically.

        Args:
            name: Strategy name for lookup
            strategy_class: Strategy class to register

        Raises:
            TypeError: If strategy_class doesn't inherit from BaseStrategy
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(f"{strategy_class.__name__} must inherit from BaseStrategy")

        cls._STRATEGIES[name] = strategy_class
        print(f"Registered strategy: {name}")
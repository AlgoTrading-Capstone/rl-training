"""
Strategy Registry

Provides dynamic loading and instantiation of trading strategies by name.
This allows configuration-driven strategy selection without code changes.
"""

from typing import Dict, Type, List
from strategies.base_strategy import BaseStrategy
from strategies.supertrend_strategy import SupertrendStrategy
from strategies.bjorgum_double_tap_strategy import BjorgumDoubleTapStrategy
from strategies.evasive_super_trend_strategy_source_select_strategy import EvasiveSuperTrendStrategySourceSelectStrategy
from strategies.kama_trend_strategy import KamaTrendStrategy
from strategies.new_tott_strategy import NewTottStrategy
from strategies.rudy_breakout_momentum_v2_strategy import RudyBreakoutMomentumV2Strategy
from strategies.trendmaster_pro_2_3_with_alerts_strategy import TrendmasterPro23WithAlertsStrategy
from strategies.all_day_futures_scalper_ema_trend_cross_atr_brackets_strategy import AllDayFuturesScalperEmaTrendCrossAtrBrackets
from strategies.threecommas_bot_strategy import ThreeCommasBotStrategy
from strategies.ny15m_orb_with_a_fixed_sl_tp_nasdaq_strategy import Ny15mOrbWithAFixedSlTpNasdaqStrategy
from strategies.traling_sl_target_strategy import TralingSLTargetStrategy


class StrategyRegistry:
    """
    Registry for trading strategies.
    Handles dynamic loading and instantiation of strategy classes by name.
    """

    # Map strategy names to classes
    _STRATEGIES: Dict[str, Type[BaseStrategy]] = {
        "SupertrendStrategy": SupertrendStrategy,
        "BjorgumDoubleTapStrategy": BjorgumDoubleTapStrategy,
        "EvasiveSuperTrendStrategySourceSelectStrategy": EvasiveSuperTrendStrategySourceSelectStrategy,
        "KamaTrendStrategy": KamaTrendStrategy,
        "NewTottStrategy": NewTottStrategy,
        "RudyBreakoutMomentumV2Strategy": RudyBreakoutMomentumV2Strategy,
        "TrendmasterPro23WithAlertsStrategy": TrendmasterPro23WithAlertsStrategy,
        "AllDayFuturesScalperEmaTrendCrossAtrBrackets": AllDayFuturesScalperEmaTrendCrossAtrBrackets,
        "ThreeCommasBotStrategy": ThreeCommasBotStrategy,
        "Ny15mOrbWithAFixedSlTpNasdaqStrategy": Ny15mOrbWithAFixedSlTpNasdaqStrategy,
        "TralingSLTargetStrategy": TralingSLTargetStrategy,
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
"""
Base Strategy Interface

This module defines the abstract base class that all trading strategies must inherit from.
It establishes a universal contract for strategy input/output and core methods.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import NamedTuple
import pandas as pd


class SignalType(Enum):
    """
    Enumeration of possible trading signals.
    """
    LONG = "LONG"  # Target position is long
    SHORT = "SHORT"  # Target position is short
    FLAT = "FLAT"  # Target position is zero (no exposure)
    HOLD = "HOLD"  # No recommendation / keep current exposure


class StrategyRecommendation(NamedTuple):
    """Standard output of every strategy."""
    signal: SignalType
    timestamp: datetime


class BaseStrategy(ABC):
    """
    Base interface for all trading strategies.

    Each strategy must define:
        - name
        - description
        - timeframe (string)
        - lookback_hours (int)

    And must implement:
        - run(df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation
    """

    def __init__(self, name: str, description: str, timeframe: str, lookback_hours: int):
        self._name = name
        self._description = description
        self._timeframe = timeframe
        self._lookback_hours = lookback_hours

    # ---- getters (read-only) ----

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def timeframe(self) -> str:
        return self._timeframe

    @property
    def lookback_hours(self) -> int:
        return self._lookback_hours

    # ---- required method ----

    @abstractmethod
    def run(self, df: pd.DataFrame, timestamp: datetime) -> StrategyRecommendation:
        """
        Execute the logic of the strategy.

        :param df: pandas DataFrame containing candle data
        :param timestamp: datetime of evaluation (UTC)
        :return: StrategyRecommendation(signal, timestamp)
        """
        pass
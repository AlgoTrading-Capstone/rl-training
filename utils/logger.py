"""
Loguru-based logger utilities for the RL training pipeline.

Design goals:
- Colorized, concise console logs (INFO+) to stderr (tqdm-friendly)
- Debug-level file logs under: {run_path}/logs/
- Strict component routing to dedicated log files
- Safe handler lifecycle to prevent duplicate logs
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from loguru import logger as _logger


class LogComponent(str, Enum):
    MAIN = "MAIN"
    DATA = "DATA"
    STRATEGY = "STRATEGY"
    TRAINING = "TRAINING"
    BACKTEST = "BACKTEST"


def _component_filter(allowed: Iterable[str]) -> Callable[[dict], bool]:
    allowed_set = set(allowed)

    def _filter(record: dict) -> bool:
        return record["extra"].get("component") in allowed_set

    return _filter


@dataclass(frozen=True)
class _LoggerView:
    """A lightweight view over the globally-configured loguru logger."""

    bound: Any

    def bind(self, **kwargs):
        return _LoggerView(self.bound.bind(**kwargs))

    def debug(self, msg: str) -> None:
        self.bound.debug(msg)

    def info(self, msg: str) -> None:
        self.bound.info(msg)

    def warning(self, msg: str) -> None:
        self.bound.warning(msg)

    def error(self, msg: str) -> None:
        self.bound.error(msg)

    def success(self, msg: str) -> None:
        self.bound.success(msg)

    def exception(self, msg: str) -> None:
        self.bound.exception(msg)

    @contextmanager
    def phase(self, name: str, phase_num: int, total_phases: int):
        """Log phase start + duration, and always log SUCCESS/FAILED appropriately."""
        self.bound.info(f"[Phase {phase_num}/{total_phases}] {name}")
        start = time.time()
        failed = False
        try:
            yield
        except Exception as e:
            failed = True
            elapsed = time.time() - start
            self.bound.error(f"[Phase {phase_num}/{total_phases}] FAILED after {elapsed:.2f}s: {e}")
            raise
        finally:
            if not failed:
                elapsed = time.time() - start
                self.bound.success(f"[Phase {phase_num}/{total_phases}] Completed in {elapsed:.2f}s")


class RLLogger:
    """
    Logger wrapper that configures loguru sinks once per instance.

    IMPORTANT:
    - Each RLLogger initialization calls loguru's global logger.remove()
      to avoid handler duplication. Do not instantiate repeatedly in tight loops.
    """

    def __init__(
        self,
        run_path: Optional[str | Path] = None,
        log_level: str = "INFO",
        file_log_level: str = "DEBUG",
        component: LogComponent = LogComponent.MAIN,
    ) -> None:
        # Prevent duplicate logs by removing ALL existing handlers.
        _logger.remove()
        self._handler_ids: list[int] = []

        self.run_path = Path(run_path) if run_path else None
        self.log_level = log_level
        self.file_log_level = file_log_level

        # Ensure a default component exists for any non-bound logs.
        _logger.configure(extra={"component": component.value})
        self._bound = _logger.bind(component=component.value)

        self._setup_sinks()

    def _setup_sinks(self) -> None:
        # Console handler: stderr to avoid corrupting tqdm on stdout.
        console_id = _logger.add(
            sys.stderr,
            level=self.log_level,
            colorize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        )
        self._handler_ids.append(console_id)

        if not self.run_path:
            return

        log_dir = self.run_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # training.log gets MAIN + TRAINING (so entrypoint + training pipeline are together)
        self._handler_ids.append(
            _logger.add(
                log_dir / "training.log",
                level=self.file_log_level,
                rotation="50 MB",
                retention="30 days",
                enqueue=True,
                filter=_component_filter([LogComponent.MAIN.value, LogComponent.TRAINING.value]),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[component]: <9} | {message}",
            )
        )

        self._handler_ids.append(
            _logger.add(
                log_dir / "data_pipeline.log",
                level=self.file_log_level,
                rotation="50 MB",
                retention="30 days",
                enqueue=True,
                filter=_component_filter([LogComponent.DATA.value]),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[component]: <9} | {message}",
            )
        )

        self._handler_ids.append(
            _logger.add(
                log_dir / "strategy_execution.log",
                level=self.file_log_level,
                rotation="50 MB",
                retention="30 days",
                enqueue=True,
                filter=_component_filter([LogComponent.STRATEGY.value]),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[component]: <9} | {message}",
            )
        )

        self._handler_ids.append(
            _logger.add(
                log_dir / "backtest.log",
                level=self.file_log_level,
                rotation="50 MB",
                retention="30 days",
                enqueue=True,
                filter=_component_filter([LogComponent.BACKTEST.value]),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[component]: <9} | {message}",
            )
        )

    # -----------------------------
    # Primary logging API
    # -----------------------------
    def view(self) -> _LoggerView:
        return _LoggerView(self._bound)

    def for_component(self, component: LogComponent) -> _LoggerView:
        """Return a view bound to a specific component without reconfiguring sinks."""
        return _LoggerView(_logger.bind(component=component.value))

    def bind(self, **kwargs) -> _LoggerView:
        """Bind extra fields (keeps current component unless overridden)."""
        return _LoggerView(self._bound.bind(**kwargs))

    def debug(self, msg: str) -> None:
        self._bound.debug(msg)

    def info(self, msg: str) -> None:
        self._bound.info(msg)

    def warning(self, msg: str) -> None:
        self._bound.warning(msg)

    def error(self, msg: str) -> None:
        self._bound.error(msg)

    def success(self, msg: str) -> None:
        self._bound.success(msg)

    def exception(self, msg: str) -> None:
        self._bound.exception(msg)

    @contextmanager
    def phase(self, name: str, phase_num: int, total_phases: int):
        with self.view().phase(name, phase_num, total_phases):
            yield

    def cleanup(self) -> None:
        """Remove only handlers added by this instance."""
        for hid in list(self._handler_ids):
            try:
                _logger.remove(hid)
            except ValueError:
                pass
        self._handler_ids.clear()



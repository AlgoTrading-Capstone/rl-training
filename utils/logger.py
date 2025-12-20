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
from typing import Any, Callable, Iterable, Optional, Iterator
from loguru import logger as _logger

# Loguru records are dicts with specific structure
Record = dict[str, Any]


class LogComponent(str, Enum):
    """Logical components for log routing."""
    # str values for loguru extra field
    MAIN = "MAIN"
    DATA = "DATA"
    STRATEGY = "STRATEGY"
    TRAINING = "TRAINING"
    BACKTEST = "BACKTEST"


def _component_filter(allowed: Iterable[str]) -> Callable[[Record], bool]:
    """
    Factory function that creates a specialized filter for loguru sinks based on component names.

    This function utilizes a Closure pattern to capture a specific set of allowed components
    and returns a filter function that validates log records against this set. It is designed
    to route logs to specific destinations (e.g., separating "DATA" logs from "TRAINING" logs).

    Args:
        allowed (Iterable[str]): A collection of component names (strings) that are permitted
            by this filter. Example: `["DATA"]`.

    Returns:
        Callable[[dict], bool]: A filter predicate function compatible with loguru's `add()` method.
            The returned function takes a log `record` (dict) and returns:
            - `True`: If the record's component is present in the `allowed` set (log is kept).
            - `False`: If the component is missing or not allowed (log is discarded).
    """
    allowed_set = set(allowed)

    def _filter(record: Record) -> bool:
        return record["extra"].get("component") in allowed_set

    return _filter


@dataclass(frozen=True)
class _LoggerView:
    """A lightweight, immutable proxy wrapper around the configured loguru logger.

    This class serves as a restricted view of the global logger, typically bound to specific
    contextual information (like a specific component name). It enforces immutability to ensure
    thread safety and predictable logging behavior.

    It delegates standard logging calls (info, debug, error) to the underlying loguru instance
    while providing additional utilities such as context managers for phase tracking.

    Attributes:
        bound (Any): The underlying loguru logger instance, potentially pre-bound with
                     context variables (e.g., `logger.bind(component='DATA')`).
"""

    bound: Any # loguru logger bound instance

    def bind(self, **kwargs):
        """
                Creates a new LoggerView with additional context variables.

                Since this class is immutable (frozen), this method does not modify the current instance.
                Instead, it returns a new instance wrapping a new loguru logger with the added context.

                Args:
                    **kwargs: Arbitrary keyword arguments to bind to the log records (e.g., step=10).

                Returns:
                    _LoggerView: A new logger view instance with the updated context.
                """
        return _LoggerView(self.bound.bind(**kwargs))

    def debug(self, msg: str) -> None:
        """Log a message with severity 'DEBUG'."""
        self.bound.debug(msg)

    def info(self, msg: str) -> None:
        """Log a message with severity 'INFO'."""
        self.bound.info(msg)

    def warning(self, msg: str) -> None:
        """Log a message with severity 'WARNING'."""
        self.bound.warning(msg)

    def error(self, msg: str) -> None:
        """Log a message with severity 'ERROR'."""
        self.bound.error(msg)

    def success(self, msg: str) -> None:
        """Log a message with severity 'SUCCESS'."""
        self.bound.success(msg)

    def exception(self, msg: str) -> None:
        """Log a message with severity 'ERROR' including the stack trace."""
        self.bound.exception(msg)

    @contextmanager
    def phase(self, name: str, phase_num: int, total_phases: int):
        """Context manager that encapsulates a distinct execution phase, providing automatic logging
        for start, completion, and failure states with duration tracking.

        This method wraps a block of code to ensure consistent logging lifecycle:
        1. **Start:** Logs an INFO message with the phase name and index.
        2. **Execution:** Measures the wall-clock time taken by the block.
        3. **Failure:** If an exception occurs, logs an ERROR message with the elapsed time
           and re-raises the exception.
        4. **Success:** If the block completes without errors, logs a SUCCESS message with
           the elapsed time.

        Args:
            name (str): The human-readable description of the phase (e.g., "Data Loading").
            phase_num (int): The current step number (1-based index).
            total_phases (int): The total number of planned phases in the workflow.

        Yields:
            None: Yields control back to the caller's context block.

        Raises:
            Exception: Any exception raised within the `with` block is caught, logged as a
                       failure, and then re-raised to propagate the error upwards."""
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

    VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "SUCCESS"}

    def __init__(
        self,
        run_path: Optional[str | Path] = None,
        log_level: str = "INFO",
        file_log_level: str = "DEBUG",
        component: LogComponent = LogComponent.MAIN,
    ) -> None:
        # Validate log levels
        if log_level not in self.VALID_LOG_LEVELS:
            raise ValueError(f"Invalid log_level: {log_level}. Must be one of {self.VALID_LOG_LEVELS}")
        if file_log_level not in self.VALID_LOG_LEVELS:
            raise ValueError(f"Invalid file_log_level: {file_log_level}. Must be one of {self.VALID_LOG_LEVELS}")

        # Prevent duplicate logs by removing ALL existing handlers.
        _logger.remove()
        self._handler_ids: list[int] = [] # Track only our own handlers

        self.run_path = Path(run_path) if run_path else None
        self.log_level = log_level
        self.file_log_level = file_log_level

        # Bind to component (no need for configure)
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
    def phase(self, name: str, phase_num: int, total_phases: int) -> Iterator[None]:
        with self.view().phase(name=name, phase_num=phase_num, total_phases=total_phases):
            yield

    def cleanup(self) -> None:
        """Remove only handlers added by this instance."""
        for hid in list(self._handler_ids):
            try:
                _logger.remove(hid)
            except ValueError:
                pass
        self._handler_ids.clear()

    def __enter__(self):
        """Support context manager usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context manager exit."""
        self.cleanup()
        return False



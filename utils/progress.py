"""
Progress tracking helpers using tqdm.

All progress output is written to stderr to avoid conflicts with loguru console logging.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Callable, Iterator, Tuple

from tqdm import tqdm


class ProgressTracker:
    @staticmethod
    def download_chunks(total: int, desc: str = "Downloading") -> tqdm:
        return tqdm(
            total=total,
            desc=desc,
            unit="chunk",
            file=sys.stderr,
            leave=False,
            dynamic_ncols=True,
        )

    @staticmethod
    def process_items(total: int, desc: str = "Processing", unit: str = "item") -> tqdm:
        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            file=sys.stderr,
            leave=False,
            dynamic_ncols=True,
        )


@contextmanager
def safe_progress(total: int, desc: str, unit: str = "item") -> Iterator[Tuple[tqdm, Callable[[int], None]]]:
    """
    Context manager ensuring the progress bar reaches 100% and closes cleanly.
    Yields: (pbar, update_fn)
    """
    pbar = tqdm(
        total=total,
        desc=desc,
        unit=unit,
        file=sys.stderr,
        leave=False,
        dynamic_ncols=True,
    )
    try:
        yield pbar, pbar.update
    finally:
        remaining = total - pbar.n
        if remaining > 0:
            pbar.update(remaining)
        pbar.close()


@contextmanager
def spinner(desc: str) -> Iterator[tqdm]:
    """
    Spinner for indeterminate operations (no total).

    Args:
        desc: Operation description

    Yields:
        tqdm progress bar in spinner mode
    """
    pbar = tqdm(
        desc=desc,
        bar_format="{desc}: {elapsed}",
        file=sys.stderr,
        leave=False,
    )
    try:
        yield pbar
    finally:
        pbar.close()



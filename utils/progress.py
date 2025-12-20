"""
Progress tracking helpers using tqdm.

All progress output is written to stderr to avoid conflicts with loguru console logging.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Callable, Iterator

from tqdm import tqdm


class ProgressTracker:
    @staticmethod
    def download_chunks(total: int, desc: str = "Downloading") -> tqdm:
        """Create a tqdm progress bar for downloading chunks."""
        return tqdm(
            total=total,# how many chunks to download
            desc=desc, # description shown on progress bar
            unit="chunk", # unit of measurement
            file=sys.stderr, # output to stderr to avoid log conflicts
            leave=False, # do not leave progress bar after completion
            dynamic_ncols=True, # adapt width to terminal size
        )

    @staticmethod
    def process_items(total: int, desc: str = "Processing", unit: str = "item") -> tqdm:
        """Create a tqdm progress bar for processing items."""
        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            file=sys.stderr,
            leave=False,
            dynamic_ncols=True,
        )


@contextmanager
def safe_progress(total: int, desc: str, unit: str = "item") -> Iterator[Callable[[int], None]]:
    """
    Context manager ensuring the progress bar reaches 100% and closes cleanly.
    Yields: update_fn (validated to prevent overflow)
    """
    pbar = tqdm(
        total=total,
        desc=desc,
        unit=unit,
        file=sys.stderr,
        leave=False,
        dynamic_ncols=True,
    )

    def safe_update(n: int):
        """Update progress bar with overflow protection."""
        if pbar.n + n > pbar.total: # check if the update would exceed total
            # Prevent overflow - only update remaining amount
            needed = pbar.total - pbar.n # calculate remaining to reach total
            if needed > 0:
                pbar.update(needed)
        else:
            pbar.update(n)

    try:
        yield safe_update
    finally: # ensure the progress bar is completed and closed
        final_remainder = total - pbar.n
        if final_remainder > 0:
            pbar.update(final_remainder)
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



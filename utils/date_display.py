"""
Date parsing and formatting utilities for the RL training system.

This module provides a centralized interface for all date handling:
- parse_date(): Public API for plotting and datetime operations
- format_*_for_display(): Formatting for console/INFO logs (DD-MM-YYYY)
- validate_iso_date(): ISO format validation
- parse_user_date(): User input parsing (DD-MM-YYYY → ISO)

Architecture:
    User Input (DD-MM-YYYY) → parse_user_date() → ISO 8601
                                                       ↓
                                                  Storage/Processing
                                                       ↓
                                      format_date_for_display() → DD-MM-YYYY (Display)
                                      parse_date() → datetime (Plotting)
"""

from __future__ import annotations

from datetime import datetime
from typing import Union
import pandas as pd

import config


class DateFormatError(ValueError):
    """Custom exception for date format errors"""
    pass


# ============================================================================
# Public API - For Plotting and Datetime Operations
# ============================================================================

def parse_date(date_input: Union[str, datetime, pd.Timestamp]) -> datetime:
    """
    Parse ISO date string or datetime-like object to datetime.

    This is the PUBLIC API for converting dates to datetime objects for:
    - Matplotlib plotting
    - Date arithmetic
    - Datetime operations

    Args:
        date_input: ISO date string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS),
                   datetime object, or pandas Timestamp

    Returns:
        datetime object

    Raises:
        DateFormatError: If format is invalid or type is unsupported

    Examples:
        >>> parse_date("2025-01-05")
        datetime(2025, 1, 5, 0, 0, 0)

        >>> dates = ["2025-01-05", "2025-01-06"]
        >>> datetimes = [parse_date(d) for d in dates]
        >>> plt.plot(datetimes, values)
    """
    # Passthrough for datetime objects
    if isinstance(date_input, datetime):
        return date_input

    # Convert pandas Timestamp to datetime
    if isinstance(date_input, pd.Timestamp):
        return date_input.to_pydatetime()

    # Parse ISO string
    if isinstance(date_input, str):
        try:
            # Try ISO date format (YYYY-MM-DD)
            return datetime.strptime(date_input, config.INTERNAL_ISO_FORMAT)
        except ValueError:
            pass

        try:
            # Try ISO datetime format (YYYY-MM-DD HH:MM:SS)
            return datetime.strptime(date_input, config.INTERNAL_ISO_DATETIME_FORMAT)
        except ValueError:
            raise DateFormatError(
                f"Invalid date format: '{date_input}'. "
                f"Expected ISO 8601 format (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)"
            )

    raise DateFormatError(
        f"Unsupported date type: {type(date_input).__name__}. "
        f"Expected str, datetime, or pd.Timestamp"
    )


# ============================================================================
# Display Formatting - For Console/INFO Logs
# ============================================================================

def format_date_for_display(date_input: Union[str, datetime, pd.Timestamp]) -> str:
    """
    Format ISO date to DD-MM-YYYY for display.

    Used for:
    - Console output
    - INFO-level logs
    - User-facing messages

    Args:
        date_input: ISO date string (YYYY-MM-DD), datetime, or pd.Timestamp

    Returns:
        Formatted date string in DD-MM-YYYY format

    Raises:
        DateFormatError: If input format is invalid

    Examples:
        >>> format_date_for_display("2025-01-05")
        "05-01-2025"

        >>> format_date_for_display(datetime(2025, 1, 5))
        "05-01-2025"
    """
    dt = parse_date(date_input)
    return dt.strftime(config.USER_DATE_FORMAT)


def format_datetime_for_display(datetime_input: Union[str, datetime, pd.Timestamp]) -> str:
    """
    Format ISO datetime to DD-MM-YYYY HH:MM:SS for display.

    Args:
        datetime_input: ISO datetime string, datetime, or pd.Timestamp

    Returns:
        Formatted datetime string in DD-MM-YYYY HH:MM:SS format

    Examples:
        >>> format_datetime_for_display("2025-01-05 14:30:00")
        "05-01-2025 14:30:00"
    """
    dt = parse_date(datetime_input)
    return dt.strftime(config.USER_DATETIME_FORMAT)


def format_date_range_for_display(
    start_date: Union[str, datetime, pd.Timestamp],
    end_date: Union[str, datetime, pd.Timestamp]
) -> str:
    """
    Format date range for display.

    Args:
        start_date: Start date (ISO string, datetime, or pd.Timestamp)
        end_date: End date (ISO string, datetime, or pd.Timestamp)

    Returns:
        Formatted range string: "DD-MM-YYYY to DD-MM-YYYY"

    Examples:
        >>> format_date_range_for_display("2025-01-05", "2025-12-31")
        "05-01-2025 to 31-12-2025"
    """
    start_str = format_date_for_display(start_date)
    end_str = format_date_for_display(end_date)
    return f"{start_str} to {end_str}"


# ============================================================================
# Validation and User Input Parsing
# ============================================================================

def validate_iso_date(date_str: str) -> str:
    """
    Validate that a date string is in valid ISO 8601 format (YYYY-MM-DD).

    Args:
        date_str: Date string to validate

    Returns:
        The same date string if valid

    Raises:
        DateFormatError: If format is invalid or date values are invalid

    Examples:
        >>> validate_iso_date("2025-01-05")
        "2025-01-05"

        >>> validate_iso_date("05-01-2025")  # Raises DateFormatError
    """
    try:
        datetime.strptime(date_str, config.INTERNAL_ISO_FORMAT)
        return date_str
    except ValueError as e:
        raise DateFormatError(
            f"Invalid ISO date format: '{date_str}'. "
            f"Expected YYYY-MM-DD. Error: {e}"
        )


def parse_user_date(user_input: str) -> str:
    """
    Parse user input date (DD-MM-YYYY) to ISO format (YYYY-MM-DD).

    This function is specifically for parsing dates entered by users
    in DD-MM-YYYY format and converting them to internal ISO format.

    Args:
        user_input: Date string in DD-MM-YYYY format

    Returns:
        Date string in ISO format (YYYY-MM-DD)

    Raises:
        DateFormatError: If input is not in DD-MM-YYYY format or date is invalid

    Examples:
        >>> parse_user_date("05-01-2025")
        "2025-01-05"

        >>> parse_user_date("29-02-2024")  # Leap year
        "2024-02-29"

        >>> parse_user_date("29-02-2025")  # Not a leap year - raises error
    """
    try:
        dt = datetime.strptime(user_input, config.USER_DATE_FORMAT)
        return dt.strftime(config.INTERNAL_ISO_FORMAT)
    except ValueError as e:
        raise DateFormatError(
            f"Invalid user date format: '{user_input}'. "
            f"Expected DD-MM-YYYY (e.g., 05-01-2025). Error: {e}"
        )
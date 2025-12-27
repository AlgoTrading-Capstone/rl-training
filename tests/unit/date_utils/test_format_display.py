"""
Unit tests for display formatting functions.
"""

import pytest
from datetime import datetime
import pandas as pd

from utils.date_display import (
    format_date_for_display,
    format_datetime_for_display,
    format_date_range_for_display,
    DateFormatError,
)


@pytest.mark.date
class TestDateFormatting:
    """Test date formatting utilities"""

    def test_format_date_from_iso_string(self):
        """Test formatting ISO string to DD-MM-YYYY"""
        assert format_date_for_display("2025-01-05") == "05-01-2025"
        assert format_date_for_display("2025-12-31") == "31-12-2025"

    def test_format_date_from_datetime(self):
        """Test formatting datetime to DD-MM-YYYY"""
        dt = datetime(2025, 1, 5)
        assert format_date_for_display(dt) == "05-01-2025"

    def test_format_date_from_timestamp(self):
        """Test formatting pandas Timestamp to DD-MM-YYYY"""
        ts = pd.Timestamp("2025-01-05")
        assert format_date_for_display(ts) == "05-01-2025"

    def test_format_datetime_with_time(self):
        """Test formatting datetime with time"""
        result = format_datetime_for_display("2025-01-05 14:30:00")
        assert result == "05-01-2025 14:30:00"

        dt = datetime(2025, 1, 5, 14, 30, 0)
        assert format_datetime_for_display(dt) == "05-01-2025 14:30:00"

    def test_format_date_range(self):
        """Test formatting date range"""
        result = format_date_range_for_display("2025-01-05", "2025-12-31")
        assert result == "05-01-2025 to 31-12-2025"

    def test_invalid_date_format_raises_error(self):
        """Test error handling for invalid format"""
        with pytest.raises(DateFormatError):
            format_date_for_display("05-01-2025")  # DD-MM-YYYY not accepted

    def test_ambiguous_dates_handled_correctly(self):
        """Test that ambiguous dates are parsed correctly as ISO"""
        result = format_date_for_display("2025-01-05")
        assert result == "05-01-2025"  # Day 5, Month 1
        assert result != "01-05-2025"  # Not May 1st

    def test_leap_year_display(self):
        """Test leap year date display"""
        display = format_date_for_display("2024-02-29")
        assert display == "29-02-2024"


@pytest.mark.date
class TestRoundTripConversion:
    """Test round-trip conversions"""

    def test_iso_to_display_to_iso(self):
        """Test conversion doesn't lose information"""
        from utils.date_display import parse_user_date, format_date_for_display

        # Start with ISO
        iso_original = "2025-01-05"

        # Convert to display format
        display = format_date_for_display(iso_original)
        assert display == "05-01-2025"

        # Convert back to ISO
        iso_result = parse_user_date(display)
        assert iso_result == iso_original
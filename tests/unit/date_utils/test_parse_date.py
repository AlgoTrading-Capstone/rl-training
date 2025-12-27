"""
Unit tests for parse_date() - public API for datetime conversion.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd

from utils.date_display import parse_date, DateFormatError


@pytest.mark.date
class TestParseDatePublicAPI:
    """Test public parse_date() API for plotting/datetime operations"""

    def test_parse_date_from_iso_string(self):
        """Test parsing ISO string to datetime"""
        dt = parse_date("2025-01-05")
        assert dt == datetime(2025, 1, 5, 0, 0, 0)
        assert isinstance(dt, datetime)

    def test_parse_date_from_iso_datetime_string(self):
        """Test parsing ISO datetime string"""
        dt = parse_date("2025-01-05 14:30:00")
        assert dt == datetime(2025, 1, 5, 14, 30, 0)

    def test_parse_date_from_datetime_object(self):
        """Test parsing datetime object (passthrough)"""
        input_dt = datetime(2025, 1, 5, 14, 30)
        output_dt = parse_date(input_dt)
        assert output_dt == input_dt
        assert output_dt is input_dt  # Same object

    def test_parse_date_from_timestamp(self):
        """Test parsing pandas Timestamp"""
        ts = pd.Timestamp("2025-01-05 14:30:00")
        dt = parse_date(ts)
        assert dt == datetime(2025, 1, 5, 14, 30, 0)
        assert isinstance(dt, datetime)

    def test_parse_date_for_plotting(self):
        """Test use case: converting ISO strings for matplotlib"""
        iso_dates = ["2025-01-05", "2025-01-06", "2025-01-07"]
        datetimes = [parse_date(d) for d in iso_dates]

        assert len(datetimes) == 3
        assert all(isinstance(dt, datetime) for dt in datetimes)
        assert datetimes[1] == datetimes[0] + timedelta(days=1)

    def test_parse_date_for_arithmetic(self):
        """Test use case: date arithmetic"""
        dt = parse_date("2025-01-05")
        tomorrow = dt + timedelta(days=1)
        assert tomorrow == datetime(2025, 1, 6, 0, 0, 0)

    def test_parse_date_invalid_format(self):
        """Test error handling for invalid formats"""
        with pytest.raises(DateFormatError, match="Invalid date format"):
            parse_date("05-01-2025")  # DD-MM-YYYY not accepted

    def test_parse_date_unsupported_type(self):
        """Test error handling for unsupported types"""
        with pytest.raises(DateFormatError, match="Unsupported date type"):
            parse_date(123456)

    def test_leap_year_date(self):
        """Test handling of leap year date"""
        dt = parse_date("2024-02-29")
        assert dt == datetime(2024, 2, 29)

    def test_end_of_year_boundary(self):
        """Test year boundary dates"""
        dt_end = parse_date("2024-12-31")
        dt_start = parse_date("2025-01-01")

        assert dt_end == datetime(2024, 12, 31)
        assert dt_start == datetime(2025, 1, 1)
        assert (dt_start - dt_end).days == 1
"""
Unit tests for date validation and parsing.
"""

import pytest
from utils.date_display import validate_iso_date, parse_user_date, DateFormatError


@pytest.mark.date
class TestDateValidation:
    """Test date validation utilities"""

    def test_validate_iso_date_valid(self):
        """Test validation of valid ISO dates"""
        assert validate_iso_date("2025-01-05") == "2025-01-05"
        assert validate_iso_date("2025-12-31") == "2025-12-31"

    def test_validate_iso_date_invalid_format(self):
        """Test validation rejects non-ISO formats"""
        with pytest.raises(DateFormatError):
            validate_iso_date("05-01-2025")  # DD-MM-YYYY

        with pytest.raises(DateFormatError):
            validate_iso_date("01/05/2025")  # MM/DD/YYYY

    def test_validate_iso_date_invalid_values(self):
        """Test validation rejects invalid date values"""
        with pytest.raises(DateFormatError):
            validate_iso_date("2025-13-45")  # Invalid month and day

        with pytest.raises(DateFormatError):
            validate_iso_date("2025-02-30")  # Invalid day for February


@pytest.mark.date
class TestUserDateParsing:
    """Test date parsing from user input"""

    def test_parse_user_date_dd_mm_yyyy(self):
        """Test parsing DD-MM-YYYY to ISO"""
        assert parse_user_date("05-01-2025") == "2025-01-05"
        assert parse_user_date("31-12-2025") == "2025-12-31"

    def test_parse_user_date_invalid_format(self):
        """Test error handling for invalid user input"""
        with pytest.raises(DateFormatError):
            parse_user_date("2025-01-05")  # ISO format

        with pytest.raises(DateFormatError):
            parse_user_date("invalid-date")

    def test_parse_user_date_ambiguous(self):
        """Ensure DD-MM-YYYY is parsed correctly (no ambiguity)"""
        # "05-01-2025" is Day 5, Month 1 (January 5th)
        result = parse_user_date("05-01-2025")
        assert result == "2025-01-05"  # ISO: Year-Month-Day

        # "12-01-2025" is Day 12, Month 1 (January 12th)
        result = parse_user_date("12-01-2025")
        assert result == "2025-01-12"  # NOT 2025-12-01

    def test_parse_user_date_leap_year(self):
        """Test leap year handling"""
        result = parse_user_date("29-02-2024")
        assert result == "2024-02-29"

        # Invalid leap year (2025 is not a leap year)
        with pytest.raises(DateFormatError):
            parse_user_date("29-02-2025")
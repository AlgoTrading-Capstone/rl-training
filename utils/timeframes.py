"""
Timeframe conversion and candle-based datetime utilities.

Provides helpers to:
- Convert timeframe strings (e.g., '1h', '15m') into minutes.
- Perform timestamp conversions (datetime <-> milliseconds).
- Align datetimes to candle boundaries.
- Offset dates by a number of candles.

Used by strategies that rely on precise multi-timeframe candle logic.
"""

from datetime import datetime, UTC, timedelta
import ccxt


# Timeframe conversion table
TIMEFRAME_MINUTES_MAP = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "1d": 1440,
}


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Convert a timeframe string (e.g. '1h', '15m') into the number of minutes it represents.
    """
    if timeframe not in TIMEFRAME_MINUTES_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    return TIMEFRAME_MINUTES_MAP[timeframe]


def timeframe_to_cron(timeframe: str) -> dict:
    """
    Convert a timeframe string (e.g., '1h', '15m') into a dictionary
    that can be passed directly into APScheduler's CronTrigger.

    Examples:
        '1h'  -> {'minute': 0}
        '4h'  -> {'minute': 0, 'hour': '*/4'}
        '15m' -> {'minute': '*/15'}
        '5m'  -> {'minute': '*/5'}
        '1d'  -> {'hour': 0, 'minute': 0}

    Supported units:
        - m: minutes
        - h: hours
        - d: days
    """
    if timeframe not in TIMEFRAME_MINUTES_MAP:
        raise ValueError(f"Unsupported timeframe for cron: {timeframe}")

    # Parse the timeframe format: number + unit
    unit = timeframe[-1]   # 'm', 'h', or 'd'
    value = int(timeframe[:-1])

    if unit == "m":
        return {"minute": f"*/{value}"}

    if unit == "h":
        # Run at minute 0 of every Nth hour
        return {"minute": 0, "hour": f"*/{value}"}

    if unit == "d":
        # Run at midnight every N days
        return {"hour": 0, "minute": 0, "day": f"*/{value}"}

    raise ValueError(f"Unsupported timeframe unit: {unit}")


def datetime_to_timestamp_ms(dt: datetime) -> int:
    """
    Convert datetime to a millisecond-based timestamp.
    """
    return int(dt.timestamp() * 1000)


def timestamp_ms_to_datetime(ts: int) -> datetime:
    """
    Convert a millisecond-based timestamp into a UTC datetime.
    """
    return datetime.fromtimestamp(ts, tz=UTC)


def timeframe_to_prev_date(timeframe: str, date: datetime | None = None) -> datetime:
    """
    Round the given datetime down to the start of the previous candle.
    """
    if date is None:
        date = datetime.now(UTC)

    timestamp_ms = datetime_to_timestamp_ms(date)
    # ccxt returns timestamp in milliseconds
    rounded_ms = ccxt.Exchange.round_timeframe(timeframe, timestamp_ms, ccxt.ROUND_DOWN)
    return timestamp_ms_to_datetime(rounded_ms // 1000)


def date_minus_candles(timeframe: str, candle_count: int, date: datetime | None = None) -> datetime:
    """
    Subtract N candles from the given date, after rounding the date down
    to the beginning of the current candle.
    """
    if date is None:
        date = datetime.now(UTC)

    tf_minutes = timeframe_to_minutes(timeframe)
    candle_start = timeframe_to_prev_date(timeframe, date)

    return candle_start - timedelta(minutes=tf_minutes * candle_count)
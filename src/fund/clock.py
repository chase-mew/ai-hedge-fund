# pyright: strict
"""Session clock and cycle slot helpers. US holidays are ignored."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from fund.contracts import SessionState

NEW_YORK = ZoneInfo("America/New_York")


def _aware(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone aware")
    return now


def session_state(now: datetime) -> SessionState:
    """Return whether US hours markets are open and whether the Friday wind down has started."""
    return SessionState(tradfi_open=tradfi_open(now), pre_weekend=pre_weekend(now))


def tradfi_open(now: datetime) -> bool:
    """Return False from Friday 20:00 to Sunday 20:05 New York time."""
    local = _aware(now).astimezone(NEW_YORK)
    weekday = local.weekday()
    minute_of_day = local.hour * 60 + local.minute
    friday_closed = weekday == 4 and minute_of_day >= 20 * 60
    sunday_closed = weekday == 6 and minute_of_day < 20 * 60 + 5
    return not (friday_closed or weekday == 5 or sunday_closed)


def pre_weekend(now: datetime) -> bool:
    """Return True from Friday 12:00 inclusive until Friday 20:00 exclusive, New York time."""
    local = _aware(now).astimezone(NEW_YORK)
    if local.weekday() != 4:
        return False
    minute_of_day = local.hour * 60 + local.minute
    return 12 * 60 <= minute_of_day < 20 * 60


def cycle_slot(now: datetime, hours: int) -> datetime:
    """Floor now to the cycle boundary in UTC."""
    utc = _aware(now).astimezone(UTC)
    floored_hour = (utc.hour // hours) * hours
    return utc.replace(hour=floored_hour, minute=0, second=0, microsecond=0)


def cycle_id(slot: datetime) -> str:
    """Format a cycle slot as YYYY-MM-DDTHH:MMZ."""
    utc = _aware(slot).astimezone(UTC)
    return utc.strftime("%Y-%m-%dT%H:%MZ")


def utc_day(now: datetime) -> str:
    """Return the UTC calendar day as YYYY-MM-DD."""
    return _aware(now).astimezone(UTC).strftime("%Y-%m-%d")

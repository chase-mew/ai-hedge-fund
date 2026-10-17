"""Clock boundaries, including both US daylight saving transitions."""

from datetime import UTC, datetime

import pytest

from fund.clock import NEW_YORK, cycle_id, cycle_slot, pre_weekend, session_state, tradfi_open, utc_day


def ny(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NEW_YORK)


def test_weekday_afternoon_is_open() -> None:
    moment = ny(2026, 10, 14, 15, 0)
    assert tradfi_open(moment) is True
    assert pre_weekend(moment) is False
    state = session_state(moment)
    assert state.tradfi_open is True
    assert state.pre_weekend is False


def test_friday_close_boundary() -> None:
    assert tradfi_open(ny(2026, 10, 16, 19, 59)) is True
    assert pre_weekend(ny(2026, 10, 16, 19, 59)) is True
    assert tradfi_open(ny(2026, 10, 16, 20, 0)) is False
    assert pre_weekend(ny(2026, 10, 16, 20, 0)) is False


def test_saturday_is_closed() -> None:
    moment = ny(2026, 10, 17, 12, 0)
    assert tradfi_open(moment) is False
    assert pre_weekend(moment) is False


def test_sunday_reopen_boundary() -> None:
    assert tradfi_open(ny(2026, 10, 18, 20, 4)) is False
    assert tradfi_open(ny(2026, 10, 18, 20, 5)) is True


def test_friday_noon_boundary() -> None:
    assert pre_weekend(ny(2026, 10, 16, 11, 59)) is False
    assert pre_weekend(ny(2026, 10, 16, 12, 0)) is True
    assert tradfi_open(ny(2026, 10, 16, 12, 0)) is True


def test_dst_end_sunday_reopen_is_0105_utc() -> None:
    closed = ny(2026, 11, 1, 20, 4)
    opened = ny(2026, 11, 1, 20, 5)
    assert opened.astimezone(UTC) == datetime(2026, 11, 2, 1, 5, tzinfo=UTC)
    assert tradfi_open(closed) is False
    assert tradfi_open(opened) is True


def test_dst_start_sunday_reopen() -> None:
    closed = ny(2026, 3, 8, 20, 4)
    opened = ny(2026, 3, 8, 20, 5)
    assert opened.astimezone(UTC) == datetime(2026, 3, 9, 0, 5, tzinfo=UTC)
    assert tradfi_open(closed) is False
    assert tradfi_open(opened) is True


def test_naive_datetimes_are_rejected() -> None:
    naive = datetime(2026, 10, 14, 15, 0)  # noqa: DTZ001  # this test rejects a naive datetime
    with pytest.raises(ValueError, match="timezone aware"):
        tradfi_open(naive)
    with pytest.raises(ValueError, match="timezone aware"):
        pre_weekend(naive)
    with pytest.raises(ValueError, match="timezone aware"):
        session_state(naive)
    with pytest.raises(ValueError, match="timezone aware"):
        cycle_slot(naive, 4)
    with pytest.raises(ValueError, match="timezone aware"):
        cycle_id(naive)
    with pytest.raises(ValueError, match="timezone aware"):
        utc_day(naive)


def test_cycle_slot_floors_to_four_hours() -> None:
    before = datetime(2026, 10, 14, 3, 59, tzinfo=UTC)
    on_boundary = datetime(2026, 10, 14, 4, 0, tzinfo=UTC)
    assert cycle_slot(before, 4) == datetime(2026, 10, 14, 0, 0, tzinfo=UTC)
    assert cycle_slot(on_boundary, 4) == datetime(2026, 10, 14, 4, 0, tzinfo=UTC)
    assert cycle_id(cycle_slot(on_boundary, 4)) == "2026-10-14T04:00Z"


def test_utc_day() -> None:
    late_new_york = ny(2026, 10, 14, 22, 0)
    assert utc_day(late_new_york) == "2026-10-15"

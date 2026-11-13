from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from inbox_recall.timeparse import TimeParseError, parse_when

TZ = ZoneInfo("America/Chicago")
NOW = datetime(2026, 11, 13, 15, 30, tzinfo=TZ)  # a Friday


def span(phrase, now=NOW):
    r = parse_when(phrase, now)
    fmt = "%Y-%m-%d"
    return (r.start.strftime(fmt) if r.start else None, r.end.strftime(fmt) if r.end else None)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("last spring", ("2026-03-01", "2026-06-01")),
        ("spring", ("2026-03-01", "2026-06-01")),
        ("last fall", ("2025-09-01", "2025-12-01")),  # this fall hasn't finished yet
        ("this fall", ("2026-09-01", "2026-12-01")),
        ("summer 2023", ("2023-06-01", "2023-09-01")),
        ("winter 2024", ("2024-12-01", "2025-03-01")),
        ("last winter", ("2025-12-01", "2026-03-01")),
        ("in march", ("2026-03-01", "2026-04-01")),
        ("last november", ("2025-11-01", "2025-12-01")),
        ("november", ("2026-11-01", "2026-12-01")),
        ("march 2024", ("2024-03-01", "2024-04-01")),
        ("march of 2024", ("2024-03-01", "2024-04-01")),
        ("dec '24", ("2024-12-01", "2025-01-01")),
        ("march 5th", ("2026-03-05", "2026-03-06")),
        ("5 march 2024", ("2024-03-05", "2024-03-06")),
        ("2024-03-05", ("2024-03-05", "2024-03-06")),
        ("3/5/2024", ("2024-03-05", "2024-03-06")),
        ("2024-03", ("2024-03-01", "2024-04-01")),
        ("2023", ("2023-01-01", "2024-01-01")),
        ("last year", ("2025-01-01", "2026-01-01")),
        ("this year", ("2026-01-01", "2027-01-01")),
        ("last month", ("2026-10-01", "2026-11-01")),
        ("last week", ("2026-11-02", "2026-11-09")),
        ("this week", ("2026-11-09", "2026-11-16")),
        ("yesterday", ("2026-11-12", "2026-11-13")),
        ("today", ("2026-11-13", "2026-11-14")),
        ("on tuesday", ("2026-11-10", "2026-11-11")),
        ("last friday", ("2026-11-06", "2026-11-07")),
        ("last weekend", ("2026-11-07", "2026-11-09")),
        ("3 weeks ago", ("2026-10-19", "2026-10-26")),
        ("two months ago", ("2026-09-01", "2026-10-01")),
        ("a year ago", ("2025-01-01", "2026-01-01")),
        ("past 2 weeks", ("2026-10-30", "2026-11-13")),
        ("the last 3 days", ("2026-11-10", "2026-11-13")),
        ("early 2024", ("2024-01-01", "2024-05-01")),
        ("late last summer", ("2026-08-01", "2026-09-01")),
        ("last christmas", ("2025-12-24", "2025-12-27")),
        ("over the holidays", ("2025-12-20", "2026-01-04")),
        ("thanksgiving 2025", ("2025-11-26", "2025-12-01")),
        ("Valentine’s Day", ("2026-02-14", "2026-02-15")),
        ("since june", ("2026-06-01", None)),
        ("after march 2024", ("2024-04-01", None)),
        ("before 2023", (None, "2023-01-01")),
        ("until last spring", (None, "2026-06-01")),
        ("from jan 2024 to march 2024", ("2024-01-01", "2024-04-01")),
        ("mid october", ("2026-10-11", "2026-10-21")),
        ("between last christmas and new year's", ("2025-12-24", "2026-01-02")),
    ],
)
def test_phrases(phrase, expected):
    assert span(phrase) == expected


def test_rolling_ranges_end_now_and_keep_time():
    r = parse_when("past 2 weeks", NOW)
    assert r.end == NOW and r.start.hour == 15


def test_seasons_roll_over_at_the_new_year():
    january = datetime(2026, 1, 20, 9, 0, tzinfo=TZ)
    assert span("this winter", january) == ("2025-12-01", "2026-03-01")
    assert span("last spring", january) == ("2025-03-01", "2025-06-01")
    assert span("december", january) == ("2025-12-01", "2026-01-01")


def test_labels_are_readable():
    assert parse_when("last spring", NOW).label == "2026-03-01 to 2026-05-31"
    assert parse_when("yesterday", NOW).label == "2026-11-12"
    assert parse_when("since june", NOW).label == "since 2026-06-01"


@pytest.mark.parametrize("phrase", ["whenever", "the other day maybe", "between june and march"])
def test_unreadable_or_backwards_phrases_raise(phrase):
    with pytest.raises(TimeParseError):
        parse_when(phrase, NOW)


def test_requires_aware_now():
    with pytest.raises(ValueError):
        parse_when("today", datetime(2026, 1, 1))

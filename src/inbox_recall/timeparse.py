"""Natural-language time ranges for queries like "what did Maya say last spring".

`parse_when("last spring", now)` returns a half-open `[start, end)` range in
`now`'s time zone. Agents pass the phrase they heard; nobody has to work out
dates by hand.

Supported, case-insensitive, with an optional leading in/on/during/around:

- today, yesterday, tonight, this/last week|weekend|month|year
- past|last N days|weeks|months|years (rolling), N days|weeks|months|years ago
- monday ... sunday (most recent), last friday
- january ... december, last march, march 2024, march of 2024
- march 5, march 5th 2024, 5 march, 2024-03-05, 2024-03, 3/5/2024
- spring, summer, fall/autumn, winter (northern meteorological seasons;
  winter 2024 is Dec 2024 to Feb 2025), last spring, summer 2023
- christmas, the holidays, new year's, thanksgiving, halloween, valentine's day
- 2023, early/mid/late <any period>
- since X, after X, before X, until X, between X and Y, from X to Y

A bare period means the latest one that has started ("march" in November is
this March). "last <period>" means the latest one that has finished, so "last
spring" asked in November is this year's spring and "last november" is a year
ago.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

MONTHS = {
    name: i
    for i, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for name in names
}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "a couple": 2, "a couple of": 2, "couple": 2, "three": 3,
    "a few": 3, "few": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}  # fmt: skip
UNITS = {"day": "day", "days": "day", "week": "week", "weeks": "week", "month": "month", "months": "month",
         "year": "year", "years": "year"}  # fmt: skip
SEASONS = {"spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12}

MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
NUM_RE = r"\d+|" + "|".join(sorted(NUMBERS, key=len, reverse=True))


class TimeParseError(ValueError):
    pass


@dataclass(frozen=True)
class TimeRange:
    start: datetime | None
    end: datetime | None  # exclusive
    phrase: str = ""

    @property
    def label(self) -> str:
        def fmt(d: datetime) -> str:
            return d.strftime("%Y-%m-%d") if d.time() == time(0) else d.strftime("%Y-%m-%d %H:%M")

        if self.start and self.end:
            last = self.end - timedelta(days=1) if self.end.time() == time(0) else self.end
            if self.end.time() == time(0) and last.date() == self.start.date():
                return fmt(self.start)
            return f"{fmt(self.start)} to {fmt(last)}"
        if self.start:
            return f"since {fmt(self.start)}"
        if self.end:
            return f"before {fmt(self.end)}"
        return "any time"


Span = tuple[datetime, datetime]


def _at(now: datetime, d: date) -> datetime:
    return datetime.combine(d, time(0), tzinfo=now.tzinfo)


def _add_months(d: date, n: int) -> date:
    month = d.month - 1 + n
    return date(d.year + month // 12, month % 12 + 1, 1)


def _month(now: datetime, year: int, month: int) -> Span:
    first = date(year, month, 1)
    return _at(now, first), _at(now, _add_months(first, 1))


def _season(now: datetime, year: int, first_month: int) -> Span:
    first = date(year, first_month, 1)
    return _at(now, first), _at(now, _add_months(first, 3))


def _days(now: datetime, d: date, n: int = 1) -> Span:
    return _at(now, d), _at(now, d + timedelta(days=n))


def _thanksgiving(year: int) -> date:
    nov1 = date(year, 11, 1)
    first_thursday = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thursday + timedelta(weeks=3)


HOLIDAYS: dict[str, Callable[[datetime, int], Span]] = {
    "christmas": lambda now, y: _days(now, date(y, 12, 24), 3),
    "xmas": lambda now, y: _days(now, date(y, 12, 24), 3),
    "christmas eve": lambda now, y: _days(now, date(y, 12, 24)),
    "the holidays": lambda now, y: _days(now, date(y, 12, 20), 15),
    "holidays": lambda now, y: _days(now, date(y, 12, 20), 15),
    "holiday season": lambda now, y: _days(now, date(y, 12, 20), 15),
    "new year's eve": lambda now, y: _days(now, date(y, 12, 31), 2),
    "new years eve": lambda now, y: _days(now, date(y, 12, 31), 2),
    "nye": lambda now, y: _days(now, date(y, 12, 31), 2),
    "new year's": lambda now, y: _days(now, date(y, 12, 31), 2),
    "new years": lambda now, y: _days(now, date(y, 12, 31), 2),
    "new year": lambda now, y: _days(now, date(y, 12, 31), 2),
    "thanksgiving": lambda now, y: _days(now, _thanksgiving(y) - timedelta(days=1), 5),
    "halloween": lambda now, y: _days(now, date(y, 10, 31)),
    "valentine's day": lambda now, y: _days(now, date(y, 2, 14)),
    "valentines day": lambda now, y: _days(now, date(y, 2, 14)),
    "valentine's": lambda now, y: _days(now, date(y, 2, 14)),
}
HOLIDAY_RE = "|".join(re.escape(h) for h in sorted(HOLIDAYS, key=len, reverse=True))


def _occurrence(now: datetime, period: Callable[[int], Span], mode: str) -> Span:
    """Pick a yearly period: '' = latest started, 'last' = latest finished, 'this' = current year's."""
    spans = [period(y) for y in range(now.year + 1, now.year - 4, -1)]
    if mode == "this":
        for start, end in spans:
            if start <= now < end:
                return start, end
        return period(now.year)
    for start, end in spans:
        if (end <= now) if mode == "last" else (start <= now):
            return start, end
    return spans[-1]


def _number(text: str) -> int:
    return int(text) if text.isdigit() else NUMBERS[text]


def _shift(now: datetime, unit: str, n: int) -> datetime:
    if unit == "day":
        return now - timedelta(days=n)
    if unit == "week":
        return now - timedelta(weeks=n)
    if unit == "month":
        target = _add_months(now.date().replace(day=1), -n)
        day = min(now.day, (_add_months(target, 1) - timedelta(days=1)).day)
        return now.replace(year=target.year, month=target.month, day=day)
    return now.replace(year=now.year - n, day=min(now.day, 28) if now.month == 2 else now.day)


def _containing(now: datetime, moment: datetime, unit: str) -> Span:
    d = moment.date()
    if unit == "day":
        return _days(now, d)
    if unit == "week":
        monday = d - timedelta(days=d.weekday())
        return _days(now, monday, 7)
    if unit == "month":
        return _month(now, d.year, d.month)
    return _at(now, date(d.year, 1, 1)), _at(now, date(d.year + 1, 1, 1))


def _thirds(now: datetime, start: datetime, end: datetime) -> Span:
    """Split points for early/mid/late: whole months for years and seasons, ~10 days for a month."""
    months = (end.year - start.year) * 12 + end.month - start.month
    if start.day == end.day == 1 and start.time() == end.time() == time(0) and months > 0:
        if months % 3 == 0:
            step = months // 3
            return _at(now, _add_months(start.date(), step)), _at(now, _add_months(start.date(), 2 * step))
        if months == 1:
            return start + timedelta(days=10), start + timedelta(days=20)
    third = (end - start) / 3
    return start + third, end - third


def _period(text: str, now: datetime) -> Span:
    """Resolve one period expression (no since/before/between) to a span."""
    today = now.date()
    t = re.sub(r"^(?:in|on|during|around|over|at)\s+(?:the\s+)?", "", text).strip()
    t = re.sub(r"^the\s+", "", t) if not t.startswith("the holidays") else t

    m = re.fullmatch(r"(early|mid|middle of|late|end of|start of|beginning of)\s+(.+)", t)
    if m:
        start, end = _period(m.group(2), now)
        a, b = _thirds(now, start, end)
        part = m.group(1)
        if part == "early" or part.startswith(("start", "beginning")):
            return start, a
        if part.startswith("mid"):
            return a, b
        return b, end

    if t in ("today", "tonight"):
        return _days(now, today)
    if t == "yesterday":
        return _days(now, today - timedelta(days=1))

    m = re.fullmatch(r"(this|last|past|previous)\s+(week|weekend|month|year)", t)
    if m:
        which, unit = m.groups()
        if which == "past":
            return _shift(now, unit if unit != "weekend" else "week", 1), now
        back = 0 if which == "this" else 1
        if unit == "weekend":
            saturday = today - timedelta(days=(today.weekday() - 5) % 7)
            if which != "this" and saturday + timedelta(days=2) > today:
                saturday -= timedelta(weeks=1)
            return _days(now, saturday, 2)
        return _containing(now, _shift(now, unit, back), unit)

    m = re.fullmatch(rf"(?:the\s+)?(?:past|last|previous)\s+({NUM_RE})\s+(days?|weeks?|months?|years?)", t)
    if m:
        return _shift(now, UNITS[m.group(2)], _number(m.group(1))), now

    m = re.fullmatch(rf"({NUM_RE})\s+(days?|weeks?|months?|years?)\s+ago", t)
    if m:
        unit = UNITS[m.group(2)]
        return _containing(now, _shift(now, unit, _number(m.group(1))), unit)

    m = re.fullmatch(r"(?:(last|this)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", t)
    if m:
        back = (today.weekday() - WEEKDAYS.index(m.group(2))) % 7 or 7
        return _days(now, today - timedelta(days=back))

    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t) or re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", t)
    if m:
        a, b, c = (int(x) for x in m.groups())
        d = date(a, b, c) if a > 31 else date(c, a, b)
        return _days(now, d)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", t)
    if m:
        return _month(now, int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"(\d{4})", t)
    if m:
        y = int(m.group(1))
        return _at(now, date(y, 1, 1)), _at(now, date(y + 1, 1, 1))

    m = re.fullmatch(rf"(?:(last|this)\s+)?({MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?", t) or (
        re.fullmatch(rf"(?:(last|this)\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTH_RE})(?:,?\s+(\d{{4}}))?", t)
    )
    if m:
        mode, a, b, year = m.groups()
        month, day = (MONTHS[a], int(b)) if a in MONTHS else (MONTHS[b], int(a))
        if year:
            return _days(now, date(int(year), month, day))
        return _occurrence(now, lambda y: _days(now, date(y, month, day)), mode or "")

    m = re.fullmatch(rf"(?:(last|this)\s+)?({MONTH_RE})\.?(?:\s+(?:of\s+)?'?(\d{{4}}|\d{{2}}))?", t)
    if m:
        mode, name, year = m.groups()
        if year:
            return _month(now, int(year) if len(year) == 4 else 2000 + int(year), MONTHS[name])
        return _occurrence(now, lambda y: _month(now, y, MONTHS[name]), mode or "")

    m = re.fullmatch(r"(?:(last|this)\s+)?(spring|summer|fall|autumn|winter)(?:\s+(?:of\s+)?(\d{4}))?", t)
    if m:
        mode, season, year = m.groups()
        first = SEASONS[season]
        if year:
            return _season(now, int(year), first)
        return _occurrence(now, lambda y: _season(now, y, first), mode or "")

    m = re.fullmatch(rf"(?:(last|this)\s+)?({HOLIDAY_RE})(?:\s+(?:of\s+)?(\d{{4}}))?", t)
    if m:
        mode, name, year = m.groups()
        holiday = HOLIDAYS[name]
        if year:
            return holiday(now, int(year))
        return _occurrence(now, lambda y: holiday(now, y), mode or "")

    raise TimeParseError(
        f"can't read {text!r} as a time; try e.g. 'last spring', 'march 2024', 'past 2 weeks', "
        "'since june', '2023-04-01', or pass after/before as ISO dates"
    )


def parse_when(text: str, now: datetime) -> TimeRange:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    phrase = " ".join(text.lower().replace("’", "'").split()).strip(" .?!")
    if not phrase:
        return TimeRange(None, None, "")

    m = re.fullmatch(r"(?:between|from)\s+(.+?)\s+(?:and|to|until|through|-)\s+(.+)", phrase)
    if m:
        start = _period(m.group(1), now)[0]
        end = _period(m.group(2), now)[1]
        if end <= start:
            raise TimeParseError(f"{text!r} ends before it starts")
        return TimeRange(start, end, text)
    m = re.fullmatch(r"(since|after|before|until|till|prior to)\s+(.+)", phrase)
    if m:
        start, end = _period(m.group(2), now)
        word = m.group(1)
        if word == "since":
            return TimeRange(start, None, text)
        if word == "after":
            return TimeRange(end, None, text)
        if word in ("until", "till"):
            return TimeRange(None, end, text)
        return TimeRange(None, start, text)
    start, end = _period(phrase, now)
    return TimeRange(start, end, text)

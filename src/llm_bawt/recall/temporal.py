"""Explicit time hints in a recall question narrow the message search (TASK-1558).

Recency ranking halves a message's rank every RECENCY_HALF_WEIGHT_DAYS, so a
question about March buries the March answer under newer chatter. When the
question names a date, a month or a plain relative day ("yesterday", "last
Tuesday"), message search is limited to that period, padded by PAD_SECONDS so a
date said in Eastern time still matches messages stored in UTC.

Only unambiguous phrases count: a bare month name that is also an English word
("may", "march") needs a day, a year or a leading preposition ("in May").
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import re

PAD_SECONDS = 86_400.0
MONTHS = {name: number for number, names in enumerate((
    ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
    ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
    ("september", "sept", "sep"), ("october", "oct"), ("november", "nov"),
    ("december", "dec")), start=1) for name in names}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTH = "(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
_DAY = r"(?:[12][0-9]|3[01]|0?[1-9])(?:st|nd|rd|th)?"
_YEAR = r"(?:19|20)[0-9]{2}"
_LEAD = (r"(?:in|during|from|since|after|before|until|till|through|throughout|"
         r"early|mid|late|end\s+of|start\s+of|of)")
# Range words turn a date into an open range: since/from/after run to now;
# until/till/through/before run from the start of history.
_RANGE_WORDS = ("since", "from", "after", "before", "until", "till", "through")
OPEN_START = date(1970, 1, 2)
# Most specific first; a matched span is consumed before broader patterns run.
_PATTERNS = (
    ("iso", re.compile(rf"\b(?P<y>{_YEAR})-(?P<m>0[1-9]|1[0-2])-(?P<d>0[1-9]|[12][0-9]|3[01])"
                       r"(?:[T ][0-9]{2}:[0-9]{2}(?::[0-9]{2})?(?:\.[0-9]+)?Z?)?\b")),
    ("month_day", re.compile(rf"\b(?P<mon>{_MONTH})\s+(?P<d>{_DAY})\b(?:,?\s+(?P<y>{_YEAR})\b)?",
                             re.IGNORECASE)),
    ("day_month", re.compile(rf"\b(?P<d>{_DAY})\s+(?:of\s+)?(?P<mon>{_MONTH})(?:,?\s+(?P<y>{_YEAR}))?\b",
                             re.IGNORECASE)),
    ("month_year", re.compile(rf"\b(?P<mon>{_MONTH})\s+(?P<y>{_YEAR})\b", re.IGNORECASE)),
    ("lead_month", re.compile(rf"\b{_LEAD}\s+(?P<mon>{_MONTH})(?![\w-])", re.IGNORECASE)),
    ("ago", re.compile(r"\b(?P<n>[1-9][0-9]?|an?|one|two|three|four|five|six|seven)\s+"
                       r"(?P<unit>days?|weeks?)\s+ago\b", re.IGNORECASE)),
    ("weekday", re.compile(rf"\b(?P<which>last|this|on)\s+(?P<wd>{'|'.join(WEEKDAYS)})\b",
                           re.IGNORECASE)),
    ("relative", re.compile(r"\b(?P<rel>today|yesterday|last\s+week|this\s+week|"
                            r"last\s+month|this\s+month)\b", re.IGNORECASE)),
)
_WORD_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
                 "five": 5, "six": 6, "seven": 7}


@dataclass(frozen=True)
class TimeWindow:
    """Half-open UTC epoch range [since, until), already padded."""
    since: float
    until: float
    # The question with every matched time phrase removed, so "October 24 2026"
    # no longer competes as three lexical terms inside the window.
    residual: str

    def label(self) -> dict:
        def day(value: float) -> str:
            return datetime.fromtimestamp(value, UTC).date().isoformat()
        return {"from": day(self.since), "to": day(self.until - 1)}


def _epoch(day: date) -> float:
    return datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()


def _month_span(year: int, month: int) -> tuple[date, date]:
    end = date(year + (month == 12), month % 12 + 1, 1)
    return date(year, month, 1), end


def _latest_year(today: date, month: int, day: int | None) -> int:
    """No year given: the most recent such date that is not in the future."""
    year = today.year
    try:
        candidate = date(year, month, day or 1)
    except ValueError:
        return year
    return year if candidate <= today else year - 1


def _span(kind: str, match: re.Match, today: date) -> tuple[date, date] | None:
    groups = match.groupdict()
    try:
        if kind == "iso":
            start = date(int(groups["y"]), int(groups["m"]), int(groups["d"]))
            return start, start + timedelta(days=1)
        if kind in {"month_day", "day_month"}:
            month = MONTHS[groups["mon"].lower().rstrip(".")]
            day = int(re.match(r"\d+", groups["d"]).group())
            year = int(groups["y"]) if groups.get("y") else _latest_year(today, month, day)
            start = date(year, month, day)
            return start, start + timedelta(days=1)
        if kind in {"month_year", "lead_month"}:
            month = MONTHS[groups["mon"].lower().rstrip(".")]
            year = int(groups["y"]) if groups.get("y") else _latest_year(today, month, None)
            return _month_span(year, month)
        if kind == "ago":
            count = _WORD_NUMBERS.get(groups["n"].lower()) or int(groups["n"])
            days = count * (7 if groups["unit"].lower().startswith("week") else 1)
            centre = today - timedelta(days=days)
            # "Three weeks ago" is approximate: allow half the distance either side.
            slack = timedelta(days=max(1, days // 2))
            return centre - slack, centre + slack + timedelta(days=1)
        if kind == "weekday":
            target = WEEKDAYS.index(groups["wd"].lower())
            back = (today.weekday() - target) % 7
            if groups["which"].lower() == "last" and back == 0:
                back = 7
            start = today - timedelta(days=back)
            return start, start + timedelta(days=1)
        rel = re.sub(r"\s+", " ", groups["rel"].lower())
        if rel == "today":
            return today, today + timedelta(days=1)
        if rel == "yesterday":
            return today - timedelta(days=1), today
        monday = today - timedelta(days=today.weekday())
        if rel == "this week":
            return monday, today + timedelta(days=1)
        if rel == "last week":
            return monday - timedelta(days=7), monday
        first = today.replace(day=1)
        if rel == "this month":
            return first, today + timedelta(days=1)
        previous = (first - timedelta(days=1)).replace(day=1)
        return previous, first
    except (ValueError, KeyError):
        return None


def _qualifier(kind: str, match: re.Match) -> str | None:
    """The range word governing this date: inside "since March", or just before it."""
    if kind == "lead_month":
        word = match.group().split()[0].lower()
    else:
        before = re.search(rf"\b({'|'.join(_RANGE_WORDS)})\s+(?:the\s+)?$",
                           match.string[:match.start()], re.IGNORECASE)
        word = before.group(1).lower() if before else None
    return word if word in _RANGE_WORDS else None


def _compose(dated: list[tuple[str | None, tuple[date, date]]],
             today: date) -> tuple[date, date] | None:
    """Combine every date on a line into one range; None when it is empty.

    since/from/after give a lower bound, until/till/through/before an upper
    bound, and a plain date covers its own span. A side with no bound stays
    open, so "from Oct 3 to Oct 9" ends with Oct 9 while "since March" runs to
    now. Each date is classified before any side is opened, so one open date
    never swallows the other end of an explicit range.
    """
    starts: list[date] = []
    ends: list[date] = []
    for word, (start, end) in dated:
        if word in {"since", "from"}:
            starts.append(start)
        elif word == "after":
            starts.append(end)
        elif word in {"until", "till", "through"}:
            ends.append(end)
        elif word == "before":
            ends.append(start)
        else:
            starts.append(start)
            ends.append(end)
    start = min(starts) if starts else OPEN_START
    end = max(ends) if ends else today + timedelta(days=1)
    return (start, end) if start < end else None


def _parse_line(line: str, today: date) -> tuple[date, date, str] | None:
    dated: list[tuple[str | None, tuple[date, date]]] = []
    residual = line
    for kind, pattern in _PATTERNS:
        def consume(match: re.Match) -> str:
            month = match.groupdict().get("mon")
            # "it may 2x", "march 3 steps": lowercase verbs, not dates.
            if (kind in {"month_day", "day_month"} and month in {"may", "march"}
                    and not match.groupdict().get("y")):
                return match.group()
            span = _span(kind, match, today)
            if span is None:
                return match.group()
            # Every date form keeps its preceding range word, so "since March
            # 2025" and "since 2025-03-01" run to now, not just that month/day.
            dated.append((_qualifier(kind, match), span))
            return "\x00"
        residual = pattern.sub(consume, residual)
    if not dated:
        return None
    span = _compose(dated, today)
    if span is None:
        return None
    # A range word before a consumed date leaves with it.
    residual = re.sub(rf"\b(?:{'|'.join(_RANGE_WORDS)})\s+(?:the\s+)?\x00", " ", residual,
                      flags=re.IGNORECASE).replace("\x00", " ")
    return span[0], span[1], re.sub(r"\s+", " ", residual).strip()


def time_window(query: str, now: float) -> TimeWindow | None:
    """Window for the newest question line that names a time, else None.

    A follow-up recall joins its questions with newlines; the latest line with
    a hint wins so an earlier "in March" never pins a later "yesterday".
    """
    today = datetime.fromtimestamp(now, UTC).date()
    lines = query.split("\n")
    for index in range(len(lines) - 1, -1, -1):
        parsed = _parse_line(lines[index], today)
        if parsed is None:
            continue
        start, end, residual = parsed
        since = _epoch(start) - PAD_SECONDS
        until = min(_epoch(end) + PAD_SECONDS, now + PAD_SECONDS)
        if until <= since:
            return None
        rest = [*lines[:index], residual, *lines[index + 1:]]
        return TimeWindow(since, until, "\n".join(line for line in rest if line.strip()))
    return None

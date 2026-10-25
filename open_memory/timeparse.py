"""Turn the time words in a query into a date window, relative to `now`.

"last week", "two months ago", "in March", "yesterday", "since June 3" ...
Returns (start, end) datetimes or None. Deliberately conservative: a wrong
window costs more than no window, so the engine only uses it as a boost.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
       "seven": 7, "eight": 8, "nine": 9, "ten": 10, "couple": 2, "few": 3, "several": 4}
UNIT = {"day": 1, "week": 7, "month": 30, "year": 365}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def parse_date(s: str | None) -> datetime | None:
    """Accepts ISO, '2023/05/30 (Tue) 23:40', '2023-05-30', and friends."""
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    s = re.sub(r"\([A-Za-z]{3}\)", "", str(s)).strip().replace("/", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19].strip(), fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except ValueError:
        return None


def _n(tok: str) -> int | None:
    return int(tok) if tok.isdigit() else NUM.get(tok)


def window(query: str, now: datetime | None) -> tuple[datetime, datetime] | None:
    if now is None:
        return None
    q = query.lower()
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if re.search(r"\btoday\b", q):
        return day, day + timedelta(days=1)
    if re.search(r"\byesterday\b", q):
        return day - timedelta(days=1), day
    m = re.search(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|couple|few|several)"
                  r"(?: of)? (day|week|month|year)s? ago\b", q)
    if m and _n(m.group(1)):
        span = _n(m.group(1)) * UNIT[m.group(2)]
        slack = max(2, span // 4)
        center = day - timedelta(days=span)
        return center - timedelta(days=slack), center + timedelta(days=slack + 1)
    m = re.search(r"\b(?:last|past|previous) (\d+|two|three|four|five|six|few|couple) (day|week|month)s?\b", q)
    if m and _n(m.group(1)):
        return day - timedelta(days=_n(m.group(1)) * UNIT[m.group(2)]), day + timedelta(days=1)
    m = re.search(r"\blast (week|month|year)\b", q)
    if m:
        span = UNIT[m.group(1)]
        return day - timedelta(days=2 * span), day - timedelta(days=0) + timedelta(days=1)
    m = re.search(r"\blast (" + "|".join(WEEKDAYS) + r")\b", q)
    if m:
        back = (now.weekday() - WEEKDAYS.index(m.group(1))) % 7 or 7
        d = day - timedelta(days=back)
        return d - timedelta(days=1), d + timedelta(days=2)
    m = re.search(r"\b(?:in|during|on|since|of) (" + "|".join(MONTHS) + r")\b(?: (\d{1,2})(?:st|nd|rd|th)?)?", q)
    if m:
        mon = MONTHS[m.group(1)]
        year = now.year if mon <= now.month else now.year - 1
        if m.group(2):
            d = datetime(year, mon, min(int(m.group(2)), 28))
            return d - timedelta(days=2), d + timedelta(days=3)
        start = datetime(year, mon, 1)
        end = datetime(year + (mon == 12), mon % 12 + 1, 1)
        return start, end
    return None

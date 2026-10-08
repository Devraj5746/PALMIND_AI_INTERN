"""Deterministic rule-based booking-field extraction.

This module backs the offline ``mock`` LLM provider: it turns a natural-language
utterance such as::

    "Hi, I'm Devraj Singh - devraj@example.com - book 2026-05-14 at 14:30"

into the same structured fields a hosted model would return.  It is a test
double: deliberately simple, fully deterministic, and it never guesses at a
field it cannot find — the field stays ``None`` so the conversational flow asks
a follow-up question instead.

It is not used when ``LLM_PROVIDER=openai``.
"""

from __future__ import annotations

import re
from datetime import date, time, timedelta
from typing import Any, Final

from app.core.enums import BookingIntent

# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------
BOOKING_KEYWORDS: Final[tuple[str, ...]] = (
    "book",
    "schedul",
    "interview",
    "appointment",
    "slot",
    "meet",
)
CANCEL_KEYWORDS: Final[tuple[str, ...]] = ("cancel", "drop", "call off")
LIST_KEYWORDS: Final[tuple[str, ...]] = ("list", "show", "what are", "see my")

# ---------------------------------------------------------------------------
# Field patterns
# ---------------------------------------------------------------------------
EMAIL_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
)

#: Intro phrases that precede a name ("I'm Devraj", "my name is Devraj").
#: Case-insensitive because users write both "I'm" and "i'm".
NAME_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:my\s+name\s+is|name\s+is|i\s+am|i\s*'m|this\s+is|myself|call\s+me)\b",
    re.IGNORECASE,
)

#: An explicitly labelled name such as "name: Devraj Singh".
NAME_LABEL_RE: Final[re.Pattern[str]] = re.compile(
    r"\bname\s*[:=]\s*([^,;\n]+)",
    re.IGNORECASE,
)

#: A capitalised, name-shaped phrase of one to four words.  Case-sensitive on
#: purpose: it must not capture "I'm booking an interview" as a name.
CAPITALISED_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"\s*([A-Z][A-Za-z'\u2019\-]+(?:\s+[A-Z][A-Za-z'\u2019\-]+){0,3})",
)

#: Fires only when the entire message is name-like (a bare slot answer).
BARE_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*([A-Z][A-Za-z'\u2019\-]+(?:\s+[A-Z][A-Za-z'\u2019\-]+){0,3})\s*[.!]?\s*$",
)

#: "notes: prefers mornings"
NOTES_RE: Final[re.Pattern[str]] = re.compile(r"\bnotes?\s*[:=]\s*(.+)$", re.IGNORECASE)


def classify_intent(text: str) -> BookingIntent:
    """Classify the user's intent from keyword presence."""
    lowered = text.lower()
    has_booking_noun = any(word in lowered for word in BOOKING_KEYWORDS)

    if any(word in lowered for word in CANCEL_KEYWORDS) and has_booking_noun:
        return BookingIntent.CANCEL_BOOKING
    if any(word in lowered for word in LIST_KEYWORDS) and has_booking_noun:
        return BookingIntent.LIST_BOOKINGS
    if has_booking_noun:
        return BookingIntent.BOOK_INTERVIEW
    return BookingIntent.OTHER


def extract_email(text: str) -> str | None:
    """Return the first e-mail address found in ``text`` (lower-cased)."""
    match = EMAIL_RE.search(text)
    return match.group(0).lower() if match else None


def _clean_name(raw: str) -> str | None:
    """Normalise a raw name capture, rejecting sentence fragments."""
    candidate = raw.strip().rstrip(".,;:!?")
    words = candidate.split()
    if not words or len(words) > 4:
        return None
    return " ".join(words)


def extract_name(text: str) -> str | None:
    """Return a candidate full name, or ``None`` when nothing convincing is found."""
    prefix = NAME_PREFIX_RE.search(text)
    if prefix:
        match = CAPITALISED_NAME_RE.match(text, prefix.end())
        if match:
            cleaned = _clean_name(match.group(1))
            if cleaned:
                return cleaned

    labelled = NAME_LABEL_RE.search(text)
    if labelled:
        cleaned = _clean_name(labelled.group(1))
        if cleaned:
            return cleaned

    bare = BARE_NAME_RE.match(text)
    if bare:
        return _clean_name(bare.group(1))

    return None


def extract_notes(text: str) -> str | None:
    """Return free-form notes explicitly labelled by the user, if any."""
    match = NOTES_RE.search(text)
    return match.group(1).strip() if match else None


# ---------------------------------------------------------------------------
# Date patterns
# ---------------------------------------------------------------------------
ISO_DATE_RE: Final[re.Pattern[str]] = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
NUMERIC_DATE_RE: Final[re.Pattern[str]] = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})\b")

MONTH_NAMES: Final[dict[str, int]] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH_ALT: Final[str] = "|".join(sorted(MONTH_NAMES, key=len, reverse=True))

MONTH_FIRST_DATE_RE: Final[re.Pattern[str]] = re.compile(
    rf"\b({_MONTH_ALT})[a-z]*\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
DAY_FIRST_DATE_RE: Final[re.Pattern[str]] = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT})[a-z]*\b",
    re.IGNORECASE,
)

WEEKDAYS: Final[dict[str, int]] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_WEEKDAY_ALT: Final[str] = "|".join(WEEKDAYS)
WEEKDAY_RE: Final[re.Pattern[str]] = re.compile(
    rf"\b(next\s+)?({_WEEKDAY_ALT})\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Time patterns
# ---------------------------------------------------------------------------
TIME_COLON_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*([ap]\.?m\.?)?\b",
    re.IGNORECASE,
)
TIME_MERIDIEM_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(\d{1,2})\s*([ap]\.?m\.?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Date and time extraction
# ---------------------------------------------------------------------------
def _coerce_year(raw: int) -> int:
    """Expand a two-digit year into the 2000s."""
    return 2000 + raw if raw < 100 else raw


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _roll_forward(candidate: date, today: date) -> date:
    """Advance a year-less date to its next occurrence on or after ``today``."""
    if candidate >= today:
        return candidate
    try:
        return candidate.replace(year=candidate.year + 1)
    except ValueError:  # 29 February in a non-leap year
        return candidate


def extract_date(text: str, *, today: date | None = None) -> date | None:
    """Extract an interview date from free text, or ``None``.

    Explicit years are honoured verbatim — a past date is *not* silently moved
    forward, because the service layer rejects it with a clear error.  Year-less
    forms (``14 May``, ``next Monday``) roll forward instead.
    """
    today = today or date.today()
    lowered = text.lower()

    if "day after tomorrow" in lowered:
        return today + timedelta(days=2)
    if "tomorrow" in lowered:
        return today + timedelta(days=1)
    if "today" in lowered:
        return today

    match = ISO_DATE_RE.search(text)
    if match:
        parsed = _safe_date(int(match[1]), int(match[2]), int(match[3]))
        if parsed is not None:
            return parsed

    match = NUMERIC_DATE_RE.search(text)
    if match:
        first, second, year = int(match[1]), int(match[2]), _coerce_year(int(match[3]))
        # Day-first unless that is impossible and month-first is not.
        if first > 12:
            day, month = first, second
        elif second > 12:
            day, month = second, first
        else:
            day, month = first, second
        parsed = _safe_date(year, month, day)
        if parsed is not None:
            return parsed

    match = MONTH_FIRST_DATE_RE.search(text)
    if match:
        parsed = _safe_date(today.year, MONTH_NAMES[match[1].lower()], int(match[2]))
        if parsed is not None:
            return _roll_forward(parsed, today)

    match = DAY_FIRST_DATE_RE.search(text)
    if match:
        parsed = _safe_date(today.year, MONTH_NAMES[match[2].lower()], int(match[1]))
        if parsed is not None:
            return _roll_forward(parsed, today)

    match = WEEKDAY_RE.search(text)
    if match:
        target = WEEKDAYS[match[2].lower()]
        delta = (target - today.weekday()) % 7 or 7
        return today + timedelta(days=delta)

    return None


def extract_time(text: str) -> time | None:
    """Extract an interview start time from free text, or ``None``."""
    match = TIME_COLON_RE.search(text)
    if match:
        hour, minute = int(match[1]), int(match[2])
        meridiem = (match[3] or "").lower().replace(".", "")
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return time(hour, minute)

    match = TIME_MERIDIEM_RE.search(text)
    if match:
        hour = int(match[1])
        meridiem = match[2].lower().replace(".", "")
        if not 1 <= hour <= 12:
            return None
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        return time(hour, 0)

    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
REQUIRED_FIELDS: Final[tuple[str, ...]] = ("name", "email", "date", "time")


def parse_booking_fields(text: str, *, today: date | None = None) -> dict[str, Any]:
    """Parse ``text`` into a partial booking payload.

    Only confidently-found fields are returned, so the caller can merge the
    result into an in-progress draft across several conversation turns.
    """
    today = today or date.today()
    fields: dict[str, Any] = {"intent": classify_intent(text)}

    name = extract_name(text)
    if name:
        fields["name"] = name

    email = extract_email(text)
    if email:
        fields["email"] = email

    parsed_date = extract_date(text, today=today)
    if parsed_date:
        fields["date"] = parsed_date

    parsed_time = extract_time(text)
    if parsed_time:
        fields["time"] = parsed_time

    notes = extract_notes(text)
    if notes:
        fields["notes"] = notes

    fields["missing_fields"] = [key for key in REQUIRED_FIELDS if key not in fields]
    return fields


__all__ = [
    "REQUIRED_FIELDS",
    "classify_intent",
    "extract_date",
    "extract_email",
    "extract_name",
    "extract_notes",
    "extract_time",
    "parse_booking_fields",
]

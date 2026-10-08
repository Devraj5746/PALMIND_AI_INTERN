"""Rule-based booking-field extraction used by the offline mock model."""

from __future__ import annotations

from datetime import date, time

import pytest

from app.core.enums import BookingIntent
from app.services.llm import rules
from app.services.llm.rules import (
    classify_intent,
    extract_date,
    extract_email,
    extract_name,
    extract_notes,
    extract_time,
    parse_booking_fields,
)

TODAY = date(2026, 1, 15)


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I want to book an interview", BookingIntent.BOOK_INTERVIEW),
        ("can we schedule a meeting", BookingIntent.BOOK_INTERVIEW),
        ("I need an appointment", BookingIntent.BOOK_INTERVIEW),
        ("cancel my interview please", BookingIntent.CANCEL_BOOKING),
        ("call off the interview", BookingIntent.CANCEL_BOOKING),
        ("show me my interviews", BookingIntent.LIST_BOOKINGS),
        ("what are my bookings", BookingIntent.LIST_BOOKINGS),
        ("list my interviews", BookingIntent.LIST_BOOKINGS),
        ("what is the leave policy", BookingIntent.OTHER),
        ("hello there", BookingIntent.OTHER),
        ("cancel it", BookingIntent.OTHER),
    ],
)
def test_classify_intent(text: str, expected: BookingIntent) -> None:
    assert classify_intent(text) == expected


# ---------------------------------------------------------------------------
# Individual fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hi, I'm Devraj Singh", "Devraj Singh"),
        ("my name is Priya Nair", "Priya Nair"),
        ("name: Devraj Singh", "Devraj Singh"),
        ("Devraj Singh", "Devraj Singh"),
        ("this is Meera", "Meera"),
        ("call me Ana-Maria", "Ana-Maria"),
    ],
)
def test_extract_name_accepts_supported_phrasings(text: str, expected: str) -> None:
    assert extract_name(text) == expected


@pytest.mark.parametrize(
    "text",
    ["devraj@example.com", "2026-12-20", "09:30", "hello there", "book an interview"],
)
def test_extract_name_declines_when_nothing_name_like_is_present(text: str) -> None:
    assert extract_name(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("devraj@example.com", "devraj@example.com"),
        ("Ping DEVRAJ@Example.COM please", "devraj@example.com"),
        ("no address here", None),
    ],
)
def test_extract_email(text: str, expected: str | None) -> None:
    assert extract_email(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-12-20", date(2026, 12, 20)),
        ("on 2026-1-5 at noon", date(2026, 1, 5)),
        ("lets do 20/12/2026", date(2026, 12, 20)),
        ("20 Dec 2026", date(2026, 12, 20)),
        ("14 May", date(2026, 5, 14)),
        ("tomorrow", date(2026, 1, 16)),
        ("today", date(2026, 1, 15)),
        ("day after tomorrow", date(2026, 1, 17)),
        ("no date mentioned", None),
    ],
)
def test_extract_date(text: str, expected: date | None) -> None:
    assert extract_date(text, today=TODAY) == expected


def test_explicit_past_date_is_not_silently_rolled_forward() -> None:
    # The service layer rejects past dates with a clear error; silently moving
    # the date would hide the user's mistake.
    assert extract_date("2020-01-01", today=TODAY) == date(2020, 1, 1)


def test_year_less_date_rolls_forward_only_when_it_has_passed() -> None:
    # Still ahead of 15 January, so the year is unchanged.
    assert extract_date("14 December", today=TODAY) == date(2026, 12, 14)
    # Already behind 1 March, so it moves to the next occurrence.
    assert extract_date("14 January", today=date(2026, 3, 1)) == date(2027, 1, 14)


def test_weekday_dates_resolve_forward() -> None:
    # 2026-01-15 is a Thursday; the next Friday is the 16th.
    assert extract_date("next Friday", today=TODAY) == date(2026, 1, 16)


def test_named_weekday_always_moves_forward_at_least_one_day() -> None:
    # Today is a Thursday, so "Thursday" must mean the following week.
    assert extract_date("Thursday", today=TODAY) == date(2026, 1, 22)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("09:30", time(9, 30)),
        ("at 14:30", time(14, 30)),
        ("2:30 pm", time(14, 30)),
        ("2:30 am", time(2, 30)),
        ("12:00 am", time(0, 0)),
        ("3 pm", time(15, 0)),
        ("11am", time(11, 0)),
        ("no time here", None),
        ("at 25:00", None),
    ],
)
def test_extract_time(text: str, expected: time | None) -> None:
    assert extract_time(text) == expected


def test_extract_notes() -> None:
    assert extract_notes("notes: prefers mornings") == "prefers mornings"
    assert extract_notes("note=needs a ramp") == "needs a ramp"
    assert extract_notes("nothing labelled") is None


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def test_parse_booking_fields_extracts_a_complete_record() -> None:
    fields = parse_booking_fields(
        "Hi, I'm Devraj Singh - devraj@example.com - please book 2026-12-20 at 09:30",
        today=TODAY,
    )

    assert fields["intent"] is BookingIntent.BOOK_INTERVIEW
    assert fields["name"] == "Devraj Singh"
    assert fields["email"] == "devraj@example.com"
    assert fields["date"] == date(2026, 12, 20)
    assert fields["time"] == time(9, 30)
    assert fields["missing_fields"] == []


def test_parse_booking_fields_reports_what_is_still_missing() -> None:
    fields = parse_booking_fields("I want to book an interview", today=TODAY)

    assert fields["intent"] is BookingIntent.BOOK_INTERVIEW
    assert "name" not in fields
    assert fields["missing_fields"] == ["name", "email", "date", "time"]


def test_parse_booking_fields_never_invents_values() -> None:
    fields = parse_booking_fields("devraj@example.com", today=TODAY)

    assert fields["email"] == "devraj@example.com"
    assert "name" not in fields
    assert "date" not in fields
    assert "time" not in fields
    assert fields["missing_fields"] == ["name", "date", "time"]


def test_parse_booking_fields_includes_labelled_notes() -> None:
    fields = parse_booking_fields(
        "book an interview on 2026-12-20 at 09:30, notes: needs a quiet room",
        today=TODAY,
    )
    assert fields["notes"] == "needs a quiet room"


def test_required_fields_are_the_documented_set() -> None:
    assert rules.REQUIRED_FIELDS == ("name", "email", "date", "time")

"""Booking request schemas and the multi-turn draft merge."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from app.core.enums import BookingIntent, BookingSource
from app.schemas.booking import (
    BookingCreate,
    BookingDraft,
    BookingRead,
    ExtractedBookingFields,
)

TOMORROW = dt.date.today() + dt.timedelta(days=1)


def valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "Devraj Singh",
        "email": "devraj@example.com",
        "date": TOMORROW,
        "time": "14:30",
        "notes": None,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# BookingCreate
# ---------------------------------------------------------------------------


def test_booking_create_accepts_a_valid_payload() -> None:
    booking = BookingCreate.model_validate(valid_payload())

    assert booking.name == "Devraj Singh"
    assert booking.date == TOMORROW
    assert booking.time == dt.time(14, 30)


def test_booking_create_strips_surrounding_whitespace_from_the_name() -> None:
    assert BookingCreate.model_validate(valid_payload(name="  Devraj  ")).name == "Devraj"


def test_booking_create_rejects_a_blank_name() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        BookingCreate.model_validate(valid_payload(name="   "))


def test_booking_create_rejects_a_past_date() -> None:
    yesterday = dt.date.today() - dt.timedelta(days=1)
    with pytest.raises(ValidationError, match="must not be in the past"):
        BookingCreate.model_validate(valid_payload(date=yesterday))


def test_booking_create_rejects_a_date_beyond_the_horizon() -> None:
    too_far = dt.date.today() + dt.timedelta(days=400)
    with pytest.raises(ValidationError, match="within 365 days"):
        BookingCreate.model_validate(valid_payload(date=too_far))


def test_booking_create_rejects_an_invalid_email() -> None:
    with pytest.raises(ValidationError):
        BookingCreate.model_validate(valid_payload(email="not-an-email"))


def test_booking_create_rejects_an_invalid_time() -> None:
    with pytest.raises(ValidationError):
        BookingCreate.model_validate(valid_payload(time="half past two"))


def test_booking_create_serialises_dates_as_iso_strings() -> None:
    dumped = BookingCreate.model_validate(valid_payload()).model_dump(mode="json")
    assert dumped["date"] == TOMORROW.isoformat()
    assert dumped["time"] == "14:30:00"


def test_json_schema_builds_despite_date_and_time_field_names() -> None:
    # Regression guard: a field named `date`/`time` whose annotation is the bare
    # `date`/`time` type shadows it and raises PydanticUserError.
    schema = BookingCreate.model_json_schema()
    assert "date" in schema["properties"]
    assert "time" in schema["properties"]


def test_booking_read_carries_the_provenance_fields() -> None:
    booking = BookingRead.model_validate(
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "Devraj",
            "email": "devraj@example.com",
            "date": TOMORROW,
            "time": dt.time(9, 0),
            "notes": None,
            "source": "chat",
            "session_id": "abc",
            "created_at": dt.datetime.now(dt.UTC),
        },
    )
    assert booking.source is BookingSource.CHAT
    assert booking.session_id == "abc"


# ---------------------------------------------------------------------------
# ExtractedBookingFields
# ---------------------------------------------------------------------------


def test_extracted_fields_default_to_other_intent_and_null_values() -> None:
    fields = ExtractedBookingFields()

    assert fields.intent is BookingIntent.OTHER
    assert fields.name is None
    assert fields.missing_fields == []


# ---------------------------------------------------------------------------
# BookingDraft
# ---------------------------------------------------------------------------


def test_draft_starts_empty_and_incomplete() -> None:
    draft = BookingDraft()

    assert draft.missing_fields == ["name", "email", "date", "time"]
    assert draft.is_complete is False


def test_draft_reports_completion_once_every_required_field_is_present() -> None:
    draft = BookingDraft(
        name="Devraj Singh",
        email="devraj@example.com",
        date=TOMORROW,
        time=dt.time(9, 0),
    )

    assert draft.missing_fields == []
    assert draft.is_complete is True


def test_merge_applies_only_the_fields_present_in_the_update() -> None:
    draft = BookingDraft(name="Devraj Singh")

    merged = draft.merge(ExtractedBookingFields(email="devraj@example.com"))

    assert merged.name == "Devraj Singh"
    assert merged.email == "devraj@example.com"
    assert merged.missing_fields == ["date", "time"]


def test_merge_does_not_mutate_the_original_draft() -> None:
    draft = BookingDraft(name="Devraj Singh")
    draft.merge(ExtractedBookingFields(email="devraj@example.com"))

    assert draft.email is None


def test_merge_across_several_turns_builds_a_complete_booking() -> None:
    draft = BookingDraft()
    draft = draft.merge(ExtractedBookingFields(intent=BookingIntent.BOOK_INTERVIEW))
    draft = draft.merge(ExtractedBookingFields(name="Devraj Singh"))
    draft = draft.merge(ExtractedBookingFields(email="devraj@example.com"))
    draft = draft.merge(ExtractedBookingFields(date=TOMORROW))
    draft = draft.merge(ExtractedBookingFields(time=dt.time(9, 30)))

    assert draft.is_complete
    assert draft.name == "Devraj Singh"
    assert draft.time == dt.time(9, 30)


def test_merge_skips_a_malformed_value_without_discarding_the_rest() -> None:
    # `model_construct` deliberately bypasses validation so the extractor can be
    # simulated returning a bad e-mail, which the merge must tolerate.
    draft = BookingDraft()
    bad_update = ExtractedBookingFields.model_construct(
        name="Devraj Singh",
        email="not-an-email",
        date=None,
        time=None,
        notes=None,
        missing_fields=[],
        intent=BookingIntent.OTHER,
    )

    merged = draft.merge(bad_update)

    assert merged.name == "Devraj Singh"
    assert merged.email is None


def test_merge_keeps_a_previously_valid_value_when_the_update_is_null() -> None:
    draft = BookingDraft(name="Devraj Singh")
    merged = draft.merge(ExtractedBookingFields())

    assert merged.name == "Devraj Singh"


def test_draft_survives_a_json_round_trip() -> None:
    draft = BookingDraft(name="Devraj Singh", date=TOMORROW, time=dt.time(9, 0))

    restored = BookingDraft.model_validate_json(draft.model_dump_json())

    assert restored == draft
    assert restored.date == TOMORROW

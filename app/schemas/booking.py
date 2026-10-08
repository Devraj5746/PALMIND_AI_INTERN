"""Schemas for interview booking (direct API and LLM-driven chat flow).

.. note::
   Date and time annotations are written as ``dt.date`` / ``dt.time`` rather
   than ``date`` / ``time``.  A field named ``date`` whose annotation is also
   ``date`` shadows the type when Pydantic resolves the (deferred) annotation
   string, which raises ``PydanticUserError``.  Qualifying the type is the
   least surprising fix.
"""

from __future__ import annotations

import datetime as dt
from typing import Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, ValidationError, field_validator

from app.core.enums import BookingIntent, BookingSource

MAX_BOOKING_HORIZON_DAYS: Final[int] = 365


class BookingCreate(BaseModel):
    """Direct REST payload to create an interview booking."""

    name: str = Field(min_length=1, max_length=120, examples=["Devraj Singh"])
    email: EmailStr = Field(examples=["devraj@example.com"])
    date: dt.date = Field(description="Interview date (YYYY-MM-DD).", examples=["2026-05-14"])
    time: dt.time = Field(description="Interview start time (HH:MM).", examples=["14:30"])
    notes: str | None = Field(default=None, max_length=1000)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("name must not be blank")
        return cleaned

    @field_validator("date")
    @classmethod
    def _not_in_the_past(cls, value: dt.date) -> dt.date:
        today = dt.date.today()
        if value < today:
            raise ValueError("date must not be in the past")
        if value > today + dt.timedelta(days=MAX_BOOKING_HORIZON_DAYS):
            raise ValueError(f"date must be within {MAX_BOOKING_HORIZON_DAYS} days from today")
        return value


class BookingRead(BaseModel):
    """A persisted interview booking."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    email: EmailStr
    date: dt.date
    time: dt.time
    notes: str | None = None
    source: BookingSource
    session_id: str | None = None
    created_at: dt.datetime


class BookingListResponse(BaseModel):
    """Offset-paginated list of bookings."""

    total: int
    limit: int
    offset: int
    items: list[BookingRead]


class ExtractedBookingFields(BaseModel):
    """Structured fields the LLM extracts from a natural-language turn.

    Every field is optional because a user may provide the details across
    several messages; the conversational flow fills the gaps over multiple
    turns and only persists once the record is complete.
    """

    intent: BookingIntent = Field(
        default=BookingIntent.OTHER,
        description="Classified intent of the user's latest message.",
    )
    name: str | None = Field(default=None, description="Full name of the candidate.")
    email: EmailStr | None = Field(default=None, description="Contact email address.")
    date: dt.date | None = Field(default=None, description="Requested interview date.")
    time: dt.time | None = Field(default=None, description="Requested interview start time.")
    notes: str | None = Field(default=None, description="Any extra context the user provided.")
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Fields the assistant still needs before the booking can be created.",
    )


class BookingDraft(BaseModel):
    """In-progress booking kept in Redis between conversation turns."""

    name: str | None = None
    email: EmailStr | None = None
    date: dt.date | None = None
    time: dt.time | None = None
    notes: str | None = None

    def merge(self, update: ExtractedBookingFields) -> BookingDraft:
        """Return a new draft with any non-null extracted values applied.

        Each field is validated independently, so one malformed value (say an
        e-mail the model mangled) is skipped rather than discarding the whole
        draft and losing the fields that were fine.
        """
        data = self.model_dump()
        for field_name in ("name", "email", "date", "time", "notes"):
            value = getattr(update, field_name)
            if value is None:
                continue
            try:
                data = BookingDraft.model_validate({**data, field_name: value}).model_dump()
            except ValidationError:
                continue
        return BookingDraft.model_validate(data)

    @property
    def missing_fields(self) -> list[str]:
        """Names of the fields still required to persist the booking."""
        return [
            field_name
            for field_name in ("name", "email", "date", "time")
            if getattr(self, field_name) is None
        ]

    @property
    def is_complete(self) -> bool:
        return not self.missing_fields

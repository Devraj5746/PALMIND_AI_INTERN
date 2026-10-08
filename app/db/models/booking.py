"""ORM model for interview bookings."""

from __future__ import annotations

from datetime import date, time

from sqlalchemy import Date, Index, String, Text, Time, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.enums import BookingSource
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Booking(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A scheduled interview slot.

    Uniqueness is enforced on ``(email, date, time)`` so the same candidate
    cannot hold two identical slots; the service layer translates the resulting
    integrity error into an HTTP 409.
    """

    __tablename__ = "bookings"
    __table_args__ = (
        UniqueConstraint("email", "date", "time", name="booking_slot"),
        Index("ix_bookings_date_time", "date", "time"),
    )

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    time: Mapped[time] = mapped_column(Time, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    source: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=BookingSource.API.value,
        doc="Origin of the booking: ``api`` or ``chat``.",
    )
    session_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        doc="Chat session that produced the booking, when created conversationally.",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Booking id={self.id} email={self.email!r} {self.date} {self.time}>"


__all__ = ["Booking"]

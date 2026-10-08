"""Repository for interview bookings."""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.booking import Booking


class BookingRepository:
    """Persistence operations for :class:`Booking`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, booking: Booking) -> Booking:
        """Stage a booking and flush so its id and timestamps are populated."""
        self._session.add(booking)
        await self._session.flush()
        return booking

    async def get(self, booking_id: uuid.UUID) -> Booking | None:
        return await self._session.get(Booking, booking_id)

    async def find_slot(
        self,
        *,
        email: str,
        on_date: date,
        at_time: object,
    ) -> Booking | None:
        """Return an existing booking for the same candidate, date and time."""
        stmt = select(Booking).where(
            Booking.email == email.lower(),
            Booking.date == on_date,
            Booking.time == at_time,
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        email: str | None = None,
        session_id: str | None = None,
    ) -> tuple[list[Booking], int]:
        """Return a page of bookings plus the total row count."""
        filters = []
        if email is not None:
            filters.append(Booking.email == email.lower())
        if session_id is not None:
            filters.append(Booking.session_id == session_id)

        items_stmt = (
            select(Booking)
            .where(*filters)
            .order_by(Booking.date, Booking.time, Booking.created_at)
            .limit(limit)
            .offset(offset)
        )
        count_stmt = select(func.count()).select_from(Booking).where(*filters)

        items = list((await self._session.execute(items_stmt)).scalars().all())
        total = int((await self._session.execute(count_stmt)).scalar_one())
        return items, total

    async def delete(self, booking: Booking) -> None:
        await self._session.delete(booking)
        await self._session.flush()

    async def count(self) -> int:
        return int(
            (await self._session.execute(select(func.count()).select_from(Booking))).scalar_one(),
        )

"""Interview booking endpoints (direct REST interface)."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from app.api.deps import BookingServiceDep
from app.schemas.booking import BookingCreate, BookingListResponse, BookingRead

router = APIRouter(prefix="/bookings", tags=["bookings"])


@router.post(
    "",
    response_model=BookingRead,
    status_code=status.HTTP_201_CREATED,
    summary="Book an interview slot",
    description=(
        "Creates an interview booking directly, bypassing the LLM. The "
        "conversational equivalent (which extracts these fields from natural "
        "language) is `POST /chat`."
    ),
)
async def create_booking(service: BookingServiceDep, payload: BookingCreate) -> BookingRead:
    """Create an interview booking."""
    return await service.create(payload)


@router.get(
    "",
    response_model=BookingListResponse,
    summary="List interview bookings",
)
async def list_bookings(
    service: BookingServiceDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    email: Annotated[str | None, Query(description="Filter by candidate email.")] = None,
    session_id: Annotated[
        str | None,
        Query(description="Filter by the chat session that created the booking."),
    ] = None,
) -> BookingListResponse:
    """Return a page of bookings, ordered by slot."""
    bookings, total = await service.list_bookings(
        limit=limit,
        offset=offset,
        email=email,
        session_id=session_id,
    )
    return BookingListResponse(total=total, limit=limit, offset=offset, items=bookings)


@router.get(
    "/{booking_id}",
    response_model=BookingRead,
    summary="Get a booking",
)
async def get_booking(service: BookingServiceDep, booking_id: uuid.UUID) -> BookingRead:
    """Return a single booking."""
    return await service.get(booking_id)


@router.delete(
    "/{booking_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a booking",
)
async def delete_booking(service: BookingServiceDep, booking_id: uuid.UUID) -> Response:
    """Delete a booking."""
    await service.delete(booking_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

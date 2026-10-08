"""Interview booking: LLM-driven extraction, persistence, and the chat flow.

Three collaborating pieces:

* :class:`BookingExtractor` — asks the chat model, through LangChain's
  ``with_structured_output``, to turn a free-text message into typed
  :class:`~app.schemas.booking.ExtractedBookingFields`.
* :class:`BookingService` — validates and persists bookings, and owns the
  uniqueness rule (no duplicate slot for the same candidate).
* :class:`BookingConversation` — the multi-turn slot-filling state machine that
  fills the gaps across turns using the Redis-backed draft.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.enums import BookingIntent, BookingSource
from app.core.errors import AppError, ConflictError, NotFoundError
from app.core.logging import get_logger
from app.db.models.booking import Booking
from app.db.repositories.booking_repository import BookingRepository
from app.schemas.booking import (
    BookingCreate,
    BookingDraft,
    BookingRead,
    ExtractedBookingFields,
)
from app.services.llm.tags import TASK_TAG_CONVERSE
from app.services.memory.base import BookingDraftStore

logger = get_logger(__name__)

BOOKING_EXTRACTION_INSTRUCTIONS = """\
You are the booking assistant's information-extraction step.

Read the user's latest message together with the conversation so far and return
the interview-booking details it contains.

Rules:
- `intent`: `book_interview` if the user wants to schedule or continues a
  scheduling conversation; `cancel_booking` to cancel; `list_bookings` to see
  existing bookings; otherwise `other`.
- `name`, `email`, `date`, `time`, `notes`: fill only fields explicitly present
  in the conversation. Leave everything else null — never invent a value.
- `date` must be ISO `YYYY-MM-DD`, `time` must be 24-hour `HH:MM`.
- Resolve relative dates ("tomorrow", "next Monday") against today's date.
- `missing_fields`: list the required fields still unknown, from
  name, email, date, time."""

BOOKING_EXTRACTION_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", f"{TASK_TAG_CONVERSE}\n{BOOKING_EXTRACTION_INSTRUCTIONS}"),
        MessagesPlaceholder("history", optional=True),
        ("human", "{message}"),
    ],
)

#: Human-readable labels used when asking for missing fields.
FIELD_LABELS: dict[str, str] = {
    "name": "full name",
    "email": "email address",
    "date": "preferred date",
    "time": "preferred time",
}


class BookingExtractor:
    """Extracts structured booking fields from natural language."""

    def __init__(self, *, chat_model: BaseChatModel, settings: Settings) -> None:
        self._settings = settings
        # Built once: ``with_structured_output`` binds the schema to the model.
        self._structured = chat_model.with_structured_output(ExtractedBookingFields)

    async def extract(
        self,
        message: str,
        *,
        history: list[BaseMessage] | None = None,
    ) -> ExtractedBookingFields:
        """Return the fields found in ``message``.

        Extraction is best-effort: if the provider fails we degrade to an
        ``other`` intent so the turn still gets answered instead of erroring.
        """
        messages = BOOKING_EXTRACTION_PROMPT.format_messages(
            message=message,
            history=history or [],
        )
        try:
            result = await self._structured.ainvoke(messages)
        except Exception:
            logger.warning("Booking extraction failed; treating as 'other'", exc_info=True)
            return ExtractedBookingFields()

        if isinstance(result, ExtractedBookingFields):
            return result
        if isinstance(result, dict):
            return ExtractedBookingFields.model_validate(result)
        raise AppError(  # pragma: no cover - defensive
            "The booking extractor returned an unexpected payload.",
            details={"type": type(result).__name__},
        )


class BookingService:
    """Persistence and validation for interview bookings."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings

    async def create(
        self,
        payload: BookingCreate,
        *,
        source: BookingSource = BookingSource.API,
        session_id: str | None = None,
    ) -> BookingRead:
        """Persist a booking, rejecting duplicate slots for the same candidate.

        ``source`` defaults to ``api``; the conversational flow passes ``chat``.
        """
        email = str(payload.email).lower()

        async with self._session_factory() as session:
            repository = BookingRepository(session)

            existing = await repository.find_slot(
                email=email,
                on_date=payload.date,
                at_time=payload.time,
            )
            if existing is not None:
                raise ConflictError(
                    "A booking already exists for that candidate at the requested time.",
                    details={
                        "booking_id": str(existing.id),
                        "email": email,
                        "date": payload.date.isoformat(),
                        "time": payload.time.strftime("%H:%M"),
                    },
                )

            booking = Booking(
                name=payload.name,
                email=email,
                date=payload.date,
                time=payload.time,
                notes=payload.notes,
                source=source.value,
                session_id=session_id,
            )
            try:
                await repository.add(booking)
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise ConflictError(
                    "That interview slot is already taken.",
                    details={"email": email, "date": payload.date.isoformat()},
                ) from exc

            logger.info(
                "Booking created (id=%s, source=%s, session=%s)",
                booking.id,
                source.value,
                session_id,
            )
            return BookingRead.model_validate(booking)

    async def get(self, booking_id: uuid.UUID) -> BookingRead:
        async with self._session_factory() as session:
            booking = await BookingRepository(session).get(booking_id)
        if booking is None:
            raise NotFoundError(
                "No booking exists with that id.",
                details={"booking_id": str(booking_id)},
            )
        return BookingRead.model_validate(booking)

    async def list_bookings(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        email: str | None = None,
        session_id: str | None = None,
    ) -> tuple[list[BookingRead], int]:
        """Return a page of bookings plus the total count.

        Named ``list_bookings`` rather than ``list`` so the method does not shadow
        the builtin ``list`` inside the class body.
        """
        async with self._session_factory() as session:
            bookings, total = await BookingRepository(session).list(
                limit=limit,
                offset=offset,
                email=email,
                session_id=session_id,
            )
        return [BookingRead.model_validate(item) for item in bookings], total

    async def delete(self, booking_id: uuid.UUID) -> None:
        async with self._session_factory() as session:
            repository = BookingRepository(session)
            booking = await repository.get(booking_id)
            if booking is None:
                raise NotFoundError(
                    "No booking exists with that id.",
                    details={"booking_id": str(booking_id)},
                )
            await repository.delete(booking)
            await session.commit()

    async def count(self) -> int:
        async with self._session_factory() as session:
            return await BookingRepository(session).count()


@dataclass(frozen=True, slots=True)
class BookingTurnOutcome:
    """What the booking flow did with one conversation turn."""

    reply: str
    booking: BookingRead | None = None
    completed: bool = False
    cancelled: bool = False
    listed: bool = False


def describe_missing(fields: list[str]) -> str:
    """Render a missing-field list as readable prose ("full name and date")."""
    labels = [FIELD_LABELS.get(name, name) for name in fields]
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


class BookingConversation:
    """Multi-turn interview-booking flow over Redis-backed drafts.

    The flow is entered when the user asks to book, and continues on subsequent
    turns while a draft exists.  A turn that carries no booking information and
    no booking intent returns ``None``, which tells the caller to answer the
    message through the retrieval pipeline instead.
    """

    def __init__(
        self,
        *,
        bookings: BookingService,
        drafts: BookingDraftStore,
        settings: Settings,
    ) -> None:
        self._bookings = bookings
        self._drafts = drafts
        self._settings = settings

    async def handle(
        self,
        *,
        session_id: str,
        message: str,
        extraction: ExtractedBookingFields,
        history: list[BaseMessage] | None = None,
    ) -> BookingTurnOutcome | None:
        """Advance the booking flow, or return ``None`` to fall back to RAG."""
        draft = await self._drafts.get(session_id)

        if extraction.intent == BookingIntent.CANCEL_BOOKING:
            await self._drafts.clear(session_id)
            if draft is None:
                return BookingTurnOutcome(
                    reply="There is no booking in progress to cancel.",
                    cancelled=True,
                )
            return BookingTurnOutcome(
                reply="No problem - I have cancelled that booking request.",
                cancelled=True,
            )

        if extraction.intent == BookingIntent.LIST_BOOKINGS:
            return await self._list_bookings(extraction, draft)

        starting = extraction.intent == BookingIntent.BOOK_INTERVIEW
        has_new_fields = any(
            getattr(extraction, name) is not None
            for name in ("name", "email", "date", "time", "notes")
        )

        if draft is None and not starting:
            return None
        if draft is not None and not has_new_fields:
            # Mid-slot-filling, but this message is about something else.
            return None

        merged = (draft or BookingDraft()).merge(extraction)

        if merged.date is not None and merged.date < dt.date.today():
            merged = merged.model_copy(update={"date": None})
            await self._drafts.save(session_id, merged)
            return BookingTurnOutcome(
                reply=(
                    "That date is already in the past. What future date works for you (YYYY-MM-DD)?"
                ),
            )

        if not merged.is_complete:
            await self._drafts.save(session_id, merged)
            return BookingTurnOutcome(reply=self._ask_for(merged))

        return await self._complete(session_id, merged)

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _ask_for(draft: BookingDraft) -> str:
        missing = describe_missing(draft.missing_fields)
        return f"Happy to book that interview. Could you share your {missing}?"

    async def _complete(self, session_id: str, draft: BookingDraft) -> BookingTurnOutcome:
        name, email, on_date, at_time = draft.name, draft.email, draft.date, draft.time
        if name is None or email is None or on_date is None or at_time is None:  # pragma: no cover
            raise AppError(
                "Internal error: attempted to persist an incomplete booking draft.",
                details={"missing_fields": draft.missing_fields},
            )

        payload = BookingCreate(
            name=name,
            email=email,
            date=on_date,
            time=at_time,
            notes=draft.notes,
        )

        try:
            booking = await self._bookings.create(
                payload,
                source=BookingSource.CHAT,
                session_id=session_id,
            )
        except ConflictError:
            await self._drafts.save(
                session_id,
                draft.model_copy(update={"date": None, "time": None}),
            )
            return BookingTurnOutcome(
                reply="That slot is already taken. Could you suggest another date and time?",
            )
        except ValidationError:
            await self._drafts.save(
                session_id,
                draft.model_copy(update={"date": None, "time": None}),
            )
            return BookingTurnOutcome(
                reply="I could not use that date and time. Could you give me another "
                "date and time (YYYY-MM-DD, HH:MM)?",
            )

        await self._drafts.clear(session_id)
        return BookingTurnOutcome(
            reply=self._confirmation(booking),
            booking=booking,
            completed=True,
        )

    @staticmethod
    def _confirmation(booking: BookingRead) -> str:
        return (
            f"Booked. {booking.name} ({booking.email}) on "
            f"{booking.date.isoformat()} at {booking.time.strftime('%H:%M')}. "
            f"Your reference is {booking.id}."
        )

    async def _list_bookings(
        self,
        extraction: ExtractedBookingFields,
        draft: BookingDraft | None,
    ) -> BookingTurnOutcome:
        email = extraction.email or (draft.email if draft else None)
        if email is None:
            return BookingTurnOutcome(
                reply="Sure - which email address should I look up bookings for?",
                listed=True,
            )

        bookings, total = await self._bookings.list_bookings(email=str(email), limit=10)
        if not bookings:
            return BookingTurnOutcome(
                reply=f"I could not find any interviews booked for {email}.",
                listed=True,
            )

        lines = [
            f"- {item.date.isoformat()} at {item.time.strftime('%H:%M')} ({item.name})"
            for item in bookings
        ]
        return BookingTurnOutcome(
            reply=f"I found {total} booking(s) for {email}:\n" + "\n".join(lines),
            listed=True,
        )


__all__ = [
    "BOOKING_EXTRACTION_PROMPT",
    "BookingConversation",
    "BookingExtractor",
    "BookingService",
    "BookingTurnOutcome",
    "describe_missing",
]

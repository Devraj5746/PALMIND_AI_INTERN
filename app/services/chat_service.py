"""Conversational turn orchestration.

One turn is handled in four steps:

1. **Load** the session transcript from Redis.
2. **Classify** the message into booking intent and extract booking fields
   (one structured call to the chat model).
3. **Route**:
   * the booking conversation handles the turn when it recognises one
     (asking for missing details or confirming a created booking);
   * otherwise the custom RAG pipeline answers from the indexed documents.
4. **Persist** the completed turn back into Redis.

Because memory is Redis-backed and keyed by ``session_id``, multi-turn
conversation works across processes and restarts.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.booking import BookingRead
from app.services.booking_service import (
    BookingConversation,
    BookingExtractor,
    BookingTurnOutcome,
)
from app.services.llm.factory import describe_chat_model
from app.services.memory.redis_memory import ChatMemoryStore
from app.services.rag.engine import ConversationalRAGEngine
from app.services.vector_store.base import RetrievedChunk

logger = get_logger(__name__)

ChatRoute = Literal["rag", "booking"]


@dataclass(frozen=True, slots=True)
class ChatOutcome:
    """Everything one conversation turn produced."""

    session_id: str
    answer: str
    route: ChatRoute
    model: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    standalone_query: str | None = None
    used_history: bool = False
    booking: BookingRead | None = None


class ChatService:
    """Coordinates memory, booking extraction and the RAG engine."""

    def __init__(
        self,
        *,
        memory: ChatMemoryStore,
        rag: ConversationalRAGEngine,
        booking_extractor: BookingExtractor,
        booking_conversation: BookingConversation,
        chat_model: BaseChatModel,
        settings: Settings,
    ) -> None:
        self._memory = memory
        self._rag = rag
        self._booking_extractor = booking_extractor
        self._booking_conversation = booking_conversation
        self._chat_model = chat_model
        self._settings = settings
        self._model_label = describe_chat_model(chat_model)

    # -- public API --------------------------------------------------------
    async def respond(
        self,
        *,
        message: str,
        session_id: str | None = None,
        top_k: int | None = None,
        use_retrieval: bool = True,
    ) -> ChatOutcome:
        """Handle one user turn and return the assistant's answer."""
        resolved_session = session_id or uuid.uuid4().hex
        memory = self._memory.for_session(resolved_session)
        history: list[BaseMessage] = await memory.get_messages()

        # 1. Booking intent + field extraction (single structured call).
        extraction = await self._booking_extractor.extract(message, history=history)

        # 2. Route the turn.
        outcome: BookingTurnOutcome | None = await self._booking_conversation.handle(
            session_id=resolved_session,
            message=message,
            extraction=extraction,
            history=history,
        )

        if outcome is not None:
            result = ChatOutcome(
                session_id=resolved_session,
                answer=outcome.reply,
                route="booking",
                model=self._model_label,
                used_history=bool(history),
                booking=outcome.booking,
            )
        else:
            rag_answer = await self._rag.answer(
                message,
                history=history,
                top_k=top_k,
                use_retrieval=use_retrieval,
            )
            result = ChatOutcome(
                session_id=resolved_session,
                answer=rag_answer.answer,
                route="rag",
                model=self._model_label,
                sources=rag_answer.sources,
                standalone_query=rag_answer.standalone_query,
                used_history=rag_answer.used_history,
            )

        # 3. Persist the turn so the next one has context.
        await memory.append_turn(user_message=message, assistant_message=result.answer)

        logger.info(
            "Turn handled (session=%s, route=%s, history=%d, sources=%d)",
            resolved_session,
            result.route,
            len(history),
            len(result.sources),
        )
        return result

    async def history(self, session_id: str) -> list[BaseMessage]:
        """Return the stored transcript for a session (oldest first)."""
        return await self._memory.for_session(session_id).get_messages()

    async def clear(self, session_id: str) -> int:
        """Clear a session's transcript and booking draft."""
        return await self._memory.clear_session(session_id)


__all__ = ["ChatOutcome", "ChatRoute", "ChatService"]

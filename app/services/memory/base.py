"""Conversation memory and in-progress booking storage.

.. note::
   LangChain's ``BaseChatMessageHistory`` is deprecated as of ``langchain-core``
   1.6.4 (scheduled for removal in 2.0.0), so this module deliberately does not
   subclass it.  The method names mirror its *async* contract
   (``aget_messages`` / ``aadd_messages`` / ``aclear``) and the payloads are
   LangChain ``BaseMessage`` objects serialised with LangChain's own
   ``message_to_dict`` / ``messages_from_dict`` helpers.  Memory is therefore
   fully LangChain-compatible without depending on a deprecated base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.schemas.booking import BookingDraft


class ChatMemory(ABC):
    """Short-term, per-session conversation memory."""

    def __init__(self, *, session_id: str, max_messages: int, ttl_seconds: int) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.session_id = session_id
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds

    @abstractmethod
    async def get_messages(self) -> list[BaseMessage]:
        """Return the stored transcript, oldest first."""

    @abstractmethod
    async def append(self, messages: Sequence[BaseMessage]) -> None:
        """Append messages and refresh the sliding window and TTL."""

    @abstractmethod
    async def clear(self) -> int:
        """Delete the transcript and return how many messages were removed."""

    @abstractmethod
    async def health(self) -> None:
        """Raise if the backing store is unreachable."""

    async def append_turn(self, *, user_message: str, assistant_message: str) -> None:
        """Convenience helper for recording one complete conversation turn."""
        await self.append(
            [
                HumanMessage(content=user_message),
                AIMessage(content=assistant_message),
            ],
        )


class BookingDraftStore(ABC):
    """Stores partially-filled booking drafts between conversation turns."""

    @abstractmethod
    async def get(self, session_id: str) -> BookingDraft | None:
        """Return the in-progress draft for a session, if one exists."""

    @abstractmethod
    async def save(self, session_id: str, draft: BookingDraft) -> None:
        """Persist (or replace) the draft for a session."""

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Discard the draft for a session."""

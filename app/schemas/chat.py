"""Schemas for the conversational RAG API."""

from __future__ import annotations

from typing import Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import ChatRole
from app.schemas.booking import BookingRead

DEFAULT_MAX_MESSAGE_CHARS: Final[int] = 8000


class ChatRequest(BaseModel):
    """A single turn of a (possibly multi-turn) conversation."""

    message: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_MESSAGE_CHARS,
        description="The user's message.",
        examples=["What does the onboarding policy say about leave?"],
    )
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Conversation identifier. Omit it to start a new session; the "
            "response returns the id to reuse for follow-up turns."
        ),
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Override the number of retrieved chunks for this turn.",
    )
    use_retrieval: bool = Field(
        default=True,
        description="Set to false to answer without document retrieval (chit-chat only).",
    )


class SourceChunk(BaseModel):
    """A retrieved chunk cited in an answer."""

    document_id: UUID
    chunk_id: UUID
    chunk_index: int
    filename: str
    score: float
    snippet: str = Field(description="Truncated chunk text for display in citations.")


class ChatMessage(BaseModel):
    """A single transcript entry."""

    role: ChatRole
    content: str


class ChatResponse(BaseModel):
    """Answer for one conversational turn."""

    session_id: str = Field(description="Reuse this id to continue the conversation.")
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    standalone_query: str | None = Field(
        default=None,
        description="The query actually embedded after condensing follow-ups against history.",
    )
    used_history: bool = Field(description="Whether prior conversation turns informed the answer.")
    booking: BookingRead | None = Field(
        default=None,
        description="Populated when this turn completed an interview booking.",
    )
    route: str = Field(
        description="Which pipeline handled the turn: ``rag`` or ``booking``.",
        examples=["rag"],
    )
    model: str = Field(description="Identifier of the model that produced the answer.")


class ChatHistoryResponse(BaseModel):
    """Stored transcript for a session."""

    model_config = ConfigDict(from_attributes=True)

    session_id: str
    messages: list[ChatMessage]


class ChatSessionDeleted(BaseModel):
    """Acknowledgement that a session's memory was cleared."""

    session_id: str
    deleted_messages: int

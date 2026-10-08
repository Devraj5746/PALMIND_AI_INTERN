"""Conversational RAG endpoints."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Path, status

from app.api.deps import ChatServiceDep
from app.core.enums import ChatRole
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSessionDeleted,
    SourceChunk,
)
from app.services.chat_service import ChatOutcome

router = APIRouter(prefix="/chat", tags=["chat"])

#: How much of a retrieved chunk is echoed back as a citation snippet.
SNIPPET_CHARS = 280

SessionIdPath = Annotated[
    str,
    Path(min_length=1, max_length=128, description="Conversation identifier."),
]


def _to_response(outcome: ChatOutcome) -> ChatResponse:
    """Map a service outcome onto the API response schema."""
    return ChatResponse(
        session_id=outcome.session_id,
        answer=outcome.answer,
        sources=[
            SourceChunk(
                document_id=uuid.UUID(chunk.document_id),
                chunk_id=uuid.UUID(chunk.chunk_id),
                chunk_index=chunk.chunk_index,
                filename=chunk.filename,
                score=round(chunk.score, 6),
                snippet=chunk.text[:SNIPPET_CHARS],
            )
            for chunk in outcome.sources
        ],
        standalone_query=outcome.standalone_query,
        used_history=outcome.used_history,
        booking=outcome.booking,
        route=outcome.route,
        model=outcome.model,
    )


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a question (multi-turn, RAG + interview booking)",
    description=(
        "Handles one conversation turn. The message is first classified: if it "
        "is an interview-booking request the slot-filling flow runs, otherwise "
        "the custom RAG pipeline answers from the indexed documents. Omit "
        "`session_id` to start a conversation and reuse the returned id for "
        "follow-up turns — history is kept in Redis."
    ),
)
async def chat(service: ChatServiceDep, payload: ChatRequest) -> ChatResponse:
    """Handle a single conversational turn."""
    outcome = await service.respond(
        message=payload.message,
        session_id=payload.session_id,
        top_k=payload.top_k,
        use_retrieval=payload.use_retrieval,
    )
    return _to_response(outcome)


@router.get(
    "/{session_id}/history",
    response_model=ChatHistoryResponse,
    summary="Get a session's transcript",
)
async def get_history(service: ChatServiceDep, session_id: SessionIdPath) -> ChatHistoryResponse:
    """Return the stored messages for a session, oldest first."""
    messages = await service.history(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            ChatMessage(
                role=ChatRole.USER if message.type == "human" else ChatRole.ASSISTANT,
                content=str(message.content),
            )
            for message in messages
        ],
    )


@router.delete(
    "/{session_id}",
    response_model=ChatSessionDeleted,
    status_code=status.HTTP_200_OK,
    summary="Clear a session",
    description="Deletes the transcript and any in-progress booking draft for the session.",
)
async def clear_session(service: ChatServiceDep, session_id: SessionIdPath) -> ChatSessionDeleted:
    """Forget a conversation."""
    deleted = await service.clear(session_id)
    return ChatSessionDeleted(session_id=session_id, deleted_messages=deleted)

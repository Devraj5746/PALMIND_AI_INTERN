"""Pydantic request/response schemas."""

from __future__ import annotations

from app.schemas.booking import (
    BookingCreate,
    BookingDraft,
    BookingListResponse,
    BookingRead,
    ExtractedBookingFields,
)
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSessionDeleted,
    SourceChunk,
)
from app.schemas.common import (
    ErrorDetail,
    ErrorResponse,
    HealthComponent,
    HealthResponse,
    MessageResponse,
)
from app.schemas.ingestion import (
    ChunkRead,
    DocumentDetail,
    DocumentListResponse,
    DocumentRead,
    IngestionOptions,
    IngestionResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

__all__ = [
    "BookingCreate",
    "BookingDraft",
    "BookingListResponse",
    "BookingRead",
    "ChatHistoryResponse",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatSessionDeleted",
    "ChunkRead",
    "DocumentDetail",
    "DocumentListResponse",
    "DocumentRead",
    "ErrorDetail",
    "ErrorResponse",
    "ExtractedBookingFields",
    "HealthComponent",
    "HealthResponse",
    "IngestionOptions",
    "IngestionResponse",
    "MessageResponse",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
    "SourceChunk",
]

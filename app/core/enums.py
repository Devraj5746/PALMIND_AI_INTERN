"""Shared enumerations used across configuration, schemas and services.

Using ``str``-based enums keeps them JSON-serialisable and directly usable as
FastAPI query parameters while still giving static type safety.  The stdlib
:class:`enum.StrEnum` (Python 3.11+) is used directly: ``str(member)``,
``f"{member}"`` and ``json.dumps(member)`` all yield the raw value.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class ChunkingStrategy(StrEnum):
    """Available text chunking strategies."""

    FIXED = "fixed"
    """Fixed-size character windows with overlap."""

    RECURSIVE = "recursive"
    """Hierarchical separator-based splitting with overlap."""


class LLMProvider(StrEnum):
    """Supported chat-completion providers."""

    OPENAI = "openai"
    MOCK = "mock"


class EmbeddingProvider(StrEnum):
    """Supported embedding providers."""

    OPENAI = "openai"
    HASH = "hash"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class Environment(StrEnum):
    """Deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ChatRole(StrEnum):
    """Roles in a chat transcript."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class BookingIntent(StrEnum):
    """Intent classification for the conversational booking flow."""

    BOOK_INTERVIEW = "book_interview"
    CANCEL_BOOKING = "cancel_booking"
    LIST_BOOKINGS = "list_bookings"
    OTHER = "other"


class DocumentStatus(StrEnum):
    """Lifecycle status of an ingested document."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class BookingSource(StrEnum):
    """How an interview booking was created."""

    API = "api"
    CHAT = "chat"


ALL_CHUNKING_STRATEGIES: Final[tuple[ChunkingStrategy, ...]] = tuple(ChunkingStrategy)

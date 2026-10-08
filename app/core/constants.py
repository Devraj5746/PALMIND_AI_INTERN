"""Application-wide constants and sentinel values."""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_TITLE: Final[str] = "RAG Backend"
API_DESCRIPTION: Final[str] = (
    "Document ingestion (PDF/TXT → chunk → embed → Qdrant) and a custom "
    "conversational RAG API with Redis-backed chat memory and LLM-driven "
    "interview booking."
)
API_VERSION: Final[str] = "1.0.0"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
DEFAULT_SEPARATORS: Final[tuple[str, ...]] = (
    "\n\n",  # paragraph
    "\n",  # line
    ". ",  # sentence
    "! ",
    "? ",
    "; ",
    ", ",
    " ",  # word
    "",  # character
)

# ---------------------------------------------------------------------------
# Redis key namespaces
# ---------------------------------------------------------------------------
REDIS_CHAT_HISTORY_PREFIX: Final[str] = "chat:history"
REDIS_BOOKING_DRAFT_PREFIX: Final[str] = "chat:booking_draft"
REDIS_ROLE_USER: Final[str] = "user"
REDIS_ROLE_ASSISTANT: Final[str] = "assistant"

# ---------------------------------------------------------------------------
# Supported uploads
# ---------------------------------------------------------------------------
MIME_BY_EXTENSION: Final[dict[str, str]] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
}

# ---------------------------------------------------------------------------
# Qdrant payload keys
# ---------------------------------------------------------------------------
PAYLOAD_DOCUMENT_ID: Final[str] = "document_id"
PAYLOAD_CHUNK_ID: Final[str] = "chunk_id"
PAYLOAD_CHUNK_INDEX: Final[str] = "chunk_index"
PAYLOAD_TEXT: Final[str] = "text"
PAYLOAD_FILENAME: Final[str] = "filename"
PAYLOAD_CHUNKING_STRATEGY: Final[str] = "chunking_strategy"

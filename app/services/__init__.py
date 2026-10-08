"""Service layer: providers, pipelines and orchestration.

Layering (outer layers depend on inner ones, never the reverse)::

    api  ->  services  ->  db / external providers

* ``extraction`` / ``chunking`` / ``embeddings`` / ``llm`` / ``vector_store`` /
  ``memory`` are infrastructure adapters behind narrow interfaces.
* ``rag`` composes retrieval + generation (hand-written, no RetrievalQAChain).
* ``ingestion_service`` / ``booking_service`` / ``chat_service`` hold the
  application workflows and own the transaction boundaries.
"""

from __future__ import annotations

from app.services.booking_service import BookingConversation, BookingExtractor, BookingService
from app.services.chat_service import ChatOutcome, ChatService
from app.services.ingestion_service import IngestionResult, IngestionService

__all__ = [
    "BookingConversation",
    "BookingExtractor",
    "BookingService",
    "ChatOutcome",
    "ChatService",
    "IngestionResult",
    "IngestionService",
]

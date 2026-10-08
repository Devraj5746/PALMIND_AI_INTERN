"""Retrieval step of the RAG pipeline.

Deliberately a plain class with one job: embed the query, ask the vector store
for the nearest chunks, and render them as numbered context blocks.

No ``RetrievalQAChain``, no ``create_retrieval_chain`` — the orchestration lives
in :mod:`app.services.rag.engine` as ordinary, readable Python.
"""

from __future__ import annotations

from uuid import UUID

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.vector_store.base import RetrievedChunk, VectorStore

logger = get_logger(__name__)


class ChunkRetriever:
    """Fetches the most relevant chunks for a query."""

    def __init__(self, *, vector_store: VectorStore, settings: Settings) -> None:
        self._vector_store = vector_store
        self._settings = settings

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: UUID | str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most similar chunks, best first."""
        resolved_k = top_k or self._settings.rag_top_k
        resolved_threshold = (
            self._settings.rag_score_threshold if score_threshold is None else score_threshold
        )

        chunks = await self._vector_store.similarity_search(
            query,
            top_k=resolved_k,
            score_threshold=resolved_threshold,
            document_id=document_id,
        )

        logger.debug(
            "Retrieved %d chunk(s) for query %r (k=%d, threshold=%.2f)",
            len(chunks),
            query[:80],
            resolved_k,
            resolved_threshold,
        )
        return chunks

    def format_context(self, chunks: list[RetrievedChunk], *, max_chars: int | None = None) -> str:
        """Render chunks as numbered, source-attributed context blocks.

        The whole block is truncated to ``max_chars`` so an unexpectedly large
        retrieval cannot blow past the model's context window.
        """
        if not chunks:
            return ""

        limit = max_chars or self._settings.rag_max_context_chars
        blocks: list[str] = []
        used = 0

        for position, chunk in enumerate(chunks, start=1):
            header = (
                f"[{position}] source={chunk.filename or 'unknown'} "
                f"chunk={chunk.chunk_index} score={chunk.score:.3f}"
            )
            block = f"{header}\n{chunk.text.strip()}"
            if used + len(block) > limit and blocks:
                break
            blocks.append(block)
            used += len(block) + 2  # account for the blank line separator

        return "\n\n".join(blocks)


__all__ = ["ChunkRetriever"]

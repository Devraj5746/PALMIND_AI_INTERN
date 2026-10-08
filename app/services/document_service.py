"""Read-side operations over ingested documents.

Keeps the document router thin and, more importantly, keeps the *ordering* of
the deletion invariant in one place: vectors are removed before the database
row, so a failure never leaves the SQL metadata describing documents whose
vectors are gone.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.enums import DocumentStatus
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.models.document import Document
from app.db.repositories.document_repository import DocumentRepository
from app.services.vector_store.base import RetrievedChunk, VectorStore

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DocumentStats:
    """Corpus-level counters."""

    documents: int
    chunks: int
    vectors: int


class DocumentService:
    """Query, search and delete ingested documents."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        vector_store: VectorStore,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._settings = settings

    async def list_documents(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: DocumentStatus | None = None,
    ) -> tuple[list[Document], int]:
        """Return a page of documents plus the total count.

        Named ``list_documents`` rather than ``list`` on purpose: a method called
        ``list`` shadows the builtin inside the class body, which silently breaks
        every later ``list[...]`` annotation in this class.
        """
        async with self._session_factory() as session:
            return await DocumentRepository(session).list(
                limit=limit,
                offset=offset,
                status=status,
            )

    async def get(self, document_id: uuid.UUID) -> Document:
        """Return one document with its chunks eagerly loaded."""
        async with self._session_factory() as session:
            document = await DocumentRepository(session).get(document_id)
        if document is None:
            raise NotFoundError(
                "No document exists with that id.",
                details={"document_id": str(document_id)},
            )
        return document

    async def delete(self, document_id: uuid.UUID) -> None:
        """Delete a document, its vectors and its chunk metadata."""
        async with self._session_factory() as session:
            repository = DocumentRepository(session)
            document = await repository.get(document_id)
            if document is None:
                raise NotFoundError(
                    "No document exists with that id.",
                    details={"document_id": str(document_id)},
                )

            # Vectors first: if this fails the metadata row survives and the
            # document remains consistent (a retry can finish the job).
            await self._vector_store.delete_document(document_id)
            await repository.delete(document)
            await session.commit()

        logger.info("Deleted document %s and its vectors", document_id)

    async def search(
        self,
        *,
        query: str,
        top_k: int,
        score_threshold: float | None = None,
        document_id: uuid.UUID | None = None,
    ) -> list[RetrievedChunk]:
        """Semantic search over the indexed corpus."""
        return await self._vector_store.similarity_search(
            query,
            top_k=top_k,
            score_threshold=(
                self._settings.rag_score_threshold if score_threshold is None else score_threshold
            ),
            document_id=document_id,
        )

    async def stats(self) -> DocumentStats:
        """Document, chunk and vector counts."""
        async with self._session_factory() as session:
            repository = DocumentRepository(session)
            documents = await repository.count()
            chunks = await repository.count_all_chunks()
        return DocumentStats(
            documents=documents,
            chunks=chunks,
            vectors=await self._vector_store.count(),
        )


__all__ = ["DocumentService", "DocumentStats"]

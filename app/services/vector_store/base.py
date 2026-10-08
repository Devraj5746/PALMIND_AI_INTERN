"""Vector store abstraction.

The application speaks in terms of LangChain ``Document`` objects and typed
:class:`RetrievedChunk` results, never in terms of a specific vector database.
Swapping Qdrant for another backend means implementing this interface and
nothing else.

Only Qdrant is implemented, per the task constraints (no FAISS, no Chroma).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

from langchain_core.documents import Document


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single retrieved chunk, ready to be cited in an answer."""

    document_id: str
    chunk_id: str
    chunk_index: int
    filename: str
    chunking_strategy: str
    text: str
    score: float

    @property
    def as_citation(self) -> dict[str, object]:
        """Compact representation used when rendering prompt context."""
        return {
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "filename": self.filename,
            "score": round(self.score, 4),
        }


class VectorStore(ABC):
    """Durable storage and similarity search for chunk embeddings."""

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Name of the underlying collection."""

    @abstractmethod
    async def ensure_ready(self, *, dimension: int) -> None:
        """Create the collection if needed and fail fast on a dimension mismatch."""

    @abstractmethod
    async def add_documents(self, documents: list[Document], ids: list[str]) -> list[str]:
        """Embed and upsert ``documents`` under the supplied point ids."""

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        *,
        top_k: int,
        score_threshold: float | None = None,
        document_id: UUID | str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks, best first."""

    @abstractmethod
    async def delete_document(self, document_id: UUID | str) -> None:
        """Delete every chunk belonging to a document."""

    @abstractmethod
    async def delete_ids(self, ids: list[str]) -> None:
        """Delete specific points by id."""

    @abstractmethod
    async def count(self) -> int:
        """Number of stored points."""

    @abstractmethod
    async def health(self) -> None:
        """Raise if the backing service is unreachable."""

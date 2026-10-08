"""Embedding provider abstraction.

Providers are LangChain :class:`~langchain_core.embeddings.Embeddings`
implementations, so they can be handed directly to
``langchain_qdrant.QdrantVectorStore``.  The thin base class only adds what
LangChain does not model: a stable provider name, the vector dimension (needed
to create the Qdrant collection) and a serialisable description that is stored
alongside every ingested document.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

from langchain_core.embeddings import Embeddings


class BaseEmbeddingProvider(Embeddings):
    """Converts text into dense vectors."""

    #: Short identifier, e.g. ``openai``, ``hash`` or ``sentence_transformers``.
    name: ClassVar[str]

    def __init__(self, *, model: str, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.model = model
        self.dimension = dimension

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents, aligned by index with ``texts``."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""

    async def health(self) -> None:
        """Verify the provider is usable. Override when a cheap probe exists."""
        return None

    def describe(self) -> dict[str, Any]:
        """Serialisable description persisted with each ingested document."""
        return {"provider": self.name, "model": self.model, "dimension": self.dimension}

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<{type(self).__name__} model={self.model!r} dim={self.dimension}>"

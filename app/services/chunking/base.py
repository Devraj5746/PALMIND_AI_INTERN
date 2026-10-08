"""Chunking abstractions built on LangChain text splitters.

Each concrete strategy owns a LangChain :class:`~langchain_text_splitters.base.TextSplitter`;
this base class is responsible for the parts the splitter does not do — turning
the resulting documents into typed :class:`Chunk` objects, resolving exact
character offsets against the source text, and producing LangChain
``Document`` objects ready for embedding.

Why wrap instead of using the splitters directly at the call sites?
``Chunk`` carries the ingestion-specific metadata (offsets, chunk index,
strategy) that the API, the database and the tests all depend on.  Keeping a
narrow domain type means the rest of the application never touches LangChain's
``Document`` shape except at the embedding boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

from app.core.enums import ChunkingStrategy

#: Characters-per-token heuristic, close to the ``cl100k_base`` average for
#: English prose.  Used only for reporting, so it stays dependency-free.
CHARS_PER_TOKEN: float = 4.0

#: Metadata key populated by LangChain when ``add_start_index=True``.
START_INDEX_KEY: str = "start_index"


def estimate_tokens(text: str) -> int:
    """Estimate the number of LLM tokens in ``text`` (characters / 4)."""
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single chunk ready for embedding and storage.

    ``start_char``/``end_char`` are exact: ``source[start_char:end_char]`` is
    always identical to :attr:`text`, because the splitter is configured with
    ``strip_whitespace=False``.
    """

    index: int
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.text)

    def to_document(self, vector_id: str, document_id: str, filename: str) -> Document:
        """Render this chunk as a LangChain ``Document`` for the vector store.

        The payload keys are flat so they can be used directly as Qdrant
        ``FieldCondition`` filters (``metadata.document_id``).
        """
        return Document(
            page_content=self.text,
            metadata={
                "document_id": document_id,
                "chunk_id": vector_id,
                "chunk_index": self.index,
                "filename": filename,
                "chunking_strategy": self.metadata.get("chunking_strategy", ""),
            },
        )


class Chunker(ABC):
    """Base class for the selectable chunking strategies."""

    #: Identifier surfaced in API responses and persisted with the document.
    strategy: ClassVar[ChunkingStrategy]

    def __init__(self, *, chunk_size: int, chunk_overlap: int = 0) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter: TextSplitter | None = None

    # -- public API --------------------------------------------------------
    @property
    def splitter(self) -> TextSplitter:
        """The underlying LangChain splitter (built lazily)."""
        if self._splitter is None:
            self._splitter = self._build_splitter()
        return self._splitter

    def chunk(self, text: str) -> list[Chunk]:
        """Split ``text`` into ordered, non-blank :class:`Chunk` objects."""
        if not text or not text.strip():
            return []

        documents = self.splitter.create_documents([text])

        chunks: list[Chunk] = []
        cursor = 0
        for document in documents:
            content = document.page_content
            if not content.strip():
                continue

            start = self._resolve_start_index(text, content, document.metadata, cursor)
            end = start + len(content)
            cursor = start

            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=content,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "chunking_strategy": self.strategy.value,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                    },
                ),
            )

        return chunks

    # -- to implement ------------------------------------------------------
    @abstractmethod
    def _build_splitter(self) -> TextSplitter:
        """Return the LangChain splitter backing this strategy."""

    # -- internals ---------------------------------------------------------
    def _resolve_start_index(
        self,
        source: str,
        content: str,
        metadata: dict[str, Any],
        cursor: int,
    ) -> int:
        """Return the offset of ``content`` in ``source``.

        Prefers LangChain's ``start_index`` metadata; falls back to a monotonic
        forward search when a splitter does not provide it.
        """
        raw = metadata.get(START_INDEX_KEY)
        if isinstance(raw, int) and source[raw : raw + len(content)] == content:
            return raw

        found = source.find(content, cursor)
        if found >= 0:
            return found

        # Should not happen for our own splitters; degrade to the last known
        # position rather than raising, so ingestion never fails over metadata.
        return cursor

    def describe(self) -> dict[str, Any]:
        """Return a serialisable description of the chunker's configuration."""
        return {
            "strategy": self.strategy.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

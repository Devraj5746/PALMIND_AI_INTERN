"""Strategy 2 — recursive separator-based chunking.

Delegates to LangChain's ``RecursiveCharacterTextSplitter``: it splits on the
most meaningful separator first (paragraph breaks), then recursively falls back
to weaker separators (lines, sentences, words, characters) only for the pieces
that are still too large.  The resulting chunks respect document structure,
which measurably improves retrieval quality, at the cost of variable sizes.
"""

from __future__ import annotations

from typing import Any, ClassVar

from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from app.core.constants import DEFAULT_SEPARATORS
from app.core.enums import ChunkingStrategy
from app.services.chunking.base import Chunker


class RecursiveChunker(Chunker):
    """Hierarchical splitter that preserves paragraphs and sentences."""

    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.RECURSIVE

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int = 0,
        separators: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = tuple(separators) if separators is not None else DEFAULT_SEPARATORS

    def _build_splitter(self) -> TextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=list(self.separators),
            # False keeps chunk text clean (no leading "\n\n" noise) while
            # LangChain still reports an accurate, verifiable start index.
            keep_separator=False,
            add_start_index=True,
            # Keep byte-for-byte fidelity so offsets map exactly onto the source.
            strip_whitespace=False,
            length_function=len,
        )

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "separators": list(self.separators)}

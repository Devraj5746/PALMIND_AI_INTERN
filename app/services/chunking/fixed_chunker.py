"""Strategy 1 — fixed-size sliding-window chunking.

Delegates to LangChain's ``CharacterTextSplitter`` configured with an empty
separator, which yields uniform windows of exactly ``chunk_size`` characters
overlapping by ``chunk_overlap``.

It is the cheapest and most predictable strategy and a good baseline: chunks
are always the same shape regardless of document structure, but sentences are
frequently cut in half, which is why both strategies are offered side by side.
"""

from __future__ import annotations

from typing import Any, ClassVar

from langchain_text_splitters import CharacterTextSplitter, TextSplitter

from app.core.enums import ChunkingStrategy
from app.services.chunking.base import Chunker


class FixedSizeChunker(Chunker):
    """Character-window chunker with configurable overlap."""

    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.FIXED

    def _build_splitter(self) -> TextSplitter:
        return CharacterTextSplitter(
            # An empty separator makes the splitter cut on raw characters.
            separator="",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            strip_whitespace=False,
            length_function=len,
        )

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "separator": ""}

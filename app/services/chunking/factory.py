"""Factory that turns a strategy identifier into a configured :class:`Chunker`."""

from __future__ import annotations

from app.core.enums import ChunkingStrategy
from app.core.errors import ValidationError
from app.services.chunking.base import Chunker
from app.services.chunking.fixed_chunker import FixedSizeChunker
from app.services.chunking.recursive_chunker import RecursiveChunker

#: Registry mapping strategy -> concrete implementation.
CHUNKER_REGISTRY: dict[ChunkingStrategy, type[Chunker]] = {
    ChunkingStrategy.FIXED: FixedSizeChunker,
    ChunkingStrategy.RECURSIVE: RecursiveChunker,
}


def build_chunker(
    strategy: ChunkingStrategy,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> Chunker:
    """Instantiate the chunker for ``strategy``.

    Args:
        strategy: Which strategy to build.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between adjacent chunks in characters.

    Raises:
        ValidationError: When the strategy is unknown or parameters are invalid.
    """
    chunker_cls = CHUNKER_REGISTRY.get(strategy)
    if chunker_cls is None:  # pragma: no cover - guarded by the enum
        raise ValidationError(
            f"Unknown chunking strategy '{strategy}'.",
            details={"supported_strategies": [s.value for s in CHUNKER_REGISTRY]},
        )

    try:
        return chunker_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except ValueError as exc:
        raise ValidationError(
            "Invalid chunking parameters.",
            details={"reason": str(exc), "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        ) from exc

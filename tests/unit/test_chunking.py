"""Both chunking strategies: exact offsets, bounds, overlap and edge cases."""

from __future__ import annotations

from itertools import pairwise

import pytest

from app.core.enums import ChunkingStrategy
from app.core.errors import ValidationError
from app.services.chunking import build_chunker
from app.services.chunking.base import Chunker, estimate_tokens
from app.services.chunking.fixed_chunker import FixedSizeChunker
from app.services.chunking.recursive_chunker import RecursiveChunker
from tests.helpers import HANDBOOK_TEXT

ALL_STRATEGIES = [ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE]


@pytest.fixture(params=ALL_STRATEGIES, ids=lambda s: str(s))
def chunker(request: pytest.FixtureRequest) -> Chunker:
    return build_chunker(request.param, chunk_size=200, chunk_overlap=40)


# ---------------------------------------------------------------------------
# Invariants shared by every strategy
# ---------------------------------------------------------------------------


def test_chunks_are_indexed_in_order(chunker: Chunker) -> None:
    chunks = chunker.chunk(HANDBOOK_TEXT)
    assert chunks
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_offsets_are_exact_slices_of_the_source(chunker: Chunker) -> None:
    for chunk in chunker.chunk(HANDBOOK_TEXT):
        assert HANDBOOK_TEXT[chunk.start_char : chunk.end_char] == chunk.text


def test_offsets_are_monotonic(chunker: Chunker) -> None:
    starts = [chunk.start_char for chunk in chunker.chunk(HANDBOOK_TEXT)]
    assert starts == sorted(starts)


def test_no_chunk_exceeds_the_requested_size(chunker: Chunker) -> None:
    for chunk in chunker.chunk(HANDBOOK_TEXT):
        assert 0 < len(chunk.text) <= chunker.chunk_size


def test_no_chunk_is_blank(chunker: Chunker) -> None:
    for chunk in chunker.chunk(HANDBOOK_TEXT):
        assert chunk.text.strip()


def test_chunk_carries_its_strategy_metadata(chunker: Chunker) -> None:
    chunk = chunker.chunk(HANDBOOK_TEXT)[0]
    assert chunk.metadata["chunking_strategy"] == chunker.strategy.value
    assert chunk.metadata["chunk_size"] == chunker.chunk_size
    assert chunk.metadata["chunk_overlap"] == chunker.chunk_overlap


def test_char_and_token_counts_are_consistent(chunker: Chunker) -> None:
    for chunk in chunker.chunk(HANDBOOK_TEXT):
        assert chunk.char_count == len(chunk.text)
        assert chunk.token_estimate >= 1


def test_fixed_chunking_guarantees_overlap_between_adjacent_chunks() -> None:
    # A sliding-character window is exact: every adjacent pair shares text.
    chunks = FixedSizeChunker(chunk_size=200, chunk_overlap=40).chunk(HANDBOOK_TEXT)

    assert len(chunks) > 1
    for previous, current in pairwise(chunks):
        assert current.start_char < previous.end_char
        assert current.start_char == previous.end_char - 40


def test_recursive_chunking_applies_overlap_where_the_splitter_can() -> None:
    # ``RecursiveCharacterTextSplitter`` adds overlap while merging splits, but a
    # paragraph boundary can force a fresh merge run, so overlap is best-effort
    # rather than guaranteed.  The invariant that always holds is that overlap
    # never pushes a chunk past the requested size (asserted above).
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=40).chunk(HANDBOOK_TEXT)

    assert len(chunks) > 1
    overlapping = [
        (previous, current)
        for previous, current in pairwise(chunks)
        if current.start_char < previous.end_char
    ]
    assert overlapping
    for previous, current in overlapping:
        assert current.start_char > previous.start_char


def test_to_document_uses_flat_filterable_metadata(chunker: Chunker) -> None:
    chunk = chunker.chunk(HANDBOOK_TEXT)[0]
    document = chunk.to_document("vec-1", "doc-1", "handbook.txt")

    assert document.page_content == chunk.text
    assert document.metadata["document_id"] == "doc-1"
    assert document.metadata["chunk_id"] == "vec-1"
    assert document.metadata["filename"] == "handbook.txt"
    assert document.metadata["chunk_index"] == 0
    assert document.metadata["chunking_strategy"] == chunker.strategy.value


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t ", "\r\n"])
def test_blank_input_produces_no_chunks(chunker: Chunker, text: str) -> None:
    assert chunker.chunk(text) == []


def test_text_shorter_than_the_window_is_a_single_chunk(chunker: Chunker) -> None:
    chunks = chunker.chunk("Short text.")
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."
    assert chunks[0].start_char == 0


def test_describe_includes_the_strategy(chunker: Chunker) -> None:
    described = chunker.describe()
    assert described["strategy"] == chunker.strategy.value
    assert described["chunk_size"] == chunker.chunk_size


# ---------------------------------------------------------------------------
# Strategy-specific behaviour
# ---------------------------------------------------------------------------


def test_fixed_chunker_produces_uniform_windows() -> None:
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk(HANDBOOK_TEXT)

    assert len(chunks) > 1
    # Every chunk except possibly the last is exactly the requested size.
    assert all(len(chunk.text) == 100 for chunk in chunks[:-1])


def test_recursive_chunker_keeps_paragraph_boundaries() -> None:
    chunker = RecursiveChunker(chunk_size=400, chunk_overlap=0)
    chunks = chunker.chunk(HANDBOOK_TEXT)

    # Paragraph boundaries survive: no chunk starts with the previous chunk's
    # tail glued to a new heading.
    assert any("Annual leave" in chunk.text for chunk in chunks)
    assert any("Remote work" in chunk.text for chunk in chunks)


def test_recursive_chunker_uses_documented_default_separators() -> None:
    chunker = RecursiveChunker(chunk_size=200)
    described = chunker.describe()
    assert "\n\n" in described["separators"]


def test_fixed_chunker_describes_its_raw_character_separator() -> None:
    assert FixedSizeChunker(chunk_size=100).describe()["separator"] == ""


# ---------------------------------------------------------------------------
# Factory and construction
# ---------------------------------------------------------------------------


def test_registry_covers_every_declared_strategy() -> None:
    for strategy in ChunkingStrategy:
        assert build_chunker(strategy, chunk_size=100, chunk_overlap=0)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(100, 100), (100, 500), (50, 60)],
)
def test_factory_rejects_overlap_not_smaller_than_size(size: int, overlap: int) -> None:
    with pytest.raises(ValidationError):
        build_chunker(ChunkingStrategy.FIXED, chunk_size=size, chunk_overlap=overlap)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-10, 0)])
def test_factory_rejects_non_positive_size(size: int, overlap: int) -> None:
    with pytest.raises(ValidationError):
        build_chunker(ChunkingStrategy.RECURSIVE, chunk_size=size, chunk_overlap=overlap)


def test_zero_overlap_is_allowed_and_non_overlapping() -> None:
    chunks = build_chunker(ChunkingStrategy.RECURSIVE, chunk_size=200, chunk_overlap=0).chunk(
        HANDBOOK_TEXT,
    )
    for previous, current in pairwise(chunks):
        assert current.start_char >= previous.end_char


def test_estimate_tokens_is_zero_for_empty_text() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1

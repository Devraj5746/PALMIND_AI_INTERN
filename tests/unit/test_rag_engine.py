"""The hand-written conversational RAG pipeline (no RetrievalQAChain)."""

from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage

from app.container import Container
from app.services.llm.mock_chat_model import NO_CONTEXT_ANSWER
from app.services.rag.retriever import ChunkRetriever
from app.services.vector_store.base import RetrievedChunk


def make_chunk(text: str, *, score: float = 0.9, index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=str(uuid.uuid4()),
        chunk_id=str(uuid.uuid4()),
        chunk_index=index,
        filename="handbook.txt",
        chunking_strategy="recursive",
        text=text,
        score=score,
    )


# ---------------------------------------------------------------------------
# Context rendering
# ---------------------------------------------------------------------------


def test_format_context_numbers_and_attributes_every_chunk(settings) -> None:
    retriever = ChunkRetriever(vector_store=None, settings=settings)  # type: ignore[arg-type]

    rendered = retriever.format_context(
        [make_chunk("first block", score=0.91), make_chunk("second block", score=0.42, index=1)],
    )

    assert "[1] source=handbook.txt chunk=0 score=0.910" in rendered
    assert "[2] source=handbook.txt chunk=1 score=0.420" in rendered
    assert "first block" in rendered
    assert "second block" in rendered


def test_format_context_returns_empty_string_for_no_chunks(settings) -> None:
    retriever = ChunkRetriever(vector_store=None, settings=settings)  # type: ignore[arg-type]
    assert retriever.format_context([]) == ""


def test_format_context_stops_before_exceeding_the_budget(settings) -> None:
    retriever = ChunkRetriever(vector_store=None, settings=settings)  # type: ignore[arg-type]
    chunks = [make_chunk("x" * 100, index=index) for index in range(10)]

    rendered = retriever.format_context(chunks, max_chars=250)

    assert len(rendered) < 10 * 100
    assert "[1]" in rendered
    assert "[10]" not in rendered


def test_format_context_always_keeps_the_first_block(settings) -> None:
    retriever = ChunkRetriever(vector_store=None, settings=settings)  # type: ignore[arg-type]
    # A single chunk larger than the budget must still be returned.
    rendered = retriever.format_context([make_chunk("y" * 5000)], max_chars=100)
    assert "y" * 100 in rendered


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


async def test_single_turn_answer_is_grounded_and_cited(seeded_container: Container) -> None:
    result = await seeded_container.rag_engine.answer("How many days of annual leave?")

    assert result.sources
    assert "annual leave" in result.answer.lower()
    assert result.used_history is False
    assert result.context_chars > 0


async def test_sources_carry_the_metadata_needed_for_citations(seeded_container: Container) -> None:
    result = await seeded_container.rag_engine.answer("How many days of annual leave?")

    source = result.sources[0]
    assert source.filename == "handbook.txt"
    assert source.chunking_strategy == "recursive"
    assert source.score > 0
    assert source.as_citation["chunk_index"] == source.chunk_index


async def test_retrieval_can_be_switched_off(seeded_container: Container) -> None:
    result = await seeded_container.rag_engine.answer(
        "hello there",
        use_retrieval=False,
    )

    assert result.sources == []
    assert NO_CONTEXT_ANSWER in result.answer


async def test_answer_declines_when_the_corpus_is_empty(container: Container) -> None:
    result = await container.rag_engine.answer("What is the leave policy?")

    assert result.sources == []
    assert NO_CONTEXT_ANSWER in result.answer


async def test_multi_turn_answers_use_the_stored_history(seeded_container: Container) -> None:
    history = [
        HumanMessage(content="How many days of annual leave?"),
        AIMessage(content="Twenty four days."),
    ]

    result = await seeded_container.rag_engine.answer("and how about remote work?", history=history)

    assert result.used_history is True
    assert result.standalone_query


async def test_condensing_is_skipped_when_history_is_empty(seeded_container: Container) -> None:
    result = await seeded_container.rag_engine.answer("How many days of annual leave?")
    assert result.standalone_query == "How many days of annual leave?"


async def test_condensing_can_be_disabled(container: Container) -> None:
    from app.services.rag.engine import ConversationalRAGEngine

    engine = ConversationalRAGEngine(
        chat_model=container.chat_model,
        retriever=container.retriever,
        settings=container.settings.model_copy(update={"rag_condense_followups": False}),
    )

    result = await engine.answer("follow up", history=[HumanMessage(content="first")])

    assert result.standalone_query == "follow up"


async def test_top_k_is_forwarded_to_retrieval(seeded_container: Container) -> None:
    single = await seeded_container.rag_engine.answer("days", top_k=1, use_retrieval=True)
    assert len(single.sources) <= 1


async def test_document_scoped_answers_cite_only_that_document(seeded_container: Container) -> None:
    documents, _ = await seeded_container.document_service.list_documents()
    assert len(documents) == 1
    target = documents[0].id

    result = await seeded_container.rag_engine.answer(
        "annual leave",
        document_id=target,
    )

    assert result.sources
    assert {source.document_id for source in result.sources} == {str(target)}


async def test_unknown_document_scope_yields_no_sources(seeded_container: Container) -> None:
    result = await seeded_container.rag_engine.answer(
        "annual leave",
        document_id=str(uuid.uuid4()),
    )

    assert result.sources == []


async def test_history_window_is_capped_by_settings(container: Container) -> None:
    long_history = [HumanMessage(content=f"turn {index}") for index in range(100)]

    window = container.rag_engine._history_window(long_history)

    assert len(window) == container.settings.chat_memory_max_messages
    assert window[-1].content == "turn 99"

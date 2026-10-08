"""Qdrant adapter: upsert, search, filtering, deletion and dimension guards."""

from __future__ import annotations

import uuid

import pytest
from langchain_core.documents import Document

from app.container import Container
from app.core.errors import ValidationError
from app.services.vector_store.qdrant_store import build_qdrant_client

DOC_A = str(uuid.uuid4())
DOC_B = str(uuid.uuid4())

#: Qdrant only accepts unsigned integers or UUIDs as point ids, which is why the
#: production code derives every point id from a ``uuid4`` chunk id.
CORPUS = [
    (
        str(uuid.uuid4()),
        DOC_A,
        "annual-leave.txt",
        "Employees get twenty four days of paid annual leave.",
    ),
    (
        str(uuid.uuid4()),
        DOC_A,
        "annual-leave.txt",
        "Unused leave days may be carried into the next year.",
    ),
    (
        str(uuid.uuid4()),
        DOC_B,
        "remote-work.txt",
        "Remote work is allowed up to three days per week.",
    ),
    (
        str(uuid.uuid4()),
        DOC_B,
        "remote-work.txt",
        "Replacement laptops arrive within five working days.",
    ),
]


def _documents() -> tuple[list[Document], list[str]]:
    ids: list[str] = []
    documents: list[Document] = []
    for index, (point_id, document_id, filename, text) in enumerate(CORPUS):
        ids.append(point_id)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "document_id": document_id,
                    "chunk_id": point_id,
                    "chunk_index": index,
                    "filename": filename,
                    "chunking_strategy": "recursive",
                },
            ),
        )
    return ids, documents


async def seed_vector_store(container: Container) -> tuple[list[Document], list[str]]:
    await container.vector_store.ensure_ready(dimension=container.embedding.dimension)
    ids, documents = _documents()
    await container.vector_store.add_documents(documents, ids=ids)
    return ids, documents


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_ensure_ready_creates_the_collection(container: Container) -> None:
    await container.vector_store.ensure_ready(dimension=container.embedding.dimension)
    assert await container.vector_store.count() == 0


async def test_ensure_ready_is_idempotent(container: Container) -> None:
    dimension = container.embedding.dimension
    await container.vector_store.ensure_ready(dimension=dimension)
    await container.vector_store.ensure_ready(dimension=dimension)
    assert container.vector_store.collection_name == container.settings.qdrant_collection


async def test_ensure_ready_rejects_a_dimension_mismatch(container: Container) -> None:
    await container.vector_store.ensure_ready(dimension=container.embedding.dimension)

    with pytest.raises(ValidationError, match="different vector size"):
        await container.vector_store.ensure_ready(dimension=container.embedding.dimension + 1)


async def test_health_check_succeeds_for_an_in_memory_client(container: Container) -> None:
    await container.vector_store.health()


def test_client_supports_the_local_forms(settings) -> None:
    assert build_qdrant_client(settings.model_copy(update={"qdrant_url": ":memory:"})) is not None
    assert build_qdrant_client(settings.model_copy(update={"qdrant_url": "memory"})) is not None


# ---------------------------------------------------------------------------
# Writes and reads
# ---------------------------------------------------------------------------


async def test_add_documents_indexes_every_chunk(container: Container) -> None:
    _, documents = await seed_vector_store(container)
    assert await container.vector_store.count() == len(documents)


async def test_add_documents_is_idempotent_for_the_same_ids(container: Container) -> None:
    ids, documents = await seed_vector_store(container)
    await container.vector_store.add_documents(documents, ids=ids)
    assert await container.vector_store.count() == len(documents)


async def test_add_documents_with_nothing_to_do_is_a_no_op(container: Container) -> None:
    await container.vector_store.ensure_ready(dimension=container.embedding.dimension)
    assert await container.vector_store.add_documents([], ids=[]) == []


async def test_add_documents_rejects_mismatched_ids(container: Container) -> None:
    await container.vector_store.ensure_ready(dimension=container.embedding.dimension)
    _, documents = _documents()

    with pytest.raises(ValueError, match="same length"):
        await container.vector_store.add_documents(documents, ids=[str(uuid.uuid4())])


async def test_similarity_search_returns_typed_hits(container: Container) -> None:
    await seed_vector_store(container)

    hits = await container.vector_store.similarity_search("paid annual leave", top_k=2)

    assert hits
    assert len(hits) <= 2
    top = hits[0]
    assert isinstance(top.document_id, str)
    assert "leave" in top.text
    assert top.filename == "annual-leave.txt"
    assert top.chunking_strategy == "recursive"
    assert top.score > 0


async def test_similarity_search_ranks_the_relevant_chunk_first(container: Container) -> None:
    await seed_vector_store(container)

    leave_hits = await container.vector_store.similarity_search(
        "how many days of annual leave", top_k=1
    )
    remote_hits = await container.vector_store.similarity_search(
        "days of remote work per week", top_k=1
    )

    assert "annual leave" in leave_hits[0].text.lower()
    assert "remote work" in remote_hits[0].text.lower()


async def test_similarity_search_respects_top_k(container: Container) -> None:
    await seed_vector_store(container)
    hits = await container.vector_store.similarity_search("days", top_k=3)
    assert len(hits) <= 3


async def test_similarity_search_can_be_scoped_to_one_document(container: Container) -> None:
    await seed_vector_store(container)

    hits = await container.vector_store.similarity_search(
        "days of leave and remote work",
        top_k=10,
        document_id=DOC_B,
    )

    assert hits
    assert {hit.document_id for hit in hits} == {DOC_B}


async def test_score_threshold_filters_weak_matches(container: Container) -> None:
    await seed_vector_store(container)

    everything = await container.vector_store.similarity_search("leave", top_k=10)
    strict = await container.vector_store.similarity_search("leave", top_k=10, score_threshold=0.99)

    assert len(strict) < len(everything)


async def test_scores_are_sorted_best_first(container: Container) -> None:
    await seed_vector_store(container)
    hits = await container.vector_store.similarity_search("annual leave days", top_k=4)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


async def test_search_on_an_empty_collection_returns_nothing(container: Container) -> None:
    await container.vector_store.ensure_ready(dimension=container.embedding.dimension)
    assert await container.vector_store.similarity_search("anything", top_k=5) == []


# ---------------------------------------------------------------------------
# Deletes
# ---------------------------------------------------------------------------


async def test_delete_document_removes_only_that_documents_chunks(container: Container) -> None:
    await seed_vector_store(container)

    await container.vector_store.delete_document(DOC_A)

    remaining = await container.vector_store.count()
    assert remaining == 2
    hits = await container.vector_store.similarity_search("leave remote equipment", top_k=10)
    assert {hit.document_id for hit in hits} == {DOC_B}


async def test_delete_document_for_an_unknown_id_is_harmless(container: Container) -> None:
    _, documents = await seed_vector_store(container)
    await container.vector_store.delete_document(str(uuid.uuid4()))
    assert await container.vector_store.count() == len(documents)


async def test_delete_ids_removes_specific_points(container: Container) -> None:
    ids, _ = await seed_vector_store(container)

    await container.vector_store.delete_ids([ids[0]])

    assert await container.vector_store.count() == len(ids) - 1


async def test_delete_ids_with_an_empty_list_is_a_no_op(container: Container) -> None:
    ids, _ = await seed_vector_store(container)
    await container.vector_store.delete_ids([])
    assert await container.vector_store.count() == len(ids)

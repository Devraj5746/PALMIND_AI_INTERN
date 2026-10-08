"""Embedding providers (offline hash provider, which is what the suite uses)."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.enums import EmbeddingProvider
from app.services.embeddings.base import BaseEmbeddingProvider
from app.services.embeddings.factory import build_embedding_provider
from app.services.embeddings.hash_embeddings import HashEmbeddings
from tests.helpers import cosine_similarity


@pytest.fixture
def embeddings() -> HashEmbeddings:
    return HashEmbeddings(dimension=128)


def test_dimension_is_honoured(embeddings: HashEmbeddings) -> None:
    assert embeddings.dimension == 128
    assert len(embeddings.embed_query("anything")) == 128
    assert all(len(vector) == 128 for vector in embeddings.embed_documents(["a", "b", "c"]))


def test_provider_is_a_langchain_embeddings_implementation(
    embeddings: HashEmbeddings,
) -> None:
    from langchain_core.embeddings import Embeddings

    assert isinstance(embeddings, Embeddings)
    assert isinstance(embeddings, BaseEmbeddingProvider)


def test_embedding_exposes_the_provider_name(embeddings: HashEmbeddings) -> None:
    assert embeddings.name == "hash"
    assert embeddings.describe() == {
        "provider": "hash",
        "model": "hash-blake2b-v1",
        "dimension": 128,
    }


def test_vectors_are_l2_normalised(embeddings: HashEmbeddings) -> None:
    vector = embeddings.embed_query("annual leave policy")
    norm = sum(value * value for value in vector) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_embeddings_are_deterministic_across_instances() -> None:
    left = HashEmbeddings(dimension=64).embed_query("remote work policy")
    right = HashEmbeddings(dimension=64).embed_query("remote work policy")
    assert left == right


def test_identical_text_is_maximally_similar(embeddings: HashEmbeddings) -> None:
    vector = embeddings.embed_query("twenty four days of annual leave")
    assert cosine_similarity(vector, vector) == pytest.approx(1.0, abs=1e-9)


def test_related_text_scores_higher_than_unrelated_text(embeddings: HashEmbeddings) -> None:
    query = embeddings.embed_query("how many days of annual leave do I get")
    related = embeddings.embed_query("entitled to days of paid annual leave per year")
    unrelated = embeddings.embed_query("replacement laptop monitor help desk")

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_empty_text_yields_a_zero_vector(embeddings: HashEmbeddings) -> None:
    assert embeddings.embed_query("") == [0.0] * 128


def test_batch_and_single_embedding_agree(embeddings: HashEmbeddings) -> None:
    batch = embeddings.embed_documents(["alpha beta"])
    assert batch[0] == embeddings.embed_query("alpha beta")


def test_embed_documents_preserves_alignment(embeddings: HashEmbeddings) -> None:
    vectors = embeddings.embed_documents(["one", "two", "three"])
    assert len(vectors) == 3
    assert vectors[0] == embeddings.embed_query("one")
    assert vectors[2] == embeddings.embed_query("three")


def test_provider_rejects_a_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        HashEmbeddings(dimension=0)


def test_factory_builds_the_hash_provider_from_settings() -> None:
    settings = Settings(
        _env_file=None,
        embedding_provider=EmbeddingProvider.HASH,
        hash_embedding_dimension=64,
    )
    provider = build_embedding_provider(settings)

    assert provider.name == "hash"
    assert provider.dimension == 64


async def test_health_check_is_a_no_op_for_local_providers() -> None:
    await HashEmbeddings(dimension=16).health()

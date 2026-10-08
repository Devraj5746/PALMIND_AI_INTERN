"""Shared pytest fixtures.

The whole suite runs **fully offline**: hash embeddings, the deterministic mock
chat model, in-memory Qdrant, ``fakeredis`` and a throwaway SQLite file.  No
network access and no API keys are required, so ``pytest`` is a complete and
reliable gate in CI.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from asgi_lifespan import LifespanManager
from fakeredis.aioredis import FakeRedis
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.container import Container, build_container
from app.core.config import Settings
from app.core.enums import EmbeddingProvider, LLMProvider
from app.main import create_app
from tests.helpers import HANDBOOK_TEXT, ingest_text

#: Vector size used by the offline hash embedder in tests.  Small keeps the
#: suite fast; large enough that distinct paragraphs do not collide.
HASH_DIMENSION = 256


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Fully offline settings pointing at throwaway storage."""
    return Settings(
        _env_file=None,
        app_env="development",
        debug=True,
        log_level="WARNING",
        embedding_provider=EmbeddingProvider.HASH,
        hash_embedding_dimension=HASH_DIMENSION,
        llm_provider=LLMProvider.MOCK,
        qdrant_url=":memory:",
        qdrant_collection=f"test_{uuid.uuid4().hex[:10]}",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        rag_score_threshold=0.05,
        rag_top_k=4,
    )


@pytest.fixture
def redis_client() -> FakeRedis:
    """An in-process Redis stand-in with the same wire shape as the real one."""
    return FakeRedis(decode_responses=True)


@pytest.fixture
async def container(settings: Settings, redis_client: FakeRedis) -> AsyncIterator[Container]:
    """A started application container, torn down after the test."""
    built = await build_container(settings, redis_client=redis_client)
    await built.startup()
    try:
        yield built
    finally:
        await built.shutdown()


@pytest.fixture
def app(settings: Settings, redis_client: FakeRedis) -> FastAPI:
    """The ASGI application wired to the offline providers."""
    return create_app(settings, redis_client=redis_client)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """An HTTP client that drives the real ASGI app through its lifespan."""
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as http:
            yield http


@pytest.fixture
async def seeded_container(container: Container) -> Container:
    """A container with the handbook already ingested."""
    await ingest_text(container, HANDBOOK_TEXT)
    return container


__all__ = ["HASH_DIMENSION"]

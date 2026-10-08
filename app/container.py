"""Application container: builds and owns every long-lived dependency.

Composition happens in exactly one place so that routers only ever ask for the
service they need, and so tests can construct a container with offline
providers (hash embeddings, mock model, in-memory Qdrant, fakeredis) without
touching the application code.

Lifecycle:

* :func:`build_container` — construct providers, create tables, verify the
  Qdrant collection.
* :meth:`Container.health` — per-dependency health report used by ``/health``.
* :meth:`Container.shutdown` — release the Redis pool and the DB engine.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.session import create_engine, create_session_factory, init_models, ping_database
from app.schemas.common import HealthComponent
from app.services.booking_service import BookingConversation, BookingExtractor, BookingService
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.embeddings.base import BaseEmbeddingProvider
from app.services.embeddings.factory import build_embedding_provider
from app.services.extraction.registry import ExtractorRegistry
from app.services.ingestion_service import IngestionService
from app.services.llm.factory import build_chat_model
from app.services.memory.redis_memory import ChatMemoryStore, RedisClient, build_redis_client
from app.services.rag.engine import ConversationalRAGEngine
from app.services.rag.retriever import ChunkRetriever
from app.services.vector_store.qdrant_store import QdrantVectorStoreAdapter, build_vector_store

logger = get_logger(__name__)


@dataclass
class Container:
    """Holds the wired object graph for the running application."""

    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    embedding: BaseEmbeddingProvider
    chat_model: BaseChatModel
    vector_store: QdrantVectorStoreAdapter
    retriever: ChunkRetriever
    rag_engine: ConversationalRAGEngine
    redis_client: RedisClient
    memory: ChatMemoryStore
    extractors: ExtractorRegistry
    ingestion_service: IngestionService
    document_service: DocumentService
    booking_service: BookingService
    booking_extractor: BookingExtractor
    booking_conversation: BookingConversation
    chat_service: ChatService

    async def health(self) -> list[HealthComponent]:
        """Probe every downstream dependency, reporting failures per component."""

        return [
            await self._check("database", lambda: ping_database(self.engine)),
            await self._check(
                "vector_store",
                self.vector_store.health,
                detail=f"collection={self.vector_store.collection_name}",
            ),
            await self._check("chat_memory", self.memory.health, detail="redis"),
            await self._check(
                "embedding_provider",
                self.embedding.health,
                detail=(
                    f"{self.embedding.name}:{self.embedding.model} (dim={self.embedding.dimension})"
                ),
            ),
            HealthComponent(
                name="chat_model",
                healthy=True,
                detail=f"{self.settings.llm_provider.value}:"
                f"{getattr(self.chat_model, 'model_name', None) or type(self.chat_model).__name__}",
            ),
        ]

    @staticmethod
    async def _check(
        name: str,
        probe: Callable[[], Awaitable[None]],
        *,
        detail: str | None = None,
    ) -> HealthComponent:
        """Run one probe, converting any failure into an unhealthy component."""
        try:
            await probe()
        except Exception as exc:
            logger.warning("Health probe '%s' failed: %s", name, exc)
            return HealthComponent(name=name, healthy=False, detail=str(exc))
        return HealthComponent(name=name, healthy=True, detail=detail)

    async def startup(self) -> None:
        """Prepare storage: create tables and verify the vector collection."""
        await init_models(self.engine)
        await self.vector_store.ensure_ready(dimension=self.embedding.dimension)
        logger.info(
            "Container ready (embeddings=%s:%s dim=%d, llm=%s, collection=%s)",
            self.embedding.name,
            self.embedding.model,
            self.embedding.dimension,
            self.settings.llm_provider.value,
            self.vector_store.collection_name,
        )

    async def shutdown(self) -> None:
        """Release pooled resources."""
        await self.memory.close()
        await self.engine.dispose()
        logger.info("Container shut down")


async def build_container(
    settings: Settings,
    *,
    redis_client: RedisClient | None = None,
) -> Container:
    """Construct the object graph for ``settings``.

    Args:
        settings: Validated application settings.
        redis_client: Optional pre-built Redis client (used by tests to inject
            ``fakeredis``); when omitted a real client is created from
            ``REDIS_URL``.
    """
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    embedding = build_embedding_provider(settings)
    chat_model = build_chat_model(settings)
    vector_store = build_vector_store(settings, embedding)
    retriever = ChunkRetriever(vector_store=vector_store, settings=settings)
    rag_engine = ConversationalRAGEngine(
        chat_model=chat_model,
        retriever=retriever,
        settings=settings,
    )

    client = redis_client or build_redis_client(settings)
    memory = ChatMemoryStore(client, settings)

    extractors = ExtractorRegistry()
    ingestion_service = IngestionService(
        session_factory=session_factory,
        extractors=extractors,
        embedding=embedding,
        vector_store=vector_store,
        settings=settings,
    )
    document_service = DocumentService(
        session_factory=session_factory,
        vector_store=vector_store,
        settings=settings,
    )
    booking_service = BookingService(session_factory=session_factory, settings=settings)
    booking_extractor = BookingExtractor(chat_model=chat_model, settings=settings)
    booking_conversation = BookingConversation(
        bookings=booking_service,
        drafts=memory.drafts,
        settings=settings,
    )
    chat_service = ChatService(
        memory=memory,
        rag=rag_engine,
        booking_extractor=booking_extractor,
        booking_conversation=booking_conversation,
        chat_model=chat_model,
        settings=settings,
    )

    return Container(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        embedding=embedding,
        chat_model=chat_model,
        vector_store=vector_store,
        retriever=retriever,
        rag_engine=rag_engine,
        redis_client=client,
        memory=memory,
        extractors=extractors,
        ingestion_service=ingestion_service,
        document_service=document_service,
        booking_service=booking_service,
        booking_extractor=booking_extractor,
        booking_conversation=booking_conversation,
        chat_service=chat_service,
    )


__all__ = ["Container", "build_container"]

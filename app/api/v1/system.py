"""Service health and corpus statistics endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from app.api.deps import BookingServiceDep, ContainerDep, DocumentServiceDep
from app.core.constants import API_VERSION
from app.core.enums import ChunkingStrategy
from app.schemas.common import HealthComponent, HealthResponse, StatsResponse

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Probes the database, Qdrant, Redis and the embedding provider. "
        "Returns HTTP 200 with `status=degraded` when any dependency is down, "
        "so orchestrators can distinguish liveness from readiness."
    ),
)
async def health(container: ContainerDep) -> HealthResponse:
    """Report the health of every downstream dependency."""
    components: list[HealthComponent] = await container.health()
    healthy = all(component.healthy for component in components)
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=API_VERSION,
        environment=container.settings.app_env.value,
        components=components,
        checked_at=datetime.now(UTC),
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Corpus and configuration summary",
)
async def stats(
    container: ContainerDep,
    documents: DocumentServiceDep,
    bookings: BookingServiceDep,
) -> StatsResponse:
    """Return counts plus the active provider configuration."""
    corpus = await documents.stats()
    return StatsResponse(
        documents=corpus.documents,
        chunks=corpus.chunks,
        vectors=corpus.vectors,
        bookings=await bookings.count(),
        embedding_provider=container.embedding.name,
        embedding_model=container.embedding.model,
        embedding_dimension=container.embedding.dimension,
        llm_provider=container.settings.llm_provider.value,
        chunking_strategies=[strategy.value for strategy in ChunkingStrategy],
        collection_name=container.vector_store.collection_name,
    )

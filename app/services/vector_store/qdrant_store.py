"""Qdrant-backed vector store, built on ``langchain_qdrant.QdrantVectorStore``.

LangChain owns the embedding + upsert + search plumbing; this adapter owns the
things the application cares about and LangChain leaves open:

* explicit collection lifecycle (create-if-missing, fail fast on a dimension
  mismatch rather than silently mixing vector sizes),
* deterministic point ids derived from chunk ids, so re-ingesting is idempotent,
* mapping raw hits back into typed :class:`RetrievedChunk` objects,
* document-scoped deletion through a payload filter.
"""

from __future__ import annotations

import warnings
from uuid import UUID

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from app.core.config import Settings
from app.core.errors import DependencyUnavailableError, ValidationError
from app.core.logging import get_logger
from app.services.embeddings.base import BaseEmbeddingProvider
from app.services.vector_store.base import RetrievedChunk, VectorStore

logger = get_logger(__name__)

#: Payload path prefix LangChain uses for document metadata.
METADATA_PREFIX = "metadata"

#: Local (in-process) Qdrant sentinels accepted in ``QDRANT_URL``.
IN_MEMORY_SENTINELS = frozenset({":memory:", "memory", "in-memory"})


def build_qdrant_client(settings: Settings) -> QdrantClient:
    """Create a Qdrant client from settings.

    ``QDRANT_URL`` supports three forms:

    * ``:memory:`` — in-process, ephemeral. Used by the test-suite.
    * ``local:/path`` — on-disk local mode, no server required.
    * ``http(s)://host:port`` — a real Qdrant service.
    """
    url = settings.qdrant_url.strip()

    if url.lower() in IN_MEMORY_SENTINELS:
        logger.debug("Using in-memory Qdrant (ephemeral)")
        return QdrantClient(location=":memory:")

    if url.lower().startswith("local:"):
        path = url.split(":", 1)[1] or "./qdrant_storage"
        logger.debug("Using on-disk local Qdrant at %s", path)
        return QdrantClient(path=path)

    return QdrantClient(
        url=url,
        api_key=settings.qdrant_api_key_value,
        prefer_grpc=settings.qdrant_prefer_grpc,
        timeout=30,
    )


class QdrantVectorStoreAdapter(VectorStore):
    """Concrete :class:`VectorStore` backed by Qdrant."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        embedding: BaseEmbeddingProvider,
        collection_name: str,
    ) -> None:
        self._client = client
        self._embedding = embedding
        self._collection = collection_name
        # validate_embeddings/config are disabled because this adapter owns both
        # concerns explicitly; enabling them would trigger a probe embedding call
        # on construction (an unnecessary paid API request).
        self._store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
            validate_embeddings=False,
            validate_collection_config=False,
        )
        #: Dimension the collection was last verified against, so repeated calls
        #: to :meth:`ensure_ready` do not issue a network round-trip every time.
        self._verified_dimension: int | None = None

    @property
    def collection_name(self) -> str:
        return self._collection

    # -- lifecycle ---------------------------------------------------------
    async def ensure_ready(self, *, dimension: int) -> None:
        if self._verified_dimension == dimension:
            return

        if not self._collection_exists():
            logger.info(
                "Creating Qdrant collection '%s' (dimension=%d, distance=cosine)",
                self._collection,
                dimension,
            )
            self._create_collection(dimension)
            self._verified_dimension = dimension
            return

        existing = self._vector_dimension()
        if existing is not None and existing != dimension:
            raise ValidationError(
                "The Qdrant collection was created with a different vector size.",
                details={
                    "collection": self._collection,
                    "existing_dimension": existing,
                    "configured_dimension": dimension,
                    "hint": (
                        "Change QDRANT_COLLECTION, or re-ingest after deleting the "
                        "collection, when switching embedding providers."
                    ),
                },
            )
        self._verified_dimension = dimension

    def _collection_exists(self) -> bool:
        try:
            return bool(self._client.collection_exists(self._collection))
        except Exception as exc:
            raise self._unavailable(exc, "checking collection existence") from exc

    def _create_collection(self, dimension: int) -> None:
        try:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            # Payload indexes keep document-scoped filtering fast.  They are a
            # no-op in local (in-memory / on-disk) mode, where the client emits
            # a UserWarning we deliberately silence.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for field in ("document_id", "chunk_id"):
                    self._client.create_payload_index(
                        collection_name=self._collection,
                        field_name=f"{METADATA_PREFIX}.{field}",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
        except Exception as exc:
            raise self._unavailable(exc, "creating the collection") from exc

    def _vector_dimension(self) -> int | None:
        try:
            info = self._client.get_collection(self._collection)
        except Exception as exc:
            raise self._unavailable(exc, "reading collection info") from exc

        vectors = info.config.params.vectors
        if isinstance(vectors, dict):  # named vectors
            first = next(iter(vectors.values()), None)
            return getattr(first, "size", None)
        return getattr(vectors, "size", None)

    # -- writes ------------------------------------------------------------
    async def add_documents(self, documents: list[Document], ids: list[str]) -> list[str]:
        if not documents:
            return []
        if len(documents) != len(ids):
            raise ValueError("documents and ids must have the same length")

        try:
            return await self._store.aadd_documents(documents, ids=ids)
        except Exception as exc:
            raise self._unavailable(exc, "upserting vectors") from exc

    async def delete_document(self, document_id: UUID | str) -> None:
        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key=f"{METADATA_PREFIX}.document_id",
                    match=models.MatchValue(value=str(document_id)),
                ),
            ],
        )
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(filter=flt),
            )
        except Exception as exc:
            raise self._unavailable(exc, "deleting a document") from exc

    async def delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        # Qdrant models point selectors as ``list[ExtendedPointId]``; the
        # annotated variable lets mypy accept the (invariant) ``list[str]``.
        selector: list[models.ExtendedPointId] = list(ids)
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=selector,
            )
        except Exception as exc:
            raise self._unavailable(exc, "deleting vectors") from exc

    # -- reads -------------------------------------------------------------
    async def similarity_search(
        self,
        query: str,
        *,
        top_k: int,
        score_threshold: float | None = None,
        document_id: UUID | str | None = None,
    ) -> list[RetrievedChunk]:
        flt: models.Filter | None = None
        if document_id is not None:
            flt = models.Filter(
                must=[
                    models.FieldCondition(
                        key=f"{METADATA_PREFIX}.document_id",
                        match=models.MatchValue(value=str(document_id)),
                    ),
                ],
            )

        try:
            hits: list[tuple[Document, float]] = await self._store.asimilarity_search_with_score(
                query,
                k=top_k,
                filter=flt,
            )
        except Exception as exc:
            raise self._unavailable(exc, "running a similarity search") from exc

        threshold = score_threshold or 0.0
        results: list[RetrievedChunk] = []
        for document, score in hits:
            if score < threshold:
                continue
            metadata = document.metadata or {}
            results.append(
                RetrievedChunk(
                    document_id=str(metadata.get("document_id", "")),
                    chunk_id=str(metadata.get("chunk_id", "")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    filename=str(metadata.get("filename", "")),
                    chunking_strategy=str(metadata.get("chunking_strategy", "")),
                    text=document.page_content,
                    score=float(score),
                ),
            )
        return results

    async def count(self) -> int:
        try:
            result = self._client.count(collection_name=self._collection, exact=True)
        except Exception as exc:
            raise self._unavailable(exc, "counting points") from exc
        return int(result.count)

    async def health(self) -> None:
        try:
            self._client.get_collections()
        except Exception as exc:
            raise self._unavailable(exc, "health check") from exc

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _unavailable(exc: Exception, operation: str) -> DependencyUnavailableError:
        logger.error("Qdrant operation failed while %s: %s", operation, exc)
        return DependencyUnavailableError(
            "The vector store is unavailable.",
            details={"operation": operation, "reason": str(exc)},
        )


def build_vector_store(
    settings: Settings,
    embedding: BaseEmbeddingProvider,
    *,
    client: QdrantClient | None = None,
) -> QdrantVectorStoreAdapter:
    """Create the Qdrant adapter, optionally reusing an existing client."""
    return QdrantVectorStoreAdapter(
        client=client or build_qdrant_client(settings),
        embedding=embedding,
        collection_name=settings.qdrant_collection,
    )


__all__ = ["QdrantVectorStoreAdapter", "build_qdrant_client", "build_vector_store"]

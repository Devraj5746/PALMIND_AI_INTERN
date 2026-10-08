"""Vector store abstraction and its Qdrant implementation."""

from __future__ import annotations

from app.services.vector_store.base import RetrievedChunk, VectorStore
from app.services.vector_store.qdrant_store import (
    QdrantVectorStoreAdapter,
    build_qdrant_client,
    build_vector_store,
)

__all__ = [
    "QdrantVectorStoreAdapter",
    "RetrievedChunk",
    "VectorStore",
    "build_qdrant_client",
    "build_vector_store",
]

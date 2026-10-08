"""Schemas for the document ingestion API."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from app.core.enums import ChunkingStrategy, DocumentStatus

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class IngestionOptions(BaseModel):
    """Tunable ingestion parameters, supplied as multipart form fields."""

    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.RECURSIVE,
        description="Chunking strategy to apply: ``fixed`` or ``recursive``.",
    )
    chunk_size: int = Field(
        default=1000,
        ge=64,
        le=8000,
        description="Target chunk size in characters.",
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=2000,
        description="Number of overlapping characters between adjacent chunks.",
    )

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> IngestionOptions:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class SearchRequest(BaseModel):
    """Semantic search over indexed chunks."""

    query: str = Field(min_length=1, max_length=4000, description="Natural-language query.")
    top_k: int = Field(default=4, ge=1, le=50, description="Number of chunks to return.")
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity. Defaults to the server-side setting.",
    )
    document_id: UUID | None = Field(
        default=None,
        description="Restrict the search to a single ingested document.",
    )


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------


class ChunkRead(BaseModel):
    """A single stored chunk of a document."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    chunk_index: int
    text: str
    char_count: int
    token_estimate: int
    start_char: int
    end_char: int
    chunking_strategy: ChunkingStrategy
    vector_id: str


class DocumentRead(BaseModel):
    """Summary of an ingested document (without its chunks)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    chunking_strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    char_count: int
    page_count: int | None = None
    embedding_provider: str
    embedding_model: str
    collection_name: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size_kb(self) -> float:
        """Upload size expressed in kibibytes, rounded to one decimal."""
        return round(self.size_bytes / 1024, 1)


class DocumentDetail(DocumentRead):
    """A document together with its stored chunks."""

    chunks: list[ChunkRead] = Field(default_factory=list)


class IngestionResponse(DocumentRead):
    """Result of a successful ingestion request."""

    duration_ms: float = Field(description="Wall-clock time spent ingesting.")


class SearchHit(BaseModel):
    """A single retrieved chunk with its relevance score."""

    document_id: UUID
    chunk_id: UUID
    chunk_index: int
    filename: str
    chunking_strategy: str
    score: float
    text: str


class SearchResponse(BaseModel):
    """Result set for a semantic search."""

    query: str
    top_k: int
    score_threshold: float
    results: list[SearchHit]


class DocumentListResponse(BaseModel):
    """Offset-paginated list of documents."""

    total: int
    limit: int
    offset: int
    items: list[DocumentRead]

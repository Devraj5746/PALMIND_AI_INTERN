"""Schemas shared across multiple routers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Machine-readable error payload."""

    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable description of the failure.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured context (e.g. per-field validation errors).",
    )


class ErrorResponse(BaseModel):
    """Uniform error envelope returned for every non-2xx response."""

    error: ErrorDetail


class HealthComponent(BaseModel):
    """Health of a single downstream dependency."""

    name: str
    healthy: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    """Aggregate health report."""

    status: str = Field(description="``ok`` when every dependency is healthy, else ``degraded``.")
    version: str
    environment: str
    components: list[HealthComponent]
    checked_at: datetime


class MessageResponse(BaseModel):
    """Generic acknowledgement payload."""

    message: str


class PaginatedResponse(BaseModel):
    """Envelope for simple offset/limit collections."""

    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class StatsResponse(BaseModel):
    """Corpus and configuration summary."""

    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    vectors: int = Field(ge=0)
    bookings: int = Field(ge=0)
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    llm_provider: str
    chunking_strategies: list[str]
    collection_name: str

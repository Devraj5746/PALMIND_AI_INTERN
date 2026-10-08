"""ORM models for ingested documents and their chunks."""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import DocumentStatus
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Document(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Metadata for a single uploaded document.

    The document's vector payloads live in Qdrant; this table is the
    authoritative record of what was ingested, how it was chunked and which
    embedding model was used.
    """

    __tablename__ = "documents"

    filename: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="SHA-256 of the raw upload, used for de-duplication and integrity checks.",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=DocumentStatus.PENDING.value,
        index=True,
    )

    # -- chunking configuration -------------------------------------------
    chunking_strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # -- extraction results ------------------------------------------------
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # -- embedding / vector store -----------------------------------------
    embedding_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    collection_name: Mapped[str] = mapped_column(String(128), nullable=False)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.chunk_index",
        lazy="selectin",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Document id={self.id} filename={self.filename!r} chunks={self.chunk_count}>"


class Chunk(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A single chunk produced by a chunking strategy."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="document_chunk_index"),
        Index("ix_chunks_document_index", "document_id", "chunk_index"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    token_estimate: Mapped[int] = mapped_column(Integer, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    chunking_strategy: Mapped[str] = mapped_column(String(32), nullable=False)

    vector_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        doc="Point id of the matching vector in Qdrant.",
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Chunk id={self.id} document_id={self.document_id} index={self.chunk_index}>"


__all__ = ["Chunk", "Document"]

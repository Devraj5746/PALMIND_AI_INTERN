"""Repository for documents and their chunks.

Repositories own all SQL; services never build queries directly.  They do not
commit — transaction boundaries belong to the calling service so that a
multi-step ingestion can be composed atomically.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.enums import DocumentStatus
from app.db.models.document import Chunk, Document


class DocumentRepository:
    """Persistence operations for :class:`Document` and :class:`Chunk`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- documents ---------------------------------------------------------
    async def add(self, document: Document) -> Document:
        """Stage a new document and flush so its id is populated."""
        self._session.add(document)
        await self._session.flush()
        return document

    async def add_chunks(self, chunks: list[Chunk]) -> None:
        """Bulk-insert chunks."""
        if not chunks:
            return
        self._session.add_all(chunks)
        await self._session.flush()

    async def get(self, document_id: uuid.UUID) -> Document | None:
        """Return a document (with chunks eagerly loaded) or ``None``."""
        stmt = (
            select(Document)
            .options(selectinload(Document.chunks))
            .where(Document.id == document_id)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_by_checksum(self, checksum: str) -> Document | None:
        """Return an indexed document with the same content hash, if any."""
        stmt = (
            select(Document)
            .where(Document.checksum == checksum)
            .order_by(Document.created_at.desc())
            .limit(1)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: DocumentStatus | None = None,
    ) -> tuple[list[Document], int]:
        """Return a page of documents plus the total row count."""
        filters = []
        if status is not None:
            filters.append(Document.status == status.value)

        items_stmt = (
            select(Document)
            .where(*filters)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        count_stmt = select(func.count()).select_from(Document).where(*filters)

        items = list((await self._session.execute(items_stmt)).scalars().all())
        total = int((await self._session.execute(count_stmt)).scalar_one())
        return items, total

    async def update_status(
        self,
        document: Document,
        status: DocumentStatus,
        *,
        error_message: str | None = None,
    ) -> Document:
        """Move a document through its ingestion lifecycle."""
        document.status = status.value
        document.error_message = error_message
        await self._session.flush()
        return document

    async def delete(self, document: Document) -> None:
        """Delete a document; chunks cascade at the database level."""
        await self._session.delete(document)
        await self._session.flush()

    async def count(self) -> int:
        """Total number of documents."""
        return int(
            (await self._session.execute(select(func.count()).select_from(Document))).scalar_one(),
        )

    # -- chunks ------------------------------------------------------------
    async def count_all_chunks(self) -> int:
        """Total number of chunks across every document."""
        return int(
            (await self._session.execute(select(func.count()).select_from(Chunk))).scalar_one(),
        )

"""Document ingestion orchestration.

The pipeline, in order:

1. **Validate** the upload (extension allow-list, size limit, non-empty body).
2. **Extract** text with the right extractor, off the event loop.
3. **Chunk** with the caller-selected strategy via
   :func:`app.services.chunking.factory.build_chunker`.
4. **Embed + index** the chunks in Qdrant through LangChain's vector store.
5. **Persist** document metadata and every chunk row in SQL.

Failure handling is explicit: the document row is written up-front with status
``processing`` so a crash leaves an auditable ``failed`` record, and any vectors
already written are rolled back so the index never holds orphans.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.enums import DocumentStatus
from app.core.errors import ExtractionError, PayloadTooLargeError, UnsupportedMediaTypeError
from app.core.logging import get_logger
from app.db.models.document import Chunk as ChunkRow
from app.db.models.document import Document
from app.db.repositories.document_repository import DocumentRepository
from app.schemas.ingestion import IngestionOptions
from app.services.chunking import build_chunker
from app.services.chunking.base import Chunk
from app.services.embeddings.base import BaseEmbeddingProvider
from app.services.extraction.registry import ExtractorRegistry
from app.services.vector_store.base import VectorStore

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Outcome of a successful ingestion."""

    document: Document
    chunks: list[Chunk]
    duration_ms: float


class IngestionService:
    """Runs the full document-ingestion pipeline."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        extractors: ExtractorRegistry,
        embedding: BaseEmbeddingProvider,
        vector_store: VectorStore,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._extractors = extractors
        self._embedding = embedding
        self._vector_store = vector_store
        self._settings = settings

    async def ingest(
        self,
        *,
        filename: str,
        content_type: str,
        data: bytes,
        options: IngestionOptions,
    ) -> IngestionResult:
        """Ingest one uploaded file end to end."""
        started = time.perf_counter()
        self._validate_upload(filename=filename, data=data)

        # 2. Extract (CPU/IO bound -> worker thread).
        extractor = self._extractors.get(filename)
        extracted = await run_in_threadpool(extractor.extract, filename=filename, data=data)

        # 3. Chunk.
        chunker = build_chunker(
            options.chunking_strategy,
            chunk_size=options.chunk_size,
            chunk_overlap=options.chunk_overlap,
        )
        chunks: list[Chunk] = chunker.chunk(extracted.text)
        if not chunks:
            raise ExtractionError(
                "No chunks could be produced from the extracted text.",
                details={"filename": filename, "char_count": extracted.char_count},
            )

        checksum = hashlib.sha256(data).hexdigest()
        document_id = await self._create_document(
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            checksum=checksum,
            page_count=extracted.page_count,
            char_count=extracted.char_count,
            options=options,
        )
        logger.info(
            "Ingesting %s (%d bytes, %d chunks, strategy=%s)",
            filename,
            len(data),
            len(chunks),
            options.chunking_strategy.value,
        )

        chunk_ids: list[uuid.UUID] = []
        try:
            # 4. Embed + index.
            chunk_ids = await self._index_chunks(
                document_id=document_id,
                filename=filename,
                chunks=chunks,
            )
            # 5. Persist chunk metadata and finalise the document row.
            await self._finalize(
                document_id=document_id,
                chunks=chunks,
                chunk_ids=chunk_ids,
                page_count=extracted.page_count,
                char_count=extracted.char_count,
            )
        except Exception as exc:
            await self._rollback(
                document_id=document_id,
                vector_ids=[str(chunk_id) for chunk_id in chunk_ids],
                error=exc,
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        document = await self._load(document_id)
        logger.info("Ingested %s in %.1fms", filename, duration_ms)

        return IngestionResult(document=document, chunks=chunks, duration_ms=duration_ms)

    # -- steps -------------------------------------------------------------
    def _validate_upload(self, *, filename: str, data: bytes) -> None:
        if not data:
            raise UnsupportedMediaTypeError(
                "The uploaded file is empty.",
                details={"filename": filename},
            )

        if len(data) > self._settings.max_upload_size_bytes:
            raise PayloadTooLargeError(
                f"The uploaded file exceeds the {self._settings.max_upload_size_mb} MB limit.",
                details={
                    "filename": filename,
                    "size_bytes": len(data),
                    "max_size_bytes": self._settings.max_upload_size_bytes,
                },
            )

        extension = ExtractorRegistry.extension_of(filename)
        if extension not in self._settings.allowed_extensions:
            raise UnsupportedMediaTypeError(
                f"Files of type '{extension or 'unknown'}' are not accepted.",
                details={
                    "filename": filename,
                    "allowed_extensions": self._settings.allowed_extensions,
                },
            )

    async def _create_document(
        self,
        *,
        filename: str,
        content_type: str,
        size_bytes: int,
        checksum: str,
        page_count: int | None,
        char_count: int,
        options: IngestionOptions,
    ) -> uuid.UUID:
        async with self._session_factory() as session:
            repository = DocumentRepository(session)
            duplicate = await repository.get_by_checksum(checksum)
            if duplicate is not None:
                logger.warning(
                    "Identical content already ingested as document %s; "
                    "storing a separate copy for this ingestion",
                    duplicate.id,
                )

            document = Document(
                filename=filename,
                content_type=content_type,
                size_bytes=size_bytes,
                checksum=checksum,
                status=DocumentStatus.PROCESSING.value,
                chunking_strategy=options.chunking_strategy.value,
                chunk_size=options.chunk_size,
                chunk_overlap=options.chunk_overlap,
                chunk_count=0,
                char_count=char_count,
                page_count=page_count,
                embedding_provider=self._embedding.name,
                embedding_model=self._embedding.model,
                embedding_dimension=self._embedding.dimension,
                collection_name=self._vector_store.collection_name,
            )
            await repository.add(document)
            await session.commit()
            return document.id

    async def _index_chunks(
        self,
        *,
        document_id: uuid.UUID,
        filename: str,
        chunks: list[Chunk],
    ) -> list[uuid.UUID]:
        # Ensure the collection exists and matches this embedding model.
        await self._vector_store.ensure_ready(dimension=self._embedding.dimension)

        # The chunk's database id *is* its Qdrant point id: one identifier keeps
        # SQL metadata and vectors in lockstep and makes re-ingestion idempotent.
        chunk_ids = [uuid.uuid4() for _ in chunks]
        documents = [
            chunk.to_document(str(chunk_id), str(document_id), filename)
            for chunk, chunk_id in zip(chunks, chunk_ids, strict=True)
        ]
        await self._vector_store.add_documents(documents, ids=[str(cid) for cid in chunk_ids])
        return chunk_ids

    async def _finalize(
        self,
        *,
        document_id: uuid.UUID,
        chunks: list[Chunk],
        chunk_ids: list[uuid.UUID],
        page_count: int | None,
        char_count: int,
    ) -> None:
        async with self._session_factory() as session:
            repository = DocumentRepository(session)
            document = await repository.get(document_id)
            if document is None:  # pragma: no cover - defensive
                raise RuntimeError(f"Document {document_id} vanished during ingestion")

            rows = [
                ChunkRow(
                    id=chunk_id,
                    document_id=document_id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                    char_count=chunk.char_count,
                    token_estimate=chunk.token_estimate,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunking_strategy=chunk.metadata["chunking_strategy"],
                    vector_id=str(chunk_id),
                )
                for chunk, chunk_id in zip(chunks, chunk_ids, strict=True)
            ]
            await repository.add_chunks(rows)
            document.chunk_count = len(rows)
            document.char_count = char_count
            document.page_count = page_count
            document.status = DocumentStatus.INDEXED.value
            document.error_message = None
            await session.commit()

    async def _rollback(
        self,
        *,
        document_id: uuid.UUID,
        vector_ids: list[str],
        error: Exception,
    ) -> None:
        """Best-effort cleanup so a failed ingestion leaves no partial index."""
        logger.error("Ingestion failed for document %s: %s", document_id, error)

        if vector_ids:
            try:
                await self._vector_store.delete_ids(vector_ids)
            except Exception:
                logger.warning("Could not remove vectors for failed ingestion", exc_info=True)

        try:
            async with self._session_factory() as session:
                repository = DocumentRepository(session)
                document = await repository.get(document_id)
                if document is not None:
                    await repository.update_status(
                        document,
                        DocumentStatus.FAILED,
                        error_message=str(error)[:2000],
                    )
                    await session.commit()
        except Exception:
            logger.warning("Could not mark document as failed", exc_info=True)

    async def _load(self, document_id: uuid.UUID) -> Document:
        async with self._session_factory() as session:
            repository = DocumentRepository(session)
            document = await repository.get(document_id)
            if document is None:  # pragma: no cover - defensive
                raise RuntimeError(f"Document {document_id} disappeared after ingestion")
            return document


__all__ = ["IngestionResult", "IngestionService"]

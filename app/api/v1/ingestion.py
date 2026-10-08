"""Document ingestion and retrieval endpoints."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status
from pydantic import ValidationError as PydanticValidationError

from app.api.deps import DocumentServiceDep, IngestionServiceDep, SettingsDep
from app.core.enums import ChunkingStrategy, DocumentStatus
from app.core.errors import PayloadTooLargeError, ValidationError
from app.schemas.ingestion import (
    DocumentDetail,
    DocumentListResponse,
    DocumentRead,
    IngestionOptions,
    IngestionResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    description=(
        "Extracts text from a PDF or TXT upload, splits it with the selected "
        "chunking strategy, embeds the chunks into Qdrant and stores the "
        "document metadata in SQL. Chunking parameters fall back to the server "
        "defaults when omitted."
    ),
)
async def ingest_document(
    service: IngestionServiceDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="A .pdf or .txt file.")],
    chunking_strategy: Annotated[
        ChunkingStrategy | None,
        Form(description="`recursive` (structure aware) or `fixed` (uniform windows)."),
    ] = None,
    chunk_size: Annotated[
        int | None, Form(ge=64, le=8000, description="Characters per chunk.")
    ] = None,
    chunk_overlap: Annotated[
        int | None,
        Form(ge=0, le=2000, description="Characters shared with the next chunk."),
    ] = None,
) -> IngestionResponse:
    """Ingest one uploaded file."""
    content_type = file.content_type or "application/octet-stream"

    # Reject oversized uploads before buffering them into memory, when the
    # multipart parser already knows the size.
    if file.size is not None and file.size > settings.max_upload_size_bytes:
        raise PayloadTooLargeError(
            f"The uploaded file exceeds the {settings.max_upload_size_mb} MB limit.",
            details={
                "filename": file.filename,
                "size_bytes": file.size,
                "max_size_bytes": settings.max_upload_size_bytes,
            },
        )

    data = await file.read()
    try:
        options = IngestionOptions(
            chunking_strategy=chunking_strategy or settings.default_chunking_strategy,
            chunk_size=chunk_size or settings.default_chunk_size,
            chunk_overlap=(
                settings.default_chunk_overlap if chunk_overlap is None else chunk_overlap
            ),
        )
    except PydanticValidationError as exc:
        # Cross-field rules (e.g. overlap < size) cannot be expressed as plain
        # Form constraints, so surface them through the standard error envelope.
        # ``exc.errors()`` embeds the raised ValueError object in its ``ctx``,
        # which is not JSON serialisable — project only the useful fields.
        issues = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())) or "body",
                "message": error.get("msg", "invalid value"),
                "type": error.get("type", "value_error"),
            }
            for error in exc.errors()
        ]
        raise ValidationError("Invalid ingestion parameters.", details={"issues": issues}) from exc

    result = await service.ingest(
        filename=file.filename or "upload",
        content_type=content_type,
        data=data,
        options=options,
    )

    payload = DocumentRead.model_validate(result.document)
    return IngestionResponse.model_validate(
        {**payload.model_dump(), "duration_ms": round(result.duration_ms, 2)},
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List ingested documents",
)
async def list_documents(
    service: DocumentServiceDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    document_status: Annotated[
        DocumentStatus | None,
        Query(alias="status", description="Filter by ingestion status."),
    ] = None,
) -> DocumentListResponse:
    """Return a page of ingested documents, newest first."""
    documents, total = await service.list_documents(
        limit=limit,
        offset=offset,
        status=document_status,
    )
    return DocumentListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[DocumentRead.model_validate(document) for document in documents],
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search over indexed chunks",
)
async def search_documents(service: DocumentServiceDep, payload: SearchRequest) -> SearchResponse:
    """Embed the query and return the nearest chunks."""
    hits = await service.search(
        query=payload.query,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
        document_id=payload.document_id,
    )
    return SearchResponse(
        query=payload.query,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold or 0.0,
        results=[
            SearchHit(
                document_id=uuid.UUID(hit.document_id),
                chunk_id=uuid.UUID(hit.chunk_id),
                chunk_index=hit.chunk_index,
                filename=hit.filename,
                chunking_strategy=hit.chunking_strategy,
                score=round(hit.score, 6),
                text=hit.text,
            )
            for hit in hits
        ],
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
    summary="Get a document with its chunks",
)
async def get_document(service: DocumentServiceDep, document_id: uuid.UUID) -> DocumentDetail:
    """Return a document and every chunk that was stored for it."""
    document = await service.get(document_id)
    return DocumentDetail.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description="Removes the vectors from Qdrant and the metadata (plus chunks) from SQL.",
)
async def delete_document(
    service: DocumentServiceDep,
    document_id: uuid.UUID,
) -> Response:
    """Delete a document and everything derived from it."""
    await service.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

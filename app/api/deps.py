"""FastAPI dependencies.

Every dependency resolves through the single :class:`~app.container.Container`
stored on ``app.state`` by the lifespan handler.  Routers therefore declare
*what* they need (a service) and never *how* it is constructed.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.container import Container
from app.core.config import Settings
from app.services.booking_service import BookingService
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService
from app.services.vector_store.base import VectorStore


def get_container(request: Request) -> Container:
    """Return the application container (fails loudly if startup did not run)."""
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - only reachable on a broken startup
        raise RuntimeError(
            "Application container is not initialised; the lifespan handler did not run.",
        )
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_settings(container: ContainerDep) -> Settings:
    """Return the active settings."""
    return container.settings


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_ingestion_service(container: ContainerDep) -> IngestionService:
    return container.ingestion_service


def get_document_service(container: ContainerDep) -> DocumentService:
    return container.document_service


def get_chat_service(container: ContainerDep) -> ChatService:
    return container.chat_service


def get_booking_service(container: ContainerDep) -> BookingService:
    return container.booking_service


def get_vector_store(container: ContainerDep) -> VectorStore:
    return container.vector_store


IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
BookingServiceDep = Annotated[BookingService, Depends(get_booking_service)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]

__all__ = [
    "BookingServiceDep",
    "ChatServiceDep",
    "ContainerDep",
    "DocumentServiceDep",
    "IngestionServiceDep",
    "SettingsDep",
    "VectorStoreDep",
    "get_container",
    "get_settings",
]

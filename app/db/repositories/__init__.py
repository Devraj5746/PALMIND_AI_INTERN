"""Data-access layer: one repository per aggregate."""

from __future__ import annotations

from app.db.repositories.booking_repository import BookingRepository
from app.db.repositories.document_repository import DocumentRepository

__all__ = ["BookingRepository", "DocumentRepository"]

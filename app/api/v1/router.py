"""Aggregate router for API version 1."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import booking, chat, ingestion, system

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(ingestion.router)
api_router.include_router(chat.router)
api_router.include_router(booking.router)

__all__ = ["api_router"]

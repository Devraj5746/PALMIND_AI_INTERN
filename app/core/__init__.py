"""Cross-cutting concerns: configuration, logging, errors, middleware."""

from __future__ import annotations

from app.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]

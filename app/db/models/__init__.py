"""ORM model registry.

Importing this package registers every table on ``Base.metadata``; the
application and ``create_all``/Alembic rely on that side effect.
"""

from __future__ import annotations

from app.db.models.booking import Booking
from app.db.models.document import Chunk, Document

__all__ = ["Booking", "Chunk", "Document"]

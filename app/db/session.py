"""Async SQLAlchemy engine and session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.base import Base

logger = get_logger(__name__)


def create_engine(settings: Settings) -> AsyncEngine:
    """Build an :class:`AsyncEngine` tuned for the configured database URL."""
    url = settings.database_url
    kwargs: dict[str, Any] = {"echo": settings.db_echo, "pool_pre_ping": True}

    if url.startswith("sqlite"):
        # aiosqlite requires this for cross-thread usage under the event loop.
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            # Each new connection would otherwise get its own empty database.
            kwargs["poolclass"] = StaticPool

    logger.debug("Creating async engine for %s", url.split("@")[-1])
    return create_async_engine(url, **kwargs)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a session factory bound to ``engine``.

    ``expire_on_commit=False`` keeps attributes usable after commit, which lets
    response models be serialised from ORM instances without an extra query.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def init_models(engine: AsyncEngine) -> None:
    """Create all tables that do not already exist.

    Suitable for local development and tests.  Production deployments are
    expected to run versioned migrations instead (see README).
    """
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("Database schema is ready (%d tables)", len(Base.metadata.tables))


async def drop_models(engine: AsyncEngine) -> None:
    """Drop all tables. Used by tests and local resets."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)


async def ping_database(engine: AsyncEngine) -> None:
    """Raise if the database is unreachable."""
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


@asynccontextmanager
async def session_scope(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Provide a transactional scope around a series of operations."""
    session = session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

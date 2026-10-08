"""FastAPI application factory.

The application is built by :func:`create_app`, which keeps configuration,
middleware, routers and the lifespan handler in one explicit place.  Tests call
it with a fully offline :class:`~app.core.config.Settings`; production calls it
with no arguments so settings come from the environment.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.container import Container, build_container
from app.core.config import Settings, get_settings
from app.core.constants import API_DESCRIPTION, API_TITLE, API_VERSION
from app.core.error_handlers import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import PROCESS_TIME_HEADER, REQUEST_ID_HEADER, RequestContextMiddleware

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, *, redis_client: Any = None) -> FastAPI:
    """Build the ASGI application.

    Args:
        settings: Explicit settings (tests, embedded use). Defaults to the
            environment-derived singleton.
        redis_client: Optional pre-built Redis client, injected by tests so the
            suite never needs a live Redis server.
    """
    resolved = settings or get_settings()
    configure_logging(resolved.log_level, json_logs=resolved.is_production)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container: Container = await build_container(resolved, redis_client=redis_client)
        await container.startup()
        app.state.container = container
        logger.info("%s started (env=%s)", API_TITLE, resolved.app_env.value)
        try:
            yield
        finally:
            await container.shutdown()
            app.state.container = None
            logger.info("%s stopped", API_TITLE)

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = resolved

    # Added first, so CORS ends up as the outermost middleware and still applies
    # its headers to responses produced by the error handlers.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER, PROCESS_TIME_HEADER],
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=resolved.api_v1_prefix)

    @app.get("/", tags=["system"], summary="Service metadata")
    async def root() -> dict[str, Any]:
        """Return basic service information and useful links."""
        return {
            "name": API_TITLE,
            "version": API_VERSION,
            "environment": resolved.app_env.value,
            "api_prefix": resolved.api_v1_prefix,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": f"{resolved.api_v1_prefix}/health",
        }

    return app


app = create_app()

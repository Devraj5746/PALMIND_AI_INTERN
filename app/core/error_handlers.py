"""Centralised FastAPI exception handlers producing a uniform error envelope."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, object] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
            },
        },
    )


async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    """Handle every :class:`AppError` subclass."""
    log = logger.warning if exc.status_code < 500 else logger.error
    log("Handled application error: %s (%s)", exc.message, exc.error_code)
    return _error_response(exc.status_code, exc.error_code, exc.message, exc.details)


async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Translate Pydantic/FastAPI validation errors into the standard envelope."""
    field_errors = [
        {
            "location": ".".join(str(part) for part in error.get("loc", ())),
            "message": error.get("msg", "invalid value"),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return _error_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "validation_error",
        "The request payload failed validation.",
        {"fields": field_errors},
    )


async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle framework-raised ``HTTPException`` (404 routes, 405, ...)."""
    return _error_response(
        exc.status_code,
        f"http_{exc.status_code}",
        str(exc.detail),
    )


async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler so clients always receive the standard envelope."""
    logger.exception("Unhandled exception: %s", exc)
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "An unexpected error occurred.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to ``app``."""
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)

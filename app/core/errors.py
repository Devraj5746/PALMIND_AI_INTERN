"""Domain exception hierarchy.

Every failure the API can produce is an :class:`AppError`, which carries an
HTTP status code and a stable machine-readable ``error_code``.  A single set of
FastAPI exception handlers (:mod:`app.core.error_handlers`) converts them into a
consistent JSON envelope, so clients never have to parse ad-hoc payloads.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all expected, handled application errors."""

    status_code: int = 500
    error_code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.details: dict[str, Any] = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
            },
        }


class ValidationError(AppError):
    """Request payload, query parameters or domain rules are invalid."""

    status_code = 422
    error_code = "validation_error"
    message = "The request could not be validated."


class NotFoundError(AppError):
    """A requested resource does not exist."""

    status_code = 404
    error_code = "not_found"
    message = "The requested resource was not found."


class ConflictError(AppError):
    """The request conflicts with the current state of the system."""

    status_code = 409
    error_code = "conflict"
    message = "The request conflicts with an existing resource."


class UnsupportedMediaTypeError(AppError):
    """The uploaded file type is not supported."""

    status_code = 415
    error_code = "unsupported_media_type"
    message = "The uploaded file type is not supported."


class PayloadTooLargeError(AppError):
    """The uploaded file exceeds the configured size limit."""

    status_code = 413
    error_code = "payload_too_large"
    message = "The uploaded file exceeds the maximum allowed size."


class DependencyUnavailableError(AppError):
    """A downstream dependency (Qdrant, Redis, database, LLM) is unreachable."""

    status_code = 503
    error_code = "dependency_unavailable"
    message = "A required downstream dependency is unavailable."


class UpstreamProviderError(AppError):
    """An external provider (OpenAI, embedding API, ...) returned an error."""

    status_code = 502
    error_code = "upstream_provider_error"
    message = "An upstream provider failed to fulfil the request."


class ExtractionError(AppError):
    """Text could not be extracted from the uploaded document."""

    status_code = 422
    error_code = "extraction_failed"
    message = "Text could not be extracted from the document."


class ConfigurationError(AppError):
    """The application was misconfigured."""

    status_code = 500
    error_code = "configuration_error"
    message = "The application is misconfigured."

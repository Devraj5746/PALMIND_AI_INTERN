"""ASGI middleware: request correlation ids and access logging."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.logging import get_logger, set_request_id

logger = get_logger("app.access")

REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id to every request and log its outcome.

    The id is echoed back through the ``X-Request-ID`` header and bound to the
    logging context so downstream records can be correlated.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        set_request_id(request_id)
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "%s %s -> unhandled error in %.1fms",
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise
        finally:
            set_request_id(None)

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{elapsed_ms:.1f}"

        logger.info(
            "%s %s -> %s in %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

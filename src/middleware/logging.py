"""Request Logging Middleware.

Logs every non-probe request with method, path, request_id, status, and
processing time. Consumes ``request.state.request_id`` set upstream by
``RequestIdMiddleware``; mints a fresh uuid7 only if that middleware wasn't
wired in.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger: logging.Logger = logging.getLogger(__name__)


_SKIP_LOG_PATHS = frozenset({"/health", "/healthz", "/ready", "/readyz"})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and adding request tracking headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in _SKIP_LOG_PATHS:
            return await call_next(request)

        request_id: str = getattr(request.state, "request_id", "") or str(uuid.uuid7())
        request.state.request_id = request_id

        start_time: float = time.time()

        logger.info(f"Request started: {request.method} {request.url.path} [{request_id}]")

        response: Response = await call_next(request)

        process_time: float = (time.time() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{process_time:.2f}ms"

        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"[{request_id}] - Status: {response.status_code} - "
            f"Time: {process_time:.2f}ms"
        )

        return response

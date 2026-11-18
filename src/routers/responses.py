"""Common response definitions for OpenAPI documentation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Inner error payload."""

    code: str = Field(..., description="Stable machine-readable error code")
    message: str = Field(..., description="Human-readable error description")
    status: int = Field(..., description="HTTP status code")
    request_id: str | None = Field(default=None, description="Request UUID7 for log correlation")
    timestamp: str | None = Field(default=None, description="ISO-8601 UTC timestamp the error was emitted")
    details: str | None = Field(
        default=None,
        description="Optional debug detail; only populated when LEAK_EXCEPTION_DETAILS=true",
    )


class ErrorResponse(BaseModel):
    """Standard error response envelope."""

    error: ErrorDetail


AUTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"description": "Authentication required", "model": ErrorResponse},
}

NOT_FOUND_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"description": "Resource not found", "model": ErrorResponse},
}

VALIDATION_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": "Bad request — invalid input", "model": ErrorResponse},
    422: {"description": "Request validation failed", "model": ErrorResponse},
}

BACKEND_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    500: {"description": "Internal server error", "model": ErrorResponse},
    503: {"description": "Backend unavailable", "model": ErrorResponse},
}

VLM_ENDPOINT_RESPONSES: dict[int | str, dict[str, Any]] = {
    **AUTH_RESPONSES,
    **VALIDATION_RESPONSES,
    **BACKEND_ERROR_RESPONSES,
}

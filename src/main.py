"""VLM Backend — FastAPI entrypoint."""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.exceptions import HTTPException as StarletteHTTPException

from auth import AuthMiddleware
from auth.dependencies import get_current_user
from auth.middleware import PROTECTED_PREFIXES, PUBLIC_ENDPOINTS
from auth.models import User
from middleware.logging import RequestLoggingMiddleware
from middleware.request_id import RequestIdMiddleware
from middleware.security_headers import SecurityHeadersMiddleware
from models import HealthResponse
from routers import vlm_routes
from services.vlm_service import get_vlm_service

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SERVICE_NAME = os.getenv("SERVICE_NAME", "vlm-backend")
VERSION = os.getenv("VERSION", "0.1.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "0195f48b-d000-7000-8000-000000000001")
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "false").lower() == "true"
docs_enabled = ENABLE_DOCS or ENVIRONMENT.lower() in {"development", "dev", "test"}
LEAK_EXCEPTION_DETAILS = os.getenv("LEAK_EXCEPTION_DETAILS", "").lower() == "true" or ENVIRONMENT.lower() in {
    "development",
    "dev",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    logger.info("Starting %s v%s (%s)", SERVICE_NAME, VERSION, ENVIRONMENT)
    svc = get_vlm_service()
    try:
        await svc.initialize()
        app.state.vlm_service = svc
        logger.info("VLM model loaded: %s", svc.model_repo)
    except Exception:
        logger.exception("VLM init failed — service starts in degraded mode")
        app.state.vlm_service = svc
    logger.info("%s ready", SERVICE_NAME)
    yield
    logger.info("Shutting down %s", SERVICE_NAME)


app = FastAPI(
    title=SERVICE_NAME,
    description=(
        "Vision-language model inference for Apple Silicon (mlx-vlm). "
        "OpenAI-vision-compatible: `POST /api/vlm/chat/completions` accepts the same "
        "messages-with-image_url payload. Authentication is via license token in "
        "`X-Tenant-Id: Bearer <token>` with RFC1918/loopback bypass for in-cluster traffic."
    ),
    version=VERSION,
    lifespan=lifespan,
    redirect_slashes=False,
    docs_url="/docs" if docs_enabled else None,
    redoc_url="/redoc" if docs_enabled else None,
    openapi_url="/openapi.json" if docs_enabled else None,
    swagger_ui_parameters={"persistAuthorization": True},
    generate_unique_id_function=lambda route: f"{route.tags[0] if route.tags else 'default'}_{route.name}",
    openapi_tags=[
        {"name": "Health", "description": "Liveness, readiness, dependency probes."},
        {"name": "VLM", "description": "Vision-language model inference operations."},
        {"name": "Identity", "description": "Current authenticated identity context."},
    ],
)


# --- CORS ---
_cors_raw = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]
if ENVIRONMENT == "production" and not CORS_ORIGINS:
    raise RuntimeError(
        "CORS_ORIGINS env var is required in production. Set it on the launchd service env (see gitops-production)."
    )

# Middleware registration — Starlette executes in reverse-registration order.
# Target execution order: RequestId → Auth → SecurityHeaders → RequestLogging → CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Processing-Time"],
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

if ENABLE_AUTH:
    app.add_middleware(AuthMiddleware)
    logger.info("Authentication middleware enabled")
else:
    logger.warning("Authentication middleware DISABLED (ENABLE_AUTH=false)")

app.add_middleware(RequestIdMiddleware)


# Prometheus /metrics endpoint (public, scraped by cluster prometheus).
Instrumentator(excluded_handlers=["/metrics", "/health.*"]).instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False, tags=["monitoring"]
)


# --- Error envelope helpers ---
def _error_envelope(
    code: str,
    message: str,
    http_status: int,
    *,
    request_id: str,
    details: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": code,
        "message": message,
        "status": http_status,
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if details is not None:
        payload["details"] = details
    return {"error": payload}


def _http_exception_to_response(exc: StarletteHTTPException, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_envelope(f"HTTP_{exc.status_code}", str(exc.detail), exc.status_code, request_id=request_id),
        headers=exc.headers,
    )


def _find_http_exception(exc: BaseException) -> HTTPException | None:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            found = _find_http_exception(sub)
            if found is not None:
                return found
    return None


def _request_id_from(request: Request) -> str:
    return request.state.request_id if hasattr(request.state, "request_id") else str(uuid.uuid7())


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _http_exception_to_response(exc, _request_id_from(request))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = _request_id_from(request)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_error_envelope(
            "VALIDATION_ERROR",
            "Request validation failed",
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            request_id=request_id,
            details=str(exc.errors()) if LEAK_EXCEPTION_DETAILS else None,
        ),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id: str = _request_id_from(request)

    http_exc = _find_http_exception(exc)
    if http_exc is not None:
        return _http_exception_to_response(http_exc, request_id)

    logger.exception("unhandled_error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_envelope(
            "INTERNAL_SERVER_ERROR",
            "An unexpected error occurred",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=request_id,
            details=str(exc) if LEAK_EXCEPTION_DETAILS else None,
        ),
    )


# --- OpenAPI schema with per-path security ---
def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )

    components = openapi_schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes.setdefault(
        "BearerAuth",
        {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "License",
            "description": (
                "License token. Transport: `X-Tenant-Id: Bearer <token>` "
                "(not the Authorization header — that is reserved for caller-side auth)."
            ),
        },
    )

    parameters = components.setdefault("parameters", {})
    parameters.setdefault(
        "TenantIdHeader",
        {
            "name": "X-Tenant-Id",
            "in": "header",
            "required": False,
            "description": "Tenant identifier (defaults to configured DEFAULT_TENANT_ID).",
            "schema": {"type": "string", "default": DEFAULT_TENANT_ID},
            "example": DEFAULT_TENANT_ID,
        },
    )

    for path, path_item in openapi_schema.get("paths", {}).items():
        normalized_path = path.rstrip("/") or "/"
        requires_auth = (
            any(normalized_path == prefix or normalized_path.startswith(prefix + "/") for prefix in PROTECTED_PREFIXES)
            and normalized_path not in PUBLIC_ENDPOINTS
        )

        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue

            if requires_auth:
                operation.setdefault("security", [{"BearerAuth": []}])
            else:
                operation.pop("security", None)

            op_params = operation.setdefault("parameters", [])
            has_tenant = any(p.get("name", "").lower() == "x-tenant-id" and p.get("in") == "header" for p in op_params)
            if not has_tenant:
                op_params.append({"$ref": "#/components/parameters/TenantIdHeader"})

    app.openapi_schema = openapi_schema
    return app.openapi_schema


object.__setattr__(app, "openapi", custom_openapi)


# --- Routers ---
app.include_router(vlm_routes.router)


# --- Service banner + health endpoints ---
@app.get(
    "/",
    operation_id="service_root",
    tags=["Health"],
    summary="Service banner",
)
async def root() -> dict[str, str]:
    """Root endpoint (public). Returns basic service information."""
    return {"service": SERVICE_NAME, "version": VERSION, "docs": "/docs", "health": "/health"}


def _models_loaded() -> list[str]:
    svc = getattr(app.state, "vlm_service", None)
    if svc is None or not svc.is_loaded:
        return []
    return [f"vlm/{svc.model_repo}"]


@app.get(
    "/health",
    response_model=HealthResponse,
    operation_id="health_check",
    tags=["Health"],
    summary="Liveness + identity probe",
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=VERSION,
        device="unified",
        models_loaded=_models_loaded(),
    )


@app.get(
    "/health/ready",
    response_model=HealthResponse,
    operation_id="health_ready",
    tags=["Health"],
    summary="Readiness probe — reports degraded if model not loaded",
)
async def ready() -> HealthResponse:
    loaded = _models_loaded()
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        service=SERVICE_NAME,
        version=VERSION,
        device="unified",
        models_loaded=loaded,
    )


@app.get(
    "/health/live",
    response_model=HealthResponse,
    operation_id="health_live",
    tags=["Health"],
    summary="Liveness probe",
)
async def live() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=VERSION,
        device="unified",
        models_loaded=[],
    )


@app.get(
    "/me",
    operation_id="get_current_identity",
    tags=["Identity"],
    summary="Current authenticated identity",
)
async def me(user: User = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "user_id": user.sub,
        "username": user.preferred_username,
        "tenant_id": user.effective_tenant_id,
        "roles": user.get_all_roles(),
    }

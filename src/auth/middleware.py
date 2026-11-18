"""License-token Authentication Middleware.

Replaces the previous Istio-header pass-through with license-token introspection
so this service honours the global ".test services use license-authz" rule
(see ``Movies/CLAUDE.md``).

Wire format
-----------

- License tokens travel **only** in ``X-Tenant-Id: Bearer <token>``. The
  ``Authorization`` header is reserved for the caller's own auth.
- Public endpoints (health, docs, root, metrics) bypass introspection.
- RFC1918 / loopback / ``testclient`` peers bypass introspection with a default
  tenant. The host runs on a Tailnet CGNAT IP (100.64.0.0/10), which is NOT in
  RFC1918 — so real cross-host east-west traffic always goes through real
  introspection.

Successful introspection responses are cached for 5 minutes by SHA-256 prefix
of the token (TTLCache).
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, cast

import httpx
from cachetools import TTLCache
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .models import User

logger = logging.getLogger(__name__)

PUBLIC_ENDPOINTS: set[str] = {
    "/",
    "/favicon.ico",
    "/health",
    "/health/ready",
    "/health/live",
    "/health/detailed",
    "/healthz",
    "/ready",
    "/readyz",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/metrics",
}

# Implicit public trees, in addition to the exact PUBLIC_ENDPOINTS above.
# Matching is anchored to a path SEGMENT: `/docs` and `/docs/oauth2-redirect`
# are public, `/docsecret` is NOT. A bare `startswith` here was an auth-bypass
# class — the same unanchored shape that let `/storage/api/buckets/ready` reach
# MinIO unauthenticated (gitops #2204), fixed in the shared lib as kag_common
# #23. This service carries its own copy and does not depend on kag_common.
# `_is_protected_path` below was ALREADY segment-anchored: the fail-closed half
# had been fixed and the fail-open half left loose. `/metrics` stays public for
# the cluster prometheus scrape, and every health path served
# (`/health`, `/health/live`, `/health/ready`, `/healthz`, `/ready`, `/readyz`)
# is either explicit in PUBLIC_ENDPOINTS or under the `/health/` segment, so
# the tightening cannot auth-wall service.yml's `curl -f .../health` check.
PUBLIC_PREFIXES: tuple[str, ...] = ("/docs", "/redoc", "/static", "/health", "/metrics")

PROTECTED_PREFIXES: set[str] = {
    "/api",
    "/me",
}

DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "0195f48b-d000-7000-8000-000000000001")

_TRUSTED_CIDRS_RAW = os.getenv(
    "TRUSTED_CIDRS",
    "127.0.0.1/32,::1/128,192.168.0.0/16,172.16.0.0/12,10.0.0.0/8",
)
TRUSTED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network(cidr.strip(), strict=False) for cidr in _TRUSTED_CIDRS_RAW.split(",") if cidr.strip()
]

_license_cache: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=1000, ttl=300)


def _license_introspect_url() -> str:
    return os.getenv("LICENSE_INTROSPECT_URL", "")


def _is_trusted_network(host: str | None) -> bool:
    if not host:
        return False
    if host == "testclient":
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(addr in net for net in TRUSTED_NETWORKS)


def _is_public_path(path: str) -> bool:
    clean = path.rstrip("/") or "/"
    if clean in PUBLIC_ENDPOINTS:
        return True
    return any(clean == prefix or clean.startswith(prefix + "/") for prefix in PUBLIC_PREFIXES)


def _is_protected_path(path: str) -> bool:
    clean = path.rstrip("/") or "/"
    return any(clean == prefix or clean.startswith(prefix + "/") for prefix in PROTECTED_PREFIXES)


def _extract_license_token(request: Request) -> str | None:
    x_tenant_id = request.headers.get("X-Tenant-Id", "")
    if x_tenant_id and x_tenant_id.lower().startswith("bearer "):
        return x_tenant_id[7:].strip() or None
    return None


async def _introspect_license(token: str) -> dict[str, Any] | None:
    cache_key = hashlib.sha256(token.encode()).hexdigest()[:16]
    cached = _license_cache.get(cache_key)
    if cached is not None:
        return cached

    url = _license_introspect_url()
    if not url:
        logger.warning("LICENSE_INTROSPECT_URL not configured; rejecting bearer token")
        return None

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(url, json={"token": token})
    except Exception as e:  # noqa: BLE001 — network failures shouldn't 500 the request
        logger.warning("license_introspect_failed", extra={"error": str(e)})
        return None

    if resp.status_code != 200:
        return None
    try:
        data: dict[str, Any] = resp.json()
    except ValueError:
        return None
    if data.get("active"):
        _license_cache[cache_key] = data
    return data


def _error_payload(code: str, message: str, http_status: int) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "status": http_status,
        }
    }


def _unauthorized(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=_error_payload("AUTHENTICATION_REQUIRED", message, 401),
        headers={"WWW-Authenticate": "Bearer"},
    )


class AuthMiddleware(BaseHTTPMiddleware):
    """License-token auth with public/trusted-network exits."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path

        if _is_public_path(path) or not _is_protected_path(path):
            return await call_next(request)

        token = _extract_license_token(request)
        if token:
            data = await _introspect_license(token)
            if not data or not data.get("active"):
                return _unauthorized("Invalid or expired license token")

            tenant_id = data.get("tenant_id")
            if not tenant_id:
                return _unauthorized("License token missing tenant context")

            subject_id = data.get("subject_id")
            subject_str: str | None = str(subject_id) if subject_id is not None else None
            license_id = data.get("license_id")
            license_str: str | None = str(license_id) if license_id is not None else None

            user = User(
                sub=subject_str or str(tenant_id),
                preferred_username=cast("str", data.get("subject_username") or data.get("tenant_name") or "license"),
                tenant_id=str(tenant_id),
                roles=list(data.get("roles", []) or []),
            )
            request.state.user = user
            request.state.tenant_id = str(tenant_id)
            request.state.tenant_name = request.headers.get("X-Tenant-Name", "")
            request.state.subject_id = subject_str
            request.state.license_id = license_str
            request.state.license = data
            return await call_next(request)

        client_host = request.client.host if request.client else None
        if _is_trusted_network(client_host):
            tenant_id = request.headers.get("X-Tenant-Id", "").strip() or DEFAULT_TENANT_ID
            if tenant_id.lower().startswith("bearer "):
                tenant_id = DEFAULT_TENANT_ID
            user = User(
                sub=tenant_id,
                preferred_username="trusted-network",
                tenant_id=tenant_id,
            )
            request.state.user = user
            request.state.tenant_id = tenant_id
            request.state.tenant_name = request.headers.get("X-Tenant-Name", "")
            request.state.subject_id = None
            request.state.license_id = None
            request.state.license = None
            return await call_next(request)

        return _unauthorized("Missing license token")

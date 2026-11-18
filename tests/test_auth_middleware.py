"""Unit tests for the license-token AuthMiddleware."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import auth.middleware as auth_middleware
from auth.middleware import AuthMiddleware, _license_cache


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/health")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/echo")
    async def _echo(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "tenant_id": getattr(request.state, "tenant_id", None),
                "subject_id": getattr(request.state, "subject_id", None),
            }
        )

    return app


@pytest.fixture(autouse=True)
def _clear_license_cache() -> None:
    _license_cache.clear()


@pytest.fixture
async def auth_client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_health_is_public(auth_client: httpx.AsyncClient) -> None:
    resp = await auth_client.get("/health")
    assert resp.status_code == 200


async def test_protected_route_from_trusted_network_bypasses_auth(
    auth_client: httpx.AsyncClient,
) -> None:
    resp = await auth_client.get("/api/echo")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "0195f48b-d000-7000-8000-000000000001"
    assert body["subject_id"] is None


async def test_protected_route_with_valid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    introspect_payload: dict[str, Any] = {
        "active": True,
        "tenant_id": "tenant-uuid-1",
        "subject_id": "subject-uuid-9",
        "license_id": "lic-77",
        "roles": ["user"],
        "tenant_name": "acme",
    }

    async def _fake(token: str) -> dict[str, Any]:
        assert token == "abc.def.ghi"
        return introspect_payload

    monkeypatch.setattr(auth_middleware, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/echo",
            headers={"X-Tenant-Id": "Bearer abc.def.ghi"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tenant_id"] == "tenant-uuid-1"
    assert body["subject_id"] == "subject-uuid-9"


async def test_protected_route_with_invalid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake(_token: str) -> dict[str, Any]:
        return {"active": False}

    monkeypatch.setattr(auth_middleware, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/echo",
            headers={"X-Tenant-Id": "Bearer dud-token"},
        )

    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"
    assert body["error"]["status"] == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"


async def test_protected_route_missing_tenant_in_introspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake(_token: str) -> dict[str, Any]:
        return {"active": True, "subject_id": "s-only"}

    monkeypatch.setattr(auth_middleware, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/echo",
            headers={"X-Tenant-Id": "Bearer no-tenant"},
        )

    assert resp.status_code == 401
    assert "tenant" in resp.json()["error"]["message"].lower()


def test_is_public_path_anchors_to_segment_boundary() -> None:
    """An unanchored prefix match is an auth-bypass class: any route merely
    STARTING with a public prefix would be exempted from auth. Anchor to a path
    SEGMENT — the same anchoring the sibling `_is_protected_path` already used.
    Same shape as the ingress carve-out that let `/storage/api/buckets/ready`
    reach MinIO unauthenticated (gitops #2204).
    """
    # Real public trees stay public.
    assert auth_middleware._is_public_path("/health/ready") is True
    assert auth_middleware._is_public_path("/docs/oauth2-redirect") is True
    assert auth_middleware._is_public_path("/static/logo.svg") is True
    # Look-alikes that merely share a prefix must NOT be exempted.
    for bypass in (
        "/healthcheck-admin",
        "/health-secrets/dump",
        "/staticfiles/../etc/passwd",
        "/docsecret",
        "/redocs-internal",
        "/metrics-admin",
    ):
        assert auth_middleware._is_public_path(bypass) is False, bypass


def test_is_public_path_keeps_probe_paths_public() -> None:
    """`service.yml`'s healthcheck is `curl -f http://localhost:9203/health`;
    /healthz and the /health/* tree must stay public too — segment anchoring
    must never auth-wall a health probe. /metrics stays public for the cluster
    prometheus scrape."""
    for probe in (
        "/health",
        "/health/live",
        "/health/ready",
        "/health/detailed",
        "/healthz",
        "/ready",
        "/readyz",
        "/metrics",
    ):
        assert auth_middleware._is_public_path(probe) is True, probe


def test_is_public_path_does_not_widen() -> None:
    """A bypass fix must not quietly widen the exemption it exists to narrow:
    /livez never matched the old rule and no probe uses it."""
    assert auth_middleware._is_public_path("/livez") is False

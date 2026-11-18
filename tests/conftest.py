"""Shared fixtures + module-level test environment for vlm_backend."""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

# --- Env setup (must happen BEFORE `main` is imported) ---
os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")
os.environ.setdefault("LICENSE_INTROSPECT_URL", "http://license-introspect.test/license/introspect")


# --- Stub Apple-Silicon-only deps so tests import on any platform ---
def _install_stubs() -> None:
    fake_vlm = cast(Any, type(sys)("mlx_vlm"))
    fake_vlm.load = lambda *_a, **_kw: (object(), object())
    fake_vlm.generate = lambda **_kw: ("stub vision response",)

    sys.modules.setdefault("mlx_vlm", fake_vlm)


_install_stubs()


@pytest.fixture
async def client() -> AsyncIterator[Any]:
    import httpx

    from main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def test_client() -> Any:
    from fastapi.testclient import TestClient

    from main import app

    return TestClient(app)

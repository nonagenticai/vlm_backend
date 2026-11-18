"""Health-endpoint smoke tests (no Apple-Silicon model load required)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(test_client: TestClient) -> None:
    r = test_client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "vlm-backend"


def test_live_ok(test_client: TestClient) -> None:
    r = test_client.get("/health/live")
    assert r.status_code == 200


def test_root_info(test_client: TestClient) -> None:
    r = test_client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "vlm-backend"


def test_chat_completions_requires_image(test_client: TestClient) -> None:
    r = test_client.post(
        "/api/vlm/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    # 400 = bad request (no image), 503 = backend not initialised. Either is
    # fine — both prove the auth layer let us through.
    assert r.status_code in (400, 503)


def test_metrics_endpoint_exposes_prometheus_format(test_client: TestClient) -> None:
    test_client.get("/health")
    r = test_client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "# HELP" in body
    assert "# TYPE" in body
    assert "http_request_duration_seconds" in body

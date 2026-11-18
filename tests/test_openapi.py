"""Guardrail tests for the OpenAPI surface."""

from __future__ import annotations

import pytest

from main import app

EXPECTED_OPERATION_IDS: frozenset[str] = frozenset(
    {
        "service_root",
        "health_check",
        "health_ready",
        "health_live",
        "get_current_identity",
        "vlm_chat_completions",
        "list_vlm_models",
    }
)


@pytest.fixture(scope="module")
def openapi_schema() -> dict:
    app.openapi_schema = None
    return app.openapi()


def test_every_route_has_explicit_operation_id(openapi_schema: dict) -> None:
    auto_generated_pattern_suffixes = ("_post", "_get", "_put", "_delete", "_patch")
    offenders: list[tuple[str, str, str]] = []
    for path, item in openapi_schema.get("paths", {}).items():
        for method, op in item.items():
            if not isinstance(op, dict):
                continue
            op_id: str = op.get("operationId", "")
            if op_id.endswith(auto_generated_pattern_suffixes) and op_id not in EXPECTED_OPERATION_IDS:
                offenders.append((method.upper(), path, op_id))
    assert not offenders, f"Routes without explicit operation_id (auto-generated): {offenders}"


def test_operation_ids_match_plan(openapi_schema: dict) -> None:
    actual: set[str] = set()
    for item in openapi_schema.get("paths", {}).values():
        for op in item.values():
            if isinstance(op, dict) and op.get("operationId"):
                actual.add(op["operationId"])

    missing = EXPECTED_OPERATION_IDS - actual
    unexpected = actual - EXPECTED_OPERATION_IDS
    assert not missing and not unexpected, (
        f"operationId drift — missing: {sorted(missing)}, unexpected: {sorted(unexpected)}"
    )


def test_operation_ids_are_unique(openapi_schema: dict) -> None:
    seen: dict[str, tuple[str, str]] = {}
    duplicates: list[tuple[str, tuple[str, str], tuple[str, str]]] = []
    for path, item in openapi_schema.get("paths", {}).items():
        for method, op in item.items():
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            if not op_id:
                continue
            if op_id in seen:
                duplicates.append((op_id, seen[op_id], (method.upper(), path)))
            else:
                seen[op_id] = (method.upper(), path)
    assert not duplicates, f"Duplicate operationIds: {duplicates}"


def test_bearer_auth_scheme_is_declared(openapi_schema: dict) -> None:
    schemes = openapi_schema.get("components", {}).get("securitySchemes", {})
    assert "BearerAuth" in schemes, "BearerAuth scheme missing from OpenAPI"
    scheme = schemes["BearerAuth"]
    assert scheme["type"] == "http"
    assert scheme["scheme"] == "bearer"


def test_tenant_id_header_parameter_is_declared(openapi_schema: dict) -> None:
    parameters = openapi_schema.get("components", {}).get("parameters", {})
    assert "TenantIdHeader" in parameters, "TenantIdHeader parameter missing"
    param = parameters["TenantIdHeader"]
    assert param["in"] == "header"
    assert param["name"].lower() == "x-tenant-id"


def test_tag_groups_are_defined(openapi_schema: dict) -> None:
    tag_names = {t["name"] for t in openapi_schema.get("tags", [])}
    assert {"Health", "VLM", "Identity"} <= tag_names, f"Missing required tag groups; have {sorted(tag_names)}"


def test_protected_routes_have_security_requirement(openapi_schema: dict) -> None:
    protected_paths = [p for p in openapi_schema.get("paths", {}) if p.startswith("/api/") or p == "/me"]
    assert protected_paths, "No protected paths found in schema"
    for path in protected_paths:
        for method, op in openapi_schema["paths"][path].items():
            if not isinstance(op, dict):
                continue
            security = op.get("security", [])
            assert any("BearerAuth" in entry for entry in security), (
                f"{method.upper()} {path} is missing BearerAuth security requirement"
            )


def test_public_routes_have_no_security_requirement(openapi_schema: dict) -> None:
    public_paths = {"/", "/health", "/health/ready", "/health/live"}
    for path in public_paths:
        item = openapi_schema.get("paths", {}).get(path)
        if item is None:
            continue
        for method, op in item.items():
            if not isinstance(op, dict):
                continue
            security = op.get("security", [])
            assert not security, f"Public route {method.upper()} {path} unexpectedly has security={security}"

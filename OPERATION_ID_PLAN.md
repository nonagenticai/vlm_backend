# Operation ID Plan — vlm_backend

Source of truth for FastAPI `operation_id=` values. Mirrored in
`tests/test_openapi.py::EXPECTED_OPERATION_IDS`. Globally unique across the
mini1/mini2 host-services fleet.

Naming convention: `verb_noun` snake_case.

## `src/main.py`

| Method | Path           | Function | operation_id              |
| ------ | -------------- | -------- | ------------------------- |
| GET    | /              | root     | service_root              |
| GET    | /health        | health   | health_check              |
| GET    | /health/ready  | ready    | health_ready              |
| GET    | /health/live   | live     | health_live               |
| GET    | /me            | me       | get_current_identity      |

## `src/routers/vlm_routes.py` (prefix `/api/vlm`, tag `VLM`)

| Method | Path                          | Function          | operation_id          |
| ------ | ----------------------------- | ----------------- | --------------------- |
| POST   | /api/vlm/chat/completions     | chat_completions  | vlm_chat_completions  |
| GET    | /api/vlm/models               | list_models       | list_vlm_models       |

Total: 7 explicit `operation_id` values.

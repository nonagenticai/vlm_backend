# vlm-backend

Vision-language model inference for Apple Silicon. Fills the platform's local-VLM gap — consumers can stop paying for hosted vision APIs.

| | |
|---|---|
| Engine | `mlx-vlm` (also pulls `torch` + `torchvision` because the `transformers` Qwen2-VL processor hard-imports them) |
| Default model | `mlx-community/Qwen2-VL-7B-Instruct-4bit` (~6 GB) |
| Port | `9203` |
| Hostname | `vlm.test` |
| API | OpenAI-vision-compatible — `POST /api/vlm/chat/completions` |
| Auth | Istio header pass-through |

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/vlm/chat/completions` | Same payload shape as OpenAI: `messages: [{role, content: [{type: "text"|"image_url", ...}]}]`. At least one `image_url` content part required. Streaming not yet supported. |
| `GET` | `/api/vlm/models` | currently loaded model |
| `GET` | `/health{,/ready,/live}` | service health |

Image URLs accept `data:image/...;base64,...` and `http(s)://`. Images are resized to fit `VLM_IMAGE_MAX_PIXELS` before being passed to the model.

## Configuration

Deployments render the production values into `~/.config/vlm-backend/env` (XDG path; `start.sh` falls back to `/etc/vlm-backend/env` for privileged-provisioning hosts). The rendering tooling is not part of this repository.

| Var | Default | Purpose |
|---|---|---|
| `VLM_MODEL` | `mlx-community/Qwen2-VL-7B-Instruct-4bit` | HF repo |
| `VLM_MAX_TOKENS` | `1024` | per-request output cap |
| `VLM_TEMPERATURE` | `0.2` | per-request default |
| `VLM_BATCH_SIZE` | `1` | currently unused; reserved |
| `VLM_IMAGE_MAX_PIXELS` | `2073600` (1920×1080) | downscale large images before encode |
| `LOG_LEVEL` | `INFO` | |
| `CORS_ORIGINS` | `https://api.nonagentic.ai,https://w2v.nonagentic.ai` | |

## Alternative models

Drop in via `VLM_MODEL`:
- `mlx-community/MiniCPM-V-2_6-4bit` (~6 GB, very strong document/chart VQA)
- `mlx-community/Qwen2-VL-2B-Instruct-4bit` (~1.6 GB, lighter / faster)
- `mlx-community/Phi-3.5-vision-instruct-4bit` (~3 GB)

## Local dev

```bash
uv sync
PYTHONPATH=src uv run pytest                                      # CI-safe
PYTHONPATH=src uv run uvicorn main:app --host 0.0.0.0 --port 9203   
```

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This service contains no licence enforcement of its own and runs no licence
server. It runs with no licence switch set; unset means off.

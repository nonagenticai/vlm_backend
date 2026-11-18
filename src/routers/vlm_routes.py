"""OpenAI-vision-compatible chat-completions route."""

from __future__ import annotations

import binascii
import logging
import time
import uuid
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from PIL import UnidentifiedImageError

from auth.dependencies import get_current_user
from auth.models import User
from models import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ContentPartImage,
    ContentPartText,
)
from routers.responses import AUTH_RESPONSES, VLM_ENDPOINT_RESPONSES
from services.vlm_service import BackendUnavailableError, VLMService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/vlm", tags=["VLM"])


def _service_from_request(request: Request) -> VLMService:
    svc = getattr(request.app.state, "vlm_service", None)
    if svc is None or not svc.is_loaded:
        raise HTTPException(status_code=503, detail="VLM backend not initialised")
    return svc


def _flatten_messages(req: ChatCompletionRequest) -> tuple[str, list[str]]:
    """Collapse OpenAI-style messages into a single prompt + ordered list of image URLs."""
    parts: list[str] = []
    image_urls: list[str] = []
    for msg in req.messages:
        if isinstance(msg.content, str):
            parts.append(f"{msg.role}: {msg.content}")
            continue
        text_chunks: list[str] = []
        for piece in msg.content:
            if isinstance(piece, ContentPartText):
                text_chunks.append(piece.text)
            elif isinstance(piece, ContentPartImage):
                image_urls.append(piece.image_url.url)
                text_chunks.append("<image>")
        parts.append(f"{msg.role}: {' '.join(text_chunks)}")
    prompt = "\n".join(parts) + "\nassistant:"
    return prompt, image_urls


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    operation_id="vlm_chat_completions",
    summary="OpenAI-vision-compatible chat completion (single-turn, non-streaming)",
    responses=VLM_ENDPOINT_RESPONSES,
)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    _user: Annotated[User, Depends(get_current_user)],
) -> ChatCompletionResponse:
    svc = _service_from_request(request)

    if body.stream:
        raise HTTPException(status_code=400, detail="Streaming is not yet supported")

    prompt, image_urls = _flatten_messages(body)
    if not image_urls:
        raise HTTPException(status_code=400, detail="VLM requires at least one image_url content part")

    try:
        images = [await svc.fetch_image(u) for u in image_urls]
    except (
        httpx.HTTPError,
        ValueError,
        OSError,
        UnidentifiedImageError,
        binascii.Error,
    ) as exc:
        raise HTTPException(status_code=400, detail=f"Image fetch failed: {exc}") from exc

    try:
        result = await svc.generate(
            prompt=prompt,
            images=images,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
        )
    except BackendUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        logger.exception("VLM generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=body.model or svc.model_repo,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionChoiceMessage(role="assistant", content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


@router.get(
    "/models",
    operation_id="list_vlm_models",
    summary="List the currently-loaded VLM model (OpenAI /v1/models analogue)",
    responses={**AUTH_RESPONSES, 503: {"description": "Backend not initialised"}},
)
async def list_models(
    request: Request,
    _user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """OpenAI-compat: GET /v1/models analogue, but under our /api prefix."""
    svc = _service_from_request(request)
    return {
        "object": "list",
        "data": [
            {
                "id": svc.model_repo,
                "object": "model",
                "owned_by": "vlm-backend",
            }
        ],
    }

"""VLM service backed by mlx-vlm."""

from __future__ import annotations

import base64
import io
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"]


class BackendUnavailableError(RuntimeError):
    pass


class VLMService:
    name: str = "mlx-vlm"

    def __init__(self) -> None:
        self.model_repo: str = os.getenv("VLM_MODEL", "mlx-community/Qwen2-VL-7B-Instruct-4bit")
        self.max_tokens_default: int = int(os.getenv("VLM_MAX_TOKENS", "1024"))
        self.temperature_default: float = float(os.getenv("VLM_TEMPERATURE", "0.2"))
        self.image_max_pixels: int = int(os.getenv("VLM_IMAGE_MAX_PIXELS", str(1920 * 1080)))
        self._loaded: bool = False
        self._model: Any = None
        self._processor: Any = None
        self._mlx_vlm: Any = None
        self._mx: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return "unified"

    async def initialize(self) -> None:
        try:
            import mlx.core as mx
            import mlx_vlm
            from mlx_vlm import load
        except ImportError as exc:
            raise BackendUnavailableError("mlx_vlm not installed; this service requires Apple Silicon.") from exc
        self._mlx_vlm = mlx_vlm
        self._mx = mx
        logger.info("Loading VLM model: %s", self.model_repo)
        self._model, self._processor = load(self.model_repo)
        self._loaded = True
        logger.info("VLM model ready")

    async def generate(
        self,
        prompt: str,
        images: list[Image.Image],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
    ) -> GenerationResult:
        if not self._loaded or self._mlx_vlm is None:
            raise BackendUnavailableError("VLMService not initialised")

        max_t = max_tokens or self.max_tokens_default
        temp = temperature if temperature is not None else self.temperature_default

        kwargs: dict[str, Any] = {
            "model": self._model,
            "processor": self._processor,
            "prompt": prompt,
            "image": images if len(images) != 1 else images[0],
            "max_tokens": max_t,
            "temperature": temp,
            "verbose": False,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p

        # MLX streams are thread-local — pin to GPU stream for worker-thread calls.
        # mlx-vlm.generate signature evolved across versions; fall back gracefully.
        with self._mx.stream(self._mx.gpu):
            try:
                output = self._mlx_vlm.generate(**kwargs)
            except TypeError:
                kwargs.pop("temperature", None)
                kwargs.pop("top_p", None)
                output = self._mlx_vlm.generate(**kwargs)

        if isinstance(output, tuple) and len(output) >= 1:
            text = str(output[0])
        elif hasattr(output, "text"):
            text = str(output.text)
        else:
            text = str(output)

        # mlx-vlm doesn't always expose token counts; estimate
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(text.split()))
        finish_reason: Literal["stop", "length"] = "length" if completion_tokens >= max_t else "stop"

        return GenerationResult(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    async def fetch_image(self, url: str) -> Image.Image:
        """Resolve a data: URI or http(s) URL into a PIL Image, capped by max-pixels."""
        if url.startswith("data:"):
            try:
                _, b64 = url.split(",", 1)
            except ValueError as exc:
                raise ValueError("Malformed data URI") from exc
            data = base64.b64decode(b64)
        elif url.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.content
        else:
            raise ValueError(f"Unsupported image URL scheme: {url[:32]}")

        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        if w * h > self.image_max_pixels:
            scale = (self.image_max_pixels / (w * h)) ** 0.5
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        return img


_singleton: VLMService | None = None


def get_vlm_service() -> VLMService:
    global _singleton
    if _singleton is None:
        _singleton = VLMService()
    return _singleton

"""`ImageProvider` via OpenRouter chat completions (image output), plain HTTP, no SDK.

Used for the character reference image of video models that need an image with audio
(Seedance 2.0 / 2.0-fast; see shot_1). Built from the OpenRouter docs and one live
call (google/gemini-2.5-flash-image, one image, usage.cost 0.0387 USD):

  POST https://openrouter.ai/api/v1/chat/completions   (Authorization: Bearer OPENROUTER_API_KEY)
       JSON: model, messages [{"role": "user", "content": prompt}], modalities ["image", "text"],
       image_config {"aspect_ratio": "9:16"}, usage {"include": true}
    -> 200 choices[0].message.images[0].image_url.url = "data:image/png;base64,..."
       usage.cost = USD (null/missing -> reported as 0)

The image is written to `out_path` as PNG (another image type is converted with ffmpeg).
Errors: 429 / 5xx / connection errors / timeouts -> retryable; moderation (code or message) and
other 4xx -> final; an answer without an image -> retryable (the model sometimes answers text
only). A failed call may still be charged (a few cents); shot_1's money cap bounds retries."""

from __future__ import annotations

import base64
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx

from hwva.providers import video_jobs as jobs
from hwva.providers.base import ImageResult, ProviderError
from hwva.schemas import ModelRef

NAME = "OpenRouter image"
_DATA_URI = re.compile(r"data:(?P<mime>image/[\w.+-]+);base64,(?P<data>.+)", re.DOTALL)
API_KEY_ENV = "OPENROUTER_API_KEY"
URL = "https://openrouter.ai/api/v1/chat/completions"
ASPECT = "9:16"
TIMEOUT_S = 120.0
FFMPEG_TIMEOUT_S = 120


class OpenRouterImage:
    """`transport` exists for tests (httpx.MockTransport); production uses the default one.
    The model sent is `ref.model`."""

    def __init__(self, ref: ModelRef, *, transport: httpx.BaseTransport | None = None) -> None:
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise ValueError(f"{API_KEY_ENV} is not set (put it in .env, see .env.example)")
        self.model = ref.model
        self._key = key
        self._client = httpx.Client(transport=transport, timeout=TIMEOUT_S)

    def generate(self, prompt: str, out_path: Path) -> ImageResult:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image", "text"],
            "image_config": {"aspect_ratio": ASPECT},
            "usage": {"include": True},
        }
        try:
            response = self._client.post(
                URL, json=body, headers={"Authorization": f"Bearer {self._key}"}
            )
        except httpx.RequestError as exc:
            raise ProviderError(f"{NAME} request failed: {type(exc).__name__}: {exc}") from exc
        data = jobs.parse(response, NAME)  # moderation / retryable / final, like the video API
        mime, payload = _image(data)
        _write(mime, payload, out_path)
        return ImageResult(path=out_path, cost_usd=_cost(data))


def _image(data: dict[str, Any]) -> tuple[str, bytes]:
    """(mime, bytes) of the first image of the answer; retryable when there is none (the model
    sometimes answers text only)."""
    try:
        url = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
    except (KeyError, IndexError, TypeError):
        url = None
    match = _DATA_URI.match(url) if isinstance(url, str) else None
    if match is None:
        raise ProviderError(f"{NAME} answered without a base64 image: {str(data)[:200]}")
    try:
        payload = base64.b64decode(match["data"], validate=True)
    except ValueError as exc:
        raise ProviderError(f"{NAME} image is not valid base64: {exc}") from exc
    if not payload:
        raise ProviderError(f"{NAME} image is empty")
    return match["mime"].lower(), payload


def _write(mime: str, payload: bytes, out_path: Path) -> None:
    """PNG as is; another type is converted to PNG with ffmpeg (the file name says .png and the
    media host serves it by name). Written via a temp file: no torn image survives a crash."""
    tmp = out_path.with_name(out_path.name + ".part")
    if mime == "image/png":
        tmp.write_bytes(payload)
    else:
        src = out_path.with_name(out_path.name + ".src")
        src.write_bytes(payload)
        try:
            out = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                 "-frames:v", "1", "-f", "image2", "-c:v", "png", str(tmp)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=FFMPEG_TIMEOUT_S, check=False,
            )  # fmt: skip
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProviderError(f"{NAME}: converting {mime} to PNG failed: {exc}") from exc
        finally:
            src.unlink(missing_ok=True)
        if out.returncode != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            raise ProviderError(f"{NAME}: converting {mime} to PNG failed: {out.stderr[-200:]}")
    tmp.replace(out_path)


def _cost(data: dict[str, Any]) -> float:
    usage = data.get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if (
        isinstance(cost, int | float)
        and not isinstance(cost, bool)
        and math.isfinite(cost)
        and cost >= 0
    ):
        return float(cost)
    return 0.0

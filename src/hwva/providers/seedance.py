"""BytePlus ModelArk Seedance video over plain HTTP (no SDK). Built against the documented API,
not yet checked by a live call (no ARK key at the time of writing):

  POST {base}/api/v3/contents/generations/tasks   (Bearer ARK_API_KEY) -> {"id": ...}
  GET  {base}/api/v3/contents/generations/tasks/{id} until succeeded / failed / expired / cancelled
  content.video_url is downloaded at once: it expires in 24 h (the task itself lives 7 days).

Never pay twice for one clip: the task file, resume, unknown-create and cost rules live in
`video_jobs` (shared with the OpenRouter adapter).

Strategy A: our TTS goes in as `audio_url` with role `reference_audio` (base64 data URL). A frame
always goes as `image_url` with role `reference_image` — `first_frame` together with
`reference_audio` is rejected by the API (3rd-party report). The model's own audio is generated
(API default) but never used: the final track is always our TTS (assemble)."""

from __future__ import annotations

import os

# the job loop lives in video_jobs; tests patch seedance.time / seedance.random (the same modules)
import random  # noqa: F401
import time  # noqa: F401
from pathlib import Path
from typing import Any

import httpx

from hwva.providers import video_jobs as jobs
from hwva.providers.base import Capabilities, ProviderError, VideoRequest
from hwva.schemas import ModelRef

API_KEY_ENV = "ARK_API_KEY"
BASE_URL_ENV = "ARK_BASE_URL"
DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com"
TASKS_PATH = "/api/v3/contents/generations/tasks"
RESOLUTIONS = ("480p", "720p", "1080p")
RATIOS = ("9:16", "16:9", "1:1", "4:3", "3:4", "21:9")

# ref.version -> what that Seedance generation accepts (BytePlus ModelArk docs)
CAPABILITIES: dict[str, Capabilities] = {
    "2.5": Capabilities(
        accepts_audio_input=True, max_duration_s=30, min_audio_s=2.0, aspect_ratios=RATIOS
    ),
    "2.0": Capabilities(
        accepts_audio_input=True,
        needs_image=True,
        max_duration_s=15,
        min_audio_s=2.0,
        aspect_ratios=RATIOS,
    ),
}


class SeedanceVideo(jobs.VideoJobProvider):
    """`VideoProvider`. The model id sent to the API is `ref.model` (pinned in profiles.yaml);
    `ref.version` selects the capabilities.

    `transport` exists for tests (httpx.MockTransport); production uses the default one.
    `usd_per_million_tokens`: no verified price yet, so cost is 0 unless it is given.
    `retry_backoff_s` / `retry_backoff_max_s`: first and largest wait after a transient poll
    error (doubles per error in a row, random in its upper half)."""

    name = "Seedance"
    console = "the ModelArk console (video generation tasks)"
    succeeded = "succeeded"

    def __init__(
        self,
        ref: ModelRef,
        *,
        transport: httpx.BaseTransport | None = None,
        poll_interval_s: float = 10.0,
        timeout_s: float = 1200.0,
        resolution: str = "720p",
        usd_per_million_tokens: float | None = None,
        retry_backoff_s: float = 2.0,
        retry_backoff_max_s: float = 60.0,
    ) -> None:
        if ref.version not in CAPABILITIES:
            raise ValueError(
                f"Unknown Seedance version {ref.version!r}; known: {', '.join(CAPABILITIES)}"
            )
        if resolution not in RESOLUTIONS:
            raise ValueError(f"Unknown resolution {resolution!r}; known: {', '.join(RESOLUTIONS)}")
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise ValueError(f"{API_KEY_ENV} is not set (put it in .env, see .env.example)")
        base = os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL  # empty in .env -> default
        if not base.lower().startswith("https://"):  # the Bearer key never goes over plain HTTP
            raise ProviderError(
                f"{BASE_URL_ENV} must be an https:// URL, got {base[:100]!r}", retryable=False
            )
        super().__init__(
            key=key,
            model=ref.model,
            resolution=resolution,
            transport=transport,
            poll_interval_s=poll_interval_s,
            timeout_s=timeout_s,
            retry_backoff_s=retry_backoff_s,
            retry_backoff_max_s=retry_backoff_max_s,
        )
        self.capabilities = CAPABILITIES[ref.version]
        self.usd_per_million_tokens = usd_per_million_tokens
        self._tasks_url = base.rstrip("/") + TASKS_PATH

    # --- video_jobs hooks ---------------------------------------------------------------------

    def _body(self, request: VideoRequest) -> dict[str, Any]:
        duration = jobs.checked_duration(self.name, self.model, self.capabilities, request)
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        if request.audio_path is not None:
            url = jobs.data_url(request.audio_path, jobs.AUDIO_MIME, "audio", self.name)
            content.append(
                {"type": "audio_url", "audio_url": {"url": url}, "role": "reference_audio"}
            )
        if request.reference_image is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": jobs.image_url(request.reference_image, self.name)},
                    "role": "reference_image",
                }
            )
        return {
            "model": self.model,
            "content": content,
            "ratio": request.aspect_ratio,
            "resolution": self.resolution,
            "duration": duration,
            "generate_audio": True,  # API default, stated explicitly; the track is discarded
        }

    def _fingerprint(self, request: VideoRequest, body: dict[str, Any]) -> dict[str, Any]:
        return jobs.fingerprint(
            request,
            model=body["model"],
            duration=body["duration"],
            resolution=body["resolution"],
            ratio=body["ratio"],
        )

    def _create_url(self) -> str:
        return self._tasks_url

    def _job_url(self, job_id: str) -> str:
        return f"{self._tasks_url}/{job_id}"

    def _video_source(self, job_id: str, job: dict[str, Any], task_file: Path) -> tuple[str, bool]:
        video_url = (job.get("content") or {}).get("video_url")
        if not isinstance(video_url, str) or not video_url:
            # the task succeeded, so it is paid for: keep its id, a rerun re-checks it
            raise ProviderError(
                f"Seedance task {job_id} succeeded without a video_url. Check it in the ModelArk "
                f"console; a rerun re-checks this task, delete {task_file} to pay for a new one.",
                retryable=False,
            )
        if not video_url.lower().startswith("https://"):
            # paid for as well: keep its id, so a rerun re-checks it instead of paying again
            raise ProviderError(
                f"Seedance task {job_id} returned a video URL that is not https "
                f"({video_url.split('?', 1)[0][:100]}); not downloaded. Check it in the ModelArk "
                f"console; delete {task_file} to pay for a new one.",
                retryable=False,
            )
        return video_url, False  # presigned on another host: no Authorization header

    def _cost(self, job: dict[str, Any], debug: dict[str, Any]) -> float:
        usage = job.get("usage")
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if self.usd_per_million_tokens is None or not isinstance(tokens, int | float):
            # TODO: set usd_per_million_tokens once the ModelArk price for the pinned model is
            # verified; the usage tokens are kept in <clip>.task.json meanwhile.
            return 0.0
        return float(tokens) * self.usd_per_million_tokens / 1_000_000

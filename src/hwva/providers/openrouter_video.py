"""Seedance video via OpenRouter over plain HTTP (no SDK). Built against the OpenRouter docs
 and fixed after 4 paid live clips the same day (the key is paid per call):

  POST https://openrouter.ai/api/v1/videos   (Authorization: Bearer OPENROUTER_API_KEY)
       JSON: model, prompt, duration (int s), resolution ("480p" | "720p"), aspect_ratio
       ("9:16"), input_references ([{"type": "audio_url", "audio_url": {"url"}},
       {"type": "image_url", "image_url": {"url"}}]), frame_images ([{"type": "image_url",
       "image_url": {"url"}, "frame_type": "first_frame"}]), generate_audio
       read timeout 180 s (live: a create took more than 60 s once; a lost answer is an
       unknown outcome, final — see video_jobs)
    -> 202 {"id": "gen-vid-...", "polling_url": "/api/v1/videos/{id}", "status": ...}
  GET  https://openrouter.ai/api/v1/videos/{id}
       status: pending | in_progress | completed | failed | cancelled | expired; "error" string
  completed: the clip is the first item of "urls" (signed, no auth, valid until "expires_at");
       if there is none, the first item of "unsigned_urls", fetched with the Bearer header.
  cost: usage.cost in USD; may be null -> reported as 0, with a note in <clip>.task.json.
  errors: {"error": {"code", "message", "metadata"}}. 400/401/402 (no credits)/403/413 final;
       429 and 5xx retryable on polls (on the create POST a 5xx is an unknown outcome, final);
       moderation text final.

Model page: bytedance/seedance-2.5 takes 4..30 s, 480p/720p, 9:16; bytedance/seedance-2.0 and
-2.0-fast take 4..15 s. The model slug sent is `ref.model`; `ref.version` selects capabilities.

LIVE:
- Media only as public HTTPS URLs: a data URI is answered 400 "Only HTTPS URLs are allowed".
  Local audio and images are uploaded through the `MediaHost` given to the constructor (profile
  option `media_host`, e.g. litterbox) right before the create POST — never on a resume of a
  saved job. An http:// or data: reference is refused before any call. The fingerprint stays
  the sha256 of the media, not the (new on every upload) hosted URL, so a resume never pays
  again. Hosted URLs and their expiry are recorded under "media" in <clip>.task.json.
- Audio + image: BOTH go in input_references (audio first), frame_images is never sent with
  audio. 2.0-fast honoured our audio only like that; with the image as first_frame it ignored
  the audio. Without audio (shot_2) the image goes as frame_images first_frame.
- 2.0 / 2.0-fast with audio only: 400 "reference_audio cannot be the only reference input" —
  hence needs_image (shot_1 makes a character image for it).
- 2.5 ignored our audio (the beak moved during silence) with audio ONLY; with audio + image in
  input_references it lip-synced (live: 480p 5 s, 0.52 USD: English SyncNet 8.6,
  Ukrainian 1.9). Hence needs_image for 2.5 too; the profile default.
- A photorealistic human image: 400 InputImageSensitiveContentDetected.PrivacyInformation
  ("may contain real person"), final (moderation). A stylized 3D cartoon character passed.
- Prices (usage.cost): 2.5 480p 4 s 0.4155 USD; 2.0-fast 480p 4 s 0.1705 USD.
generate_audio is false: the final track is always our TTS (assemble).
ASSUMED, not in the OpenRouter facts: reference audio of 2..30 s (min_audio_s=2.0, the BytePlus
limit of the same model).

Never pay twice for one clip: the task file, resume, unknown-create and cost-once rules are
those of the Seedance adapter, shared in `video_jobs`. Three stricter rules here, because the
paths are not confirmed live and the env is not pinned: a 404 polling a saved job is final (not
"gone": the job may be paid); a 403/404 downloading a finished job is final (not "URL expired":
the poll just returned fresh URLs, a real expiry is status "expired"); a saved job that differs
only in resolution is refused, not replaced (a resume without the same HWVA_VIDEO_RESOLUTION
would otherwise pay again; to switch resolution on purpose, delete <clip>.task.json).

Resolution: constructor option, else env HWVA_VIDEO_RESOLUTION (e.g. 480p for cheap tests),
else 720p."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from hwva.providers import video_jobs as jobs
from hwva.providers.base import Capabilities, MediaHost, ProviderError, VideoRequest
from hwva.schemas import ModelRef

API_KEY_ENV = "OPENROUTER_API_KEY"
RESOLUTION_ENV = "HWVA_VIDEO_RESOLUTION"
BASE_URL = "https://openrouter.ai"
VIDEOS_PATH = "/api/v1/videos"
API_HOST = "openrouter.ai"  # the only host the Bearer key is sent to
DEFAULT_RESOLUTION = "720p"
RESOLUTIONS = ("480p", "720p")
CREATE_TIMEOUT_S = 180.0  # live: a create POST took more than 60 s once

# ref.version -> what that Seedance generation accepts via OpenRouter (model pages; the
# image-with-audio rule of 2.0 is from the BytePlus docs, same model)
CAPABILITIES: dict[str, Capabilities] = {
    # 2.5 accepts audio alone but ignores it (live): the image is needed to lip-sync
    "2.5": Capabilities(
        accepts_audio_input=True, needs_image=True, max_duration_s=30, min_audio_s=2.0
    ),
    "2.0": Capabilities(
        accepts_audio_input=True, needs_image=True, max_duration_s=15, min_audio_s=2.0
    ),
    "2.0-fast": Capabilities(
        accepts_audio_input=True, needs_image=True, max_duration_s=15, min_audio_s=2.0
    ),
}


class OpenRouterVideo(jobs.VideoJobProvider):
    """`VideoProvider` for Seedance via OpenRouter's video API.

    `transport` exists for tests (httpx.MockTransport); production uses the default one.
    `resolution`: None -> env HWVA_VIDEO_RESOLUTION -> 720p.
    `retry_backoff_s` / `retry_backoff_max_s`: first and largest wait after a transient poll
    error (doubles per error in a row, random in its upper half)."""

    name = "OpenRouter"
    console = "the OpenRouter activity page (openrouter.ai/activity)"
    gone_on_404 = False  # poll path not confirmed live: a 404 must not buy a second clip
    replace_on_expired_url = False  # each poll returns fresh URLs; a dead one is not expiry
    pinned = ("resolution",)
    pinned_hint = f"Set {RESOLUTION_ENV} back to the saved resolution"
    succeeded = "completed"
    create_timeout_s = CREATE_TIMEOUT_S

    def __init__(
        self,
        ref: ModelRef,
        *,
        transport: httpx.BaseTransport | None = None,
        poll_interval_s: float = 10.0,
        timeout_s: float = 1200.0,
        resolution: str | None = None,
        retry_backoff_s: float = 2.0,
        retry_backoff_max_s: float = 60.0,
        media_host: MediaHost | None = None,
    ) -> None:
        if ref.version not in CAPABILITIES:
            raise ValueError(
                f"Unknown Seedance version {ref.version!r} for OpenRouter; "
                f"known: {', '.join(CAPABILITIES)}"
            )
        resolution = resolution or os.environ.get(RESOLUTION_ENV) or DEFAULT_RESOLUTION
        if resolution not in RESOLUTIONS:
            raise ValueError(
                f"Unknown resolution {resolution[:20]!r} (argument or {RESOLUTION_ENV}); "
                f"known: {', '.join(RESOLUTIONS)}"
            )
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise ValueError(f"{API_KEY_ENV} is not set (put it in .env, see .env.example)")
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
        self.media_host = media_host
        self._videos_url = BASE_URL + VIDEOS_PATH

    # --- video_jobs hooks ---------------------------------------------------------------------

    def _body(self, request: VideoRequest) -> dict[str, Any]:
        """Validated body; local media stand as their paths until `_with_media` uploads them."""
        duration = jobs.checked_duration(self.name, self.model, self.capabilities, request)
        audio, image = self._media(request)
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": request.prompt,
            "duration": duration,
            "resolution": self.resolution,
            "aspect_ratio": request.aspect_ratio,
            "generate_audio": False,  # our TTS is the audio
        }
        _place(body, None if audio is None else str(audio), None if image is None else str(image))
        return body

    def _with_media(
        self, request: VideoRequest, body: dict[str, Any], debug: dict[str, Any]
    ) -> dict[str, Any]:
        """Uploads the local media (audio first) and puts their https URLs into the body."""
        audio, image = self._media(request)
        hosted: list[dict[str, Any]] = []
        audio_url = None if audio is None else self._upload(audio, hosted)
        image_url = self._upload(image, hosted) if isinstance(image, Path) else image
        if hosted:
            debug["media"] = hosted
        body = {k: v for k, v in body.items() if k not in ("input_references", "frame_images")}
        _place(body, audio_url, image_url)
        return body

    def _media(self, request: VideoRequest) -> tuple[Path | None, Path | str | None]:
        """(local audio, local image path or https URL). Raises final before any call on what
        OpenRouter refuses (non-https URL, unknown type) or what we cannot send (no file, no
        media host)."""
        audio = request.audio_path
        if audio is not None:
            self._check_local(audio, jobs.AUDIO_MIME, "audio")
        ref = request.reference_image
        image: Path | str | None = None
        if ref is not None:
            if ref.startswith("https://"):
                image = ref
            elif ref.startswith(("http://", "data:")):
                raise ProviderError(
                    f"{self.name} takes reference images only as https URLs, got "
                    f"{ref.split(',', 1)[0][:40]!r}",
                    retryable=False,
                )
            else:
                image = Path(ref)
                self._check_local(image, jobs.IMAGE_MIME, "image")
        if self.media_host is None and (audio is not None or isinstance(image, Path)):
            raise ProviderError(
                f"{self.name} needs local media at a public https URL but no media host is "
                f"configured (profile option media_host, e.g. litterbox)",
                retryable=False,
            )
        return audio, image

    def _check_local(self, path: Path, mimes: dict[str, str], kind: str) -> None:
        if path.suffix.lower() not in mimes:
            raise ProviderError(
                f"{self.name}: unsupported {kind} type {path.suffix!r}", retryable=False
            )
        if not path.is_file():
            raise ProviderError(f"{self.name}: {kind} file not found: {path}", retryable=False)

    def _upload(self, path: Path, hosted: list[dict[str, Any]]) -> str:
        assert self.media_host is not None  # checked in _media
        media = self.media_host.upload(path)
        if not media.url.startswith("https://"):
            raise ProviderError(
                f"{self.name}: media host gave a non-https URL for {path.name}", retryable=False
            )
        expires = None if media.expires_at is None else media.expires_at.isoformat()
        hosted.append({"file": path.name, "url": media.url, "expires_at": expires})
        return media.url

    def _fingerprint(self, request: VideoRequest, body: dict[str, Any]) -> dict[str, Any]:
        return jobs.fingerprint(
            request,
            model=body["model"],
            duration=body["duration"],
            resolution=body["resolution"],
            ratio=body["aspect_ratio"],
        )

    def _create_url(self) -> str:
        return self._videos_url

    def _job_url(self, job_id: str) -> str:
        return f"{self._videos_url}/{quote(job_id, safe='')}"

    def _video_source(self, job_id: str, job: dict[str, Any], task_file: Path) -> tuple[str, bool]:
        # a completed job is paid for: every refusal below keeps its id, a rerun re-checks it
        signed = _first(job.get("urls"))
        if signed is not None:
            if not signed.lower().startswith("https://"):
                raise self._unusable(job_id, task_file, "a signed URL that is not https", signed)
            return signed, False  # signed: no Authorization header
        unsigned = _first(job.get("unsigned_urls"))
        if unsigned is None:
            raise self._unusable(job_id, task_file, "no urls / unsigned_urls", None)
        url = httpx.URL(BASE_URL).join(unsigned)  # a relative path is on the API host
        if url.scheme != "https" or url.host != API_HOST:
            # the Bearer key goes to the API host only
            raise self._unusable(job_id, task_file, "an unsigned URL off the API host", unsigned)
        return str(url), True

    def _cost(self, job: dict[str, Any], debug: dict[str, Any]) -> float:
        usage = job.get("usage")
        cost = usage.get("cost") if isinstance(usage, dict) else None
        if (
            isinstance(cost, int | float)
            and not isinstance(cost, bool)
            and math.isfinite(cost)
            and cost >= 0
        ):
            return float(cost)
        debug["cost_note"] = (
            f"usage.cost is {cost!r}: reported as 0 USD; check the real charge on "
            f"openrouter.ai/activity"
        )
        return 0.0

    def _unusable(self, job_id: str, task_file: Path, what: str, url: str | None) -> ProviderError:
        shown = "" if url is None else f" ({url.split('?', 1)[0][:100]})"
        return ProviderError(
            f"OpenRouter task {job_id} completed with {what}{shown}; not downloaded. Check it "
            f"on {self.console}; a rerun re-checks this task, delete {task_file} to pay for a "
            f"new one.",
            retryable=False,
        )


def _place(body: dict[str, Any], audio: str | None, image: str | None) -> None:
    """Audio + image both in input_references (live: the only shape whose audio was honoured);
    an image alone as first frame."""
    if audio is not None:
        refs: list[dict[str, Any]] = [{"type": "audio_url", "audio_url": {"url": audio}}]
        if image is not None:
            refs.append({"type": "image_url", "image_url": {"url": image}})
        body["input_references"] = refs
    elif image is not None:
        body["frame_images"] = [
            {"type": "image_url", "image_url": {"url": image}, "frame_type": "first_frame"}
        ]


def _first(urls: object) -> str | None:
    if isinstance(urls, list) and urls and isinstance(urls[0], str) and urls[0]:
        return urls[0]
    return None

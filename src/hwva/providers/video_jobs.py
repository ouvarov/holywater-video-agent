"""Never pay twice for one clip: the job lifecycle shared by the async video APIs (Seedance on
BytePlus ModelArk, Seedance via OpenRouter). A subclass builds the request body and says where
jobs are created, polled and downloaded; everything that decides whether money is spent is here.

<clip>.task.json (written atomically) keeps a fingerprint of the request (hashes, no base64),
the job id ("task_id"), and after the download status "downloaded" with the clip's sha256 and
size:
- the same request with a local clip matching that record returns it: no API call, cost 0;
- otherwise a retry polls the saved job instead of paying for a new one; only a job that is
  gone, ended badly, or whose video URL has expired is replaced by a new job. A job that failed
  on moderation or invalid input is not replaced: its error is re-raised as final;
- a create POST whose outcome is unknown (sent, answer lost / 408 / 5xx / non-JSON / 2xx
  without a job id) is final: the `create_pending` marker saved before the POST stays, and
  every later call for the same request refuses to POST until the human has checked the
  provider's console (see `MaybeCreated`);
- a task file that cannot be parsed is final too: it may hold a pending create or a live job;
- a clip's cost is reported once, by the call that first downloads it.
A succeeded job is reused: delete <clip>.task.json to get a new take of the same request.
Polls are idempotent: transient errors are retried in place (capped backoff + jitter).
Downloads go to <clip>.part and are renamed when complete."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
import re
import time
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import httpx

from hwva.providers.base import Capabilities, ProviderError, VideoRequest, VideoResult

REQUEST_TIMEOUT_S = 60.0
DOWNLOAD_TIMEOUT_S = 300.0
MIN_DURATION_S = 4  # both APIs: integer seconds, 4..max
RETRYABLE_STATUS = {408, 429}  # plus every 5xx
DONE_FAIL = {"failed", "expired", "cancelled"}
URL_EXPIRED = {403, 404}  # presigned video URL past its lifetime
# what a task file says about a possibly paid job; kept under "abandoned" when another request
# reuses the file
_RECORD_KEYS = ("fingerprint", "task_id", "create_pending", "status", "clip", "abandoned")
# a failed job / 4xx whose code or message has one of these is a moderation rejection
# ("face" only as a whole word: "interface timeout" is not a moderation rejection)
_MODERATION_RE = re.compile(
    r"sensitive|moderation|\bfaces?\b|privacy|content[ _-]?policy|safety|\bflagged\b"
)
IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
AUDIO_MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg"}


class VideoJobProvider:
    """Base of a `VideoProvider` over an async create / poll / download API.

    Subclasses set `name` (prefix of every error), `console` (where the human checks for a
    possibly paid job), `succeeded` (the job status meaning "clip ready") and implement
    `_body`, `_fingerprint`, `_create_url`, `_job_url`, `_video_source` and `_cost`.

    `gone_on_404`: a 404 polling a saved job means it is gone, so a new job replaces it. False
    where the poll path is not confirmed live: the 404 is final (the job may be paid).
    `replace_on_expired_url`: a 403/404 downloading a resumed, finished job means its URL has
    expired, so a new job replaces it. False where the poll just returned fresh URLs (a real
    expiry shows as job status "expired"): the 403/404 is final, the job id is kept.
    `pinned`: fingerprint fields set outside the request (env); a saved, possibly paid job
    that differs only in those is refused, not replaced (a resume without the same env would
    pay again). `pinned_hint` tells the human how to set them back.
    `create_timeout_s`: read timeout of the create POST (a lost answer is an unknown outcome,
    final), so it is set generously per API; polls and other calls use REQUEST_TIMEOUT_S.
    `_with_media`: called only when a new job is about to be created, before the
    `create_pending` marker; free work such as uploading media (a resume never repeats it)."""

    name: ClassVar[str]
    console: ClassVar[str]
    succeeded: ClassVar[str]
    gone_on_404: ClassVar[bool] = True
    replace_on_expired_url: ClassVar[bool] = True
    pinned: ClassVar[tuple[str, ...]] = ()
    pinned_hint: ClassVar[str] = ""
    create_timeout_s: ClassVar[float] = REQUEST_TIMEOUT_S
    capabilities: Capabilities

    def __init__(
        self,
        *,
        key: str,
        model: str,
        resolution: str,
        transport: httpx.BaseTransport | None,
        poll_interval_s: float,
        timeout_s: float,
        retry_backoff_s: float,
        retry_backoff_max_s: float,
    ) -> None:
        self.model = model
        self.resolution = resolution
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self.retry_backoff_s = retry_backoff_s
        self.retry_backoff_max_s = retry_backoff_max_s
        self._key = key
        self._client = httpx.Client(transport=transport, timeout=REQUEST_TIMEOUT_S)

    # --- what a subclass provides -------------------------------------------------------------

    def _body(self, request: VideoRequest) -> dict[str, Any]:
        """The JSON create body; validates everything (raises final) before any paid call."""
        raise NotImplementedError

    def _fingerprint(self, request: VideoRequest, body: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _with_media(
        self, request: VideoRequest, body: dict[str, Any], debug: dict[str, Any]
    ) -> dict[str, Any]:
        """The body to POST. Default: `body` as built. Raises before anything is paid."""
        return body

    def _create_url(self) -> str:
        raise NotImplementedError

    def _job_url(self, job_id: str) -> str:
        raise NotImplementedError

    def _video_source(self, job_id: str, job: dict[str, Any], task_file: Path) -> tuple[str, bool]:
        """(URL of the finished clip, whether it needs the Bearer header). Raises final when a
        succeeded job has no usable URL (the job id is kept: it is paid for)."""
        raise NotImplementedError

    def _cost(self, job: dict[str, Any], debug: dict[str, Any]) -> float:
        """USD for a job delivered for the first time; may leave a note in `debug`."""
        raise NotImplementedError

    # --- VideoProvider ------------------------------------------------------------------------

    def generate(self, request: VideoRequest, out_path: Path) -> VideoResult:
        body = self._body(request)  # validates everything before any paid call
        fingerprint = self._fingerprint(request, body)
        debug_path = out_path.with_suffix(".task.json")
        saved = load_saved(debug_path, self.name, self.console) or {}  # unparsable -> final
        abandoned = saved.get("abandoned")
        if saved.get("fingerprint") != fingerprint:
            self._refuse_pinned_drift(saved, fingerprint, debug_path)
            # another request's file: nothing to reuse, but a possibly paid job stays on record
            if "task_id" in saved or "create_pending" in saved:
                abandoned = {k: saved[k] for k in _RECORD_KEYS if k in saved}
            saved = {}
        if clip_matches(out_path, saved):
            return VideoResult(video_path=out_path, last_frame=None, cost_usd=0.0)
        saved_id = saved.get("task_id")
        saved_id = saved_id if isinstance(saved_id, str) and saved_id else None
        if saved_id is None and "create_pending" in saved:
            pending = saved["create_pending"]
            at = pending.get("at") if isinstance(pending, dict) else None
            raise self._maybe_created(debug_path, f"an earlier create at {at} has no known job")
        out_path.unlink(missing_ok=True)  # a stale clip must never look like this job's output
        # "task_id" is in `debug` only while that job may still give us the clip: every
        # failure path that kills the job drops it, so the file never pins a dead job.
        debug: dict[str, Any] = {"request": redact(body), "fingerprint": fingerprint}
        if abandoned:
            debug["abandoned"] = abandoned
        delivered = saved_id is not None and saved.get("status") == "downloaded"
        if delivered:  # its cost was reported when it was first downloaded
            debug["status"], debug["clip"] = saved["status"], saved.get("clip")

        try:
            job = self._resume(saved_id, debug, debug_path, out_path) if saved_id else None
            resumed = job is not None
            if job is None:
                job = self._create(request, body, debug, debug_path, out_path)
            cost = 0.0 if resumed and delivered else self._cost(job, debug)
            debug["cost_usd"] = cost
        finally:
            write_json(debug_path, debug)
        return VideoResult(video_path=out_path, last_frame=None, cost_usd=cost)

    # --- job ----------------------------------------------------------------------------------

    def _resume(
        self, job_id: str, debug: dict[str, Any], debug_path: Path, out_path: Path
    ) -> dict[str, Any] | None:
        """The saved job's clip, or None when a new job is needed (gone, ended badly, video
        URL expired). A 5xx/timeout raises and keeps the saved id for the next attempt; a
        moderation / invalid-input failure raises final (a new job would fail the same way)."""
        debug["task_id"] = job_id
        debug["resumed"] = True
        try:
            job = self._wait(job_id, debug, debug_path)
        except HttpError as exc:
            if exc.status != 404:
                raise
            if not self.gone_on_404:  # task_id stays: a rerun re-checks it
                raise ProviderError(
                    f"{self.name} task {job_id}: HTTP 404 polling it; it may still exist and "
                    f"be paid. Check {self.console}; delete {debug_path} to pay for a new one.",
                    retryable=False,
                ) from exc
            _drop(debug, f"gone: {exc}")
            return None
        status = str(job.get("status"))
        if status == "failed":
            error = self._job_error(job_id, status, job.get("error"))
            if not error.retryable:  # moderation / invalid input: a new job would be rejected too
                debug.pop("task_id")
                raise error
        if status in DONE_FAIL:
            _drop(debug, status)
            return None
        try:
            clip = self._download(*self._video_source(job_id, job, debug_path), out_path)
        except HttpError as exc:
            if exc.status not in URL_EXPIRED:
                raise
            if not self.replace_on_expired_url:  # task_id stays: a rerun re-checks it
                raise ProviderError(
                    f"{self.name} task {job_id} is done but its video download is HTTP "
                    f"{exc.status}; the task is paid. Check {self.console}; a rerun re-checks "
                    f"this task, delete {debug_path} to pay for a new one.",
                    retryable=False,
                ) from exc
            _drop(debug, f"video URL expired: {exc}")
            return None
        debug["status"], debug["clip"] = "downloaded", clip
        return job

    def _create(
        self,
        request: VideoRequest,
        body: dict[str, Any],
        debug: dict[str, Any],
        debug_path: Path,
        out_path: Path,
    ) -> dict[str, Any]:
        # free (e.g. uploads); a failure here leaves no marker: no job can exist yet
        body = self._with_media(request, body, debug)
        debug["request"] = redact(body)
        # on disk before the POST: if its answer is lost, no later call may POST again
        debug["create_pending"] = {
            "fingerprint": debug["fingerprint"],
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        write_json(debug_path, debug)
        try:
            created = self._post_create(body, debug_path)
        except MaybeCreated:
            raise  # the marker stays
        except ProviderError:
            debug.pop("create_pending")  # rejected or never sent: no job exists
            raise
        debug["create"] = created
        job_id = created.get("id")
        if not isinstance(job_id, str) or not job_id:  # accepted, but which job? marker stays
            raise self._maybe_created(debug_path, f"no task id in the answer: {created!r:.300}")
        debug.pop("create_pending")
        debug["task_id"] = job_id
        write_json(debug_path, debug)  # the paid job is on disk before the first poll
        job = self._wait(job_id, debug, debug_path)
        status = str(job.get("status"))
        if status in DONE_FAIL:
            debug.pop("task_id")
            raise self._job_error(job_id, status, job.get("error"))
        clip = self._download(*self._video_source(job_id, job, debug_path), out_path)
        debug["status"], debug["clip"] = "downloaded", clip
        return job

    def _post_create(self, body: dict[str, Any], debug_path: Path) -> dict[str, Any]:
        """The create answer. Retryable only when no job can exist: nothing was sent (connect
        error/timeout) or HTTP 429. Once sent, a lost answer, a 408 (a proxy may have forwarded
        it), a 5xx or a non-JSON body means the job may exist -> `MaybeCreated` (final). Other
        4xx are rejections (final)."""
        try:
            response = self._request(
                "POST", self._create_url(), json=body, timeout=self.create_timeout_s
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ProviderError(
                f"{self.name} create not sent: {type(exc).__name__}: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise self._maybe_created(debug_path, f"{type(exc).__name__}: {exc}") from exc
        status = response.status_code
        ok = 200 <= status < 300
        if status == 408 or not (ok or 400 <= status < 500):  # 408: maybe forwarded
            raise self._maybe_created(debug_path, f"HTTP {status}: {error_reason(response)}")
        if ok:
            try:
                data = response.json()
            except ValueError:
                data = None
            if not isinstance(data, dict):
                raise self._maybe_created(debug_path, f"non-JSON answer: {response.text[:300]}")
        return parse(response, self.name)

    def _refuse_pinned_drift(
        self, saved: dict[str, Any], fingerprint: dict[str, Any], task_file: Path
    ) -> None:
        """Final when the saved, possibly paid job differs from this request only in `pinned`
        fields: it is the same clip asked with another env, not another request."""
        old = saved.get("fingerprint")
        live = "task_id" in saved or "create_pending" in saved
        if not self.pinned or not live or not isinstance(old, dict):
            return
        keys = old.keys() | fingerprint.keys()
        changed = sorted(k for k in keys if old.get(k) != fingerprint.get(k))
        if not changed or not set(changed) <= set(self.pinned):
            return
        what = ", ".join(f"{k} {old.get(k)!r} -> {fingerprint.get(k)!r}" for k in changed)
        raise ProviderError(
            f"{self.name}: {task_file} holds a possibly paid task for this request with other "
            f"settings ({what}). {self.pinned_hint} to resume it, or delete {task_file} to pay "
            f"for a new one.",
            retryable=False,
        )

    def _maybe_created(self, task_file: Path, reason: str) -> MaybeCreated:
        return MaybeCreated(self.name, self.console, task_file, reason)

    def _job_error(self, job_id: str, status: str, error: object) -> ProviderError:
        return job_error(self.name, job_id, status, error)

    # --- HTTP ---------------------------------------------------------------------------------

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        return self._client.request(
            method, url, headers={"Authorization": f"Bearer {self._key}"}, **kwargs
        )

    def _api(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._request(method, url, **kwargs)
        except httpx.RequestError as exc:  # timeouts, connection errors
            raise ProviderError(f"{self.name} request failed: {type(exc).__name__}: {exc}") from exc
        return parse(response, self.name)

    def _wait(self, job_id: str, debug: dict[str, Any], debug_path: Path) -> dict[str, Any]:
        """Polls until the job is done: the final job JSON, succeeded or ended badly. A GET
        changes nothing, so a transient error (retryable) is retried here until the deadline."""
        deadline = time.monotonic() + self.timeout_s
        polls = errors = 0
        while True:
            try:
                job = self._api("GET", self._job_url(job_id))
            except ProviderError as exc:
                if not exc.retryable:  # 404 (the resume tells "gone"), moderation, 4xx
                    raise
                errors += 1
                debug["poll_errors"] = errors
                write_json(debug_path, debug)  # visible during a long outage too
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProviderError(
                        f"{self.name} task {job_id}: polling failed until the {self.timeout_s:g} s "
                        f"deadline ({exc}); a rerun resumes this task instead of creating a new one"
                    ) from exc
                time.sleep(min(remaining, self._backoff(errors)))
                continue
            errors = 0
            polls += 1
            debug["poll"] = job
            debug["polls"] = polls
            write_json(debug_path, debug)  # visible while a long job is still running
            status = job.get("status")
            if status == self.succeeded or status in DONE_FAIL:
                return job
            if time.monotonic() >= deadline:
                raise ProviderError(
                    f"{self.name} task {job_id} still {status!r} after {self.timeout_s:g} s "
                    f"({polls} polls); a rerun resumes this task instead of creating a new one"
                )
            time.sleep(self.poll_interval_s)

    def _backoff(self, errors: int) -> float:
        """Capped exponential wait after `errors` transient errors in a row, with jitter."""
        delay = min(self.retry_backoff_max_s, self.retry_backoff_s * 2 ** min(errors - 1, 30))
        return random.uniform(delay / 2, delay)

    def _download(self, url: str, auth: bool, out_path: Path) -> dict[str, Any]:
        """The clip's sha256 and size. A presigned URL (`auth` False) lives on another host:
        no Authorization header goes there. Redirects are followed (e.g. API -> storage); httpx
        drops the Authorization header when a redirect leaves the origin."""
        part = out_path.with_name(out_path.name + ".part")
        digest = hashlib.sha256()
        headers = {"Authorization": f"Bearer {self._key}"} if auth else None
        try:
            with self._client.stream(
                "GET", url, headers=headers, timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True
            ) as response:
                if response.status_code != 200:
                    status = response.status_code
                    retryable = status in RETRYABLE_STATUS or status >= 500
                    raise HttpError(
                        f"{self.name} video download HTTP {status}", status, retryable=retryable
                    )
                with part.open("wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                        digest.update(chunk)
        except httpx.RequestError as exc:
            part.unlink(missing_ok=True)
            raise ProviderError(
                f"{self.name} video download failed: {type(exc).__name__}: {exc}"
            ) from exc
        except ProviderError:
            part.unlink(missing_ok=True)
            raise
        size = part.stat().st_size
        if size == 0:
            part.unlink()
            raise ProviderError(f"{self.name} video download is 0 bytes")
        part.replace(out_path)
        return {"sha256": digest.hexdigest(), "size": size}


# --- request helpers --------------------------------------------------------------------------


def checked_duration(name: str, model: str, caps: Capabilities, request: VideoRequest) -> int:
    """The integer duration to send (>= 4 s, covers the whole audio). Raises final, before any
    paid call, on what the model can't take: aspect ratio, audio without a needed image, audio
    out of range, clip too long."""
    if request.aspect_ratio not in caps.aspect_ratios:
        raise ProviderError(
            f"{name} does not support aspect ratio {request.aspect_ratio!r}", retryable=False
        )
    duration = max(MIN_DURATION_S, math.ceil(request.duration_s))
    if request.audio_path is not None:
        if caps.needs_image and request.reference_image is None:
            raise ProviderError(
                f"{name} {model} needs a reference image together with audio", retryable=False
            )
        audio_s = wav_seconds(request.audio_path, name)
        if audio_s is not None:
            if not caps.min_audio_s <= audio_s <= caps.max_duration_s:
                raise ProviderError(
                    f"{name} reference audio must be {caps.min_audio_s:g}-"
                    f"{caps.max_duration_s:g} s, got {audio_s:.2f} s",
                    retryable=False,
                )
            duration = max(duration, math.ceil(audio_s))
    if duration > caps.max_duration_s:
        raise ProviderError(
            f"{name} {model} clips are at most {caps.max_duration_s:g} s, asked for {duration} s",
            retryable=False,
        )
    return duration


def fingerprint(
    request: VideoRequest, *, model: str, duration: int, resolution: str, ratio: str
) -> dict[str, Any]:
    """What makes two requests the same job: what is sent, with the media as sha256 only."""
    audio = None if request.audio_path is None else sha256(request.audio_path.read_bytes())
    image = request.reference_image
    if image is not None:
        local = not image.startswith(("http://", "https://", "data:"))
        image = sha256(Path(image).read_bytes() if local else image.encode("utf-8"))
    return {
        "prompt": request.prompt,
        "audio_sha256": audio,
        "image_sha256": image,
        "duration": duration,
        "resolution": resolution,
        "model": model,
        "ratio": ratio,
    }


def wav_seconds(path: Path, name: str) -> float | None:
    """Duration of a WAV file; None if it isn't one (then the API is the judge)."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, EOFError):
        return None
    except (OSError, ZeroDivisionError) as exc:  # missing/unreadable file, 0 Hz header
        raise ProviderError(f"{name}: cannot read audio {path}: {exc}", retryable=False) from exc


def data_url(path: Path, mimes: dict[str, str], kind: str, name: str) -> str:
    mime = mimes.get(path.suffix.lower())
    if mime is None:
        raise ProviderError(f"{name}: unsupported {kind} type {path.suffix!r}", retryable=False)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ProviderError(f"{name}: cannot read {kind} {path}: {exc}", retryable=False) from exc
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def image_url(ref: str, name: str) -> str:
    if ref.startswith(("http://", "https://", "data:")):
        return ref
    return data_url(Path(ref), IMAGE_MIME, "image", name)


def redact(value: Any) -> Any:
    """Copy for the debug file with every (large) base64 data URL shortened to its type."""
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str) and value.startswith("data:"):
        return f"{value.split(',', 1)[0]},<{len(value)} chars>"
    return value


# --- answer helpers ---------------------------------------------------------------------------


class HttpError(ProviderError):
    """A non-2xx answer; `status` lets the resume tell "job gone / URL expired" apart."""

    def __init__(self, message: str, status: int, *, retryable: bool) -> None:
        super().__init__(message, retryable=retryable)
        self.status = status


class MaybeCreated(ProviderError):
    """A create POST was sent but its outcome is unknown: a paid job may exist. Final — the
    APIs have no idempotency key, so only the human can tell (provider console)."""

    def __init__(self, name: str, console: str, task_file: Path, reason: str) -> None:
        super().__init__(
            f"{name} create outcome unknown ({reason[:300]}): a paid task may exist. Check "
            f"{console} before rerunning. If the task is there, put its id as "
            f'"task_id" into {task_file} to resume it; if not, delete {task_file} to allow a '
            f"new paid task.",
            retryable=False,
        )


def parse(response: httpx.Response, name: str) -> dict[str, Any]:
    """A 2xx JSON object, else the mapped error (moderation / retryable / final)."""
    status = response.status_code
    if not 200 <= status < 300:
        reason = error_reason(response)
        # moderation is judged on code + message only: metadata (e.g. a raw upstream error)
        # mentioning "safety" must not turn a retryable 429 final
        if status < 500 and is_moderation(error_reason(response, metadata=False)):
            raise ProviderError(
                f"{name} HTTP {status}, rejected by moderation: {reason}", retryable=False
            )
        retryable = status in RETRYABLE_STATUS or status >= 500
        raise HttpError(f"{name} HTTP {status}: {reason}", status, retryable=retryable)
    try:
        data = response.json()
    except ValueError as exc:
        raise ProviderError(f"{name} returned non-JSON: {response.text[:300]}") from exc
    if not isinstance(data, dict):
        raise ProviderError(f"{name} returned unexpected JSON: {data!r:.300}")
    return data


def error_reason(response: httpx.Response, *, metadata: bool = True) -> str:
    """`code: message` of an {"error": {...}} envelope (+ its metadata, e.g. the upstream
    provider's raw error, unless `metadata` is False), else the start of the body."""
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        error = None
    if isinstance(error, dict):
        reason = f"{error.get('code', '')}: {error.get('message', '')}".strip(": ")
        extra = error.get("metadata")
        if metadata and extra:
            reason += f" ({json.dumps(extra, ensure_ascii=False)[:200]})"
        return reason[:300]
    return response.text[:300]


def job_error(name: str, job_id: str, status: str, error: object) -> ProviderError:
    if isinstance(error, dict):
        reason = f"{error.get('code', '')}: {error.get('message', '')}".strip(": ")
    else:
        reason = str(error or "")
    reason = reason or "no reason given"
    if is_moderation(reason):
        return ProviderError(
            f"{name} task {job_id} {status}, rejected by moderation: {reason}", retryable=False
        )
    invalid = "invalid" in reason.lower() or status == "cancelled"
    return ProviderError(f"{name} task {job_id} {status}: {reason}", retryable=not invalid)


def is_moderation(reason: str) -> bool:
    return _MODERATION_RE.search(reason.lower()) is not None


# --- task file --------------------------------------------------------------------------------


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_saved(path: Path, name: str, console: str) -> dict[str, Any] | None:
    """The saved task file, None if there is none. One that exists but can't be parsed may hold
    a pending create or a live job id: reading it as "no job" could pay twice, so it is final."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        saved = None
    if not isinstance(saved, dict):
        raise ProviderError(
            f"{name}: cannot parse {path}; it may record a paid task (pending create or task "
            f"id). Check it and {console}, then fix or delete the file and rerun.",
            retryable=False,
        )
    return saved


def clip_matches(out_path: Path, saved: dict[str, Any]) -> bool:
    """The local clip is the one this task file recorded as downloaded."""
    clip = saved.get("clip")
    if saved.get("status") != "downloaded" or not isinstance(clip, dict):
        return False
    try:
        if out_path.stat().st_size != clip.get("size"):
            return False
        return sha256(out_path.read_bytes()) == clip.get("sha256")
    except OSError:
        return False


def _drop(debug: dict[str, Any], reason: str) -> None:
    """The saved job cannot give the clip: forget it (a new job replaces it)."""
    debug["replaced"] = {"task_id": debug.pop("task_id"), "reason": reason[:300]}
    for key in ("resumed", "status", "clip"):
        debug.pop(key, None)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic (like manifest.save): a crash never leaves a torn task file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)

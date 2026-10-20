"""SeedanceVideo with the HTTP layer mocked (httpx.MockTransport, no network). One opt-in live test.

No ARK key existed when this was written: request/response shapes follow the BytePlus ModelArk
docs (task create + poll), so the live smoke test is the first real check."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import wave
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

from hwva.providers import registry, seedance
from hwva.providers.base import ProviderError, VideoProvider, VideoRequest
from hwva.schemas import ModelRef
from hwva.stages.shot_2 import MODERATION_MARKERS

ROOT = Path(__file__).resolve().parents[2]
REF = ModelRef(provider="seedance", model="dreamina-seedance-2-5-260628", version="2.5")
REF_20 = ModelRef(provider="seedance", model="seedance-2-0-test-id", version="2.0")
KEY = "test-ark-key-not-real-0123456789"
TASKS = seedance.DEFAULT_BASE_URL + seedance.TASKS_PATH
VIDEO_URL = "https://tos.example.com/clip.mp4?X-Tos-Signature=abc"
CLIP = b"\x00\x00\x00\x18ftypmp42 fake mp4 bytes"


def wav(seconds: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(seconds * rate))
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(seedance.API_KEY_ENV, KEY)
    monkeypatch.delenv(seedance.BASE_URL_ENV, raising=False)


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "line.wav"
    path.write_bytes(wav(3.2))
    return path


@pytest.fixture
def frame(tmp_path: Path) -> Path:
    path = tmp_path / "shot_1.last.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake png")
    return path


def task(status: str, **extra: object) -> httpx.Response:
    return httpx.Response(200, json={"id": "cgt-1", "model": REF.model, "status": status, **extra})


def done(**extra: object) -> httpx.Response:
    return task("succeeded", content={"video_url": VIDEO_URL}, **extra)


class Api:
    """MockTransport handler: POST -> create answer, GET task -> next poll answer, GET video ->
    the clip. Every request is recorded.

    Without `create`, the n-th POST answers {"id": "cgt-<n>"} (the first one is cgt-1), so a
    second paid task is distinguishable from the first. `creates` maps a POST number (1-based)
    to a special answer for that POST only. `tasks` gives per-task-id poll queues; a task id not
    in it uses `polls`. A queue's last answer repeats. `downloads` maps a video URL to its
    answer; other URLs get `download`."""

    def __init__(
        self,
        polls: list[httpx.Response] | None = None,
        *,
        create: httpx.Response | Exception | None = None,
        download: httpx.Response | Exception | None = None,
        tasks: dict[str, list[httpx.Response | Exception]] | None = None,
        creates: dict[int, httpx.Response | Exception] | None = None,
        downloads: dict[str, httpx.Response | Exception] | None = None,
    ) -> None:
        self.create = create
        self.polls: list[httpx.Response | Exception] = [*(polls if polls is not None else [done()])]
        self.download = download if download is not None else httpx.Response(200, content=CLIP)
        self.tasks = tasks if tasks is not None else {}
        self.creates = creates if creates is not None else {}
        self.downloads = downloads if downloads is not None else {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        url = str(request.url)
        answer: httpx.Response | Exception
        if request.method == "POST":
            n = len(self.posts())
            if n in self.creates:
                answer = self.creates[n]
            elif self.create is not None:
                answer = self.create
            else:
                answer = httpx.Response(200, json={"id": f"cgt-{n}"})
        elif url.startswith(TASKS):
            queue = self.tasks.get(url.removeprefix(TASKS + "/"), self.polls)
            answer = queue.pop(0) if len(queue) > 1 else queue[0]
        else:
            answer = self.downloads.get(url, self.download)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def posts(self, since: int = 0) -> list[httpx.Request]:
        return [r for r in self.requests[since:] if r.method == "POST"]

    def polled(self, since: int = 0) -> list[str]:
        """Task ids GET-polled from request number `since` on, in order."""
        prefix = TASKS + "/"
        return [
            str(r.url).removeprefix(prefix)
            for r in self.requests[since:]
            if r.method == "GET" and str(r.url).startswith(prefix)
        ]

    def body(self) -> dict[str, object]:
        posts = self.posts()
        assert len(posts) == 1
        data: dict[str, object] = json.loads(posts[0].content)
        return data

    def content(self) -> list[dict[str, object]]:
        items = self.body()["content"]
        assert isinstance(items, list)
        return items


def video(api: Api, ref: ModelRef = REF, **kwargs: object) -> seedance.SeedanceVideo:
    params: dict[str, object] = {"poll_interval_s": 0.0, "timeout_s": 5.0, **kwargs}
    return seedance.SeedanceVideo(ref, transport=httpx.MockTransport(api), **params)  # type: ignore[arg-type]


def req(audio: Path | None = None, image: str | None = None, seconds: float = 5.0) -> VideoRequest:
    return VideoRequest(
        prompt="A woman on a rooftop", duration_s=seconds, audio_path=audio, reference_image=image
    )


# --- contract & capabilities --------------------------------------------------------------------


def test_is_a_video_provider() -> None:
    assert isinstance(video(Api()), VideoProvider)


def test_version_25_takes_audio_without_an_image_up_to_30_s() -> None:
    caps = video(Api()).capabilities
    assert caps.accepts_audio_input and not caps.needs_image
    assert caps.max_duration_s == 30
    assert caps.min_audio_s == 2.0
    assert "9:16" in caps.aspect_ratios


def test_version_20_needs_an_image_with_audio_up_to_15_s() -> None:
    caps = video(Api(), REF_20).capabilities
    assert caps.accepts_audio_input and caps.needs_image
    assert caps.max_duration_s == 15


def test_unknown_version_is_rejected_at_build() -> None:
    with pytest.raises(ValueError, match="version"):
        video(Api(), ModelRef(provider="seedance", model="x", version="1.0"))


def test_unknown_resolution_is_rejected_at_build() -> None:
    with pytest.raises(ValueError, match="resolution"):
        video(Api(), resolution="4k")


def test_missing_key_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(seedance.API_KEY_ENV)
    with pytest.raises(ValueError, match=seedance.API_KEY_ENV):
        seedance.SeedanceVideo(REF)


def test_registry_builds_seedance_from_the_ref() -> None:
    provider = registry._FACTORIES["video"]["seedance"](REF)
    assert isinstance(provider, seedance.SeedanceVideo)
    assert provider.model == REF.model


def test_premium_profile_pins_model_and_version() -> None:
    ref = registry.load_profile("premium", ROOT / "config" / "profiles.yaml")["video"]
    assert ref == REF


def test_model_id_lives_in_the_profile_not_in_code() -> None:
    assert REF.model not in (ROOT / "src/hwva/providers/seedance.py").read_text(encoding="utf-8")


# --- request ------------------------------------------------------------------------------------


def test_create_is_a_bearer_post_to_the_tasks_endpoint(audio: Path, tmp_path: Path) -> None:
    api = Api()
    video(api).generate(req(audio), tmp_path / "shot_1.mp4")
    post = api.requests[0]
    assert post.method == "POST"
    assert str(post.url) == TASKS
    assert post.headers["authorization"] == f"Bearer {KEY}"


def test_base_url_comes_from_env(
    audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(seedance.BASE_URL_ENV, "https://ark.example.com/")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-1"})
        if "tasks" in str(request.url):
            return done()
        return httpx.Response(200, content=CLIP)

    provider = seedance.SeedanceVideo(REF, transport=httpx.MockTransport(handler))
    provider.generate(req(audio), tmp_path / "shot_1.mp4")
    assert seen[0] == "https://ark.example.com/api/v3/contents/generations/tasks"
    assert seen[1] == "https://ark.example.com/api/v3/contents/generations/tasks/cgt-1"


@pytest.mark.parametrize("base", ["http://ark.example.com", "ark.example.com", "ftp://x"])
def test_base_url_must_be_https(monkeypatch: pytest.MonkeyPatch, base: str) -> None:
    # the Bearer key must never go over plain HTTP
    monkeypatch.setenv(seedance.BASE_URL_ENV, base)
    with pytest.raises(ProviderError, match=seedance.BASE_URL_ENV) as err:
        video(Api())
    assert not err.value.retryable
    assert KEY not in str(err.value)


def test_video_url_must_be_https(audio: Path, tmp_path: Path) -> None:
    plain = "http://tos.example.com/clip.mp4?X-Tos-Signature=abc"
    api = Api([task("succeeded", content={"video_url": plain})])
    out = tmp_path / "shot_1.mp4"
    with pytest.raises(ProviderError, match="https") as err:
        video(api).generate(req(audio), out)
    assert not err.value.retryable
    assert plain not in [str(r.url) for r in api.requests]  # nothing downloaded
    assert not out.exists()


def test_empty_base_url_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(seedance.BASE_URL_ENV, "")
    assert video(Api())._tasks_url == TASKS


def test_body_pins_model_ratio_resolution_and_audio_flag(audio: Path, tmp_path: Path) -> None:
    api = Api()
    video(api, resolution="1080p").generate(req(audio), tmp_path / "shot_1.mp4")
    body = api.body()
    assert body["model"] == REF.model
    assert body["ratio"] == "9:16"
    assert body["resolution"] == "1080p"
    assert body["generate_audio"] is True
    assert "return_last_frame" not in body


def test_audio_goes_as_reference_audio_data_url_with_the_exact_bytes(
    audio: Path, tmp_path: Path
) -> None:
    api = Api()
    video(api).generate(req(audio), tmp_path / "shot_1.mp4")
    content = api.content()
    assert content[0] == {"type": "text", "text": "A woman on a rooftop"}
    (item,) = [c for c in content if c["type"] == "audio_url"]
    assert item["role"] == "reference_audio"
    url = item["audio_url"]["url"]  # type: ignore[index]
    prefix, payload = url.split(",", 1)
    assert prefix == "data:audio/wav;base64"
    assert base64.b64decode(payload) == audio.read_bytes()


def test_local_frame_goes_as_reference_image_never_first_frame(
    audio: Path, frame: Path, tmp_path: Path
) -> None:
    api = Api()
    video(api).generate(req(audio, str(frame)), tmp_path / "shot_2.mp4")
    (item,) = [c for c in api.content() if c["type"] == "image_url"]
    assert item["role"] == "reference_image"
    url = item["image_url"]["url"]  # type: ignore[index]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == frame.read_bytes()
    assert "first_frame" not in json.dumps(api.body())


def test_image_url_is_passed_through(tmp_path: Path) -> None:
    api = Api()
    video(api).generate(req(image="https://cdn.example.com/f.png"), tmp_path / "shot_2.mp4")
    (item,) = [c for c in api.content() if c["type"] == "image_url"]
    assert item["image_url"] == {"url": "https://cdn.example.com/f.png"}


def test_silent_shot_sends_no_audio(frame: Path, tmp_path: Path) -> None:
    api = Api()
    video(api).generate(req(image=str(frame)), tmp_path / "shot_2.mp4")
    assert [c["type"] for c in api.content()] == ["text", "image_url"]


@pytest.mark.parametrize(("asked", "sent"), [(1.0, 4), (4.0, 4), (5.2, 6), (7.0, 7), (29.5, 30)])
def test_duration_is_an_integer_rounded_up_at_least_four(
    asked: float, sent: int, tmp_path: Path
) -> None:
    api = Api()
    video(api).generate(req(seconds=asked), tmp_path / "c.mp4")
    assert api.body()["duration"] == sent


def test_duration_covers_the_whole_audio(tmp_path: Path) -> None:
    path = tmp_path / "line.wav"
    path.write_bytes(wav(6.3))
    api = Api()
    video(api).generate(req(path, seconds=5.0), tmp_path / "c.mp4")
    assert api.body()["duration"] == 7


# --- rejected before any paid call --------------------------------------------------------------


def _rejected_without_calls(api: Api, call: object) -> ProviderError:
    with pytest.raises(ProviderError) as info:
        call()  # type: ignore[operator]
    assert not info.value.retryable
    assert api.requests == []
    return info.value


def test_version_20_with_audio_and_no_image_fails_before_spend(audio: Path, tmp_path: Path) -> None:
    api = Api()
    err = _rejected_without_calls(
        api, lambda: video(api, REF_20).generate(req(audio), tmp_path / "c.mp4")
    )
    assert "image" in str(err)


def test_version_20_with_audio_and_image_is_sent(audio: Path, frame: Path, tmp_path: Path) -> None:
    api = Api()
    video(api, REF_20).generate(req(audio, str(frame)), tmp_path / "c.mp4")
    assert {c["type"] for c in api.content()} == {"text", "audio_url", "image_url"}
    assert api.body()["model"] == REF_20.model


@pytest.mark.parametrize(("ref", "seconds"), [(REF, 31.0), (REF_20, 16.0)])
def test_too_long_clip_fails_before_spend(ref: ModelRef, seconds: float, tmp_path: Path) -> None:
    api = Api()
    _rejected_without_calls(
        api, lambda: video(api, ref).generate(req(seconds=seconds), tmp_path / "c.mp4")
    )


@pytest.mark.parametrize("seconds", [1.5, 30.5])
def test_audio_out_of_2_to_30_s_fails_before_spend(seconds: float, tmp_path: Path) -> None:
    path = tmp_path / "line.wav"
    path.write_bytes(wav(seconds))
    api = Api()
    err = _rejected_without_calls(api, lambda: video(api).generate(req(path), tmp_path / "c.mp4"))
    assert "audio" in str(err)


@pytest.mark.parametrize("case", ["missing", "zero_hz"])
def test_missing_or_broken_audio_file_fails_before_spend(case: str, tmp_path: Path) -> None:
    path = tmp_path / "line.wav"
    if case == "zero_hz":
        raw = bytearray(wav(3.0))
        raw[24:28] = (0).to_bytes(4, "little")  # WAV header sample-rate field -> 0 Hz
        path.write_bytes(bytes(raw))
    api = Api()
    err = _rejected_without_calls(api, lambda: video(api).generate(req(path), tmp_path / "c.mp4"))
    assert "audio" in str(err)


def test_unsupported_aspect_ratio_fails_before_spend(tmp_path: Path) -> None:
    api = Api()
    request = VideoRequest(prompt="p", duration_s=5, aspect_ratio="2:7")
    _rejected_without_calls(api, lambda: video(api).generate(request, tmp_path / "c.mp4"))


def test_missing_frame_file_fails_before_spend(tmp_path: Path) -> None:
    api = Api()
    missing = str(tmp_path / "nope.png")
    _rejected_without_calls(
        api, lambda: video(api).generate(req(image=missing), tmp_path / "c.mp4")
    )


# --- polling & download -------------------------------------------------------------------------


def test_polls_until_succeeded_then_downloads_the_clip(audio: Path, tmp_path: Path) -> None:
    api = Api([task("queued"), task("running"), task("running"), done()])
    out = tmp_path / "shot_1.mp4"
    result = video(api).generate(req(audio), out)
    assert result.video_path == out
    assert out.read_bytes() == CLIP
    assert result.last_frame is None  # shot_1 extracts the frame locally
    gets = [str(r.url) for r in api.requests if r.method == "GET"]
    assert gets == [f"{TASKS}/cgt-1"] * 4 + [VIDEO_URL]
    assert not list(tmp_path.glob("*.part"))


def test_polls_carry_the_key_but_the_presigned_download_does_not(
    audio: Path, tmp_path: Path
) -> None:
    api = Api()
    video(api).generate(req(audio), tmp_path / "shot_1.mp4")
    poll, download = api.requests[1], api.requests[2]
    assert poll.headers["authorization"] == f"Bearer {KEY}"
    assert "authorization" not in download.headers


def test_stale_clip_is_removed_even_when_the_task_fails(audio: Path, tmp_path: Path) -> None:
    out = tmp_path / "shot_1.mp4"
    out.write_bytes(b"old clip")
    api = Api([task("failed", error={"code": "InternalError", "message": "boom"})])
    with pytest.raises(ProviderError):
        video(api).generate(req(audio), out)
    assert not out.exists()


def test_poll_timeout_is_retryable_and_names_the_task(audio: Path, tmp_path: Path) -> None:
    api = Api([task("running")])
    with pytest.raises(ProviderError) as info:
        video(api, timeout_s=0.0).generate(req(audio), tmp_path / "c.mp4")
    assert info.value.retryable
    assert "cgt-1" in str(info.value)


def test_success_without_video_url_is_a_provider_error(audio: Path, tmp_path: Path) -> None:
    api = Api([task("succeeded", content={})])
    with pytest.raises(ProviderError, match="video_url"):
        video(api).generate(req(audio), tmp_path / "c.mp4")


def test_create_without_task_id_is_a_provider_error(audio: Path, tmp_path: Path) -> None:
    api = Api(create=httpx.Response(200, json={"error": None}))
    with pytest.raises(ProviderError, match="task id"):
        video(api).generate(req(audio), tmp_path / "c.mp4")


def test_empty_download_is_a_provider_error(audio: Path, tmp_path: Path) -> None:
    api = Api(download=httpx.Response(200, content=b""))
    out = tmp_path / "c.mp4"
    with pytest.raises(ProviderError, match="0 bytes"):
        video(api).generate(req(audio), out)
    assert not out.exists()
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize(("status", "retryable"), [(403, False), (404, False), (503, True)])
def test_download_http_errors(status: int, retryable: bool, audio: Path, tmp_path: Path) -> None:
    api = Api(download=httpx.Response(status))
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), tmp_path / "c.mp4")
    assert info.value.retryable is retryable


def test_download_timeout_is_retryable(audio: Path, tmp_path: Path) -> None:
    api = Api(download=httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), tmp_path / "c.mp4")
    assert info.value.retryable


# --- debug file ---------------------------------------------------------------------------------


def test_raw_create_and_poll_json_saved_next_to_the_clip(audio: Path, tmp_path: Path) -> None:
    api = Api([task("running"), done(usage={"completion_tokens": 1234})])
    video(api).generate(req(audio), tmp_path / "shot_1.mp4")
    debug = json.loads((tmp_path / "shot_1.task.json").read_text(encoding="utf-8"))
    assert debug["create"] == {"id": "cgt-1"}
    assert debug["poll"]["status"] == "succeeded"
    assert debug["poll"]["usage"] == {"completion_tokens": 1234}
    assert debug["polls"] == 2
    assert debug["request"]["model"] == REF.model


def test_debug_file_has_no_key_and_no_base64_payload(
    audio: Path, frame: Path, tmp_path: Path
) -> None:
    api = Api()
    video(api).generate(req(audio, str(frame)), tmp_path / "shot_1.mp4")
    text = (tmp_path / "shot_1.task.json").read_text(encoding="utf-8")
    assert KEY not in text
    assert base64.b64encode(audio.read_bytes()).decode()[:40] not in text
    assert "data:audio/wav;base64," in text  # the type stays visible


def test_debug_file_written_when_the_task_fails(audio: Path, tmp_path: Path) -> None:
    api = Api([task("failed", error={"code": "InternalError", "message": "boom"})])
    with pytest.raises(ProviderError):
        video(api).generate(req(audio), tmp_path / "shot_1.mp4")
    debug = json.loads((tmp_path / "shot_1.task.json").read_text(encoding="utf-8"))
    assert debug["poll"]["error"]["code"] == "InternalError"


# --- failed tasks -------------------------------------------------------------------------------

MODERATION_ERRORS = [
    {"code": "InputImageSensitiveContentDetected.PrivacyInformation", "message": "real person"},
    {"code": "OutputVideoSensitiveContentDetected", "message": "may contain sensitive information"},
    {"code": "InputImageSensitiveContentDetected", "message": "face detected in the input"},
    {"code": "ContentModeration", "message": "rejected"},
]


@pytest.mark.parametrize("error", MODERATION_ERRORS)
def test_moderation_failure_is_final_keeps_the_reason_and_shot_2_recognises_it(
    error: dict[str, str], frame: Path, tmp_path: Path
) -> None:
    api = Api([task("failed", error=error)])
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(image=str(frame)), tmp_path / "shot_2.mp4")
    message = str(info.value)
    assert not info.value.retryable
    assert error["code"] in message and error["message"] in message
    assert any(marker in message.lower() for marker in MODERATION_MARKERS)


def test_moderation_rejection_at_create_is_final_and_recognised(
    frame: Path, tmp_path: Path
) -> None:
    body = {"error": {"code": "InputImageSensitiveContentDetected", "message": "face"}}
    api = Api(create=httpx.Response(400, json=body))
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(image=str(frame)), tmp_path / "shot_2.mp4")
    assert not info.value.retryable
    assert "InputImageSensitiveContentDetected" in str(info.value)
    assert any(marker in str(info.value).lower() for marker in MODERATION_MARKERS)


@pytest.mark.parametrize(
    ("status", "error", "retryable"),
    [
        ("failed", {"code": "InternalServiceError", "message": "try later"}, True),
        ("failed", None, True),
        ("expired", None, True),
        ("failed", {"code": "InvalidParameter", "message": "duration"}, False),
        ("cancelled", None, False),
        ("failed", {"code": "InternalServiceError", "message": "upstream interface timeout"}, True),
        ("failed", {"code": "InternalServiceError", "message": "surface render crashed"}, True),
    ],
)
def test_other_terminal_states(
    status: str, error: object, retryable: bool, audio: Path, tmp_path: Path
) -> None:
    api = Api([task(status, error=error)])
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), tmp_path / "c.mp4")
    assert info.value.retryable is retryable
    assert status in str(info.value)
    assert "moderation" not in str(info.value)


# --- HTTP errors --------------------------------------------------------------------------------


def test_rate_limit_on_create_is_retryable(audio: Path, tmp_path: Path) -> None:
    """429 on create: the task was not created, a retry may POST again. (5xx on create is an
    unknown outcome, see test_unknown_create_outcome_is_final_and_says_check_the_console.)"""
    api = Api(create=httpx.Response(429, json={"error": {"code": "X", "message": "busy"}}))
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), tmp_path / "c.mp4")
    assert info.value.retryable
    assert "429" in str(info.value)


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_rate_limit_and_server_errors_on_a_poll_are_retryable(
    status: int, audio: Path, tmp_path: Path
) -> None:
    api = Api([httpx.Response(status, json={"error": {"code": "X", "message": "busy"}})])
    with pytest.raises(ProviderError) as info:
        video(api, timeout_s=0.0).generate(req(audio), tmp_path / "c.mp4")
    assert info.value.retryable
    assert str(status) in str(info.value)


@pytest.mark.parametrize("status", [400, 401, 402, 403])
def test_client_errors_are_not_retryable(status: int, audio: Path, tmp_path: Path) -> None:
    body = {"error": {"code": "AuthenticationError", "message": "bad request or key"}}
    api = Api(create=httpx.Response(status, json=body))
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), tmp_path / "c.mp4")
    assert not info.value.retryable
    assert "AuthenticationError" in str(info.value)
    assert KEY not in str(info.value)


def test_poll_http_error_is_mapped_too(audio: Path, tmp_path: Path) -> None:
    """A poll 5xx that lasts until the deadline (0 s here) -> retryable."""
    api = Api([httpx.Response(503, text="unavailable")])
    with pytest.raises(ProviderError) as info:
        video(api, timeout_s=0.0).generate(req(audio), tmp_path / "c.mp4")
    assert info.value.retryable


@pytest.mark.parametrize("exc", [httpx.ConnectTimeout("t"), httpx.ConnectError("c")])
def test_create_connect_errors_are_retryable(exc: Exception, audio: Path, tmp_path: Path) -> None:
    """Nothing was sent: no task can exist, a retry may POST again."""
    api = Api(create=exc)
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), tmp_path / "c.mp4")
    assert info.value.retryable


def test_non_json_answer_is_a_provider_error(audio: Path, tmp_path: Path) -> None:
    api = Api(create=httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(ProviderError, match="non-JSON"):
        video(api).generate(req(audio), tmp_path / "c.mp4")


# --- resume of a saved task (no second paid task on retry) --------------------------------------
#
# A saved task is seeded the way the pipeline produces it: a first `generate` whose poll times
# out (timeout_s=0.0) leaves <clip>.task.json behind. The file format is never hand-written,
# except for the "unreadable" case.


def seed(api: Api, request: VideoRequest, out: Path, **kwargs: object) -> None:
    """First attempt: the task is created, polled once, still running -> retryable timeout."""
    with pytest.raises(ProviderError) as info:
        video(api, REF, **{**kwargs, "timeout_s": 0.0}).generate(request, out)
    assert info.value.retryable
    assert len(api.posts()) == 1


def test_retry_after_poll_timeout_resumes_the_task_instead_of_paying_again(
    audio: Path, tmp_path: Path
) -> None:
    """Main scenario: one Api, one out_path; attempt 1 times out, attempt 2 (same request)
    polls cgt-1, which has now succeeded, downloads it, and never POSTs."""
    api = Api(tasks={"cgt-1": [task("running"), done()]})
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    result = video(api).generate(req(audio), out)

    assert len(api.posts()) == 1  # exactly one paid task in total
    assert api.posts(mark) == []
    assert api.polled(mark) == ["cgt-1"]
    assert str(api.requests[-1].url) == VIDEO_URL
    assert result.video_path == out
    assert out.read_bytes() == CLIP
    text = out.with_suffix(".task.json").read_text(encoding="utf-8")
    assert base64.b64encode(audio.read_bytes()).decode()[:40] not in text
    assert KEY not in text


def test_resumed_queued_or_running_task_is_polled_with_the_normal_timeout(
    audio: Path, tmp_path: Path
) -> None:
    api = Api(tasks={"cgt-1": [task("running"), task("queued"), task("running"), done()]})
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    video(api).generate(req(audio), out)

    assert api.posts(mark) == []
    assert api.polled(mark) == ["cgt-1"] * 3
    assert out.read_bytes() == CLIP


@pytest.mark.parametrize("image", ["local", "https"])
def test_resume_also_works_with_a_reference_image(
    image: str, audio: Path, frame: Path, tmp_path: Path
) -> None:
    ref = str(frame) if image == "local" else "https://cdn.example.com/f.png"
    api = Api(tasks={"cgt-1": [task("running"), done()]})
    out = tmp_path / "shot_2.mp4"
    seed(api, req(audio, ref), out)
    mark = len(api.requests)

    video(api).generate(req(audio, ref), out)

    assert api.posts(mark) == []
    assert api.polled(mark) == ["cgt-1"]
    assert out.read_bytes() == CLIP


def test_same_sent_duration_counts_as_the_same_request(tmp_path: Path) -> None:
    """5.1 s and 5.9 s both go out as duration 6: the fingerprint is on what is sent."""
    api = Api(tasks={"cgt-1": [task("running"), done()]})
    out = tmp_path / "c.mp4"
    seed(api, req(seconds=5.1), out)
    mark = len(api.requests)

    video(api).generate(req(seconds=5.9), out)

    assert api.posts(mark) == []
    assert api.polled(mark) == ["cgt-1"]


@pytest.mark.parametrize("status", ["failed", "expired", "cancelled"])
def test_saved_task_that_ended_badly_is_checked_then_replaced_by_a_new_task(
    status: str, audio: Path, tmp_path: Path
) -> None:
    """The saved task is polled first; it ended badly -> exactly one new POST (cgt-2), and
    cgt-2 is what gets polled and saved: a further retry resumes cgt-2, not cgt-1."""
    api = Api(
        tasks={
            "cgt-1": [task("running"), task(status, error={"code": "InternalError"})],
            "cgt-2": [task("running", id="cgt-2"), done(id="cgt-2")],
        }
    )
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    # attempt 2: cgt-1 is found dead -> new task cgt-2, which in turn times out
    with pytest.raises(ProviderError) as info:
        video(api, timeout_s=0.0).generate(req(audio), out)
    assert info.value.retryable
    assert api.polled(mark) == ["cgt-1", "cgt-2"]
    assert len(api.posts(mark)) == 1
    assert api.requests.index(api.posts(mark)[0]) > mark  # the POST came after checking cgt-1
    mark = len(api.requests)

    # attempt 3: resumes cgt-2
    video(api).generate(req(audio), out)
    assert api.posts(mark) == []
    assert api.polled(mark) == ["cgt-2"]
    assert out.read_bytes() == CLIP


@pytest.mark.parametrize(
    "error",
    [*MODERATION_ERRORS, {"code": "InvalidParameter", "message": "duration"}],
)
def test_saved_task_rejected_by_moderation_or_invalid_input_is_final_not_replaced(
    error: dict[str, str], audio: Path, tmp_path: Path
) -> None:
    """The same request would be rejected again: re-raise the saved task's reason as a final
    error instead of paying for a new task."""
    api = Api(tasks={"cgt-1": [task("running"), task("failed", error=error)]})
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)

    assert not info.value.retryable
    assert error["code"] in str(info.value) and error["message"] in str(info.value)
    assert api.posts(mark) == []
    assert api.polled(mark) == ["cgt-1"]
    saved = json.loads(out.with_suffix(".task.json").read_text(encoding="utf-8"))
    assert "task_id" not in saved and "replaced" not in saved  # like a fresh task's final failure


def test_saved_task_gone_404_is_replaced_by_a_new_task(audio: Path, tmp_path: Path) -> None:
    gone = httpx.Response(404, json={"error": {"code": "NotFound", "message": "no such task"}})
    api = Api(
        tasks={"cgt-1": [task("running"), gone], "cgt-2": [done(id="cgt-2")]},
    )
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    video(api).generate(req(audio), out)

    assert api.polled(mark) == ["cgt-1", "cgt-2"]
    assert len(api.posts(mark)) == 1
    assert out.read_bytes() == CLIP


VIDEO_URL_2 = "https://tos.example.com/clip-2.mp4?X-Tos-Signature=def"


@pytest.mark.parametrize("status", [403, 404])
def test_resumed_succeeded_task_with_an_expired_video_url_is_replaced_by_a_new_task(
    status: int, audio: Path, tmp_path: Path
) -> None:
    """The presigned video_url lives 24 h, the task 7 days: a resumed `succeeded` task whose
    download is 403/404 gets one new task. (A fresh task's 403/404 download stays final, see
    test_download_http_errors.)"""
    api = Api(
        tasks={
            "cgt-1": [task("running"), done()],
            "cgt-2": [task("succeeded", id="cgt-2", content={"video_url": VIDEO_URL_2})],
        },
        downloads={VIDEO_URL: httpx.Response(status)},
    )
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    result = video(api).generate(req(audio), out)

    assert api.polled(mark) == ["cgt-1", "cgt-2"]
    assert len(api.posts(mark)) == 1
    assert str(api.requests[-1].url) == VIDEO_URL_2
    assert result.video_path == out
    assert out.read_bytes() == CLIP


@pytest.mark.parametrize(
    "error",
    [httpx.Response(503, text="unavailable"), httpx.ReadTimeout("slow")],
    ids=["503", "read_timeout"],
)
def test_transient_error_polling_the_saved_task_is_retryable_and_keeps_it(
    error: httpx.Response | Exception, audio: Path, tmp_path: Path
) -> None:
    """A 5xx/timeout on the saved task that lasts until the deadline (0 s here) says nothing
    about the task: retryable, no POST, and the next attempt resumes cgt-1 again."""
    api = Api(tasks={"cgt-1": [task("running"), error, done()]})
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    with pytest.raises(ProviderError) as info:
        video(api, timeout_s=0.0).generate(req(audio), out)
    assert info.value.retryable
    assert api.posts(mark) == []
    assert api.polled(mark) == ["cgt-1"]
    mark = len(api.requests)

    video(api).generate(req(audio), out)

    assert len(api.posts()) == 1
    assert api.polled(mark) == ["cgt-1"]
    assert out.read_bytes() == CLIP


def test_failed_create_after_a_dead_saved_task_does_not_pin_the_dead_task(
    audio: Path, tmp_path: Path
) -> None:
    """Attempt 2 finds cgt-1 failed and its new POST gets a 429 (not created). Attempt 3 must
    create a task (it may re-check cgt-1 first), not just poll the dead id again."""
    api = Api(
        tasks={"cgt-1": [task("running"), task("failed", error={"code": "InternalError"})]},
        creates={2: httpx.Response(429, json={"error": {"code": "X", "message": "busy"}})},
    )
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)
    assert info.value.retryable
    assert api.polled(mark) == ["cgt-1"]  # the dead task was found first
    assert len(api.posts(mark)) == 1
    mark = len(api.requests)

    video(api).generate(req(audio), out)  # POST 3 -> cgt-3, default polls -> succeeded

    assert len(api.posts(mark)) == 1
    assert api.polled(mark)[-1] == "cgt-3"
    assert out.read_bytes() == CLIP


def test_poll_timeout_message_names_the_task_and_says_a_rerun_resumes(
    audio: Path, tmp_path: Path
) -> None:
    api = Api([task("running")])
    with pytest.raises(ProviderError) as info:
        video(api, timeout_s=0.0).generate(req(audio), tmp_path / "c.mp4")
    msg = str(info.value)
    assert info.value.retryable
    assert "cgt-1" in msg
    assert "resum" in msg.lower()


def test_after_a_resume_the_saved_task_still_resumes_on_the_next_attempt(
    audio: Path, tmp_path: Path
) -> None:
    """Attempt 2 resumes and times out again; attempt 3 must still resume cgt-1."""
    api = Api(tasks={"cgt-1": [task("running"), task("running"), done()]})
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)

    with pytest.raises(ProviderError) as info:
        video(api, timeout_s=0.0).generate(req(audio), out)
    assert info.value.retryable
    mark = len(api.requests)

    video(api).generate(req(audio), out)

    assert len(api.posts()) == 1
    assert api.polled(mark) == ["cgt-1"]
    assert out.read_bytes() == CLIP


def test_after_a_successful_resume_the_saved_task_is_kept(audio: Path, tmp_path: Path) -> None:
    """The clip is downloaded and verified: a further call makes no API call at all, and the
    task stays saved (deleting the local clip would resume it, not create a new one)."""
    api = Api(tasks={"cgt-1": [task("running"), done()]})
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    video(api).generate(req(audio), out)
    mark = len(api.requests)

    video(api).generate(req(audio), out)

    assert len(api.posts()) == 1
    assert api.requests[mark:] == []
    saved = json.loads(out.with_suffix(".task.json").read_text(encoding="utf-8"))
    assert saved["task_id"] == "cgt-1"


# Plain regression tests: these criteria already hold today (every generate POSTs), so they
# must not be xfail. They guard the resume fix against over-eager reuse of a saved task.


def _changed(
    case: str, audio: Path, frame: Path
) -> tuple[VideoRequest, dict[str, object], VideoRequest, dict[str, object]]:
    """(first request, first provider kwargs, second request, second provider kwargs)."""
    base = req(audio, str(frame))
    if case == "prompt":
        return base, {}, replace(base, prompt="A man in a kitchen"), {}
    if case == "audio_bytes":
        return base, {}, base, {"_rewrite_audio": True}
    if case == "duration":
        return base, {}, replace(base, duration_s=9.0), {}
    if case == "resolution":
        return base, {}, base, {"resolution": "1080p"}
    if case == "model":
        other = ModelRef(provider="seedance", model="seedance-other-test-id", version="2.5")
        return base, {}, base, {"ref": other}
    if case == "image_bytes":
        return base, {}, base, {"_rewrite_frame": True}
    if case == "image_url":
        first = req(audio, "https://cdn.example.com/a.png")
        return first, {}, req(audio, "https://cdn.example.com/b.png"), {}
    if case == "ratio":
        return base, {}, replace(base, aspect_ratio="16:9"), {}
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "prompt",
        "audio_bytes",
        "duration",
        "resolution",
        "model",
        "image_bytes",
        "image_url",
        "ratio",
    ],
)
def test_changed_request_creates_a_new_task_never_polls_the_old_one(
    case: str, audio: Path, frame: Path, tmp_path: Path
) -> None:
    first, first_kw, second, second_kw = _changed(case, audio, frame)
    api = Api(tasks={"cgt-1": [task("running"), done()], "cgt-2": [done(id="cgt-2")]})
    out = tmp_path / "shot_2.mp4"
    seed(api, first, out, **first_kw)
    if second_kw.pop("_rewrite_audio", False):
        audio.write_bytes(wav(3.5))  # same path, same sent duration, other bytes
    if second_kw.pop("_rewrite_frame", False):
        frame.write_bytes(b"\x89PNG\r\n\x1a\nanother fake png")
    mark = len(api.requests)

    ref = second_kw.pop("ref", REF)
    assert isinstance(ref, ModelRef)
    video(api, ref, **second_kw).generate(second, out)

    assert len(api.posts(mark)) == 1
    assert "cgt-1" not in api.polled(mark)
    assert api.polled(mark) == ["cgt-2"]
    assert out.read_bytes() == CLIP


@pytest.mark.parametrize("case", ["garbage", "torn_pending", "torn_known_task"])
def test_unparsable_task_file_is_final_and_names_it_never_a_new_post(
    case: str, audio: Path, tmp_path: Path
) -> None:
    """It may have held a pending create or a live task id: the human decides, the provider
    never reads it as "no task"."""
    out = tmp_path / "shot_1.mp4"
    path = out.with_suffix(".task.json")
    if case == "garbage":
        path.write_text("{not json", encoding="utf-8")
    else:
        if case == "torn_pending":
            with pytest.raises(ProviderError):
                video(Api(create=httpx.ReadTimeout("lost"))).generate(req(audio), out)
        else:
            seed(Api(tasks={"cgt-1": [task("running")]}), req(audio), out)
        text = path.read_text(encoding="utf-8")
        path.write_text(text[: len(text) // 2], encoding="utf-8")
    api = Api()

    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)

    assert not info.value.retryable
    assert str(path) in str(info.value)
    assert api.requests == []


def test_create_answer_without_a_task_id_is_final_until_checked(
    audio: Path, tmp_path: Path
) -> None:
    """The POST was accepted (200) but no id came back: the task may exist and be billed. The
    next call does not POST again; the human checks the console."""
    out = tmp_path / "shot_1.mp4"
    no_id = Api(create=httpx.Response(200, json={"error": None}))
    with pytest.raises(ProviderError, match="task id") as first:
        video(no_id).generate(req(audio), out)
    _assert_check_console(first.value, out.with_suffix(".task.json"))

    api = Api()
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)
    _assert_check_console(info.value, out.with_suffix(".task.json"))
    assert api.requests == []


def test_no_task_file_means_create_normally(audio: Path, tmp_path: Path) -> None:
    out = tmp_path / "shot_1.mp4"
    assert not out.with_suffix(".task.json").exists()
    api = Api()
    video(api).generate(req(audio), out)
    assert len(api.posts()) == 1
    assert api.polled() == ["cgt-1"]


# --- never pay twice: unknown create outcome ----------------------------------------------------
#
# Once the POST was sent, the task may exist (and be paid for) even if the answer was lost. The
# API has no idempotency key, so the only safe move is to stop and let the human check.

UNKNOWN_CREATE_OUTCOMES: list[httpx.Response | Exception] = [
    httpx.ReadTimeout("lost answer"),
    httpx.WriteTimeout("half sent"),
    httpx.RemoteProtocolError("server disconnected"),
    httpx.Response(500, json={"error": {"code": "InternalError", "message": "boom"}}),
    httpx.Response(502, text="bad gateway"),
    httpx.Response(503, json={"error": {"code": "X", "message": "busy"}}),
    httpx.Response(504, text="gateway timeout"),
    httpx.Response(200, text="<html>gateway</html>"),
    httpx.Response(408, text="request timeout"),  # a proxy may have forwarded the body
    httpx.Response(200, json={"data": {"id": "cgt-1"}}),  # accepted, but no id where we look
]
UNKNOWN_IDS = ["read_timeout", "write_timeout", "remote_protocol", "500", "502", "503", "504",
               "non_json", "408", "no_task_id"]  # fmt: skip


def _assert_check_console(err: ProviderError, task_file: Path) -> None:
    msg = str(err)
    assert not err.retryable
    assert "ModelArk console" in msg
    assert str(task_file) in msg
    assert "delete" in msg.lower()


@pytest.mark.parametrize("answer", UNKNOWN_CREATE_OUTCOMES, ids=UNKNOWN_IDS)
def test_unknown_create_outcome_is_final_and_says_check_the_console(
    answer: httpx.Response | Exception, audio: Path, tmp_path: Path
) -> None:
    out = tmp_path / "shot_1.mp4"
    api = Api(create=answer)
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)
    _assert_check_console(info.value, out.with_suffix(".task.json"))
    assert len(api.posts()) == 1
    saved = json.loads(out.with_suffix(".task.json").read_text(encoding="utf-8"))
    assert "task_id" not in saved
    assert saved["create_pending"]["fingerprint"] == saved["fingerprint"]
    assert datetime.fromisoformat(saved["create_pending"]["at"]).utcoffset() == timedelta(0)


@pytest.mark.parametrize("answer", UNKNOWN_CREATE_OUTCOMES[:4], ids=UNKNOWN_IDS[:4])
def test_pending_create_for_the_same_request_blocks_a_second_post(
    answer: httpx.Response | Exception, audio: Path, tmp_path: Path
) -> None:
    out = tmp_path / "shot_1.mp4"
    with pytest.raises(ProviderError):
        video(Api(create=answer)).generate(req(audio), out)
    api = Api()

    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)

    _assert_check_console(info.value, out.with_suffix(".task.json"))
    assert api.requests == []
    saved = json.loads(out.with_suffix(".task.json").read_text(encoding="utf-8"))
    assert "create_pending" in saved  # still pending until the human resolves it


def test_create_pending_marker_is_on_disk_before_the_post(audio: Path, tmp_path: Path) -> None:
    out = tmp_path / "shot_1.mp4"
    path = out.with_suffix(".task.json")
    api = Api()
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen.append(json.loads(path.read_text(encoding="utf-8")))
        return api(request)

    provider = seedance.SeedanceVideo(
        REF, transport=httpx.MockTransport(handler), poll_interval_s=0.0, timeout_s=5.0
    )
    provider.generate(req(audio), out)

    (before,) = seen
    pending = before["create_pending"]
    assert isinstance(pending, dict)
    assert pending["fingerprint"] == before["fingerprint"]
    after = json.loads(path.read_text(encoding="utf-8"))
    assert "create_pending" not in after and after["task_id"] == "cgt-1"


def test_task_id_added_by_hand_to_a_pending_file_is_resumed(audio: Path, tmp_path: Path) -> None:
    """The other documented way out: the human found the task in the console."""
    out = tmp_path / "shot_1.mp4"
    path = out.with_suffix(".task.json")
    with pytest.raises(ProviderError):
        video(Api(create=httpx.ReadTimeout("lost"))).generate(req(audio), out)
    saved = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**saved, "task_id": "cgt-7"}), encoding="utf-8")
    api = Api(tasks={"cgt-7": [done(id="cgt-7", usage={"completion_tokens": 200_000})]})

    result = video(api, usd_per_million_tokens=2.5).generate(req(audio), out)

    assert api.posts() == []
    assert api.polled() == ["cgt-7"]
    assert out.read_bytes() == CLIP
    assert result.cost_usd == pytest.approx(0.5)  # first delivery of that task


def test_another_requests_pending_create_is_kept_as_abandoned(audio: Path, tmp_path: Path) -> None:
    out = tmp_path / "shot_1.mp4"
    path = out.with_suffix(".task.json")
    with pytest.raises(ProviderError):
        video(Api(create=httpx.ReadTimeout("lost"))).generate(req(audio), out)
    old = json.loads(path.read_text(encoding="utf-8"))

    video(Api()).generate(replace(req(audio), prompt="Another shot"), out)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["abandoned"]["fingerprint"] == old["fingerprint"]
    assert saved["abandoned"]["create_pending"] == old["create_pending"]


def test_succeeded_task_without_video_url_is_final_and_kept(audio: Path, tmp_path: Path) -> None:
    """The task is billed: a rerun re-checks it instead of paying for a new one."""
    out = tmp_path / "shot_1.mp4"
    api = Api([task("succeeded", content={})])
    with pytest.raises(ProviderError, match="video_url") as info:
        video(api).generate(req(audio), out)
    assert not info.value.retryable
    mark = len(api.requests)

    with pytest.raises(ProviderError):
        video(api).generate(req(audio), out)

    assert api.posts(mark) == []
    assert api.polled(mark) == ["cgt-1"]


def test_deleting_the_task_file_allows_a_new_post(audio: Path, tmp_path: Path) -> None:
    """The documented way out after the human checked the console."""
    out = tmp_path / "shot_1.mp4"
    with pytest.raises(ProviderError):
        video(Api(create=httpx.ReadTimeout("lost"))).generate(req(audio), out)
    out.with_suffix(".task.json").unlink()
    api = Api()
    video(api).generate(req(audio), out)
    assert len(api.posts()) == 1
    assert out.read_bytes() == CLIP


@pytest.mark.parametrize(
    "answer",
    [
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("t"),
        httpx.Response(429, json={"error": {"code": "X", "message": "busy"}}),
        httpx.Response(400, json={"error": {"code": "InvalidParameter", "message": "bad"}}),
    ],
    ids=["connect_error", "connect_timeout", "429", "400"],
)
def test_create_known_not_to_exist_does_not_block_the_next_post(
    answer: httpx.Response | Exception, audio: Path, tmp_path: Path
) -> None:
    out = tmp_path / "shot_1.mp4"
    with pytest.raises(ProviderError):
        video(Api(create=answer)).generate(req(audio), out)
    api = Api()
    video(api).generate(req(audio), out)
    assert len(api.posts()) == 1
    assert out.read_bytes() == CLIP


def test_pending_create_of_another_request_does_not_block(audio: Path, tmp_path: Path) -> None:
    out = tmp_path / "shot_1.mp4"
    with pytest.raises(ProviderError):
        video(Api(create=httpx.ReadTimeout("lost"))).generate(req(audio), out)
    api = Api()
    video(api).generate(replace(req(audio), prompt="Another shot"), out)
    assert len(api.posts()) == 1


# --- never pay twice: downloaded clip -----------------------------------------------------------


def test_download_records_status_sha256_and_size(audio: Path, tmp_path: Path) -> None:
    out = tmp_path / "shot_1.mp4"
    video(Api()).generate(req(audio), out)
    saved = json.loads(out.with_suffix(".task.json").read_text(encoding="utf-8"))
    assert saved["status"] == "downloaded"
    assert saved["task_id"] == "cgt-1"
    assert saved["clip"] == {"sha256": hashlib.sha256(CLIP).hexdigest(), "size": len(CLIP)}


def test_verified_local_clip_is_returned_without_any_call_after_the_url_expired(
    audio: Path, tmp_path: Path
) -> None:
    """Shot stage failed after the download (e.g. frame step), resume a day later: the video
    URL is 403 now, but the local clip matches the record -> same file, no API call, cost 0."""
    api = Api([done(usage={"completion_tokens": 200_000})])
    out = tmp_path / "shot_1.mp4"
    first = video(api, usd_per_million_tokens=2.5).generate(req(audio), out)
    assert first.cost_usd == pytest.approx(0.5)
    api.download = httpx.Response(403)
    mark = len(api.requests)

    result = video(api, usd_per_million_tokens=2.5).generate(req(audio), out)

    assert api.requests[mark:] == []
    assert result.video_path == out
    assert out.read_bytes() == CLIP
    assert result.cost_usd == 0.0


@pytest.mark.parametrize("local", ["tampered", "deleted"])
def test_local_clip_not_matching_the_record_is_fetched_again_at_no_cost(
    local: str, audio: Path, tmp_path: Path
) -> None:
    api = Api([done(usage={"completion_tokens": 200_000})])
    out = tmp_path / "shot_1.mp4"
    video(api, usd_per_million_tokens=2.5).generate(req(audio), out)
    if local == "tampered":
        out.write_bytes(b"not the downloaded clip")
    else:
        out.unlink()
    mark = len(api.requests)

    result = video(api, usd_per_million_tokens=2.5).generate(req(audio), out)

    assert api.posts(mark) == []
    assert api.polled(mark) == ["cgt-1"]
    assert out.read_bytes() == CLIP
    assert result.cost_usd == 0.0  # this clip's cost was reported by the first call


def test_resumed_task_not_yet_downloaded_reports_its_cost_once(audio: Path, tmp_path: Path) -> None:
    """The first attempt timed out before the download, so nothing was reported yet: the resume
    is the first call that delivers the clip and reports its cost."""
    api = Api(tasks={"cgt-1": [task("running"), done(usage={"completion_tokens": 200_000})]})
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out, usd_per_million_tokens=2.5)
    result = video(api, usd_per_million_tokens=2.5).generate(req(audio), out)
    assert result.cost_usd == pytest.approx(0.5)
    assert len(api.posts()) == 1


# --- never pay twice: atomic task file ----------------------------------------------------------


def test_task_file_is_written_via_a_temp_file_and_os_replace(
    audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "shot_1.mp4"
    path = out.with_suffix(".task.json")
    real = os.replace
    moves: list[tuple[Path, Path]] = []

    def spy(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        moves.append((Path(src), Path(dst)))
        real(src, dst)

    monkeypatch.setattr(seedance.os, "replace", spy)
    video(Api()).generate(req(audio), out)

    into_task_file = [src for src, dst in moves if dst == path]
    assert into_task_file, "task.json was not written via os.replace"
    assert all(src.parent == path.parent and src != path for src in into_task_file)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["line.wav", "shot_1.mp4", path.name]


# --- polls retry transient errors in place ------------------------------------------------------


def test_transient_poll_errors_are_retried_in_place_with_capped_backoff(
    audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors: list[httpx.Response | Exception] = [
        httpx.Response(503, text="unavailable"),
        httpx.ReadTimeout("slow"),
        httpx.Response(429, json={"error": {"code": "X", "message": "busy"}}),
        httpx.RemoteProtocolError("dropped"),
        httpx.Response(500, text="boom"),
        httpx.ConnectError("refused"),
    ]
    api = Api(tasks={"cgt-1": [*errors, done()]})
    sleeps: list[float] = []
    monkeypatch.setattr(seedance.time, "sleep", sleeps.append)
    monkeypatch.setattr(seedance.random, "uniform", lambda low, high: high)

    provider = video(api, timeout_s=1000.0, retry_backoff_s=1.0, retry_backoff_max_s=8.0)
    result = provider.generate(req(audio), tmp_path / "shot_1.mp4")

    assert result.video_path.read_bytes() == CLIP
    assert len(api.posts()) == 1
    assert api.polled() == ["cgt-1"] * 7
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]  # exponential, capped


def test_poll_backoff_has_jitter_in_the_lower_half(
    audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = Api(tasks={"cgt-1": [httpx.Response(503), httpx.Response(503), done()]})
    sleeps: list[float] = []
    monkeypatch.setattr(seedance.time, "sleep", sleeps.append)
    monkeypatch.setattr(seedance.random, "uniform", lambda low, high: low)

    provider = video(api, timeout_s=1000.0, retry_backoff_s=1.0, retry_backoff_max_s=8.0)
    provider.generate(req(audio), tmp_path / "shot_1.mp4")

    assert sleeps == [0.5, 1.0]


def test_resumed_task_poll_error_is_retried_in_place_not_a_failed_attempt(
    audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = Api(tasks={"cgt-1": [task("running"), httpx.ReadTimeout("slow"), done()]})
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    monkeypatch.setattr(seedance.time, "sleep", lambda s: None)
    mark = len(api.requests)

    video(api).generate(req(audio), out)

    assert api.posts(mark) == []
    assert api.polled(mark) == ["cgt-1", "cgt-1"]
    assert out.read_bytes() == CLIP


def test_poll_backoff_never_sleeps_past_the_deadline(
    audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient error that lasts until the deadline raises retryable with the task id."""
    api = Api([httpx.Response(503)])
    sleeps: list[float] = []
    clock = iter(range(0, 10_000, 1))
    monkeypatch.setattr(seedance.time, "monotonic", lambda: float(next(clock)))
    monkeypatch.setattr(seedance.time, "sleep", sleeps.append)

    with pytest.raises(ProviderError) as info:
        video(api, timeout_s=3.0).generate(req(audio), tmp_path / "c.mp4")

    assert info.value.retryable
    assert "cgt-1" in str(info.value)
    assert all(s <= 3.0 for s in sleeps)


# --- cost ---------------------------------------------------------------------------------------


def test_cost_is_zero_without_a_verified_price(audio: Path, tmp_path: Path) -> None:
    api = Api([done(usage={"completion_tokens": 100_000})])
    assert video(api).generate(req(audio), tmp_path / "c.mp4").cost_usd == 0.0


def test_cost_from_usage_tokens_when_a_price_is_given(audio: Path, tmp_path: Path) -> None:
    api = Api([done(usage={"completion_tokens": 200_000, "total_tokens": 200_000})])
    result = video(api, usd_per_million_tokens=2.5).generate(req(audio), tmp_path / "c.mp4")
    assert result.cost_usd == pytest.approx(0.5)


def test_cost_is_zero_when_usage_is_missing(audio: Path, tmp_path: Path) -> None:
    api = Api([done()])
    result = video(api, usd_per_million_tokens=2.5).generate(req(audio), tmp_path / "c.mp4")
    assert result.cost_usd == 0.0


# --- live ---------------------------------------------------------------------------------------


@pytest.mark.live
def test_live_generate_a_short_speaking_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke (paid): key, endpoint, body shape and polling are accepted; a 4 s 480p clip with a
    3 s tone as reference audio comes back and is downloaded."""
    monkeypatch.delenv(seedance.API_KEY_ENV, raising=False)  # drop the autouse fake key
    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get(seedance.API_KEY_ENV):
        pytest.skip(f"{seedance.API_KEY_ENV} not set")
    path = tmp_path / "line.wav"
    path.write_bytes(wav(3.0))
    provider = seedance.SeedanceVideo(REF, resolution="480p", poll_interval_s=5.0)
    out = tmp_path / "shot_1.mp4"
    result = provider.generate(
        VideoRequest(prompt="A person says a short line", duration_s=4, audio_path=path),
        out,
    )
    assert result.video_path == out and out.stat().st_size > 0

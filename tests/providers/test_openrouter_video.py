"""OpenRouterVideo with the HTTP layer mocked (httpx.MockTransport, no network). One opt-in live
test — paid per call: never run it without the human.

Request/response shapes follow the OpenRouter video docs; the live smoke test is
the first real check. The never-pay-twice logic is shared with Seedance (`video_jobs`) and
tested in depth in test_seedance.py; here every rule is checked once against OpenRouter's
shapes (202 create, "completed", urls / unsigned_urls, usage.cost)."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import wave
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

from hwva.providers import fakes, litterbox, openrouter_video, registry, video_jobs
from hwva.providers.base import HostedMedia, ProviderError, VideoProvider, VideoRequest
from hwva.schemas import ModelRef
from hwva.stages.shot_2 import MODERATION_MARKERS

ROOT = Path(__file__).resolve().parents[2]
EXPIRES = datetime(2030, 1, 1, 13, 0, tzinfo=UTC)
REF_25 = ModelRef(provider="openrouter-video", model="bytedance/seedance-2.5", version="2.5")
# The job-lifecycle tests (resume, cost once, retries, polling, download) are about the adapter's
# mechanics, not one model: they run on a test-only version that takes audio WITHOUT an image
# (2.5 needs one since the live finding), registered by the autouse fixture below.
AUDIO_ONLY = "test-audio-only"
REF = ModelRef(provider="openrouter-video", model="bytedance/seedance-2.5", version=AUDIO_ONLY)
REF_20 = ModelRef(provider="openrouter-video", model="bytedance/seedance-2.0", version="2.0")
REF_20F = ModelRef(
    provider="openrouter-video", model="bytedance/seedance-2.0-fast", version="2.0-fast"
)
KEY = "test-openrouter-key-not-real-0123456789"
VIDEOS = "https://openrouter.ai/api/v1/videos"
SIGNED = "https://storage.example.com/clip.mp4?X-Signature=abc"
SIGNED_2 = "https://storage.example.com/clip-2.mp4?X-Signature=def"
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
    monkeypatch.setenv(openrouter_video.API_KEY_ENV, KEY)
    monkeypatch.delenv(openrouter_video.RESOLUTION_ENV, raising=False)


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


@pytest.fixture(autouse=True)
def audio_only_version(monkeypatch: pytest.MonkeyPatch) -> None:
    caps = openrouter_video.Capabilities(
        accepts_audio_input=True, max_duration_s=30, min_audio_s=2.0
    )
    monkeypatch.setitem(openrouter_video.CAPABILITIES, AUDIO_ONLY, caps)


def job(status: str, job_id: str = "gen-vid-1", **extra: object) -> httpx.Response:
    return httpx.Response(200, json={"id": job_id, "status": status, **extra})


def done(job_id: str = "gen-vid-1", url: str = SIGNED, **extra: object) -> httpx.Response:
    return job("completed", job_id, urls=[url], expires_at="2030-01-02T00:00:00Z", **extra)


def created(n: int) -> httpx.Response:
    job_id = f"gen-vid-{n}"
    body = {"id": job_id, "polling_url": f"/api/v1/videos/{job_id}", "status": "pending"}
    return httpx.Response(202, json=body)


class Api:
    """MockTransport handler: POST -> create answer, GET job -> next poll answer, other GET ->
    the clip. Every request is recorded.

    The n-th POST answers 202 {"id": "gen-vid-<n>"} unless `create` (every POST) or `creates`
    (POST number n, 1-based) says otherwise. `jobs` gives per-id poll queues; other ids use
    `polls`. A queue's last answer repeats. `downloads` maps a URL to its answer (checked
    first: an unsigned URL lives under /videos/ too); other non-job URLs get `download`."""

    def __init__(
        self,
        polls: list[httpx.Response | Exception] | None = None,
        *,
        create: httpx.Response | Exception | None = None,
        download: httpx.Response | Exception | None = None,
        jobs: dict[str, list[httpx.Response | Exception]] | None = None,
        creates: dict[int, httpx.Response | Exception] | None = None,
        downloads: dict[str, httpx.Response | Exception] | None = None,
    ) -> None:
        self.create = create
        self.polls: list[httpx.Response | Exception] = [*(polls if polls is not None else [done()])]
        self.download = download if download is not None else httpx.Response(200, content=CLIP)
        self.jobs = jobs if jobs is not None else {}
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
            answer = self.creates.get(n, self.create if self.create is not None else created(n))
        elif url in self.downloads:
            answer = self.downloads[url]
        elif url.startswith(VIDEOS + "/"):
            queue = self.jobs.get(url.removeprefix(VIDEOS + "/"), self.polls)
            answer = queue.pop(0) if len(queue) > 1 else queue[0]
        else:
            answer = self.downloads.get(url, self.download)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def posts(self, since: int = 0) -> list[httpx.Request]:
        return [r for r in self.requests[since:] if r.method == "POST"]

    def polled(self, since: int = 0) -> list[str]:
        prefix = VIDEOS + "/"
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


class Host:
    """MediaHost spy: every upload gets a NEW url (like a real host), so a fingerprint built on
    the URL would change between attempts. `fail` raises on upload."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.uploads: list[Path] = []
        self.fail = fail

    def upload(self, path: Path) -> HostedMedia:
        if self.fail is not None:
            raise self.fail
        self.uploads.append(path)
        n = len(self.uploads)
        return HostedMedia(url=f"https://media.test/{n}/{path.name}", expires_at=EXPIRES)


def video(
    api: Api, ref: ModelRef = REF, host: Host | None = None, **kwargs: object
) -> openrouter_video.OpenRouterVideo:
    params: dict[str, object] = {"poll_interval_s": 0.0, "timeout_s": 5.0, **kwargs}
    transport = httpx.MockTransport(api)
    return openrouter_video.OpenRouterVideo(
        ref,
        transport=transport,
        media_host=host if host is not None else Host(),
        **params,  # type: ignore[arg-type]
    )


def req(audio: Path | None = None, image: str | None = None, seconds: float = 5.0) -> VideoRequest:
    return VideoRequest(
        prompt="A woman on a rooftop", duration_s=seconds, audio_path=audio, reference_image=image
    )


def saved(out: Path) -> dict[str, object]:
    data: dict[str, object] = json.loads(out.with_suffix(".task.json").read_text("utf-8"))
    return data


# --- contract, capabilities, config -------------------------------------------------------------


def test_is_a_video_provider() -> None:
    assert isinstance(video(Api()), VideoProvider)


def test_version_25_needs_an_image_with_audio_up_to_30_s() -> None:
    """Live: 2.5 ignored audio sent alone, lip-synced with audio + image."""
    caps = video(Api(), REF_25).capabilities
    assert caps.accepts_audio_input and caps.needs_image
    assert caps.max_duration_s == 30
    assert "9:16" in caps.aspect_ratios


@pytest.mark.parametrize("ref", [REF_20, REF_20F], ids=["2.0", "2.0-fast"])
def test_versions_20_need_an_image_with_audio_up_to_15_s(ref: ModelRef) -> None:
    caps = video(Api(), ref).capabilities
    assert caps.accepts_audio_input and caps.needs_image
    assert caps.max_duration_s == 15


def test_unknown_version_is_rejected_at_build() -> None:
    with pytest.raises(ValueError, match="version"):
        video(Api(), ModelRef(provider="openrouter-video", model="x", version="1.0"))


def test_missing_key_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(openrouter_video.API_KEY_ENV)
    with pytest.raises(ValueError, match=openrouter_video.API_KEY_ENV):
        openrouter_video.OpenRouterVideo(REF)


def test_resolution_defaults_to_720p() -> None:
    assert video(Api()).resolution == "720p"


def test_resolution_env_overrides_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(openrouter_video.RESOLUTION_ENV, "480p")
    assert video(Api()).resolution == "480p"


def test_resolution_argument_beats_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(openrouter_video.RESOLUTION_ENV, "480p")
    assert video(Api(), resolution="720p").resolution == "720p"


def test_empty_resolution_env_falls_back_to_720p(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(openrouter_video.RESOLUTION_ENV, "")
    assert video(Api()).resolution == "720p"


@pytest.mark.parametrize("bad", ["1080p", "4k"])
def test_unknown_resolution_is_rejected_at_build(bad: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="resolution"):
        video(Api(), resolution=bad)
    monkeypatch.setenv(openrouter_video.RESOLUTION_ENV, bad)
    with pytest.raises(ValueError, match=openrouter_video.RESOLUTION_ENV):
        video(Api())


def test_registry_builds_it_from_the_ref() -> None:
    provider = registry._FACTORIES["video"]["openrouter-video"](REF)
    assert isinstance(provider, openrouter_video.OpenRouterVideo)
    assert provider.model == REF.model


def test_premium_openrouter_profile_is_premium_with_openrouter_video_image_and_host() -> None:
    """Video default is 2.5 with the character image (live; 2.0-fast one line away)."""
    path = ROOT / "config" / "profiles.yaml"
    premium = registry.load_profile("premium", path)
    profile = registry.load_profile("premium-openrouter", path)
    assert profile["video"] == REF_25
    assert profile["image"] == ModelRef(
        provider="openrouter-image",
        model="google/gemini-2.5-flash-image",
        version="google/gemini-2.5-flash-image",
    )
    assert profile["media_host"].provider == "uguu"  # litterbox was down for hours
    own = ("video", "image", "media_host")
    assert {k: v for k, v in profile.items() if k not in own} == {
        k: v for k, v in premium.items() if k not in own
    }
    assert premium["video"].provider == "seedance"  # premium (BytePlus) untouched
    assert "media_host" not in premium and "image" not in premium


def test_registry_gives_the_profile_media_host_to_openrouter_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-not-real")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-eleven-key-not-real")
    providers = registry.build("premium-openrouter")
    assert isinstance(providers.video, openrouter_video.OpenRouterVideo)
    assert providers.video.media_host is providers.media_host
    assert providers.media_host is not None and providers.image is not None


# --- request ------------------------------------------------------------------------------------


def test_create_is_a_bearer_post_to_the_videos_endpoint(audio: Path, tmp_path: Path) -> None:
    api = Api()
    video(api).generate(req(audio), tmp_path / "shot_1.mp4")
    post = api.requests[0]
    assert post.method == "POST"
    assert str(post.url) == VIDEOS
    assert post.headers["authorization"] == f"Bearer {KEY}"


def test_body_has_model_prompt_duration_resolution_ratio_and_no_model_audio(
    audio: Path, tmp_path: Path
) -> None:
    api = Api()
    video(api, resolution="480p").generate(req(audio), tmp_path / "shot_1.mp4")
    body = api.body()
    assert body["model"] == "bytedance/seedance-2.5"
    assert body["prompt"] == "A woman on a rooftop"
    assert body["duration"] == 5
    assert body["resolution"] == "480p"
    assert body["aspect_ratio"] == "9:16"
    assert body["generate_audio"] is False  # our TTS is the audio
    assert "frame_images" not in body


def test_audio_goes_as_an_audio_url_at_the_media_host_never_a_data_uri(
    audio: Path, tmp_path: Path
) -> None:
    """Live: OpenRouter answers 400 "Only HTTPS URLs are allowed" to data URIs."""
    api, host = Api(), Host()
    video(api, host=host).generate(req(audio), tmp_path / "shot_1.mp4")
    (item,) = api.body()["input_references"]  # type: ignore[misc]
    assert item == {"type": "audio_url", "audio_url": {"url": "https://media.test/1/line.wav"}}
    assert host.uploads == [audio]
    assert "data:" not in api.posts()[0].content.decode()


def test_local_frame_without_audio_goes_as_first_frame_at_the_media_host(
    frame: Path, tmp_path: Path
) -> None:
    """shot_2: no audio, so the frame may (and does) stay a first frame."""
    api, host = Api(), Host()
    video(api, host=host).generate(req(image=str(frame)), tmp_path / "shot_2.mp4")
    body = api.body()
    (item,) = body["frame_images"]  # type: ignore[misc]
    assert item == {
        "type": "image_url",
        "image_url": {"url": "https://media.test/1/shot_1.last.png"},
        "frame_type": "first_frame",
    }
    assert host.uploads == [frame]
    assert "input_references" not in body  # silent shot: no audio


@pytest.mark.parametrize("ref", [REF_25, REF_20, REF_20F], ids=["2.5", "2.0", "2.0-fast"])
def test_audio_and_image_both_go_in_input_references_and_never_frame_images(
    ref: ModelRef, audio: Path, frame: Path, tmp_path: Path
) -> None:
    """Live (2.0-fast): the audio is honoured only with audio AND image both in
    input_references and no frame_images; with the image as first_frame it was ignored."""
    api, host = Api(), Host()
    video(api, ref, host).generate(req(audio, str(frame)), tmp_path / "shot_1.mp4")
    body = api.body()
    assert body["input_references"] == [
        {"type": "audio_url", "audio_url": {"url": "https://media.test/1/line.wav"}},
        {"type": "image_url", "image_url": {"url": "https://media.test/2/shot_1.last.png"}},
    ]
    assert "frame_images" not in body
    assert host.uploads == [audio, frame]


@pytest.mark.parametrize(
    "image", ["http://cdn.example.com/f.png", "data:image/png;base64,AAAA"], ids=["http", "data"]
)
def test_non_https_reference_image_fails_before_spend(image: str, tmp_path: Path) -> None:
    api, host = Api(), Host()
    with pytest.raises(ProviderError) as info:
        video(api, host=host).generate(req(image=image), tmp_path / "shot_2.mp4")
    assert not info.value.retryable and "https" in str(info.value).lower()
    assert api.requests == [] and host.uploads == []


def test_local_media_without_a_media_host_fails_before_spend(audio: Path, tmp_path: Path) -> None:
    api = Api()
    provider = openrouter_video.OpenRouterVideo(
        REF, transport=httpx.MockTransport(api), poll_interval_s=0.0, timeout_s=5.0
    )
    with pytest.raises(ProviderError) as info:
        provider.generate(req(audio), tmp_path / "shot_1.mp4")
    assert not info.value.retryable and "media host" in str(info.value).lower()
    assert api.requests == []
    assert not (tmp_path / "shot_1.task.json").exists()


def test_failed_upload_is_retryable_and_nothing_is_posted_or_pinned(
    audio: Path, tmp_path: Path
) -> None:
    """Uploading is free: a failure before the create POST leaves no pending marker, so the
    pipeline retry may create the job."""
    out = tmp_path / "shot_1.mp4"
    api = Api()
    with pytest.raises(ProviderError) as info:
        video(api, host=Host(fail=ProviderError("litterbox HTTP 503"))).generate(req(audio), out)
    assert info.value.retryable
    assert api.requests == []
    task = out.with_suffix(".task.json")
    assert not task.exists() or not {"create_pending", "task_id"} & saved(out).keys()
    video(api).generate(req(audio), out)
    assert len(api.posts()) == 1


def test_resume_never_uploads_again_and_never_pays_again_although_the_url_changes(
    audio: Path, frame: Path, tmp_path: Path
) -> None:
    """The fingerprint is the media's sha256, not its (changing) hosted URL."""
    api = Api(jobs={"gen-vid-1": [job("in_progress"), done(usage={"cost": 0.17})]})
    host = Host()
    out = tmp_path / "shot_1.mp4"
    request = req(audio, str(frame))
    with pytest.raises(ProviderError):
        video(api, REF_20F, host, timeout_s=0.0).generate(request, out)
    assert len(host.uploads) == 2
    fingerprint = saved(out)["fingerprint"]

    result = video(api, REF_20F, host).generate(request, out)

    assert len(api.posts()) == 1 and len(host.uploads) == 2  # resumed: no upload, no POST
    assert saved(out)["fingerprint"] == fingerprint
    assert result.cost_usd == pytest.approx(0.17)
    video(api, REF_20F, host).generate(request, out)  # verified local clip: nothing at all
    assert len(api.posts()) == 1 and len(host.uploads) == 2


def test_create_post_waits_180_s_polls_keep_the_short_timeout(audio: Path, tmp_path: Path) -> None:
    """Live: a create POST took more than 60 s (ReadTimeout -> unknown outcome)."""
    api = Api([job("in_progress"), done()])
    video(api).generate(req(audio), tmp_path / "shot_1.mp4")
    post = api.posts()[0]
    assert post.extensions["timeout"]["read"] == 180.0
    poll = next(r for r in api.requests if r.method == "GET")
    assert poll.extensions["timeout"]["read"] == video_jobs.REQUEST_TIMEOUT_S


def test_image_url_is_passed_through(tmp_path: Path) -> None:
    api = Api()
    video(api).generate(req(image="https://cdn.example.com/f.png"), tmp_path / "shot_2.mp4")
    (item,) = api.body()["frame_images"]  # type: ignore[misc]
    assert item["image_url"] == {"url": "https://cdn.example.com/f.png"}


@pytest.mark.parametrize(("asked", "sent"), [(1.0, 4), (4.0, 4), (5.2, 6), (29.5, 30)])
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


@pytest.mark.parametrize("ref", [REF_20, REF_20F], ids=["2.0", "2.0-fast"])
def test_versions_20_with_audio_and_no_image_fail_before_spend(
    ref: ModelRef, audio: Path, tmp_path: Path
) -> None:
    api = Api()
    err = _rejected_without_calls(api, lambda: video(api, ref).generate(req(audio), tmp_path / "c"))
    assert "image" in str(err)


@pytest.mark.parametrize(("ref", "seconds"), [(REF, 31.0), (REF_20, 16.0), (REF_20F, 16.0)])
def test_too_long_clip_fails_before_spend(ref: ModelRef, seconds: float, tmp_path: Path) -> None:
    api = Api()
    _rejected_without_calls(
        api, lambda: video(api, ref).generate(req(seconds=seconds), tmp_path / "c.mp4")
    )


@pytest.mark.parametrize("seconds", [1.5, 30.5])
def test_audio_out_of_range_fails_before_spend(seconds: float, tmp_path: Path) -> None:
    path = tmp_path / "line.wav"
    path.write_bytes(wav(seconds))
    api = Api()
    err = _rejected_without_calls(api, lambda: video(api).generate(req(path), tmp_path / "c.mp4"))
    assert "audio" in str(err)


def test_unsupported_aspect_ratio_fails_before_spend(tmp_path: Path) -> None:
    api = Api()
    request = VideoRequest(prompt="p", duration_s=5, aspect_ratio="16:9")
    _rejected_without_calls(api, lambda: video(api).generate(request, tmp_path / "c.mp4"))


def test_missing_frame_file_fails_before_spend(tmp_path: Path) -> None:
    api = Api()
    missing = str(tmp_path / "nope.png")
    _rejected_without_calls(
        api, lambda: video(api).generate(req(image=missing), tmp_path / "c.mp4")
    )


# --- polling & download -------------------------------------------------------------------------


def test_polls_until_completed_then_downloads_the_signed_url_without_the_key(
    audio: Path, tmp_path: Path
) -> None:
    api = Api([job("pending"), job("in_progress"), done()])
    out = tmp_path / "shot_1.mp4"
    result = video(api).generate(req(audio), out)
    assert result.video_path == out and result.last_frame is None
    assert out.read_bytes() == CLIP
    gets = [r for r in api.requests if r.method == "GET"]
    assert [str(r.url) for r in gets] == [f"{VIDEOS}/gen-vid-1"] * 3 + [SIGNED]
    assert all(r.headers["authorization"] == f"Bearer {KEY}" for r in gets[:3])
    assert "authorization" not in gets[-1].headers
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize(
    ("unsigned", "fetched"),
    [
        ("https://openrouter.ai/api/v1/videos/gen-vid-1/content", None),
        (
            "/api/v1/videos/gen-vid-1/content",
            "https://openrouter.ai/api/v1/videos/gen-vid-1/content",
        ),
    ],
    ids=["absolute", "relative"],
)
def test_without_signed_urls_the_unsigned_url_is_fetched_with_the_key(
    unsigned: str, fetched: str | None, audio: Path, tmp_path: Path
) -> None:
    content = fetched or unsigned
    api = Api(
        [job("completed", unsigned_urls=[unsigned])],
        downloads={content: httpx.Response(200, content=CLIP)},
    )
    out = tmp_path / "shot_1.mp4"
    video(api).generate(req(audio), out)
    download = api.requests[-1]
    assert str(download.url) == content
    assert download.headers["authorization"] == f"Bearer {KEY}"
    assert out.read_bytes() == CLIP


@pytest.mark.parametrize(
    "answer",
    [
        {"unsigned_urls": ["https://evil.example.com/clip.mp4"]},
        {"unsigned_urls": ["http://openrouter.ai/api/v1/videos/gen-vid-1/content"]},
        {"urls": ["http://storage.example.com/clip.mp4"]},
        {"urls": [], "unsigned_urls": []},
        {},
    ],
    ids=["unsigned_off_host", "unsigned_plain_http", "signed_plain_http", "empty", "none"],
)
def test_completed_job_without_a_usable_url_is_final_kept_and_never_leaks_the_key(
    answer: dict[str, object], audio: Path, tmp_path: Path
) -> None:
    """Paid for: a rerun re-checks the same job (no new POST); nothing is downloaded."""
    api = Api([job("completed", **answer)])
    out = tmp_path / "shot_1.mp4"
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)
    assert not info.value.retryable
    assert "gen-vid-1" in str(info.value)
    assert [r.method for r in api.requests] == ["POST", "GET"]  # no download attempted
    assert not out.exists()
    mark = len(api.requests)

    with pytest.raises(ProviderError):
        video(api).generate(req(audio), out)
    assert api.posts(mark) == []
    assert api.polled(mark) == ["gen-vid-1"]


def test_download_follows_a_redirect_to_storage_without_the_key(
    audio: Path, tmp_path: Path
) -> None:
    unsigned = "https://openrouter.ai/api/v1/videos/gen-vid-1/content"
    storage = "https://storage.example.com/clip.mp4?X-Signature=xyz"
    api = Api(
        [job("completed", unsigned_urls=[unsigned])],
        downloads={
            unsigned: httpx.Response(302, headers={"Location": storage}),
            storage: httpx.Response(200, content=CLIP),
        },
    )
    out = tmp_path / "shot_1.mp4"
    video(api).generate(req(audio), out)
    first, second = api.requests[-2:]
    assert str(first.url) == unsigned and first.headers["authorization"] == f"Bearer {KEY}"
    assert str(second.url) == storage and "authorization" not in second.headers
    assert out.read_bytes() == CLIP


def test_empty_download_is_a_provider_error(audio: Path, tmp_path: Path) -> None:
    api = Api(download=httpx.Response(200, content=b""))
    out = tmp_path / "c.mp4"
    with pytest.raises(ProviderError, match="0 bytes"):
        video(api).generate(req(audio), out)
    assert not out.exists() and not list(tmp_path.glob("*.part"))


def test_poll_timeout_is_retryable_names_the_job_and_says_a_rerun_resumes(
    audio: Path, tmp_path: Path
) -> None:
    api = Api([job("in_progress")])
    with pytest.raises(ProviderError) as info:
        video(api, timeout_s=0.0).generate(req(audio), tmp_path / "c.mp4")
    assert info.value.retryable
    assert "gen-vid-1" in str(info.value) and "resum" in str(info.value).lower()


# --- cost ---------------------------------------------------------------------------------------


def test_cost_is_usage_cost_in_usd(audio: Path, tmp_path: Path) -> None:
    api = Api([done(usage={"cost": 1.25})])
    out = tmp_path / "c.mp4"
    assert video(api).generate(req(audio), out).cost_usd == pytest.approx(1.25)
    assert saved(out)["cost_usd"] == pytest.approx(1.25)
    assert "cost_note" not in saved(out)


@pytest.mark.parametrize("usage", [{"cost": None}, {}, None, {"cost": "1.2"}, {"cost": True}])
def test_missing_cost_is_zero_and_said_in_the_task_file(
    usage: object, audio: Path, tmp_path: Path
) -> None:
    api = Api([done(usage=usage)])
    out = tmp_path / "c.mp4"
    assert video(api).generate(req(audio), out).cost_usd == 0.0
    note = saved(out)["cost_note"]
    assert isinstance(note, str) and "0" in note


# --- errors -------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 402, 403, 413])
def test_client_errors_on_create_are_final(status: int, audio: Path, tmp_path: Path) -> None:
    body = {"error": {"code": status, "message": "Insufficient credits or bad request"}}
    out = tmp_path / "c.mp4"
    api = Api(create=httpx.Response(status, json=body))
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)
    assert not info.value.retryable
    assert str(status) in str(info.value) and "Insufficient credits" in str(info.value)
    assert KEY not in str(info.value)
    assert "create_pending" not in saved(out)  # rejected: no job exists, a later POST may go


def test_rate_limit_on_create_is_retryable_and_does_not_block_the_next_post(
    audio: Path, tmp_path: Path
) -> None:
    out = tmp_path / "c.mp4"
    busy = Api(create=httpx.Response(429, json={"error": {"code": 429, "message": "busy"}}))
    with pytest.raises(ProviderError) as info:
        video(busy).generate(req(audio), out)
    assert info.value.retryable
    api = Api()
    video(api).generate(req(audio), out)
    assert len(api.posts()) == 1 and out.read_bytes() == CLIP


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_rate_limit_and_server_errors_on_a_poll_are_retryable(
    status: int, audio: Path, tmp_path: Path
) -> None:
    api = Api([httpx.Response(status, json={"error": {"code": status, "message": "busy"}})])
    with pytest.raises(ProviderError) as info:
        video(api, timeout_s=0.0).generate(req(audio), tmp_path / "c.mp4")
    assert info.value.retryable


def test_transient_poll_errors_are_retried_in_place_with_backoff(
    audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors: list[httpx.Response | Exception] = [
        httpx.Response(503, text="unavailable"),
        httpx.ReadTimeout("slow"),
        httpx.Response(429, json={"error": {"code": 429, "message": "busy"}}),
    ]
    api = Api(jobs={"gen-vid-1": [*errors, done()]})
    sleeps: list[float] = []
    monkeypatch.setattr(video_jobs.time, "sleep", sleeps.append)
    monkeypatch.setattr(video_jobs.random, "uniform", lambda low, high: high)

    provider = video(api, timeout_s=1000.0, retry_backoff_s=1.0, retry_backoff_max_s=2.0)
    provider.generate(req(audio), tmp_path / "shot_1.mp4")

    assert len(api.posts()) == 1
    assert api.polled() == ["gen-vid-1"] * 4
    assert sleeps == [1.0, 2.0, 2.0]


MODERATION = [
    httpx.Response(
        403,
        json={
            "error": {
                "code": 403,
                "message": "Your input was flagged",
                "metadata": {"reasons": ["violence"], "flagged_input": "..."},
            }
        },
    ),
    httpx.Response(400, json={"error": {"code": 400, "message": "content_policy_violation"}}),
]


@pytest.mark.parametrize("answer", MODERATION, ids=["flagged_403", "content_policy_400"])
def test_moderation_at_create_is_final_and_shot_2_recognises_it(
    answer: httpx.Response, frame: Path, tmp_path: Path
) -> None:
    api = Api(create=answer)
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(image=str(frame)), tmp_path / "shot_2.mp4")
    assert not info.value.retryable
    assert any(marker in str(info.value).lower() for marker in MODERATION_MARKERS)


def test_metadata_words_do_not_make_a_rate_limit_final(audio: Path, tmp_path: Path) -> None:
    """Moderation is judged on code + message; metadata is shown but not matched."""
    body = {
        "error": {
            "code": 429,
            "message": "Rate limited upstream",
            "metadata": {"raw": "safety system busy", "provider_name": "ByteDance"},
        }
    }
    api = Api(create=httpx.Response(429, json=body))
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), tmp_path / "c.mp4")
    assert info.value.retryable
    assert "ByteDance" in str(info.value)  # metadata kept for the human


def test_moderation_failure_of_the_job_is_final_and_keeps_the_reason(
    frame: Path, tmp_path: Path
) -> None:
    reason = "Output video flagged by the provider's content moderation"
    api = Api([job("failed", error=reason)])
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(image=str(frame)), tmp_path / "shot_2.mp4")
    assert not info.value.retryable
    assert reason in str(info.value)
    assert any(marker in str(info.value).lower() for marker in MODERATION_MARKERS)


@pytest.mark.parametrize(
    ("status", "error", "retryable"),
    [
        ("failed", "upstream provider error", True),
        ("failed", None, True),
        ("expired", None, True),
        ("failed", "Invalid duration", False),
        ("cancelled", None, False),
    ],
)
def test_other_terminal_states(
    status: str, error: object, retryable: bool, audio: Path, tmp_path: Path
) -> None:
    api = Api([job(status, error=error)])
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), tmp_path / "c.mp4")
    assert info.value.retryable is retryable
    assert status in str(info.value)


# --- never pay twice: unknown create outcome ----------------------------------------------------

UNKNOWN_CREATE_OUTCOMES: list[httpx.Response | Exception] = [
    httpx.ReadTimeout("lost answer"),
    httpx.RemoteProtocolError("server disconnected"),
    httpx.Response(500, json={"error": {"code": 500, "message": "boom"}}),
    httpx.Response(502, text="bad gateway"),
    httpx.Response(408, text="request timeout"),
    httpx.Response(202, text="<html>gateway</html>"),
    httpx.Response(202, json={"status": "pending"}),  # accepted, but no id
]
UNKNOWN_IDS = ["read_timeout", "remote_protocol", "500", "502", "408", "non_json", "no_id"]


@pytest.mark.parametrize("answer", UNKNOWN_CREATE_OUTCOMES, ids=UNKNOWN_IDS)
def test_unknown_create_outcome_is_final_and_blocks_a_second_post(
    answer: httpx.Response | Exception, audio: Path, tmp_path: Path
) -> None:
    out = tmp_path / "shot_1.mp4"
    task_file = out.with_suffix(".task.json")
    first = Api(create=answer)
    with pytest.raises(ProviderError) as info:
        video(first).generate(req(audio), out)
    assert not info.value.retryable
    assert "openrouter.ai/activity" in str(info.value) and str(task_file) in str(info.value)
    assert len(first.posts()) == 1
    assert "create_pending" in saved(out) and "task_id" not in saved(out)

    api = Api()
    with pytest.raises(ProviderError) as again:
        video(api).generate(req(audio), out)
    assert not again.value.retryable
    assert api.requests == []


def test_create_pending_marker_is_on_disk_before_the_post(audio: Path, tmp_path: Path) -> None:
    out = tmp_path / "shot_1.mp4"
    path = out.with_suffix(".task.json")
    api = Api()
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen.append(json.loads(path.read_text(encoding="utf-8")))
        return api(request)

    provider = openrouter_video.OpenRouterVideo(
        REF,
        transport=httpx.MockTransport(handler),
        poll_interval_s=0.0,
        timeout_s=5.0,
        media_host=Host(),
    )
    provider.generate(req(audio), out)

    (before,) = seen
    assert before["create_pending"]["fingerprint"] == before["fingerprint"]  # type: ignore[index]
    after = saved(out)
    assert "create_pending" not in after and after["task_id"] == "gen-vid-1"


@pytest.mark.parametrize("exc", [httpx.ConnectTimeout("t"), httpx.ConnectError("c")])
def test_create_never_sent_is_retryable(exc: Exception, audio: Path, tmp_path: Path) -> None:
    out = tmp_path / "c.mp4"
    with pytest.raises(ProviderError) as info:
        video(Api(create=exc)).generate(req(audio), out)
    assert info.value.retryable
    api = Api()
    video(api).generate(req(audio), out)
    assert len(api.posts()) == 1


# --- never pay twice: resume of a saved job ------------------------------------------------------


def seed(api: Api, request: VideoRequest, out: Path) -> None:
    """First attempt: the job is created, polled once, still running -> retryable timeout."""
    with pytest.raises(ProviderError) as info:
        video(api, timeout_s=0.0).generate(request, out)
    assert info.value.retryable
    assert len(api.posts()) == 1


def test_retry_after_poll_timeout_resumes_the_job_instead_of_paying_again(
    audio: Path, tmp_path: Path
) -> None:
    api = Api(jobs={"gen-vid-1": [job("in_progress"), done(usage={"cost": 0.8})]})
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    result = video(api).generate(req(audio), out)

    assert len(api.posts()) == 1
    assert api.polled(mark) == ["gen-vid-1"]
    assert out.read_bytes() == CLIP
    assert result.cost_usd == pytest.approx(0.8)  # first delivery of that job reports its cost


def test_verified_local_clip_is_reused_without_any_call_at_cost_0(
    audio: Path, tmp_path: Path
) -> None:
    api = Api([done(usage={"cost": 0.8})])
    out = tmp_path / "shot_1.mp4"
    assert video(api).generate(req(audio), out).cost_usd == pytest.approx(0.8)
    assert saved(out)["clip"] == {"sha256": hashlib.sha256(CLIP).hexdigest(), "size": len(CLIP)}
    mark = len(api.requests)

    result = video(api).generate(req(audio), out)

    assert api.requests[mark:] == []
    assert result.cost_usd == 0.0


def test_deleted_local_clip_is_fetched_again_from_the_same_job_at_cost_0(
    audio: Path, tmp_path: Path
) -> None:
    api = Api([done(usage={"cost": 0.8})])
    out = tmp_path / "shot_1.mp4"
    video(api).generate(req(audio), out)
    out.unlink()
    mark = len(api.requests)

    result = video(api).generate(req(audio), out)

    assert api.posts(mark) == []
    assert api.polled(mark) == ["gen-vid-1"]
    assert out.read_bytes() == CLIP
    assert result.cost_usd == 0.0


@pytest.mark.parametrize("status", ["failed", "expired", "cancelled"])
def test_saved_job_that_ended_badly_is_replaced_by_one_new_job(
    status: str, audio: Path, tmp_path: Path
) -> None:
    api = Api(
        jobs={
            "gen-vid-1": [job("in_progress"), job(status, error="upstream error")],
            "gen-vid-2": [done("gen-vid-2")],
        }
    )
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    video(api).generate(req(audio), out)

    assert api.polled(mark) == ["gen-vid-1", "gen-vid-2"]
    assert len(api.posts(mark)) == 1
    assert saved(out)["task_id"] == "gen-vid-2"


GONE = httpx.Response(404, json={"error": {"code": 404, "message": "not found"}})


def test_saved_job_404_is_final_and_kept_never_a_new_post(audio: Path, tmp_path: Path) -> None:
    """The poll path is not confirmed live: a 404 may mean "wrong path", not "gone". Treating
    it as gone would pay for a new clip on every resume."""
    api = Api(jobs={"gen-vid-1": [job("in_progress"), GONE]})
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)

    assert not info.value.retryable
    assert "openrouter.ai/activity" in str(info.value) and "gen-vid-1" in str(info.value)
    assert api.posts(mark) == []
    assert saved(out)["task_id"] == "gen-vid-1"


def test_404_right_after_create_then_resumes_never_pay_again(audio: Path, tmp_path: Path) -> None:
    api = Api(jobs={"gen-vid-1": [GONE]})
    out = tmp_path / "shot_1.mp4"
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)
    assert not info.value.retryable
    for _ in range(2):  # every --resume re-checks the same job
        with pytest.raises(ProviderError):
            video(api).generate(req(audio), out)
    assert len(api.posts()) == 1
    assert api.polled() == ["gen-vid-1"] * 3


@pytest.mark.parametrize("status", [403, 404])
def test_resumed_completed_job_whose_download_fails_is_final_not_replaced(
    status: int, audio: Path, tmp_path: Path
) -> None:
    """The poll just returned the URLs, so a 403/404 is not expiry (that is status "expired"):
    replacing the job would pay again on every resume."""
    api = Api(
        jobs={"gen-vid-1": [job("in_progress"), done()]},
        downloads={SIGNED: httpx.Response(status)},
    )
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)

    assert not info.value.retryable
    assert "openrouter.ai/activity" in str(info.value) and "gen-vid-1" in str(info.value)
    assert api.posts(mark) == []
    assert saved(out)["task_id"] == "gen-vid-1"


def test_unsigned_content_404_then_resumes_never_pay_again(audio: Path, tmp_path: Path) -> None:
    unsigned = "https://openrouter.ai/api/v1/videos/gen-vid-1/content"
    api = Api(
        [job("completed", unsigned_urls=[unsigned])],
        downloads={unsigned: httpx.Response(404)},
    )
    out = tmp_path / "shot_1.mp4"
    for _ in range(3):  # first run + two --resume
        with pytest.raises(ProviderError) as info:
            video(api).generate(req(audio), out)
        assert not info.value.retryable
    assert len(api.posts()) == 1
    assert saved(out)["task_id"] == "gen-vid-1"


def test_saved_job_rejected_by_moderation_is_final_not_replaced(
    audio: Path, tmp_path: Path
) -> None:
    api = Api(jobs={"gen-vid-1": [job("in_progress"), job("failed", error="content moderation")]})
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)

    assert not info.value.retryable
    assert api.posts(mark) == []
    assert "task_id" not in saved(out)


def test_changed_request_creates_a_new_job_never_polls_the_old_one(
    audio: Path, tmp_path: Path
) -> None:
    api = Api(jobs={"gen-vid-1": [job("in_progress"), done()], "gen-vid-2": [done("gen-vid-2")]})
    out = tmp_path / "shot_1.mp4"
    seed(api, req(audio), out)
    mark = len(api.requests)

    video(api).generate(replace(req(audio), prompt="A man in a kitchen"), out)

    assert len(api.posts(mark)) == 1
    assert api.polled(mark) == ["gen-vid-2"]


@pytest.mark.parametrize("state", ["running", "downloaded", "pending_create"])
def test_saved_job_differing_only_in_resolution_is_refused_not_paid_again(
    state: str, audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run with HWVA_VIDEO_RESOLUTION=480p, --resume in a shell without it (-> 720p): the same
    clip, so no new POST; the error says how to resume it."""
    monkeypatch.setenv(openrouter_video.RESOLUTION_ENV, "480p")
    out = tmp_path / "shot_1.mp4"
    if state == "running":
        seed(Api(jobs={"gen-vid-1": [job("in_progress")]}), req(audio), out)
    elif state == "downloaded":
        video(Api()).generate(req(audio), out)
        out.unlink()  # e.g. the run dir was cleaned
    else:
        with pytest.raises(ProviderError):
            video(Api(create=httpx.ReadTimeout("lost"))).generate(req(audio), out)
    monkeypatch.delenv(openrouter_video.RESOLUTION_ENV)
    api = Api()

    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)

    msg = str(info.value)
    assert not info.value.retryable
    assert "480p" in msg and "720p" in msg and openrouter_video.RESOLUTION_ENV in msg
    assert api.requests == []
    monkeypatch.setenv(openrouter_video.RESOLUTION_ENV, "480p")  # set back -> resumes
    if state != "pending_create":
        video(api).generate(req(audio), out)
        assert api.posts() == [] and out.read_bytes() == CLIP


def test_resolution_change_of_a_dead_saved_job_creates_a_new_one(
    audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing possibly paid is pinned (the job failed): another resolution is just a request."""
    out = tmp_path / "shot_1.mp4"
    with pytest.raises(ProviderError):
        video(Api([job("failed", error="upstream error")])).generate(req(audio), out)
    monkeypatch.setenv(openrouter_video.RESOLUTION_ENV, "480p")
    api = Api()
    video(api).generate(req(audio), out)
    assert len(api.posts()) == 1 and api.body()["resolution"] == "480p"


def test_unparsable_task_file_is_final_and_never_posts(audio: Path, tmp_path: Path) -> None:
    out = tmp_path / "shot_1.mp4"
    out.with_suffix(".task.json").write_text("{not json", encoding="utf-8")
    api = Api()
    with pytest.raises(ProviderError) as info:
        video(api).generate(req(audio), out)
    assert not info.value.retryable
    assert "openrouter.ai/activity" in str(info.value)
    assert api.requests == []


# --- task file ----------------------------------------------------------------------------------


def test_task_file_has_no_key_no_base64_the_hosted_urls_and_is_written_atomically(
    audio: Path, frame: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "shot_1.mp4"
    path = out.with_suffix(".task.json")
    real = os.replace
    moves: list[Path] = []

    def spy(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(dst) == path:
            moves.append(Path(src))
        real(src, dst)

    monkeypatch.setattr(video_jobs.os, "replace", spy)
    video(Api()).generate(req(audio, str(frame)), out)

    text = path.read_text(encoding="utf-8")
    assert KEY not in text
    assert base64.b64encode(audio.read_bytes()).decode()[:40] not in text
    assert "data:" not in text
    task = saved(out)
    assert task["request"]["input_references"][0]["audio_url"]["url"].startswith(  # type: ignore[index]
        "https://media.test/"
    )
    media = task["media"]
    assert isinstance(media, list) and len(media) == 2
    assert {m["expires_at"] for m in media} == {EXPIRES.isoformat()}
    assert moves and all(src != path and src.parent == path.parent for src in moves)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["line.wav", "shot_1.last.png", "shot_1.mp4", path.name]
    )


# --- live ---------------------------------------------------------------------------------------


@pytest.mark.live
def test_live_generate_a_short_speaking_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke (PAID ~0.17 USD, run only when the human says so): key, endpoint, body shape,
    litterbox-hosted audio + image in input_references and polling are accepted; a 4 s 480p
    2.0-fast clip with a 3 s tone as audio reference comes back."""
    monkeypatch.delenv(openrouter_video.API_KEY_ENV, raising=False)  # drop the autouse fake key
    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get(openrouter_video.API_KEY_ENV):
        pytest.skip(f"{openrouter_video.API_KEY_ENV} not set")
    path = tmp_path / "line.wav"
    path.write_bytes(wav(3.0))
    image = fakes.FakeImage().generate("a cartoon cat", tmp_path / "character.png").path
    provider = openrouter_video.OpenRouterVideo(
        REF_20F, resolution="480p", poll_interval_s=5.0, media_host=litterbox.Litterbox()
    )
    out = tmp_path / "shot_1.mp4"
    result = provider.generate(
        VideoRequest(
            prompt="A cartoon cat says a short line",
            duration_s=4,
            audio_path=path,
            reference_image=str(image),
        ),
        out,
    )
    assert result.video_path == out and out.stat().st_size > 0

"""uguu.se media host with the HTTP layer mocked (httpx.MockTransport, no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from hwva.providers import registry, uguu
from hwva.providers.base import MediaHost, ProviderError
from hwva.schemas import ModelRef

HOSTED = "https://h.uguu.se/abc123.wav"
OK = {"success": True, "files": [{"hash": "x", "filename": "abc123.wav", "url": HOSTED}]}


class Api:
    def __init__(self, answer: httpx.Response | Exception | None = None) -> None:
        self.answer = answer if answer is not None else httpx.Response(200, json=OK)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def host(api: Api) -> uguu.Uguu:
    return uguu.Uguu(transport=httpx.MockTransport(api))


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "line.wav"
    path.write_bytes(b"RIFF fake wav bytes")
    return path


def test_is_a_media_host() -> None:
    assert isinstance(host(Api()), MediaHost)


def test_upload_is_a_multipart_post_of_files_array(audio: Path) -> None:
    api = Api()
    host(api).upload(audio)
    (request,) = api.requests
    assert request.method == "POST"
    assert str(request.url) == "https://uguu.se/upload"
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b'name="files[]"; filename="line.wav"' in request.content
    assert audio.read_bytes() in request.content
    assert "authorization" not in request.headers  # no account, no key


def test_the_url_comes_from_the_json_and_expires_in_three_hours(audio: Path) -> None:
    before = datetime.now(UTC)
    hosted = host(Api()).upload(audio)
    assert hosted.url == HOSTED
    assert hosted.expires_at is not None
    assert before + timedelta(minutes=179) <= hosted.expires_at <= before + timedelta(minutes=181)


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(200, json={"success": True, "files": [{"url": "http://h.uguu.se/a.wav"}]}),
        httpx.Response(200, json={"success": False, "errorcode": 400}),
        httpx.Response(200, json={"success": True, "files": []}),
        httpx.Response(200, text="<html>error</html>"),
    ],
    ids=["http", "no-files-key", "empty-files", "not-json"],
)
def test_an_answer_without_an_https_url_is_final(answer: httpx.Response, audio: Path) -> None:
    with pytest.raises(ProviderError) as info:
        host(Api(answer)).upload(audio)
    assert not info.value.retryable


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(429, text="slow down"),
        httpx.Response(503, text="unavailable"),
        httpx.ConnectError("no route"),
        httpx.ReadTimeout("slow"),
    ],
    ids=["429", "503", "connect", "read_timeout"],
)
def test_transient_failures_are_retryable(answer: httpx.Response | Exception, audio: Path) -> None:
    """Uploading is free: retrying it never pays twice."""
    with pytest.raises(ProviderError) as info:
        host(Api(answer)).upload(audio)
    assert info.value.retryable


def test_other_client_errors_are_final(audio: Path) -> None:
    with pytest.raises(ProviderError) as info:
        host(Api(httpx.Response(413, text="too big"))).upload(audio)
    assert not info.value.retryable and "413" in str(info.value)


def test_missing_file_is_final_and_nothing_is_sent(tmp_path: Path) -> None:
    api = Api()
    with pytest.raises(ProviderError) as info:
        host(api).upload(tmp_path / "nope.wav")
    assert not info.value.retryable
    assert api.requests == []


def test_registry_builds_it_and_the_profile_uses_it() -> None:
    ref = ModelRef(provider="uguu", model="uguu", version="uguu")
    assert isinstance(registry._FACTORIES["media_host"]["uguu"](ref), uguu.Uguu)
    root = Path(__file__).resolve().parents[2]
    profile = registry.load_profile("premium-openrouter", root / "config" / "profiles.yaml")
    assert profile["media_host"].provider == "uguu"

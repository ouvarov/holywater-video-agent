"""Litterbox media host with the HTTP layer mocked (httpx.MockTransport, no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from hwva.providers import litterbox, registry
from hwva.providers.base import MediaHost, ProviderError
from hwva.schemas import ModelRef

HOSTED = "https://litter.catbox.moe/abc123.wav"


class Api:
    def __init__(self, answer: httpx.Response | Exception | None = None) -> None:
        self.answer = answer if answer is not None else httpx.Response(200, text=HOSTED + "\n")
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def host(api: Api) -> litterbox.Litterbox:
    return litterbox.Litterbox(transport=httpx.MockTransport(api))


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "line.wav"
    path.write_bytes(b"RIFF fake wav bytes")
    return path


def test_is_a_media_host() -> None:
    assert isinstance(host(Api()), MediaHost)


def test_upload_is_a_multipart_post_with_a_1h_lifetime(audio: Path) -> None:
    api = Api()
    host(api).upload(audio)
    (request,) = api.requests
    assert request.method == "POST"
    assert str(request.url) == "https://litterbox.catbox.moe/resources/internals/api.php"
    assert request.headers["content-type"].startswith("multipart/form-data")
    body = request.content
    assert b'name="reqtype"' in body and b"fileupload" in body
    assert b'name="time"' in body and b"1h" in body
    assert b'name="fileToUpload"; filename="line.wav"' in body
    assert audio.read_bytes() in body
    assert "authorization" not in request.headers  # no account, no key


def test_the_answer_body_is_the_https_url_and_it_expires_in_an_hour(audio: Path) -> None:
    before = datetime.now(UTC)
    hosted = host(Api()).upload(audio)
    assert hosted.url == HOSTED
    assert hosted.expires_at is not None
    assert before + timedelta(minutes=59) <= hosted.expires_at <= before + timedelta(minutes=61)


@pytest.mark.parametrize(
    "body", ["http://litter.catbox.moe/abc.wav", "<html>error</html>", ""], ids=str
)
def test_an_answer_that_is_not_an_https_url_is_final(body: str, audio: Path) -> None:
    with pytest.raises(ProviderError) as info:
        host(Api(httpx.Response(200, text=body))).upload(audio)
    assert not info.value.retryable and "https" in str(info.value)


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


def test_registry_builds_it_from_the_profile_option() -> None:
    ref = ModelRef(provider="litterbox", model="litterbox", version="1h")
    assert isinstance(registry._FACTORIES["media_host"]["litterbox"](ref), litterbox.Litterbox)

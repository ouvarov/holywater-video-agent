"""OpenRouterImage with the HTTP layer mocked (httpx.MockTransport, no network)."""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import httpx
import pytest
from conftest import needs_ffmpeg, probe

from hwva.providers import openrouter_image, registry
from hwva.providers.base import ImageProvider, ProviderError
from hwva.schemas import ModelRef

REF = ModelRef(
    provider="openrouter-image",
    model="google/gemini-2.5-flash-image",
    version="google/gemini-2.5-flash-image",
)
KEY = "test-openrouter-key-not-real-0123456789"
URL = "https://openrouter.ai/api/v1/chat/completions"
PNG = b"\x89PNG\r\n\x1a\n fake png bytes"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(openrouter_image.API_KEY_ENV, KEY)


def answer(
    image: bytes = PNG, mime: str = "image/png", cost: object = 0.0387, images: bool = True
) -> httpx.Response:
    url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
    message: dict[str, object] = {"role": "assistant", "content": ""}
    if images:
        message["images"] = [{"type": "image_url", "image_url": {"url": url}}]
    return httpx.Response(200, json={"choices": [{"message": message}], "usage": {"cost": cost}})


class Api:
    def __init__(self, response: httpx.Response | Exception | None = None) -> None:
        self.response = response if response is not None else answer()
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def provider(api: Api) -> openrouter_image.OpenRouterImage:
    return openrouter_image.OpenRouterImage(REF, transport=httpx.MockTransport(api))


def test_is_an_image_provider() -> None:
    assert isinstance(provider(Api()), ImageProvider)


def test_request_is_a_bearer_chat_completion_asking_for_a_9_16_image(tmp_path: Path) -> None:
    api = Api()
    provider(api).generate("a cartoon cat", tmp_path / "character.png")
    (request,) = api.requests
    assert request.method == "POST" and str(request.url) == URL
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["model"] == "google/gemini-2.5-flash-image"
    assert body["messages"] == [{"role": "user", "content": "a cartoon cat"}]
    assert body["modalities"] == ["image", "text"]
    assert body["image_config"] == {"aspect_ratio": "9:16"}
    assert body["usage"] == {"include": True}


def test_the_data_uri_image_is_written_and_usage_cost_reported(tmp_path: Path) -> None:
    out = tmp_path / "character.png"
    result = provider(Api()).generate("a cartoon cat", out)
    assert result.path == out and out.read_bytes() == PNG
    assert result.cost_usd == pytest.approx(0.0387)


@pytest.mark.parametrize("cost", [None, "0.04", True, -1])
def test_unusable_cost_is_zero(cost: object, tmp_path: Path) -> None:
    result = provider(Api(answer(cost=cost))).generate("p", tmp_path / "c.png")
    assert result.cost_usd == 0.0


@needs_ffmpeg
def test_a_jpeg_answer_is_converted_to_png(tmp_path: Path) -> None:
    jpeg = tmp_path / "src.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=90x160",
         "-frames:v", "1", str(jpeg)],
        check=True,
    )  # fmt: skip
    out = tmp_path / "character.png"
    provider(Api(answer(jpeg.read_bytes(), "image/jpeg"))).generate("p", out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    stream = probe(out)["streams"][0]
    assert (stream["width"], stream["height"]) == (90, 160)


def test_an_answer_without_an_image_is_retryable_and_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "c.png"
    with pytest.raises(ProviderError) as info:
        provider(Api(answer(images=False))).generate("p", out)
    assert info.value.retryable
    assert not out.exists()


def test_moderation_is_final(tmp_path: Path) -> None:
    body = {"error": {"code": 400, "message": "Request flagged by content moderation"}}
    with pytest.raises(ProviderError) as info:
        provider(Api(httpx.Response(400, json=body))).generate("p", tmp_path / "c.png")
    assert not info.value.retryable and "moderation" in str(info.value)
    assert KEY not in str(info.value)


@pytest.mark.parametrize(
    ("response", "retryable"),
    [
        (httpx.Response(429, json={"error": {"code": 429, "message": "busy"}}), True),
        (httpx.Response(502, text="bad gateway"), True),
        (httpx.ReadTimeout("slow"), True),
        (httpx.Response(402, json={"error": {"code": 402, "message": "no credits"}}), False),
    ],
    ids=["429", "502", "timeout", "402"],
)
def test_errors(response: httpx.Response | Exception, retryable: bool, tmp_path: Path) -> None:
    with pytest.raises(ProviderError) as info:
        provider(Api(response)).generate("p", tmp_path / "c.png")
    assert info.value.retryable is retryable


def test_missing_key_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(openrouter_image.API_KEY_ENV)
    with pytest.raises(ValueError, match=openrouter_image.API_KEY_ENV):
        openrouter_image.OpenRouterImage(REF)


def test_registry_builds_it_from_the_ref() -> None:
    built = registry._FACTORIES["image"]["openrouter-image"](REF)
    assert isinstance(built, openrouter_image.OpenRouterImage) and built.model == REF.model

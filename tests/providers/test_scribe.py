"""ScribeStt with the HTTP layer mocked (httpx.MockTransport, no network). One opt-in live test."""

from __future__ import annotations

import email.policy
import io
import os
import wave
from collections.abc import Callable
from email.parser import BytesParser
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

from hwva.providers import registry, scribe
from hwva.providers.base import ProviderError, SttProvider
from hwva.schemas import ModelRef
from hwva.text import diff_words

ROOT = Path(__file__).resolve().parents[2]
REF = ModelRef(provider="scribe", model="scribe", version="scribe_v2")
KEY = "test-key-not-real-0123456789"
LINE = "Я ніколи не здамся, навіть якщо впаду сто разів — пам'ятай це!"

Handler = Callable[[httpx.Request], httpx.Response]


def wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(seconds * rate))
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(scribe.API_KEY_ENV, KEY)


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "line.wav"
    path.write_bytes(wav())
    return path


class Recorder:
    """MockTransport handler: records requests, answers with a fixed response or raises."""

    def __init__(self, answer: httpx.Response | Exception) -> None:
        self.answer = answer
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def stt(answer: httpx.Response | Exception) -> tuple[scribe.ScribeStt, Recorder]:
    recorder = Recorder(answer)
    return scribe.ScribeStt(REF, transport=httpx.MockTransport(recorder)), recorder


def ok(text: object = LINE) -> httpx.Response:
    return httpx.Response(200, json={"language_code": "ukr", "text": text, "words": []})


def form(request: httpx.Request) -> dict[str, tuple[str | None, str | None, bytes]]:
    """Multipart body -> {field: (filename, content type, bytes)}."""
    head = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    message = BytesParser(policy=email.policy.default).parsebytes(head + request.content)
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        assert isinstance(name, str) and isinstance(payload, bytes)
        fields[name] = (part.get_filename(), part.get_content_type(), payload)
    return fields


def fails(answer: httpx.Response | Exception, audio: Path) -> ProviderError:
    provider, _ = stt(answer)
    with pytest.raises(ProviderError) as info:
        provider.transcribe(audio, "uk")
    assert KEY not in str(info.value)
    return info.value


# --- request ------------------------------------------------------------------------------------


def test_is_an_stt_provider() -> None:
    provider, _ = stt(ok())
    assert isinstance(provider, SttProvider)


def test_request_is_a_post_with_the_key_header(audio: Path) -> None:
    provider, rec = stt(ok())
    provider.transcribe(audio, "uk")
    (request,) = rec.requests
    assert request.method == "POST"
    assert str(request.url) == "https://api.elevenlabs.io/v1/speech-to-text"
    assert request.headers["xi-api-key"] == KEY
    assert request.headers["content-type"].startswith("multipart/form-data")


@pytest.mark.parametrize("lang", ["uk", "en"])
def test_multipart_fields_pinned_model_language_and_file(audio: Path, lang: str) -> None:
    provider, rec = stt(ok())
    provider.transcribe(audio, lang)  # type: ignore[arg-type]
    fields = form(rec.requests[0])
    assert set(fields) == {"model_id", "language_code", "file"}
    assert fields["model_id"][2] == b"scribe_v2"  # = REF.version, the exact model code
    assert fields["language_code"][2] == lang.encode()
    filename, mime, content = fields["file"]
    assert filename == "line.wav"
    assert mime == "audio/wav"
    assert content == audio.read_bytes()


def test_mp3_is_sent_as_audio_mpeg(tmp_path: Path) -> None:
    mp3 = tmp_path / "final_audio.mp3"
    mp3.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 400)
    provider, rec = stt(ok())
    provider.transcribe(mp3, "uk")
    assert form(rec.requests[0])["file"][:2] == ("final_audio.mp3", "audio/mpeg")


def test_timeout_is_sixty_seconds(audio: Path) -> None:
    provider, rec = stt(ok())
    provider.transcribe(audio, "uk")
    timeout = rec.requests[0].extensions["timeout"]
    assert timeout == {"connect": 60.0, "read": 60.0, "write": 60.0, "pool": 60.0}


# --- response -----------------------------------------------------------------------------------


def test_returns_the_text_field_untouched(audio: Path) -> None:
    heard = "  Я ніколи не здамся, навіть якщо впаду сто разів. Пам'ятай це?  "
    provider, _ = stt(ok(heard))
    assert provider.transcribe(audio, "uk") == heard


@pytest.mark.parametrize(
    ("line", "heard"),
    [
        (LINE, "Я ніколи не здамся, навіть якщо впаду сто разів. Пам'ятай це."),
        ("Ти справді прийшов — сам?", "Ти справді прийшов? Сам?"),
        ("Не зараз — потім", "Не зараз, потім."),
    ],
    ids=["dash-to-period", "dash-to-question", "trailing-period"],
)
def test_scribe_punctuation_differences_pass_the_verbatim_diff(
    line: str, heard: str, audio: Path
) -> None:
    """What Scribe returned in the live check: the words, with its own punctuation."""
    provider, _ = stt(ok(heard))
    assert diff_words(line, provider.transcribe(audio, "uk"), "uk") == []


def test_a_real_word_change_is_still_caught(audio: Path) -> None:
    provider, _ = stt(ok("Я ніколи не здамся, навіть якщо впаду двісті разів. Пам'ятай це."))
    assert diff_words(LINE, provider.transcribe(audio, "uk"), "uk") != []


def test_empty_transcript_is_returned_not_raised(audio: Path) -> None:
    """Silence is an answer; the voice stage's diff reports it as missing words."""
    provider, _ = stt(ok(""))
    assert provider.transcribe(audio, "uk") == ""


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>gateway</html>"),
        httpx.Response(200, json={"detail": "no text here"}),
        httpx.Response(200, json=["text"]),
        httpx.Response(200, json={"text": None}),
        httpx.Response(200, json={"text": 42}),
    ],
    ids=["not-json", "no-text-field", "not-an-object", "text-null", "text-number"],
)
def test_unreadable_response_is_a_provider_error(response: httpx.Response, audio: Path) -> None:
    fails(response, audio)


# --- error mapping ------------------------------------------------------------------------------


def api_error(status: int) -> httpx.Response:
    return httpx.Response(status, json={"detail": {"status": "x", "message": f"boom {status}"}})


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_rate_limit_and_server_errors_are_retryable(status: int, audio: Path) -> None:
    error = fails(api_error(status), audio)
    assert error.retryable is True
    assert str(status) in str(error)
    assert f"boom {status}" in str(error)  # API's message is kept for the manifest


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_client_errors_are_not_retryable(status: int, audio: Path) -> None:
    error = fails(api_error(status), audio)
    assert error.retryable is False
    assert str(status) in str(error)


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.WriteTimeout("write timed out"),
        httpx.ConnectError("connection refused"),
        httpx.RemoteProtocolError("server disconnected"),
    ],
)
def test_timeouts_and_connection_errors_are_retryable(exc: Exception, audio: Path) -> None:
    assert fails(exc, audio).retryable is True


def test_programming_errors_are_not_disguised_as_provider_errors(audio: Path) -> None:
    provider, _ = stt(TypeError("bug"))
    with pytest.raises(TypeError):
        provider.transcribe(audio, "uk")


# --- key and registry ---------------------------------------------------------------------------


def test_missing_key_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(scribe.API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=scribe.API_KEY_ENV):
        scribe.ScribeStt(REF)


def test_registry_builds_scribe_from_the_ref() -> None:
    provider = registry._FACTORIES["stt"]["scribe"](REF)
    assert isinstance(provider, scribe.ScribeStt)
    assert provider.model == REF.version


def test_premium_profile_pins_model_and_exact_version() -> None:
    ref = registry.load_profile("premium", ROOT / "config" / "profiles.yaml")["stt"]
    assert ref == REF


# --- live ---------------------------------------------------------------------------------------


@pytest.mark.live
def test_live_transcribe_accepts_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke: key, URL and multipart shape are accepted (1 s of a quiet tone, so the transcript
    itself is meaningless). Word-for-word accuracy is the voice / final_check stages' job."""
    monkeypatch.delenv(scribe.API_KEY_ENV, raising=False)  # drop the autouse fake key
    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get(scribe.API_KEY_ENV):
        pytest.skip(f"{scribe.API_KEY_ENV} not set")
    path = tmp_path / "tone.wav"
    path.write_bytes(wav(1.0))
    assert isinstance(scribe.ScribeStt(REF).transcribe(path, "uk"), str)

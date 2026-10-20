"""ElevenLabsTts with the HTTP layer mocked (httpx.MockTransport, no network). One live test."""

from __future__ import annotations

import json
import os
import wave
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

from hwva import voices
from hwva.providers import elevenlabs_tts, registry
from hwva.providers.base import ProviderError, TtsProvider
from hwva.schemas import ModelRef, VoicePersona

ROOT = Path(__file__).resolve().parents[2]
REF = ModelRef(provider="elevenlabs-tts", model="eleven_v4", version="eleven_v4")
KEY = "test-key-not-real-0123456789"
LINE = "Я ніколи не здамся, навіть якщо впаду сто разів — пам'ятай це!"
WILL, SARAH, HARRY, LAURA = (
    "bIHbv24MWmeRgasZH58o",
    "EXAVITQu4vr4xnSDxMaL",
    "SOYHLrjzK2X1ezoPC6cr",
    "FGY2WhTYpPnrIDTdsKH5",
)


def pcm(seconds: float = 0.5, rate: int = 24000) -> bytes:
    return b"\x01\x00" * int(seconds * rate)


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(elevenlabs_tts.API_KEY_ENV, KEY)


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


def tts(answer: httpx.Response | Exception) -> tuple[elevenlabs_tts.ElevenLabsTts, Recorder]:
    recorder = Recorder(answer)
    return elevenlabs_tts.ElevenLabsTts(REF, transport=httpx.MockTransport(recorder)), recorder


def ok(body: bytes | None = None) -> httpx.Response:
    return httpx.Response(
        200, content=pcm() if body is None else body, headers={"content-type": "audio/pcm"}
    )


def body(request: httpx.Request) -> dict[str, object]:
    data = json.loads(request.content)
    assert isinstance(data, dict)
    return data


def fails(answer: httpx.Response | Exception, out: Path) -> ProviderError:
    provider, _ = tts(answer)
    with pytest.raises(ProviderError) as info:
        provider.synthesize(LINE, "uk", out)
    assert KEY not in str(info.value)
    assert not out.exists()
    return info.value


# --- request ------------------------------------------------------------------------------------


def test_is_a_tts_provider() -> None:
    provider, _ = tts(ok())
    assert isinstance(provider, TtsProvider)


def test_request_is_a_post_with_key_header_and_pcm_format(tmp_path: Path) -> None:
    provider, rec = tts(ok())
    provider.synthesize(LINE, "uk", tmp_path / "line.wav")
    (request,) = rec.requests
    assert request.method == "POST"
    assert request.url.scheme == "https"
    assert request.url.host == "api.elevenlabs.io"
    assert request.url.path == f"/v1/text-to-speech/{elevenlabs_tts.VOICE}"
    assert dict(request.url.params) == {"output_format": "pcm_24000"}
    assert request.headers["xi-api-key"] == KEY
    assert request.headers["content-type"] == "application/json"


@pytest.mark.parametrize("lang", ["uk", "en"])
def test_body_is_the_exact_line_pinned_model_and_language(tmp_path: Path, lang: str) -> None:
    provider, rec = tts(ok())
    provider.synthesize(LINE, lang, tmp_path / "line.wav")  # type: ignore[arg-type]
    assert body(rec.requests[0]) == {"text": LINE, "model_id": "eleven_v4", "language_code": lang}


@pytest.mark.parametrize(
    "line",
    ["  Я тут головний  ", "Пам’ятай це!", "Пам'ятай це!", "Ну...\nі що?"],
    ids=["outer-spaces", "typographic-apostrophe", "ascii-apostrophe", "newline"],
)
def test_odd_whitespace_and_apostrophes_are_sent_untouched(line: str, tmp_path: Path) -> None:
    provider, rec = tts(ok())
    provider.synthesize(line, "uk", tmp_path / "line.wav")
    assert body(rec.requests[0])["text"] == line


def test_timeout_is_sixty_seconds(tmp_path: Path) -> None:
    provider, rec = tts(ok())
    provider.synthesize(LINE, "uk", tmp_path / "line.wav")
    timeout = rec.requests[0].extensions["timeout"]
    assert timeout == {"connect": 60.0, "read": 60.0, "write": 60.0, "pool": 60.0}


# --- voice --------------------------------------------------------------------------------------


def test_the_passed_voice_is_the_one_in_the_url(
    tmp_path: Path, voices_yaml: Callable[[str], Path]
) -> None:
    """The stage's pinned mapping wins; voices.yaml says something else and is not read."""
    voices_yaml(f"elevenlabs-tts:\n  male-adult: {{voice: {WILL}, style: null}}\n")
    pinned = voices.Voice(key="male-adult", voice="pinnedVoiceId123", style="ignored style")
    provider, rec = tts(ok())
    provider.synthesize(
        LINE, "uk", tmp_path / "line.wav", VoicePersona(gender="male", age="adult"), voice=pinned
    )
    (request,) = rec.requests
    assert request.url.path == "/v1/text-to-speech/pinnedVoiceId123"
    assert "ignored style" not in request.content.decode()  # no style field, never spoken


def test_pinned_voice_does_not_read_voices_yaml(
    tmp_path: Path, voices_yaml: Callable[[str], Path]
) -> None:
    voices_yaml("elevenlabs-tts: [this is: not a mapping\n")  # would raise if it were read
    pinned = voices.Voice(key="female-adult", voice=SARAH, style=None)
    provider, rec = tts(ok())
    provider.synthesize(LINE, "uk", tmp_path / "line.wav", voice=pinned)
    assert rec.requests[0].url.path == f"/v1/text-to-speech/{SARAH}"


def test_voice_id_is_url_quoted(tmp_path: Path) -> None:
    provider, rec = tts(ok())
    odd = voices.Voice(key="male-adult", voice="a/b?c", style=None)
    provider.synthesize(LINE, "uk", tmp_path / "line.wav", voice=odd)
    assert rec.requests[0].url.raw_path.startswith(b"/v1/text-to-speech/a%2Fb%3Fc?")


# Seeded config/voices.yaml: premade voices only; child / elderly -> same gender adult.
SEEDED = {
    "male-adult": WILL,
    "female-adult": SARAH,
    "male-teen": HARRY,
    "female-teen": LAURA,
    "male-child": WILL,
    "female-child": SARAH,
    "male-elderly": WILL,
    "female-elderly": SARAH,
}


@pytest.mark.parametrize("key", list(SEEDED))
def test_seeded_config_maps_every_persona_to_a_premade_voice(key: str, tmp_path: Path) -> None:
    provider, rec = tts(ok())
    provider.synthesize(LINE, "uk", tmp_path / "line.wav", voices.persona_of(key))  # repo config
    assert rec.requests[0].url.path == f"/v1/text-to-speech/{SEEDED[key]}"
    assert body(rec.requests[0])["text"] == LINE


def test_seeded_config_has_null_styles_and_no_child_or_elderly_entries() -> None:
    seeded = voices.load("elevenlabs-tts", ROOT / "config" / "voices.yaml")
    assert set(seeded) == {"male-adult", "female-adult", "male-teen", "female-teen"}
    assert all(v.style is None for v in seeded.values())


def test_gemini_tts_voices_are_kept_as_the_alternative() -> None:
    assert len(voices.load("gemini-tts", ROOT / "config" / "voices.yaml")) == 8


def test_provider_without_voices_is_not_retryable(
    tmp_path: Path, voices_yaml: Callable[[str], Path]
) -> None:
    voices_yaml("fake:\n  male-adult: {voice: sine-140hz, style: null}\n")
    provider, rec = tts(ok())
    with pytest.raises(ProviderError) as info:
        provider.synthesize(
            LINE, "uk", tmp_path / "line.wav", VoicePersona(gender="male", age="adult")
        )
    assert info.value.retryable is False
    assert rec.requests == []


# --- output -------------------------------------------------------------------------------------


def test_wraps_the_pcm_into_a_24khz_mono_16bit_wav(tmp_path: Path) -> None:
    raw = pcm(1.0)
    provider, _ = tts(ok(raw))
    out = tmp_path / "line.wav"
    assert provider.synthesize(LINE, "uk", out) == out
    with wave.open(str(out)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
        assert w.readframes(w.getnframes()) == raw


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"", headers={"content-type": "audio/pcm"}),
        httpx.Response(200, content=b"\x01\x00\x02", headers={"content-type": "audio/pcm"}),
        httpx.Response(200, json={"detail": "not audio"}),
        httpx.Response(200, text="<html>gateway</html>"),
    ],
    ids=["empty", "odd-length", "json", "html"],
)
def test_no_readable_audio_is_a_provider_error_and_nothing_is_written(
    response: httpx.Response, tmp_path: Path
) -> None:
    fails(response, tmp_path / "line.wav")


def test_a_stale_file_is_removed_when_synthesis_fails(tmp_path: Path) -> None:
    out = tmp_path / "line.wav"
    out.write_bytes(b"old audio from an earlier attempt")
    fails(httpx.Response(500, text="boom"), out)


# --- error mapping ------------------------------------------------------------------------------


def api_error(status: int, code: str = "x") -> httpx.Response:
    return httpx.Response(status, json={"detail": {"status": code, "message": f"boom {status}"}})


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_rate_limit_and_server_errors_are_retryable(status: int, tmp_path: Path) -> None:
    error = fails(api_error(status), tmp_path / "line.wav")
    assert error.retryable is True
    assert str(status) in str(error)
    assert f"boom {status}" in str(error)  # API's message is kept for the manifest


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_client_errors_are_not_retryable(status: int, tmp_path: Path) -> None:
    error = fails(api_error(status), tmp_path / "line.wav")
    assert error.retryable is False
    assert str(status) in str(error)


def test_library_voice_on_free_tier_is_paid_plan_required_not_retryable(tmp_path: Path) -> None:
    error = fails(api_error(402, "paid_plan_required"), tmp_path / "line.wav")
    assert error.retryable is False
    assert "paid_plan_required" in str(error)
    assert elevenlabs_tts.VOICE in str(error)  # names the voice that needs the plan


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
def test_timeouts_and_connection_errors_are_retryable(exc: Exception, tmp_path: Path) -> None:
    assert fails(exc, tmp_path / "line.wav").retryable is True


def test_programming_errors_are_not_disguised_as_provider_errors(tmp_path: Path) -> None:
    provider, _ = tts(TypeError("bug"))
    with pytest.raises(TypeError):
        provider.synthesize(LINE, "uk", tmp_path / "line.wav")


# --- key, registry, profiles --------------------------------------------------------------------


def test_missing_key_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(elevenlabs_tts.API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=elevenlabs_tts.API_KEY_ENV):
        elevenlabs_tts.ElevenLabsTts(REF)


def test_registry_builds_elevenlabs_tts_from_the_ref() -> None:
    provider = registry._FACTORIES["tts"]["elevenlabs-tts"](REF)
    assert isinstance(provider, elevenlabs_tts.ElevenLabsTts)
    assert provider.model == REF.version


@pytest.mark.parametrize("profile", ["premium", "real-no-video"])
def test_real_profiles_default_to_elevenlabs_with_pinned_version(profile: str) -> None:
    ref = registry.load_profile(profile, ROOT / "config" / "profiles.yaml")["tts"]
    assert ref == REF


def test_gemini_tts_stays_registered() -> None:
    assert "gemini-tts" in registry._FACTORIES["tts"]


# --- live ---------------------------------------------------------------------------------------


@pytest.mark.live
def test_live_synthesize_ukrainian(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke: key, URL, body and pcm_24000 are accepted on this plan and give a WAV. Whether the
    words are right is the `voice` stage's STT check."""
    monkeypatch.delenv(elevenlabs_tts.API_KEY_ENV, raising=False)  # drop the autouse fake key
    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get(elevenlabs_tts.API_KEY_ENV):
        pytest.skip(f"{elevenlabs_tts.API_KEY_ENV} not set")
    out = elevenlabs_tts.ElevenLabsTts(REF).synthesize(
        "Я тут головний.", "uk", tmp_path / "line.wav", VoicePersona(gender="male", age="adult")
    )
    with wave.open(str(out)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() / w.getframerate() > 0.5

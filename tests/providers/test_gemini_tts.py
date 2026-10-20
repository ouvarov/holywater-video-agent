"""GeminiTts with the Interactions client faked (no network). One opt-in live smoke test."""

from __future__ import annotations

import base64
import io
import os
import wave
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from dotenv import load_dotenv
from google.genai._gaos import google_genai
from google.genai._gaos.lib.compat_errors import APIConnectionError, APIError, APITimeoutError
from google.genai.interactions import CreateModelInteraction

from hwva.providers import gemini, gemini_tts, registry
from hwva.providers.base import ProviderError, TtsProvider
from hwva.schemas import ModelRef, VoicePersona

ROOT = Path(__file__).resolve().parents[2]
REF = ModelRef(provider="gemini-tts", model="gemini-3.8-flash-tts", version="gemini-3.8-flash-tts")
LINE = "Я ніколи не здамся, навіть якщо впаду сто разів — пам'ятай це!"


def wav(seconds: float = 0.5, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(seconds * rate))
    return buf.getvalue()


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def done(data: str | None, mime: str | None = "audio/wav", errors: Any = None) -> SimpleNamespace:
    audio = SimpleNamespace(type="audio", data=data, mime_type=mime)
    return SimpleNamespace(status="completed", output_audio=audio, errors=errors)


class FakeInteractions:
    """Records every create() call; returns the scripted answer, or raises it."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def tts(answer: Any) -> tuple[gemini_tts.GeminiTts, FakeInteractions]:
    fake = FakeInteractions(answer)
    return gemini_tts.GeminiTts(REF, interactions=fake), fake


def http_error(status: int, message: str = "boom") -> APIError:
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/interactions")
    response = httpx.Response(status, request=request, json={"error": {"message": message}})
    return APIError.generate(status, {"error": {"message": message}}, message, response)


def fails(answer: Any, out: Path) -> ProviderError:
    provider, _ = tts(answer)
    with pytest.raises(ProviderError) as info:
        provider.synthesize(LINE, "uk", out)
    assert not out.exists()
    return info.value


# --- request ------------------------------------------------------------------------------------


def test_is_a_tts_provider() -> None:
    provider, _ = tts(done(b64(wav())))
    assert isinstance(provider, TtsProvider)


def test_request_uses_interactions_api_pinned_model_voice_and_audio(tmp_path: Path) -> None:
    provider, fake = tts(done(b64(wav())))
    provider.synthesize(LINE, "uk", tmp_path / "line.wav")
    (call,) = fake.calls
    assert call["model"] == REF.version
    assert call["response_format"] == {"type": "audio"}
    assert call["generation_config"] == {"speech_config": [{"voice": "Kore"}]}
    assert call["timeout"] > 0


def test_the_only_text_sent_is_the_line_byte_exact(tmp_path: Path) -> None:
    provider, fake = tts(done(b64(wav())))
    provider.synthesize(LINE, "uk", tmp_path / "line.wav")
    (turn,) = fake.calls[0]["input"]
    assert turn["type"] == "user_input"
    (block,) = turn["content"]
    assert block["type"] == "text"
    assert block["text"] == LINE  # no prefix, no instruction, no normalisation
    (note,) = block["annotations"]
    assert note["type"] == "speech_metadata"
    assert note["style"]
    assert LINE not in note["style"]


@pytest.mark.parametrize(
    "line",
    ["  Памʼятай це!  ", "Пам’ятай,\nце  важливо…", "Мама сказала: «Ні»."],
    ids=["outer-spaces-u02bc", "u2019-newline-double-space", "guillemets"],
)
def test_odd_whitespace_and_apostrophes_are_sent_untouched(line: str, tmp_path: Path) -> None:
    provider, fake = tts(done(b64(wav())))
    provider.synthesize(line, "uk", tmp_path / "line.wav")
    assert fake.calls[0]["input"][0]["content"][0]["text"] == line


def test_lang_is_not_sent_as_a_supported_language(tmp_path: Path) -> None:
    """Ukrainian isn't in the official TTS language list: we don't pass a language code."""
    provider, fake = tts(done(b64(wav())))
    provider.synthesize(LINE, "uk", tmp_path / "line.wav")
    assert "language" not in str(fake.calls[0])
    assert "'uk'" not in str(fake.calls[0])


def test_request_is_accepted_by_the_sdk_request_model(tmp_path: Path) -> None:
    """The fake accepts any kwargs; the SDK's own request model doesn't. Offline."""
    provider, fake = tts(done(b64(wav())))
    provider.synthesize(LINE, "uk", tmp_path / "line.wav")
    body = {k: v for k, v in fake.calls[0].items() if k != "timeout"}
    assert set(body) <= google_genai._CREATE_BODY_KEYS
    request = CreateModelInteraction(**body)
    assert request.generation_config is not None
    assert request.generation_config.speech_config is not None
    dumped = request.model_dump(by_alias=True, exclude_none=True)
    assert dumped["input"][0]["content"][0]["text"] == LINE
    assert dumped["input"][0]["content"][0]["annotations"][0]["type"] == "speech_metadata"


# --- voice persona (specs/voice.md) -------------------------------------------

CHEERFUL = "a cheerful man in his thirties"
EXCITED_GIRL = "an excited 7-year-old girl"
# The `fake` entry must be ignored: GeminiTts reads its own provider key, `gemini-tts`.
TEST_VOICES = (
    "fake:\n"
    '  male-adult: {voice: NotAGeminiVoice, style: "wrong provider"}\n'
    "gemini-tts:\n"
    f'  male-adult: {{voice: Puck, style: "{CHEERFUL}"}}\n'
    f'  female-child: {{voice: Leda, style: "{EXCITED_GIRL}"}}\n'
    "  female-adult: {voice: Aoede, style: null}\n"
)
STYLED = [
    (VoicePersona(gender="male", age="adult"), "Puck", CHEERFUL),
    (VoicePersona(gender="female", age="child"), "Leda", EXCITED_GIRL),
]
STYLED_IDS = ["male-adult", "female-child"]


def _persona_call(persona: VoicePersona, tmp_path: Path, line: str = LINE) -> dict[str, Any]:
    provider, fake = tts(done(b64(wav())))
    provider.synthesize(line, "uk", tmp_path / "line.wav", persona=persona)
    (call,) = fake.calls
    return call


@pytest.mark.parametrize(("persona", "voice", "style"), STYLED, ids=STYLED_IDS)
def test_persona_selects_the_voice_from_config(
    persona: VoicePersona,
    voice: str,
    style: str,
    tmp_path: Path,
    voices_yaml: Callable[[str], Path],
) -> None:
    voices_yaml(TEST_VOICES)
    call = _persona_call(persona, tmp_path)
    assert call["generation_config"] == {"speech_config": [{"voice": voice}]}


@pytest.mark.parametrize(("persona", "voice", "style"), STYLED, ids=STYLED_IDS)
def test_persona_style_goes_only_into_speech_metadata(
    persona: VoicePersona,
    voice: str,
    style: str,
    tmp_path: Path,
    voices_yaml: Callable[[str], Path],
) -> None:
    voices_yaml(TEST_VOICES)
    call = _persona_call(persona, tmp_path)
    (turn,) = call["input"]
    (block,) = turn["content"]
    assert block["text"] == LINE  # the spoken text is the line, byte-exact
    assert style not in block["text"]
    (note,) = block["annotations"]
    assert note == {"type": "speech_metadata", "style": style}
    assert str(call).count(style) == 1  # nowhere else in the request


def test_null_style_falls_back_to_the_default_style(
    tmp_path: Path, voices_yaml: Callable[[str], Path]
) -> None:
    voices_yaml(TEST_VOICES)
    call = _persona_call(VoicePersona(gender="female", age="adult"), tmp_path)
    assert call["generation_config"] == {"speech_config": [{"voice": "Aoede"}]}
    note = call["input"][0]["content"][0]["annotations"][0]
    assert note == {"type": "speech_metadata", "style": gemini_tts.STYLE}
    assert call["input"][0]["content"][0]["text"] == LINE


def test_persona_request_is_accepted_by_the_sdk_request_model(
    tmp_path: Path, voices_yaml: Callable[[str], Path]
) -> None:
    voices_yaml(TEST_VOICES)
    call = _persona_call(VoicePersona(gender="female", age="child"), tmp_path)
    body = {k: v for k, v in call.items() if k != "timeout"}
    dumped = CreateModelInteraction(**body).model_dump(by_alias=True, exclude_none=True)
    assert dumped["input"][0]["content"][0]["text"] == LINE
    assert dumped["input"][0]["content"][0]["annotations"][0]["style"] == EXCITED_GIRL


# Seeded config/voices.yaml: persona -> Gemini voice.
SEEDED = {
    "male-adult": "Puck",
    "female-adult": "Aoede",
    "male-teen": "Fenrir",
    "female-teen": "Leda",
    "male-child": "Puck",
    "female-child": "Leda",
    "male-elderly": "Charon",
    "female-elderly": "Aoede",
}
OWN_STYLE = {"male-child", "male-elderly", "female-elderly"}  # child / elderly styles in the spec


@pytest.mark.parametrize("key", list(SEEDED))
def test_seeded_config_maps_every_persona_to_its_gemini_voice(key: str, tmp_path: Path) -> None:
    gender, age = key.split("-")
    persona = VoicePersona.model_validate({"gender": gender, "age": age})
    call = _persona_call(persona, tmp_path)  # repo's config/voices.yaml (cwd = repo root)
    assert call["generation_config"] == {"speech_config": [{"voice": SEEDED[key]}]}
    block = call["input"][0]["content"][0]
    assert block["text"] == LINE
    style = block["annotations"][0]["style"]
    assert style and style not in LINE
    if key in OWN_STYLE:
        assert style != gemini_tts.STYLE


# --- output -------------------------------------------------------------------------------------


def test_writes_the_returned_wav_to_out_path(tmp_path: Path) -> None:
    raw = wav(1.0)
    provider, _ = tts(done(b64(raw)))
    out = tmp_path / "line.wav"
    assert provider.synthesize(LINE, "uk", out) == out
    assert out.read_bytes() == raw
    with wave.open(str(out)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)


def test_wav_is_accepted_whatever_mime_type_says(tmp_path: Path) -> None:
    """The bytes decide, not the label."""
    provider, _ = tts(done(b64(wav()), mime=None))
    assert provider.synthesize(LINE, "uk", tmp_path / "line.wav").exists()


@pytest.mark.parametrize(
    "answer",
    [
        done(None),
        done(""),
        SimpleNamespace(status="completed", output_audio=None, errors=None),
        SimpleNamespace(status="completed", errors=None),  # no output_audio attribute at all
    ],
    ids=["data-none", "data-empty", "audio-none", "audio-missing"],
)
def test_no_audio_is_a_retryable_provider_error(answer: Any, tmp_path: Path) -> None:
    assert fails(answer, tmp_path / "line.wav").retryable is True


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff\xfb\x90\x00" + b"\x00" * 400,  # MP3 frame
        b"\x01\x00" * 2400,  # raw 16-bit PCM without a header (audio/l16)
        b"OggS" + b"\x00" * 100,
        wav()[:20],  # RIFF/WAVE header, then cut
    ],
    ids=["mp3", "raw-pcm", "ogg", "truncated-wav"],
)
def test_non_wav_audio_is_a_provider_error_and_nothing_is_written(
    raw: bytes, tmp_path: Path
) -> None:
    error = fails(done(b64(raw)), tmp_path / "line.wav")
    assert "WAV" in str(error)


def test_wav_with_zero_frames_is_a_provider_error(tmp_path: Path) -> None:
    error = fails(done(b64(wav(0.0))), tmp_path / "line.wav")
    assert "empty" in str(error).lower()


def test_wav_with_a_short_data_chunk_is_a_provider_error(tmp_path: Path) -> None:
    """Header says 0.5 s, half the samples arrived (cut-off download)."""
    full = wav(0.5)
    error = fails(done(b64(full[: len(full) // 2])), tmp_path / "line.wav")
    assert "truncated" in str(error)


def test_invalid_base64_is_a_provider_error(tmp_path: Path) -> None:
    fails(done("not base64 at all!!"), tmp_path / "line.wav")


def test_a_stale_file_is_removed_when_synthesis_fails(tmp_path: Path) -> None:
    out = tmp_path / "line.wav"
    out.write_bytes(wav())  # from an earlier attempt
    fails(done(None), out)  # `fails` asserts the file is gone


def test_no_audio_with_safety_detail_is_not_retryable(tmp_path: Path) -> None:
    errors = [SimpleNamespace(code="SAFETY", message="Response blocked")]
    assert fails(done(None, errors=errors), tmp_path / "line.wav").retryable is False


# --- error mapping ------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_rate_limit_and_server_errors_are_retryable(status: int, tmp_path: Path) -> None:
    error = fails(http_error(status), tmp_path / "line.wav")
    assert error.retryable is True
    assert str(status) in str(error)


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_not_retryable(status: int, tmp_path: Path) -> None:
    assert fails(http_error(status), tmp_path / "line.wav").retryable is False


def test_safety_block_on_http_error_is_not_retryable(tmp_path: Path) -> None:
    error = fails(http_error(400, "Request blocked due to SAFETY"), tmp_path / "line.wav")
    assert error.retryable is False
    assert "safety" in str(error).lower()


def test_safety_block_on_failed_interaction_is_not_retryable(tmp_path: Path) -> None:
    failed = SimpleNamespace(
        status="failed",
        output_audio=None,
        errors=[SimpleNamespace(code="SAFETY", message="Response blocked")],
    )
    assert fails(failed, tmp_path / "line.wav").retryable is False


def test_failed_interaction_without_safety_is_retryable(tmp_path: Path) -> None:
    failed = SimpleNamespace(
        status="failed", output_audio=None, errors=[SimpleNamespace(code="INTERNAL", message="x")]
    )
    assert fails(failed, tmp_path / "line.wav").retryable is True


@pytest.mark.parametrize(
    "exc",
    [
        APITimeoutError(request=httpx.Request("POST", "https://x")),
        APIConnectionError(request=httpx.Request("POST", "https://x")),
        httpx.ReadTimeout("read timed out"),
    ],
)
def test_timeouts_and_connection_errors_are_retryable(exc: BaseException, tmp_path: Path) -> None:
    assert fails(exc, tmp_path / "line.wav").retryable is True


def test_programming_errors_are_not_disguised_as_provider_errors(tmp_path: Path) -> None:
    provider, _ = tts(TypeError("create() got an unexpected keyword argument 'x'"))
    with pytest.raises(TypeError):
        provider.synthesize(LINE, "uk", tmp_path / "line.wav")


# --- key and registry ---------------------------------------------------------------------------


def test_missing_key_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gemini.API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=gemini.API_KEY_ENV):
        gemini_tts.GeminiTts(REF)


def test_registry_builds_gemini_tts_from_the_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gemini.API_KEY_ENV, "test-key-not-real")  # Client() makes no request
    provider = registry._FACTORIES["tts"]["gemini-tts"](REF)
    assert isinstance(provider, gemini_tts.GeminiTts)
    assert provider.model == REF.version


@pytest.mark.parametrize("profile", ["premium", "real-no-video"])
def test_gemini_tts_is_the_alternative_not_the_default(profile: str) -> None:
    """Human decision: elevenlabs-tts is the real profiles' default TTS (free tier
    ~10k chars/month vs 10 requests/day); gemini-tts stays registered (test above)."""
    ref = registry.load_profile(profile, ROOT / "config" / "profiles.yaml")["tts"]
    assert ref.provider == "elevenlabs-tts"
    assert "gemini-tts" in registry._FACTORIES["tts"]


# --- live ---------------------------------------------------------------------------------------


@pytest.mark.live
def test_live_synthesize_ukrainian(tmp_path: Path) -> None:
    """Ukrainian is not officially supported: this only shows the API returns a WAV for it.
    Whether the words are right is the `voice` stage's STT check."""
    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get(gemini.API_KEY_ENV):
        pytest.skip(f"{gemini.API_KEY_ENV} not set")
    out = gemini_tts.GeminiTts(REF).synthesize(LINE, "uk", tmp_path / "line.wav")
    with wave.open(str(out)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() / w.getframerate() > 1.0

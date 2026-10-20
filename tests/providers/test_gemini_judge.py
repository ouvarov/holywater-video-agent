"""GeminiJudge with the Interactions client faked (no network). One opt-in live smoke test."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from conftest import make_video
from dotenv import load_dotenv
from google.genai._gaos import google_genai
from google.genai._gaos.lib.compat_errors import APIConnectionError, APIError, APITimeoutError
from google.genai.interactions import CreateModelInteraction

from hwva.providers import gemini, gemini_judge, registry
from hwva.providers.base import JudgeProvider, ProviderError
from hwva.schemas import JudgeVerdict, LineSource, ModelRef, Script, Shot

ROOT = Path(__file__).resolve().parents[2]
REF = ModelRef(provider="gemini", model="gemini-3.8-flash", version="gemini-3.8-flash")
CRITERIA = (
    "face_present",
    "mouth_moves_when_speaking",
    "matches_character",
    "same_character_across_cut",
    "matches_prompt",
)
SCRIPT = Script(
    title="Не здамся",
    line="Я ніколи не здамся, навіть якщо впаду сто разів.",
    line_source=LineSource.INPUT,
    lang="uk",
    character="a young woman in a red raincoat",
    shots=[
        Shot(visual_prompt="she speaks in the rain", duration_s=6, has_line=True),
        Shot(visual_prompt="she walks away", duration_s=4, camera="wide shot"),
    ],
)
VIDEO = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1000  # bytes only; the fake never decodes it


def verdict_json(**overrides: Any) -> str:
    criterion = {"verdict": "pass", "score": 5, "evidence": "clear face", "timestamp": "00:01"}
    data: dict[str, Any] = {name: dict(criterion) for name in CRITERIA}
    data["defects"] = [{"kind": "flicker", "timestamp": "00:07", "note": "background flickers"}]
    data.update(overrides)
    return json.dumps(data)


def done(text: str | None, errors: Any = None) -> SimpleNamespace:
    return SimpleNamespace(status="completed", output_text=text, errors=errors)


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


def judge(answer: Any) -> tuple[gemini_judge.GeminiJudge, FakeInteractions]:
    fake = FakeInteractions(answer)
    return gemini_judge.GeminiJudge(REF, interactions=fake), fake


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "final.mp4"
    path.write_bytes(VIDEO)
    return path


def http_error(status: int, message: str = "boom") -> APIError:
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/interactions")
    response = httpx.Response(status, request=request, json={"error": {"message": message}})
    return APIError.generate(status, {"error": {"message": message}}, message, response)


def fails(answer: Any, video: Path) -> ProviderError:
    provider, _ = judge(answer)
    with pytest.raises(ProviderError) as info:
        provider.judge(video, SCRIPT)
    return info.value


def blocks(call: dict[str, Any]) -> list[dict[str, Any]]:
    (turn,) = call["input"]
    assert turn["type"] == "user_input"
    content: list[dict[str, Any]] = turn["content"]
    return content


# --- request ------------------------------------------------------------------------------------


def test_is_a_judge_provider() -> None:
    provider, _ = judge(done(verdict_json()))
    assert isinstance(provider, JudgeProvider)


def test_request_uses_pinned_model_temperature_zero_and_verdict_schema(video: Path) -> None:
    provider, fake = judge(done(verdict_json()))
    provider.judge(video, SCRIPT)
    (call,) = fake.calls
    assert call["model"] == REF.version
    assert call["generation_config"] == {"temperature": 0}
    assert call["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": gemini_judge.verdict_schema(),
    }
    assert call["timeout"] >= 300  # live: a 9 s final took 95 s, once > 120 s


def test_video_is_sent_inline_as_base64_mp4_at_two_fps(video: Path) -> None:
    provider, fake = judge(done(verdict_json()))
    provider.judge(video, SCRIPT)
    (clip,) = [b for b in blocks(fake.calls[0]) if b["type"] == "video"]
    assert base64.b64decode(clip["data"]) == VIDEO
    assert clip["mime_type"] == "video/mp4"
    assert clip["processing"] == {"type": "static", "fps": 2}
    assert "uri" not in clip


def test_prompt_carries_the_script(video: Path) -> None:
    provider, fake = judge(done(verdict_json()))
    provider.judge(video, SCRIPT)
    (text,) = [b["text"] for b in blocks(fake.calls[0]) if b["type"] == "text"]
    assert SCRIPT.character in text
    assert SCRIPT.line in text
    assert "Number of shots: 2" in text
    for shot in SCRIPT.shots:
        assert shot.visual_prompt in text
        assert shot.camera in text


def test_rubric_has_anchors_and_asks_for_timestamps_everywhere(video: Path) -> None:
    provider, fake = judge(done(verdict_json()))
    provider.judge(video, SCRIPT)
    rubric = fake.calls[0]["system_instruction"]
    for name in CRITERIA:
        assert name in rubric
        line = rubric[rubric.index(f"- {name}:") :]
        assert "5 =" in line and "3 =" in line and "1 =" in line
    assert "EVERY criterion" in rubric
    assert "MM:SS" in rubric
    assert "defects" in rubric


def test_schema_is_flat_and_requires_a_timestamp_on_every_criterion_and_defect() -> None:
    schema = gemini_judge.verdict_schema()
    assert "$ref" not in json.dumps(schema)
    for name in CRITERIA:
        assert "timestamp" in schema["properties"][name]["required"]
    assert "timestamp" in schema["properties"]["defects"]["items"]["required"]
    assert set(schema["required"]) == {*CRITERIA, "defects"}  # "none found" = [], not omitted


def test_request_is_accepted_by_the_sdk_request_model(video: Path) -> None:
    """The fake accepts any kwargs; the SDK's own request model doesn't. Offline."""
    provider, fake = judge(done(verdict_json()))
    provider.judge(video, SCRIPT)
    body = {k: v for k, v in fake.calls[0].items() if k != "timeout"}
    assert set(body) <= google_genai._CREATE_BODY_KEYS
    request = CreateModelInteraction(**body)
    dumped = request.model_dump(by_alias=True, exclude_none=True)
    assert dumped["generation_config"] == {"temperature": 0}  # SDK 2.31 marks it deprecated
    clip = dumped["input"][0]["content"][1]
    assert clip["type"] == "video"
    assert clip["processing"] == {"type": "static", "fps": 2}


def test_file_over_the_inline_limit_is_not_retryable_and_not_sent(
    video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The limit is on the base64 payload: a file under it raw but over it encoded is refused."""
    monkeypatch.setattr(gemini_judge, "MAX_INLINE_BYTES", len(VIDEO) + 1)
    provider, fake = judge(done(verdict_json()))
    with pytest.raises(ProviderError) as info:
        provider.judge(video, SCRIPT)
    assert info.value.retryable is False
    assert fake.calls == []


def test_file_exactly_at_the_encoded_limit_is_sent(
    video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gemini_judge, "MAX_INLINE_BYTES", len(base64.b64encode(VIDEO)))
    provider, fake = judge(done(verdict_json()))
    provider.judge(video, SCRIPT)
    assert len(fake.calls) == 1


# --- output -------------------------------------------------------------------------------------


def test_returns_the_validated_verdict(video: Path) -> None:
    provider, _ = judge(done(verdict_json()))
    verdict = provider.judge(video, SCRIPT)
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.face_present.score == 5
    assert verdict.defects[0].timestamp == "00:07"


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        verdict_json(face_present={"verdict": "pass", "score": 9, "evidence": "x"}),
        verdict_json(defects=[{"kind": "x", "timestamp": "7 sec", "note": "y"}]),
        json.dumps({"face_present": {}}),
    ],
    ids=["not-json", "score-out-of-range", "bad-timestamp", "missing-criteria"],
)
def test_invalid_verdict_is_a_provider_error_after_one_call(text: str, video: Path) -> None:
    provider, fake = judge(done(text))
    with pytest.raises(ProviderError, match="invalid verdict") as info:
        provider.judge(video, SCRIPT)
    assert info.value.retryable is False
    assert len(fake.calls) == 1  # soft signal: no re-ask


@pytest.mark.parametrize("text", [None, ""], ids=["none", "empty"])
def test_no_text_is_a_provider_error(text: str | None, video: Path) -> None:
    assert fails(done(text), video).retryable is True


def test_no_text_with_safety_detail_is_not_retryable(video: Path) -> None:
    errors = [SimpleNamespace(code="SAFETY", message="Response blocked")]
    assert fails(done(None, errors=errors), video).retryable is False


# --- error mapping ------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_rate_limit_and_server_errors_are_retryable(status: int, video: Path) -> None:
    error = fails(http_error(status), video)
    assert error.retryable is True
    assert str(status) in str(error)


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_not_retryable(status: int, video: Path) -> None:
    assert fails(http_error(status), video).retryable is False


def test_safety_block_on_http_error_is_not_retryable(video: Path) -> None:
    error = fails(http_error(400, "Request blocked due to SAFETY"), video)
    assert error.retryable is False
    assert "safety" in str(error).lower()


def test_safety_block_on_failed_interaction_is_not_retryable(video: Path) -> None:
    failed = SimpleNamespace(
        status="failed",
        output_text=None,
        errors=[SimpleNamespace(code="SAFETY", message="Response blocked")],
    )
    assert fails(failed, video).retryable is False


@pytest.mark.parametrize(
    "exc",
    [
        APITimeoutError(request=httpx.Request("POST", "https://x")),
        APIConnectionError(request=httpx.Request("POST", "https://x")),
        httpx.ReadTimeout("read timed out"),
    ],
)
def test_timeouts_and_connection_errors_are_retryable(exc: BaseException, video: Path) -> None:
    assert fails(exc, video).retryable is True


def test_programming_errors_are_not_disguised_as_provider_errors(video: Path) -> None:
    provider, _ = judge(TypeError("create() got an unexpected keyword argument 'x'"))
    with pytest.raises(TypeError):
        provider.judge(video, SCRIPT)


# --- key and registry ---------------------------------------------------------------------------


def test_missing_key_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gemini.API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=gemini.API_KEY_ENV):
        gemini_judge.GeminiJudge(REF)


def test_registry_builds_gemini_judge_from_the_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gemini.API_KEY_ENV, "test-key-not-real")  # Client() makes no request
    provider = registry._FACTORIES["judge"]["gemini"](REF)
    assert isinstance(provider, gemini_judge.GeminiJudge)
    assert provider.model == REF.version


def test_premium_profile_pins_judge_model_and_exact_version() -> None:
    ref = registry.load_profile("premium", ROOT / "config" / "profiles.yaml")["judge"]
    assert ref == REF


def test_dev_profile_has_no_judge() -> None:
    assert "judge" not in registry.load_profile("dev", ROOT / "config" / "profiles.yaml")


# --- live ---------------------------------------------------------------------------------------


@pytest.mark.live
def test_live_judge_a_rendered_clip(tmp_path: Path) -> None:
    """Only shows the API accepts our request and answers with a valid `JudgeVerdict`."""
    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get(gemini.API_KEY_ENV):
        pytest.skip(f"{gemini.API_KEY_ENV} not set")
    clip = make_video(tmp_path / "final.mp4", 3.0, audio=True)
    verdict = gemini_judge.GeminiJudge(REF).judge(clip, SCRIPT)
    assert isinstance(verdict, JudgeVerdict)

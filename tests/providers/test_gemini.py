"""GeminiLlm with the Interactions client faked (no network). One opt-in live smoke test."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from dotenv import load_dotenv
from google.genai._gaos import google_genai
from google.genai._gaos.lib.compat_errors import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APITimeoutError,
)
from google.genai.interactions import CreateModelInteraction

from hwva import cli
from hwva.providers import gemini, registry
from hwva.providers.base import LlmProvider, ProviderError
from hwva.schemas import LineSource, ModelRef, Script

ROOT = Path(__file__).resolve().parents[2]
REF = ModelRef(provider="gemini", model="gemini-3.8-flash", version="gemini-3.8-flash")
LINE = "Я ніколи не здамся, навіть якщо впаду сто разів."


def script_json(line: str = LINE, **overrides: Any) -> str:
    data: dict[str, Any] = {
        "title": "Не здамся",
        "line": line,
        "line_source": "input",
        "lang": "uk",
        "character": "a young woman in a red raincoat",
        "shots": [
            {"visual_prompt": "she speaks in the rain", "duration_s": 6, "has_line": True},
            {"visual_prompt": "she walks away", "duration_s": 4},
        ],
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def done(text: str | None) -> SimpleNamespace:
    return SimpleNamespace(status="completed", output_text=text, errors=None)


class FakeInteractions:
    """Records every create() call; returns the scripted answers in order, or raises them."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        answer = self.answers[min(len(self.calls), len(self.answers)) - 1]
        if isinstance(answer, BaseException):
            raise answer
        return answer


def llm(*answers: Any) -> tuple[gemini.GeminiLlm, FakeInteractions]:
    fake = FakeInteractions(*answers)
    return gemini.GeminiLlm(REF, interactions=fake), fake


def http_error(status: int, message: str = "boom") -> APIError:
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/interactions")
    response = httpx.Response(status, request=request, json={"error": {"message": message}})
    return APIError.generate(status, {"error": {"message": message}}, message, response)


# --- request ------------------------------------------------------------------------------------


def test_is_an_llm_provider() -> None:
    provider, _ = llm(done(script_json()))
    assert isinstance(provider, LlmProvider)


def test_uses_interactions_api_with_pinned_model_code_and_script_schema() -> None:
    provider, fake = llm(done(script_json()))
    provider.write_script("дівчина під дощем", LINE, "uk")
    (call,) = fake.calls
    assert call["model"] == REF.version
    fmt = call["response_format"]
    assert fmt["type"] == "text"
    assert fmt["mime_type"] == "application/json"
    assert fmt["schema"] == gemini.script_schema()
    assert call["system_instruction"]
    assert call["timeout"] > 0


def test_request_is_accepted_by_the_sdk_request_model() -> None:
    """The fake accepts any kwargs; the SDK's own request model doesn't. Offline."""
    provider, fake = llm(done(script_json()))
    provider.write_script("idea", LINE, "uk")
    body = {k: v for k, v in fake.calls[0].items() if k != "timeout"}
    assert set(body) <= google_genai._CREATE_BODY_KEYS
    request = CreateModelInteraction(**body)
    assert request.response_format is not None


def test_prompt_carries_idea_line_and_lang_verbatim() -> None:
    provider, fake = llm(done(script_json()))
    provider.write_script("дівчина під дощем", LINE, "uk")
    prompt = fake.calls[0]["input"]
    assert "дівчина під дощем" in prompt
    assert LINE in prompt
    assert "uk" in prompt


def test_prompt_without_line_asks_to_write_one() -> None:
    provider, fake = llm(done(script_json(line_source="generated")))
    provider.write_script("кіт у космосі", None, None)
    prompt = fake.calls[0]["input"]
    assert "No dialogue line" in prompt


def test_system_instruction_forbids_paraphrase_and_digits() -> None:
    text = gemini.SYSTEM.lower()
    assert "exactly" in text and "paraphrase" in text
    assert "digits" in text


def test_schema_is_flat_and_matches_script() -> None:
    schema = gemini.script_schema()
    dumped = json.dumps(schema)
    assert "$ref" not in dumped and "$defs" not in dumped
    assert set(schema["required"]) == set(Script.model_json_schema()["required"])
    assert schema["properties"]["line_source"]["enum"] == [s.value for s in LineSource]
    assert schema["properties"]["shots"]["items"]["required"] == ["visual_prompt", "duration_s"]


# --- structured output validation ---------------------------------------------------------------


def test_valid_answer_becomes_a_script() -> None:
    provider, fake = llm(done(script_json()))
    script = provider.write_script("idea", LINE, "uk")
    assert isinstance(script, Script)
    assert script.line == LINE
    assert len(fake.calls) == 1


def test_invalid_answer_is_retried_once_with_the_validation_error() -> None:
    bad = script_json(shots=[])  # min_length=1
    provider, fake = llm(done(bad), done(script_json()))
    script = provider.write_script("idea", LINE, "uk")
    assert script.line == LINE
    assert len(fake.calls) == 2
    repair = fake.calls[1]["input"]
    assert "shots" in repair and "validation" in repair.lower()
    assert bad in repair
    assert fake.calls[1]["response_format"] == fake.calls[0]["response_format"]


def test_invalid_twice_is_a_retryable_provider_error_with_the_validation_error() -> None:
    provider, fake = llm(done(script_json(shots=[])), done(script_json(lang="de")))
    with pytest.raises(ProviderError) as info:
        provider.write_script("idea", LINE, "uk")
    assert info.value.retryable is True
    assert "shots" in str(info.value)  # pydantic's message for the first answer
    assert "lang" in str(info.value)  # ... and for the re-asked one
    assert len(fake.calls) == 2


def test_non_json_answer_counts_as_invalid() -> None:
    provider, fake = llm(done("Sure! Here is your script: ..."), done("still not json"))
    with pytest.raises(ProviderError) as info:
        provider.write_script("idea", LINE, "uk")
    assert info.value.retryable is True
    assert len(fake.calls) == 2


def test_empty_output_is_retryable() -> None:
    provider, _ = llm(done(None))
    with pytest.raises(ProviderError) as info:
        provider.write_script("idea", LINE, "uk")
    assert info.value.retryable is True


# --- error mapping ------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_rate_limit_and_server_errors_are_retryable(status: int) -> None:
    provider, _ = llm(http_error(status))
    with pytest.raises(ProviderError) as info:
        provider.write_script("idea", LINE, "uk")
    assert info.value.retryable is True
    assert str(status) in str(info.value)


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_not_retryable(status: int) -> None:
    provider, _ = llm(http_error(status))
    with pytest.raises(ProviderError) as info:
        provider.write_script("idea", LINE, "uk")
    assert info.value.retryable is False


def test_safety_block_on_http_error_is_not_retryable() -> None:
    provider, _ = llm(http_error(400, "Request blocked due to SAFETY"))
    with pytest.raises(ProviderError) as info:
        provider.write_script("idea", LINE, "uk")
    assert info.value.retryable is False
    assert "safety" in str(info.value).lower()


def test_safety_block_on_failed_interaction_is_not_retryable() -> None:
    failed = SimpleNamespace(
        status="failed",
        output_text=None,
        errors=[SimpleNamespace(code="SAFETY", message="Response blocked")],
    )
    provider, _ = llm(failed)
    with pytest.raises(ProviderError) as info:
        provider.write_script("idea", LINE, "uk")
    assert info.value.retryable is False


@pytest.mark.parametrize("status", ["budget_exceeded", "cancelled"])
def test_final_interaction_statuses_are_not_retryable(status: str) -> None:
    provider, _ = llm(SimpleNamespace(status=status, output_text=None, errors=None))
    with pytest.raises(ProviderError) as info:
        provider.write_script("idea", LINE, "uk")
    assert info.value.retryable is False
    assert status in str(info.value)


def test_unreadable_response_is_not_retryable() -> None:
    request = httpx.Request("POST", "https://x")
    exc = APIResponseValidationError(httpx.Response(200, request=request), body="{}")
    provider, _ = llm(exc)
    with pytest.raises(ProviderError) as info:
        provider.write_script("idea", LINE, "uk")
    assert info.value.retryable is False
    assert "HTTP 200" not in str(info.value)


def test_programming_errors_are_not_disguised_as_provider_errors() -> None:
    provider, _ = llm(TypeError("create() got an unexpected keyword argument 'x'"))
    with pytest.raises(TypeError):
        provider.write_script("idea", LINE, "uk")


def test_failed_interaction_without_safety_is_retryable() -> None:
    failed = SimpleNamespace(
        status="failed", output_text=None, errors=[SimpleNamespace(code="INTERNAL", message="x")]
    )
    provider, _ = llm(failed)
    with pytest.raises(ProviderError) as info:
        provider.write_script("idea", LINE, "uk")
    assert info.value.retryable is True


@pytest.mark.parametrize(
    "exc",
    [
        APITimeoutError(request=httpx.Request("POST", "https://x")),
        APIConnectionError(request=httpx.Request("POST", "https://x")),
        httpx.ReadTimeout("read timed out"),
    ],
)
def test_timeouts_and_connection_errors_are_retryable(exc: BaseException) -> None:
    provider, _ = llm(exc)
    with pytest.raises(ProviderError) as info:
        provider.write_script("idea", LINE, "uk")
    assert info.value.retryable is True


# --- key and registry ---------------------------------------------------------------------------


def test_missing_key_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gemini.API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=gemini.API_KEY_ENV):
        gemini.GeminiLlm(REF)


def test_registry_builds_gemini_llm_from_the_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gemini.API_KEY_ENV, "test-key-not-real")  # Client() makes no request
    provider = registry._FACTORIES["llm"]["gemini"](REF)
    assert isinstance(provider, gemini.GeminiLlm)
    assert provider.model == REF.version


def test_premium_profile_pins_model_and_exact_version() -> None:
    ref = registry.load_profile("premium", ROOT / "config" / "profiles.yaml")["llm"]
    assert ref == REF


# --- .env loading -------------------------------------------------------------------------------

VAR = "HWVA_TEST_DOTENV_VAR"


@pytest.fixture
def dotenv_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".env"
    path.write_text(f"{VAR}=from-file\n", encoding="utf-8")
    monkeypatch.setenv("HWVA_DOTENV", str(path))
    monkeypatch.setenv(VAR, "placeholder")  # registers the var so monkeypatch restores it
    monkeypatch.delenv(VAR)
    return path


def test_load_env_reads_the_file(dotenv_file: Path) -> None:
    cli.load_env()
    assert os.environ[VAR] == "from-file"


def test_load_env_never_overrides_a_set_variable(
    dotenv_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(VAR, "from-shell")
    cli.load_env()
    assert os.environ[VAR] == "from-shell"


def test_load_env_without_file_is_fine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HWVA_DOTENV", str(tmp_path / "missing.env"))
    cli.load_env()


def test_cli_loads_env_at_startup(
    dotenv_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HWVA_RUNS", str(tmp_path / "runs"))
    cli.main([])  # usage error, but .env is loaded before anything else
    assert os.environ[VAR] == "from-file"


# --- live ---------------------------------------------------------------------------------------


@pytest.mark.live
def test_live_write_script_ukrainian() -> None:
    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get(gemini.API_KEY_ENV):
        pytest.skip(f"{gemini.API_KEY_ENV} not set")
    script = gemini.GeminiLlm(REF).write_script("Дівчина під дощем на зупинці", LINE, "uk")
    assert isinstance(script, Script)
    assert script.lang == "uk"
    assert sum(shot.has_line for shot in script.shots) == 1

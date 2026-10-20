"""Gemini LLM via the Interactions API (`client.interactions.create`, google-genai >= 2.25).

`write_script` asks for JSON constrained by the `Script` schema and validates the answer with
pydantic. One invalid answer is sent back to the model once with the validation error; a second
invalid answer is a retryable `ProviderError`. The `script` stage still re-checks everything —
in particular a known line is restored byte-exact whatever the model returns."""

from __future__ import annotations

import os
from typing import Any, Protocol

import httpx

# The Interactions API's error hierarchy has no public re-export in google-genai 2.31:
# APIStatusError (status_code), APITimeoutError/APIConnectionError, APIResponseValidationError.
from google.genai._gaos.lib.compat_errors import APIError, APIResponseValidationError
from pydantic import BaseModel, ValidationError

from hwva.providers.base import ProviderError
from hwva.schemas import Lang, ModelRef, Script

API_KEY_ENV = "GEMINI_API_KEY"
# per request; the SDK itself also retries 408/409/429/5xx with backoff before we see an error
TIMEOUT_S = 120.0
SAFETY_MARKERS = ("safety", "blocked", "prohibited", "blocklist", "spii")
FINAL_STATUSES = ("budget_exceeded", "cancelled")  # retrying won't change the outcome

SYSTEM = """You write scripts for short vertical (9:16) videos: one character, 1-2 shots.
Rules:
- If a dialogue line is given, copy it into "line" EXACTLY, character for character: same words,
  word order, punctuation and case. Never translate, shorten, correct or paraphrase it.
- If no line is given, invent one short line (at most ~20 words) in the requested language.
  Write every number in words, never with digits.
- "lang" is the language of the line: "uk" (Ukrainian) or "en" (English).
- "line_source": "input" if a line was given, otherwise "generated".
- "character": one concrete appearance description (in English), reused in every shot.
- "shots": exactly one shot has "has_line": true and it comes first; its duration must fit the
  spoken line (~0.4 s per word plus a pause). A second shot, if any, is a silent reaction or
  cutaway. Total duration 8-30 seconds. "visual_prompt" and "camera" are in English and describe
  only what is seen (no dialogue text in the prompt).
- "voice": who speaks the line, {"gender": "male"|"female", "age": "child"|"teen"|"adult"|
  "elderly"}. Pick the persona from the character. When the line is grammatically gendered
  (Ukrainian past tense "зробив/зробила", short adjectives "головний/головна", "сам/сама"), the
  gender MUST agree with the line. Never change the line to fit the persona."""


class _Interactions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


def script_schema() -> dict[str, Any]:
    """`Script`'s JSON schema with `$ref`s inlined: structured output gets one flat schema."""
    return flat_schema(Script)


def flat_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A pydantic model's JSON schema with `$ref`s inlined."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                return inline(defs[ref.rsplit("/", 1)[-1]])
            return {k: inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    result: dict[str, Any] = inline(schema)
    return result


def map_error(exc: BaseException) -> ProviderError:
    """SDK/HTTP error -> ProviderError. 429/5xx/408/timeouts/connection: retryable;
    400/401/403/other 4xx and safety blocks: not."""
    status = getattr(exc, "status_code", None)
    text = str(exc)
    if isinstance(exc, APIResponseValidationError):
        return ProviderError(f"Gemini response not readable by the SDK: {text}", retryable=False)
    if isinstance(status, int):
        if is_safety(text):
            return ProviderError(f"Gemini blocked the request (safety): {text}", retryable=False)
        retryable = status in (408, 429) or status >= 500
        return ProviderError(f"Gemini HTTP {status}: {text}", retryable=retryable)
    # no status: timeout / connection error / no response (APITimeoutError, APIConnectionError)
    return ProviderError(f"Gemini request failed ({type(exc).__name__}): {text}", retryable=True)


def is_safety(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in SAFETY_MARKERS)


def interactions_from_env() -> _Interactions:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise ValueError(f"{API_KEY_ENV} is not set (put it in .env, see .env.example)")
    from google import genai

    # Keep the Client alive: if only `.interactions` is kept, the Client is garbage-collected and
    # closes its HTTP connection ("Cannot send a request, as the client has been closed").
    client = genai.Client(api_key=key)
    _CLIENTS.append(client)
    interactions: _Interactions = client.interactions
    return interactions


_CLIENTS: list[object] = []


class GeminiLlm:
    """`LlmProvider`. The model code sent to the API is `ref.version` (the pinned exact code)."""

    def __init__(self, ref: ModelRef, interactions: _Interactions | None = None) -> None:
        self.model = ref.version
        self._interactions = interactions or interactions_from_env()

    def write_script(self, idea: str, line: str | None, lang: Lang | None) -> Script:
        prompt = _prompt(idea, line, lang)
        text = self._ask(prompt)
        try:
            return Script.model_validate_json(text)
        except ValidationError as first:
            repair = (
                f"{prompt}\n\nYour previous answer was:\n{text}\n\n"
                f"It failed schema validation:\n{first}\nReturn a corrected JSON object."
            )
            text = self._ask(repair)
            try:
                return Script.model_validate_json(text)
            except ValidationError as second:
                raise ProviderError(
                    f"Gemini returned a schema-invalid script twice. "
                    f"First: {first}\nAfter re-ask: {second}",
                    retryable=True,
                ) from second

    def _ask(self, prompt: str) -> str:
        try:
            interaction = self._interactions.create(
                model=self.model,
                system_instruction=SYSTEM,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": script_schema(),
                },
                timeout=TIMEOUT_S,
            )
        except (APIError, httpx.HTTPError) as exc:  # anything else is a bug, not a network issue
            raise map_error(exc) from exc
        return output_text(interaction)


def check_completed(interaction: Any) -> str:
    """Raise unless the interaction completed. Returns the error detail (may be empty) so a
    caller that finds no output can tell a safety block from a plain empty answer."""
    status = str(getattr(interaction, "status", "completed"))
    errors = getattr(interaction, "errors", None) or []
    detail = "; ".join(f"{getattr(e, 'code', '')} {getattr(e, 'message', '')}" for e in errors)
    if status != "completed":
        message = f"Gemini interaction {status}: {detail or 'no details'}"
        final = status in FINAL_STATUSES or is_safety(detail)
        raise ProviderError(message, retryable=not final)
    return detail


def output_text(interaction: Any) -> str:
    detail = check_completed(interaction)
    text = getattr(interaction, "output_text", None)
    if not text:
        message = f"Gemini returned no text{': ' + detail if detail else ''}"
        raise ProviderError(message, retryable=not is_safety(detail))
    return str(text)


def _prompt(idea: str, line: str | None, lang: Lang | None) -> str:
    parts = [f"Idea: {idea}"]
    if line is not None:
        parts.append(f'Dialogue line (copy into "line" exactly as written):\n{line}')
    else:
        parts.append("No dialogue line is given: write one.")
    if lang is not None:
        parts.append(f"Language of the line: {lang}")
    return "\n\n".join(parts)

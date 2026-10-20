"""Gemini video judge via the Interactions API (`client.interactions.create`, google-genai >= 2.25).

`judge` sends `final.mp4` inline (base64, sampled at `FPS` frames per second, audio included),
the script and the `JudgeVerdict` JSON schema; temperature 0. The answer is validated with
pydantic; an invalid answer is a `ProviderError` — no re-ask: the verdict is a soft signal and
`quality_check` records any failure instead of retrying.

Only inline upload is implemented (the API accepts it under ~20 MB; our clips are ~1 MB). A larger
file is a non-retryable `ProviderError` rather than a silent switch to the Files API."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx
from google.genai._gaos.lib.compat_errors import APIError
from pydantic import ValidationError

from hwva.providers.base import ProviderError
from hwva.providers.gemini import (
    _Interactions,
    flat_schema,
    interactions_from_env,
    map_error,
    output_text,
)
from hwva.schemas import JudgeVerdict, ModelRef, Script

FPS = 2  # frames sampled per second (API default is 1)
# the API's ~20 MB inline limit is on the request; base64 grows the file by 4/3, ~1 MB left for
# the prompt and schema
MAX_INLINE_BYTES = 19 * 1024 * 1024
MIME = "video/mp4"
# the whole final goes up and is watched: live a 9 s final took 95 s and once ran
# past the shared 120 s (APITimeoutError), so the judge gets its own, longer timeout
TIMEOUT_S = 300.0

RUBRIC = """You review a short vertical video generated from a script. Watch every frame you are
given and listen to the audio. Judge only what you can see and hear; never guess.

Score each criterion 1-5 using these anchors, and set "verdict" to "pass" (score 4-5),
"fail" (score 1-2) or "unsure" (score 3, or you cannot tell):
- face_present: 5 = one clear human face visible in every shot; 3 = face small, partly hidden
  or missing in part of the video; 1 = no face.
- mouth_moves_when_speaking: 5 = the mouth visibly moves whenever the line is heard and is still
  otherwise; 3 = mouth moves only part of the time, or moves when nothing is said; 1 = the mouth
  never moves while the line is heard. You see sampled frames: do not judge exact lip-sync.
- matches_character: 5 = the person matches every detail of the script's "character"; 3 = some
  details differ (clothes, hair, age); 1 = a clearly different person.
- same_character_across_cut: 5 = after each cut it is evidently the same person; 3 = similar but
  some details changed; 1 = a different person. With a single shot there is no cut: "unsure",
  score 3, evidence "single shot, no cut".
- matches_prompt: 5 = each shot shows its "visual_prompt" and "camera"; 3 = partly; 1 = unrelated.

Timestamps are "MM:SS" from the start of the video. Give a timestamp for EVERY criterion: the
moment that best shows your evidence. "evidence" says concretely what you saw or heard there.

"defects": every visual or audio defect you notice (distorted face or hands, flicker, frozen
frames, morphing, text artifacts, audio glitches), each with "kind", the "MM:SS" timestamp where
it appears and a short "note". Only report defects inside the video. Empty list if none."""


class GeminiJudge:
    """`JudgeProvider`. The model code sent to the API is `ref.version` (the pinned exact code)."""

    def __init__(self, ref: ModelRef, interactions: _Interactions | None = None) -> None:
        self.model = ref.version
        self._interactions = interactions or interactions_from_env()

    def judge(self, video_path: Path, script: Script) -> JudgeVerdict:
        size = video_path.stat().st_size
        encoded = 4 * -(-size // 3)
        if encoded > MAX_INLINE_BYTES:
            raise ProviderError(
                f"{video_path.name} is {size} bytes ({encoded} as base64), over the "
                f"{MAX_INLINE_BYTES} inline limit (Files API upload not implemented)",
                retryable=False,
            )
        data = base64.b64encode(video_path.read_bytes()).decode("ascii")
        try:
            interaction = self._interactions.create(
                model=self.model,
                system_instruction=RUBRIC,
                input=[
                    {
                        "type": "user_input",
                        "content": [
                            {"type": "text", "text": _prompt(script)},
                            {
                                "type": "video",
                                "data": data,
                                "mime_type": MIME,
                                "processing": {"type": "static", "fps": FPS},
                            },
                        ],
                    }
                ],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": verdict_schema(),
                },
                # spec: temperature 0. SDK 2.31 marks the field deprecated but still sends it.
                generation_config={"temperature": 0},
                timeout=TIMEOUT_S,
            )
        except (APIError, httpx.HTTPError) as exc:  # anything else is a bug, not a network issue
            raise map_error(exc) from exc
        text = output_text(interaction)
        try:
            return JudgeVerdict.model_validate_json(text)
        except ValidationError as exc:
            message = f"Gemini judge returned an invalid verdict: {exc}"
            raise ProviderError(message, retryable=False) from exc  # soft signal: no re-ask


def verdict_schema() -> dict[str, Any]:
    """`JudgeVerdict`'s flat schema, with `timestamp` required on every criterion (the model may
    still answer null; a missing key is what we rule out) and `defects` required (empty list,
    not omitted, means none found)."""
    schema = flat_schema(JudgeVerdict)
    if "defects" not in schema["required"]:
        schema["required"].append("defects")
    for prop in schema["properties"].values():
        if prop.get("type") == "object" and "timestamp" in prop.get("properties", {}):
            required = prop.setdefault("required", [])
            if "timestamp" not in required:
                required.append("timestamp")
    return schema


def _prompt(script: Script) -> str:
    shots = "\n".join(
        f"Shot {i}: {shot.duration_s:g} s, camera: {shot.camera}, "
        f"{'the line is spoken' if shot.has_line else 'silent'}. {shot.visual_prompt}"
        for i, shot in enumerate(script.shots, 1)
    )
    return (
        f"Character: {script.character}\n"
        f"Number of shots: {len(script.shots)}\n{shots}\n"
        f"Spoken line ({script.lang}): {script.line}\n\n"
        "Return the verdict as JSON matching the schema."
    )

from pathlib import Path

import pytest
from conftest import dominant_hz, duration, make_audio, needs_ffmpeg, probe

from hwva.providers import fakes
from hwva.providers.base import VideoRequest
from hwva.schemas import Age, Gender, VoicePersona


@needs_ffmpeg
def test_fake_tts_and_stt_round_trip(tmp_path: Path) -> None:
    audio = fakes.FakeTts().synthesize("Я тут головний", "uk", tmp_path / "line.wav")
    assert audio.exists()
    assert fakes.FakeStt().transcribe(audio, "uk") == "Я тут головний"


@needs_ffmpeg
def test_fake_stt_hears_last_line_in_extracted_audio(tmp_path: Path) -> None:
    fakes.FakeTts().synthesize("Я тут головний", "uk", tmp_path / "line.wav")
    extracted = make_audio(tmp_path / "final_audio.wav", 2)  # no sidecar, like audio from mp4
    assert fakes.FakeStt().transcribe(extracted, "uk") == "Я тут головний"


@needs_ffmpeg
def test_fake_stt_can_mishear(tmp_path: Path) -> None:
    audio = fakes.FakeTts().synthesize("Я тут головний", "uk", tmp_path / "line.wav")
    stt = fakes.FakeStt(mishear={"головний": "найголовніший"})
    assert stt.transcribe(audio, "uk") == "Я тут найголовніший"


@needs_ffmpeg
def test_sidecars_do_not_clash_between_formats(tmp_path: Path) -> None:
    tts = fakes.FakeTts()
    tts.synthesize("перша", "uk", tmp_path / "line.wav")
    tts.synthesize("друга", "uk", tmp_path / "line.mp3")
    assert fakes.FakeStt().transcribe(tmp_path / "line.wav", "uk") == "перша"


@needs_ffmpeg
def test_fake_video_is_vertical_with_last_frame(tmp_path: Path) -> None:
    result = fakes.FakeVideo().generate(VideoRequest("cat", 2), tmp_path / "shot.mp4")
    stream = probe(result.video_path)["streams"][0]
    assert (stream["width"], stream["height"]) == (720, 1280)
    assert duration(result.video_path) == 2.0
    assert result.last_frame is not None and Path(result.last_frame).exists()


@needs_ffmpeg
def test_fake_image_is_a_free_vertical_png_and_records_the_prompt(tmp_path: Path) -> None:
    image = fakes.FakeImage()
    out = tmp_path / "character.png"
    result = image.generate("a cartoon cat", out)
    assert result.path == out and result.cost_usd == 0.0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    stream = probe(out)["streams"][0]
    assert (stream["width"], stream["height"]) == (720, 1280)
    assert image.prompts == ["a cartoon cat"]


def test_fake_media_host_gives_a_new_https_url_per_upload_without_network(
    tmp_path: Path,
) -> None:
    path = tmp_path / "line.wav"
    path.write_bytes(b"x")
    host = fakes.FakeMediaHost()
    first, second = host.upload(path), host.upload(path)
    assert first.url.startswith("https://") and first.url != second.url
    assert first.url.endswith("/line.wav")
    assert first.expires_at is not None
    assert host.uploads == [path, path]


# --- FakeTts pitch per persona (specs/voice.md, voice persona) ----------------

# male-adult 140 Hz, female-adult 220 Hz; teen +60, child +160, elderly -20
PERSONA_HZ = {
    ("male", "adult"): 140,
    ("male", "teen"): 200,
    ("male", "child"): 300,
    ("male", "elderly"): 120,
    ("female", "adult"): 220,
    ("female", "teen"): 280,
    ("female", "child"): 380,
    ("female", "elderly"): 200,
}


@needs_ffmpeg
@pytest.mark.parametrize("persona", list(PERSONA_HZ), ids=lambda p: "-".join(p))
def test_fake_tts_pitch_follows_persona(tmp_path: Path, persona: tuple[Gender, Age]) -> None:
    gender, age = persona
    audio = fakes.FakeTts().synthesize(
        "Я тут головний",
        "uk",
        tmp_path / "line.wav",
        persona=VoicePersona(gender=gender, age=age),
    )
    assert dominant_hz(audio) == pytest.approx(PERSONA_HZ[persona], abs=3)


@needs_ffmpeg
def test_fake_tts_with_persona_still_round_trips_the_text(tmp_path: Path) -> None:
    persona = VoicePersona(gender="female", age="child")
    out = tmp_path / "line.wav"
    audio = fakes.FakeTts().synthesize("Я тут головна", "uk", out, persona=persona)
    assert fakes.FakeStt().transcribe(audio, "uk") == "Я тут головна"  # sidecar = text only


def test_fake_llm_keeps_quoted_line_verbatim() -> None:
    script = fakes.FakeLlm().write_script('Кіт каже: "Я тут головний"', None, None)
    assert script.line == "Я тут головний"
    assert script.lang == "uk"

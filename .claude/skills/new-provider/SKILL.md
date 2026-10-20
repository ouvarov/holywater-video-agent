---
name: new-provider
description: Add a real model provider (e.g. gemini llm, seedance video) behind an existing Protocol. Use when asked to connect/add a model or API.
---

Add provider `$ARGUMENTS` (format: `<role> <name>`, role ∈ llm | tts | stt | video).

1. Read `src/hwva/providers/base.py` (Protocol + Capabilities for the role) and `research/` notes in the task folder if referenced in the spec.
2. Write `src/hwva/providers/<name>.py` implementing the Protocol. Keys only from env; add them to `.env.example`. For video: declare honest `Capabilities` (does it accept audio? need an image? max duration?).
   **Capabilities depend on the version** from `ModelRef.version` — the factory receives the ref. Seedance: `2.5` → accepts audio without an image, ≤30 s; `2.0` → `needs_image=True`, ≤15 s. Test both versions.
3. Register it in `_FACTORIES` in `src/hwva/providers/registry.py` and reference it from a profile in `config/profiles.yaml` with a pinned model **and version**.
4. Tests: unit tests with the HTTP layer mocked (no network) in `tests/providers/test_<name>.py`; one optional `@pytest.mark.live` smoke test.
5. Gate: `uv run check` green. Then spawn `reviewer` on the diff.
6. Log in `AI-LOG.md`.

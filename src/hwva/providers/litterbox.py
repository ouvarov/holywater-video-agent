"""`MediaHost` on litterbox.catbox.moe: a local file -> a public https URL for 1 hour.

Why: OpenRouter's video API takes media only as public HTTPS URLs (live: a data URI
in input_references is answered 400 "Only HTTPS URLs are allowed").

Chosen for the test task: free, no account, no key, files are deleted by the host after 1 h.
The catch: while alive, the file is public to anyone who has its random URL (our TTS line and
the character image; nothing secret). In production this would be Cloudflare R2 (or S3) with
short-lived signed URLs from our own bucket — not worth setting up for the test.

  POST https://litterbox.catbox.moe/resources/internals/api.php
       multipart: reqtype=fileupload, time=1h, fileToUpload=<file>
    -> 200, the body is the file's URL (e.g. https://litter.catbox.moe/abc123.wav)

Errors: a connection error / timeout, 429 or 5xx -> retryable (uploading is free, nothing is
paid); other 4xx, a body that is not an https URL, an unreadable file -> final."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from hwva.providers.base import HostedMedia, ProviderError

API_URL = "https://litterbox.catbox.moe/resources/internals/api.php"
LIFETIME = "1h"
LIFETIME_S = 3600
TIMEOUT_S = 120.0


class Litterbox:
    """`transport` exists for tests (httpx.MockTransport); production uses the default one."""

    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(transport=transport, timeout=TIMEOUT_S)

    def upload(self, path: Path) -> HostedMedia:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ProviderError(f"litterbox: cannot read {path}: {exc}", retryable=False) from exc
        expires_at = datetime.now(UTC) + timedelta(seconds=LIFETIME_S)  # before the upload
        try:
            response = self._client.post(
                API_URL,
                data={"reqtype": "fileupload", "time": LIFETIME},
                files={"fileToUpload": (path.name, data)},
            )
        except httpx.RequestError as exc:  # free to retry: nothing is paid
            raise ProviderError(f"litterbox upload failed: {type(exc).__name__}: {exc}") from exc
        status = response.status_code
        if status != 200:
            retryable = status == 429 or status >= 500
            raise ProviderError(
                f"litterbox upload HTTP {status}: {response.text[:200]}", retryable=retryable
            )
        url = response.text.strip()
        if not url.startswith("https://") or any(c.isspace() for c in url):
            raise ProviderError(
                f"litterbox answered something that is not an https URL: {url[:200]!r}",
                retryable=False,
            )
        return HostedMedia(url=url, expires_at=expires_at)

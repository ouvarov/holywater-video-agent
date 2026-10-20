"""`MediaHost` on uguu.se: a local file -> a public https URL for 3 hours.

Why a second host: litterbox (the first choice, `litterbox.py`) answered 500 / 412 to uploads for
hours while every OpenRouter video call needs our audio and image as public HTTPS
URLs. uguu.se worked the same day for the paid test clips (scratch runs), so it became the
profile default; litterbox stays one line away.

Same trade-off as litterbox: free, no account, no key, files deleted by the host after ~3 h, and
while alive public to anyone who has the random URL (our TTS line and the character image;
nothing secret). In production: our own bucket (Cloudflare R2 or S3) with short-lived signed URLs.

  POST https://uguu.se/upload
       multipart: files[]=<file>
    -> 200, JSON {"success": true, "files": [{"url": "https://h.uguu.se/abc.wav", ...}]}

Errors: a connection error / timeout, 429 or 5xx -> retryable (uploading is free, nothing is
paid); other 4xx, an answer without an https URL, an unreadable file -> final."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from hwva.providers.base import HostedMedia, ProviderError

API_URL = "https://uguu.se/upload"
LIFETIME_S = 3 * 3600
TIMEOUT_S = 120.0


class Uguu:
    """`transport` exists for tests (httpx.MockTransport); production uses the default one."""

    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(transport=transport, timeout=TIMEOUT_S)

    def upload(self, path: Path) -> HostedMedia:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ProviderError(f"uguu: cannot read {path}: {exc}", retryable=False) from exc
        expires_at = datetime.now(UTC) + timedelta(seconds=LIFETIME_S)  # before the upload
        try:
            response = self._client.post(API_URL, files={"files[]": (path.name, data)})
        except httpx.RequestError as exc:  # free to retry: nothing is paid
            raise ProviderError(f"uguu upload failed: {type(exc).__name__}: {exc}") from exc
        status = response.status_code
        if status != 200:
            retryable = status == 429 or status >= 500
            raise ProviderError(
                f"uguu upload HTTP {status}: {response.text[:200]}", retryable=retryable
            )
        try:
            url = str(response.json()["files"][0]["url"]).strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"uguu answered without a file URL: {response.text[:200]!r}", retryable=False
            ) from exc
        if not url.startswith("https://") or any(c.isspace() for c in url):
            raise ProviderError(
                f"uguu answered something that is not an https URL: {url[:200]!r}",
                retryable=False,
            )
        return HostedMedia(url=url, expires_at=expires_at)

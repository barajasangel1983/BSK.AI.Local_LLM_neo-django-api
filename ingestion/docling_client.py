"""Docling Serve client (BSK ingestion desktop).

Async conversion flow against the live Docling Serve v1 API
(verified against the server's OpenAPI spec, 2026-09-29):

    1. POST {base}/v1/convert/source/async
       body: {"sources": [{"kind": "file", "filename": ..., "base64_string": ...}]}
       -> TaskStatusResponse {task_id, task_type, task_status}
    2. GET {base}/v1/status/poll/{task_id}
       -> TaskStatusResponse; ``task_status`` is e.g. queued/processing/... finished/failed
    3. GET {base}/v1/result/{task_id}
       -> conversion result document (inbody target by default)

The base URL comes from ``settings.DOCLING_URL`` (default
``http://100.86.26.4:5001``). Docling is a single-worker service, so
jobs are processed sequentially — callers should not expect parallelism.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Callable, Optional

import requests
from django.conf import settings


class DoclingError(Exception):
    """Raised when a Docling conversion task fails or times out."""


class DoclingClient:
    """Synchronous client for Docling Serve's async conversion API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or settings.DOCLING_URL).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        return headers

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def submit_file(
        self,
        file_bytes: bytes,
        filename: str,
        progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Submit a file for conversion. Returns the task id.

        Uses a ``kind: "file"`` source with the payload base64-encoded
        (per ``FileSourceRequest`` in the OpenAPI spec). The default
        ``target`` is ``inbody``, so the result document comes back via
        ``/v1/result/{task_id}``.
        """
        url = f"{self.base_url}/v1/convert/source/async"
        payload = {
            "sources": [
                {
                    "kind": "file",
                    "filename": filename,
                    "base64_string": base64.b64encode(file_bytes).decode("ascii"),
                }
            ],
        }
        resp = requests.post(
            url,
            json=payload,
            headers=self._headers(),
            timeout=max(self.timeout, 60.0),
        )
        if resp.status_code >= 300:
            raise DoclingError(
                f"Docling submit failed: HTTP {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        task_id = data.get("task_id") if isinstance(data, dict) else None
        if not task_id:
            raise DoclingError(
                f"Docling submit returned no task id: {resp.text[:500]}"
            )
        if progress:
            progress(f"submitted task {task_id}")
        return task_id

    def poll_status(
        self,
        task_id: str,
        timeout_s: float = 600.0,
        initial_delay: float = 1.0,
        max_delay: float = 15.0,
        backoff_factor: float = 1.5,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
        progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Poll ``/v1/status/poll/{task_id}`` until finished or timeout.

        Returns the final ``TaskStatusResponse`` payload. Raises
        :class:`DoclingError` if the task reports a failure state or the
        timeout is exceeded.
        """
        url = f"{self.base_url}/v1/status/poll/{task_id}"
        delay = initial_delay
        deadline = now() + timeout_s
        while True:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
            if resp.status_code >= 300:
                raise DoclingError(
                    f"Docling status poll failed: HTTP {resp.status_code}: "
                    f"{resp.text[:500]}"
                )
            status = resp.json()
            state = self._status_state(status)
            if state in ("failed", "error", "cancelled", "canceled"):
                raise DoclingError(
                    f"Docling task {task_id} failed: {str(status)[:500]}"
                )
            if state in ("finished", "completed", "done", "ready"):
                return status
            if now() >= deadline:
                raise DoclingError(
                    f"Docling task {task_id} timed out after {timeout_s}s "
                    f"(last state: {state or 'unknown'})"
                )
            if progress:
                position = status.get("task_position") if isinstance(status, dict) else None
                suffix = f" (queue position {position})" if position is not None else ""
                progress(f"polling (state={state or 'unknown'}{suffix})")
            sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    def fetch_result(self, task_id: str) -> dict:
        """Fetch the conversion result for a finished task."""
        url = f"{self.base_url}/v1/result/{task_id}"
        resp = requests.get(
            url, headers=self._headers(), timeout=max(self.timeout, 120.0)
        )
        if resp.status_code >= 300:
            raise DoclingError(
                f"Docling result fetch failed: HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )
        return resp.json()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _status_state(payload: Any) -> Optional[str]:
        """Normalize a TaskStatusResponse to a single state string."""
        if isinstance(payload, str):
            return payload.lower()
        if isinstance(payload, dict):
            for key in ("task_status", "state", "status"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    return value.lower()
        return None


# ---------------------------------------------------------------------- #
# Module-level convenience functions (used by the pipeline)
# ---------------------------------------------------------------------- #

def submit_file(file_bytes: bytes, filename: str) -> str:
    """Submit a file to Docling and return the task id."""
    return DoclingClient().submit_file(file_bytes, filename)


def poll_status(task_id: str, **kwargs) -> dict:
    """Poll a Docling task until finished. See :meth:`DoclingClient.poll_status`."""
    return DoclingClient().poll_status(task_id, **kwargs)


def fetch_result(task_id: str) -> dict:
    """Fetch a finished Docling task's result. See :meth:`DoclingClient.fetch_result`."""
    return DoclingClient().fetch_result(task_id)

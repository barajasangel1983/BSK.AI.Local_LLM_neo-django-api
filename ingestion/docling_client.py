"""Docling Serve client (BSK ingestion desktop).

Sync conversion flow against the live Docling Serve v1 API:

    POST {base}/v1/convert/source
    body: {"sources": [{"kind": "file", "filename": ..., "base64_string": ...}],
           "options": CONVERT_OPTIONS}
    -> Conversion result (inbody target)

The sync endpoint is more reliable than the async endpoint for our use case
(verified: async has intermittent PDF parse failures, sync works consistently).

The base URL comes from ``settings.DOCLING_URL`` (default ``http://100.86.26.4:5001``).
Docling is a single-worker service, so jobs are processed sequentially.
"""

from __future__ import annotations

import base64
from typing import Optional

import requests

from gpu import orchestrator as gpu
from usage import recorder as usage
from django.conf import settings

from .chunker import PAGE_BREAK

# Markdown with page-break markers (the chunker derives page numbers from them)
# and image placeholders instead of embedded base64 figures.
CONVERT_OPTIONS = {
    "to_formats": ["md"],
    "md_page_break_placeholder": PAGE_BREAK,
    "image_export_mode": "placeholder",
}


class DoclingError(Exception):
    """Raised when a Docling conversion fails."""


class DoclingClient:
    """Synchronous client for Docling Serve's sync conversion API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or settings.DOCLING_URL).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        return headers

    def convert_file(self, file_bytes: bytes, filename: str, wait: Optional[float] = None,
                     with_layout: bool = False) -> dict:
        """Convert a file and return the result document.

        `wait` is how long to wait for the BSK GPU when another Hub call holds it
        (default: GPU_LOCK_WAIT, for pipeline jobs; chat passes a short wait).
        `with_layout` also asks for Docling's JSON (`document.json_content`): the
        page and box of every figure, used by Describe figures.

        Uses a ``kind: "file"`` source with the payload base64-encoded.
        Returns the full conversion result dict (including 'document', 'status', 'errors').
        """
        url = f"{self.base_url}/v1/convert/source"
        payload = {
            "sources": [
                {
                    "kind": "file",
                    "filename": filename,
                    "base64_string": base64.b64encode(file_bytes).decode("ascii"),
                }
            ],
            "options": {**CONVERT_OPTIONS, "to_formats": ["md", "json"]} if with_layout else CONVERT_OPTIONS,
        }
        try:
            # Hold the BSK GPU (activating Docling through the orchestrator when enabled).
            with gpu.use("docling", wait=wait), usage.track("parse", "docling"):
                resp = self._post(url, payload)
        except gpu.GpuError as exc:
            raise DoclingError(f"Docling unavailable: {exc}") from exc
        result = resp.json()

        # Check for errors in the response
        if result.get("status") != "success" and result.get("errors"):
            raise DoclingError(f"Docling returned errors: {result['errors']}")

        return result

    def _post(self, url: str, payload: dict):
        resp = requests.post(
            url,
            json=payload,
            headers=self._headers(),
            timeout=max(self.timeout, 300.0),
        )
        if resp.status_code >= 300:
            raise DoclingError(
                f"Docling conversion failed: HTTP {resp.status_code}: {resp.text[:500]}"
            )
        return resp


# Module-level convenience function (used by pipeline)
_client = DoclingClient()


def convert_file(file_bytes: bytes, filename: str, wait: Optional[float] = None, with_layout: bool = False) -> dict:
    """Convert a file and return the result document."""
    return _client.convert_file(file_bytes, filename, wait=wait, with_layout=with_layout)

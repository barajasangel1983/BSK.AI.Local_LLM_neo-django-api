"""Client for the BSK GPU orchestrator (contract: claude/VLM_service_brief.md, v1.2).

The BSK desktop has one 8 GB GPU that runs either Docling or the VLM
(Qwen3-VL-4B), never both. Its orchestrator (:5003) starts the requested
service and stops the other; a switch stops whatever is running, mid-request
included. So:

- `use(service)` holds a Hub-wide lock (a file lock shared by the API and the
  library worker) for the whole call. Nothing in the Hub switches the GPU
  while another Hub call is using it, and interactive requests wait
  ("GPU busy") instead of interrupting a parse (DEFER rule).
- Inside the lock it activates the service and waits until it is healthy:
  POST /gpu/activate; while `health` is "starting", poll GET /gpu/status
  every 2 s; 409 (another transition owns the orchestrator's lock) → back off
  and retry; `health: "error"` → one retry, then BSK counts as unavailable.
- With GPU_ORCHESTRATOR_ENABLED off (until BSK's orchestrator is live) the
  Hub calls services directly, as before. When enabled but the orchestrator
  can't be reached, Docling is still called directly if its own health check
  answers (backwards compatible during the switch-over).
"""

from __future__ import annotations

import fcntl
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import requests
from django.conf import settings

from usage import recorder as usage

logger = logging.getLogger("gpu")

SERVICES = ("docling", "vl", "idle")
POLL_SECONDS = 2.0
CONNECT_TIMEOUT = 5.0
CONFLICT_BACKOFF_SECONDS = 2.0
MAX_ACTIVATION_FAILURES = 2


class GpuError(RuntimeError):
    """The BSK GPU service could not be used."""


class GpuBusy(GpuError):
    """Another Hub call is using the GPU and the wait limit passed."""


class GpuUnavailable(GpuError):
    """BSK is asleep / unreachable, or the service failed to start (twice)."""


class OrchestratorUnreachable(GpuUnavailable):
    """The orchestrator itself didn't answer (connection error or timeout)."""


def enabled() -> bool:
    return bool(settings.GPU_ORCHESTRATOR_ENABLED)


def _url(path: str) -> str:
    return settings.GPU_ORCHESTRATOR_URL.rstrip("/") + path


def status(timeout: float = 5.0) -> dict:
    """GET /gpu/status → {"active", "health", "idle_reset_at", "last_activation"}."""
    try:
        resp = requests.get(_url("/gpu/status"), timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise OrchestratorUnreachable(f"GPU orchestrator unreachable: {exc}") from exc


def activate(service: str) -> dict:
    """Make `service` the active GPU service and wait until it is healthy."""
    if service not in SERVICES:
        raise ValueError(f"unknown GPU service {service!r}")
    deadline = time.monotonic() + settings.GPU_ACTIVATE_TIMEOUT
    failures = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GpuUnavailable(f"{service} did not become healthy within {settings.GPU_ACTIVATE_TIMEOUT:.0f} s")
        try:
            resp = requests.post(_url("/gpu/activate"), json={"service": service},
                                 # connect fast (BSK asleep = dropped packets); activate itself may block ~40 s
                                 timeout=(CONNECT_TIMEOUT, min(45.0, max(remaining, 1.0))))
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise OrchestratorUnreachable(f"GPU orchestrator unreachable: {exc}") from exc
        if resp.status_code == 409:      # another transition owns the orchestrator's lock
            time.sleep(CONFLICT_BACKOFF_SECONDS)
            continue
        if resp.status_code == 422:
            raise ValueError(f"orchestrator rejected service {service!r}")
        resp.raise_for_status()
        body = resp.json()
        while body.get("health") == "starting" and time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            body = status()
        if body.get("health") == "ok" and body.get("active") == service:
            return body
        if body.get("health") == "starting":
            continue  # deadline check at the top of the loop
        failures += 1
        detail = body.get("error_detail") or f"health={body.get('health')}, active={body.get('active')}"
        logger.warning("gpu activation failed service=%s attempt=%d: %s", service, failures, str(detail)[:300])
        if failures >= MAX_ACTIVATION_FAILURES:
            raise GpuUnavailable(f"{service} failed to start on BSK: {str(detail)[:300]}")


def _docling_answers() -> bool:
    try:
        return requests.get(settings.DOCLING_URL.rstrip("/") + "/health", timeout=5).status_code == 200
    except requests.RequestException:
        return False


@contextmanager
def _lock(wait: float):
    """Hub-wide GPU lock (flock on a shared file: works across the API and the worker)."""
    path = Path(settings.GPU_LOCK_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + wait
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise GpuBusy(f"GPU busy: another job is using BSK (waited {wait:.0f} s)")
                time.sleep(0.5)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextmanager
def use(service: str, wait: float | None = None):
    """Hold the GPU for `service` for the duration of the block (see module docstring)."""
    with _lock(settings.GPU_LOCK_WAIT if wait is None else wait):
        if enabled():
            try:
                with usage.track("gpu", f"gpu:{service}"):
                    activate(service)
            except OrchestratorUnreachable:
                if service == "docling" and _docling_answers():
                    logger.warning("gpu orchestrator unreachable; calling Docling directly (it answers)")
                else:
                    raise
        yield


def health_summary() -> dict:
    """For the Health page: orchestrator state without probing the services themselves."""
    if not enabled():
        return {"enabled": False}
    try:
        start = time.monotonic()
        body = status()
        return {"enabled": True, "reachable": True, "latency_ms": round((time.monotonic() - start) * 1000), **body}
    except GpuError as exc:
        return {"enabled": True, "reachable": False, "error": str(exc)}

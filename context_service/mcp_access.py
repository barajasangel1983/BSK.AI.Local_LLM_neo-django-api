"""Tokens and exposure of the Context MCP server (managed in the Studio's Settings → MCP)."""

from __future__ import annotations

import hashlib
import secrets
from datetime import timedelta

from django.conf import settings
from django.utils import timezone

from .models import McpSettings, McpToken

TOKEN_PREFIX = "bsk_"
LAST_USED_EVERY = timedelta(minutes=1)      # don't write the database on every call


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_token(name: str) -> tuple[McpToken, str]:
    """A new token for a client. Returns (record, the token itself — shown once)."""
    name = " ".join(str(name or "").split())[:80]
    if not name:
        raise ValueError("name is required (which client is the token for?)")
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    record = McpToken.objects.create(name=name, prefix=token[:10], token_hash=_hash(token))
    return record, token


def verify(token: str) -> McpToken | None:
    """The active token record for a presented token, or None."""
    if not token or not token.startswith(TOKEN_PREFIX):
        return None
    record = McpToken.objects.filter(token_hash=_hash(token), revoked_at__isnull=True).first()
    if record and (record.last_used_at is None or timezone.now() - record.last_used_at > LAST_USED_EVERY):
        McpToken.objects.filter(pk=record.pk).update(last_used_at=timezone.now())
    return record


def revoke(token_id) -> bool:
    return bool(McpToken.objects.filter(pk=token_id, revoked_at__isnull=True).update(revoked_at=timezone.now()))


def token_json(record: McpToken) -> dict:
    return {"id": str(record.id), "name": record.name, "prefix": record.prefix,
            "created_at": record.created_at.isoformat(),
            "last_used_at": record.last_used_at.isoformat() if record.last_used_at else None,
            "revoked_at": record.revoked_at.isoformat() if record.revoked_at else None}


def listen_hosts(exposure: str | None = None) -> list[str]:
    """Addresses the server listens on for an exposure setting. Never all interfaces:
    this host has a public address."""
    exposure = exposure or McpSettings.get().exposure
    if exposure == McpSettings.Exposure.OFF:
        return []
    hosts = ["127.0.0.1"]
    if exposure == McpSettings.Exposure.TAILSCALE and settings.MCP_TAILSCALE_HOST:
        hosts.append(settings.MCP_TAILSCALE_HOST)
    return hosts


EXPOSURE_OPTIONS = [
    {"key": "off", "label": "Off", "available": True, "note": "The MCP server accepts no connections."},
    {"key": "local", "label": "This machine", "available": True,
     "note": "Only programs on the Studio's own machine can connect."},
    {"key": "tailscale", "label": "Tailscale", "available": True,
     "note": "Devices in your Tailscale network can connect (BSKLAB EDGE, Claude Desktop or Claude Code on your PC)."},
    {"key": "lan", "label": "LAN / WLAN", "available": False,
     "note": "Not available on this host: it has no plant network. This is the option for the Jetson Thor."},
    {"key": "internet", "label": "Internet", "available": False,
     "note": "Not built yet. It needs HTTPS and stronger sign-in than a token, and plant knowledge would leave the site."},
]

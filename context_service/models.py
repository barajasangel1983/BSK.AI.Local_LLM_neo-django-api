import uuid

from django.db import models


class McpSettings(models.Model):
    """Where the Context MCP server listens (one row; edited in the Studio's Settings → MCP)."""

    class Exposure(models.TextChoices):
        OFF = "off", "Off"
        LOCAL = "local", "This machine"
        TAILSCALE = "tailscale", "Tailscale"

    exposure = models.CharField(max_length=16, choices=Exposure.choices, default=Exposure.LOCAL)
    updated_at = models.DateTimeField(auto_now=True)

    @classmethod
    def get(cls) -> "McpSettings":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class McpToken(models.Model):
    """A client's access token for the MCP server. Only its hash is stored; the token itself
    is shown once, when it is created."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=80)                     # the client: "BSKLAB EDGE", "Claude Desktop (Angel)"
    prefix = models.CharField(max_length=12)                   # first characters, to recognise it in the list
    token_hash = models.CharField(max_length=64, unique=True)  # sha256
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)
    revoked_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.prefix}…)"

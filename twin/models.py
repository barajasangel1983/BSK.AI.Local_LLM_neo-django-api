"""The digital twin's manifest per machine (P32): kept by the Studio, generated from the graph, read by the Studio's
Home and by EDGE (MCP). `enabled` hides the twin everywhere without a deploy."""

from django.db import models


class TwinManifest(models.Model):
    asset_id = models.CharField(max_length=512, unique=True)
    manifest = models.JSONField(default=dict)
    enabled = models.BooleanField(default=True)
    generated_at = models.DateTimeField(null=True, blank=True)
    updated_by = models.CharField(max_length=200, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

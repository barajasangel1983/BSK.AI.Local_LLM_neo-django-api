from django.db import models


class GraphSchemaVersion(models.Model):
    """One version of the global Context Graph schema (exactly one is active).

    The definition is the YAML/JSON document parsed by context_graph.schema.
    Versions are append-only, so every extraction or import can record the
    schema version it used.
    """

    version = models.PositiveIntegerField(unique=True)
    name = models.CharField(max_length=100)
    definition = models.JSONField()
    note = models.CharField(max_length=255, blank=True)
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-version"]
        constraints = [
            models.UniqueConstraint(fields=["is_active"], condition=models.Q(is_active=True),
                                    name="one_active_graph_schema"),
        ]

    def __str__(self) -> str:
        return f"{self.name} v{self.version}{' (active)' if self.is_active else ''}"

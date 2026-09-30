import uuid

from django.db import models


class IngestionJob(models.Model):
    """Tracks one document through the ingestion pipeline.

    Idempotency: a job is unique per (source_sha256, document_revision,
    config_version). Re-uploading the same file at the same revision under
    the same config returns the existing job instead of creating a new one.
    """

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        PARSING = "parsing", "Parsing"
        CHUNKING = "chunking", "Chunking"
        EMBEDDING = "embedding", "Embedding"
        STORED = "stored", "Stored"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    source_filename = models.CharField(max_length=512)
    source_sha256 = models.CharField(max_length=64, db_index=True)
    document_id = models.CharField(
        max_length=512,
        help_text="Stable document identifier derived from the filename.",
    )
    document_revision = models.CharField(max_length=64, default="1")
    asset_id = models.CharField(
        max_length=128,
        blank=True,
        help_text="Free-text asset reference (validated at the API layer).",
    )
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.QUEUED,
        db_index=True,
    )
    error = models.TextField(null=True, blank=True)
    chunk_count = models.PositiveIntegerField(null=True, blank=True)
    config_version = models.CharField(max_length=32, default="v1-2026-09-29")
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "ingestion_job"
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["source_sha256", "document_revision", "config_version"],
                name="uniq_ingestion_job_source_revision_config",
            ),
        ]

    def __str__(self):
        return f"{self.id} {self.source_filename} [{self.status}]"

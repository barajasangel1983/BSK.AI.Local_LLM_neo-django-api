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


class Document(models.Model):
    """A file in the shared document library (used by RAG Lab and GraphLab).

    Uploaded and parsed (Docling) once; each pipeline is then run per file on
    request: "Generate embeddings" (RAG, bsk_rag_v2) now, "Generate triples"
    (GraphLab) in a later phase. `doc_key` is the key stored as `asset_id` in
    Chroma chunk metadata (a document key, not a machine asset).
    """

    class ParseStatus(models.TextChoices):
        PENDING = "pending", "Pending"
        QUEUED = "queued", "Queued"
        PARSING = "parsing", "Parsing"
        PARSED = "parsed", "Parsed"
        FAILED = "failed", "Failed"

    class PipelineStatus(models.TextChoices):
        NONE = "none", "Not generated"
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    class FiguresStatus(models.TextChoices):
        NONE = "none", "Not described"
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        PENDING = "pending", "Visual interpretation pending"    # BSK was unreachable
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=512)
    doc_key = models.CharField(max_length=128, unique=True)
    sha256 = models.CharField(max_length=64, db_index=True)
    size = models.PositiveBigIntegerField(default=0)
    document_revision = models.CharField(max_length=64, default="1")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    parse_status = models.CharField(max_length=16, choices=ParseStatus.choices, default=ParseStatus.PENDING)
    parse_error = models.TextField(blank=True, default="")
    parsed_at = models.DateTimeField(null=True, blank=True)
    page_count = models.PositiveIntegerField(null=True, blank=True)

    rag_status = models.CharField(max_length=16, choices=PipelineStatus.choices, default=PipelineStatus.NONE)
    rag_strategy = models.CharField(max_length=32, blank=True, default="")
    rag_params = models.JSONField(default=dict, blank=True)
    rag_chunk_count = models.PositiveIntegerField(default=0)
    rag_error = models.TextField(blank=True, default="")
    rag_updated_at = models.DateTimeField(null=True, blank=True)

    # GraphLab "Generate triples": status of the latest extraction run.
    graph_status = models.CharField(max_length=16, choices=PipelineStatus.choices, default=PipelineStatus.NONE)
    graph_error = models.TextField(blank=True, default="")
    graph_updated_at = models.DateTimeField(null=True, blank=True)

    # "Describe figures" (VLM on BSK): status of the latest run. The figures themselves
    # (position, description) are in LIBRARY_BASE/<id>/figures.json.
    figures_status = models.CharField(max_length=16, choices=FiguresStatus.choices, default=FiguresStatus.NONE)
    figures_error = models.TextField(blank=True, default="")
    figures_updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "library_document"
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.doc_key} ({self.filename})"


class Job(models.Model):
    """Background task for the library worker (`manage.py library_worker`)."""

    class Kind(models.TextChoices):
        PARSE = "parse", "Parse"
        EMBED = "embed", "Generate embeddings"
        EXTRACT = "extract", "Generate triples"
        FIGURES = "figures", "Describe figures"

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="jobs")
    kind = models.CharField(max_length=16, choices=Kind.choices)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED, db_index=True)
    params = models.JSONField(default=dict, blank=True)
    progress_done = models.PositiveIntegerField(default=0)
    progress_total = models.PositiveIntegerField(default=0)
    message = models.CharField(max_length=255, blank=True, default="")
    error = models.TextField(blank=True, default="")
    attempts = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    heartbeat_at = models.DateTimeField(null=True, blank=True)
    # A queued job is not started before this time (retry with back-off).
    run_after = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "library_job"
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.kind} {self.document_id} [{self.status}]"

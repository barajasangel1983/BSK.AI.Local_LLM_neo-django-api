import uuid

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


class PromptPreset(models.Model):
    """Prompt template for triple extraction (GraphLab).

    Placeholders: {entity_types}, {relationships} (schema mode, rendered from the
    active schema), {document}, {section}, {text}. Editing bumps `version`, which
    every extracted triple records.
    """

    class Mode(models.TextChoices):
        SCHEMA = "schema", "Schema-guided"
        FREEFORM = "freeform", "Free-form"

    name = models.CharField(max_length=100, unique=True)
    mode = models.CharField(max_length=16, choices=Mode.choices)
    template = models.TextField()
    version = models.PositiveIntegerField(default=1)
    is_default = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["mode", "name"]

    def __str__(self) -> str:
        return f"{self.name} ({self.mode} v{self.version})"


class DataFile(models.Model):
    """A structured data file (CSV / Excel: tag list, alarm list, BOM) for GraphLab Import.

    Kept apart from the document library: rows are mapped to triples by columns
    (no LLM). The file lives under GRAPH_DATA_BASE/<id>/.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=512)
    key = models.CharField(max_length=128, unique=True)     # scope for proposed ids when no asset is chosen
    kind = models.CharField(max_length=8)                   # csv | xlsx
    size = models.PositiveBigIntegerField(default=0)
    sha256 = models.CharField(max_length=64, unique=True)
    sheets = models.JSONField(default=list, blank=True)     # sheet names (xlsx)
    sheet = models.CharField(max_length=255, blank=True, default="")
    header_row = models.PositiveIntegerField(default=1)     # 1-based row holding the column names
    columns = models.JSONField(default=list, blank=True)
    row_count = models.PositiveIntegerField(default=0)
    asset_id = models.CharField(max_length=512, blank=True, default="")
    mapping = models.JSONField(default=dict, blank=True)    # last mapping used to stage triples
    uploaded_at = models.DateTimeField(auto_now_add=True)
    staged_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self) -> str:
        return self.filename


class ImportMapping(models.Model):
    """A saved column mapping, reusable for files with the same layout."""

    name = models.CharField(max_length=100, unique=True)
    mapping = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Evidence(models.Model):
    """One place a fact was found (P8): a document window / page / region, a data-file row,
    a figure (VLM), a human edit, later an OPC UA mapping.

    Evidence rows belong to the triples staged from them; approved triples copy their
    evidence IDs onto the Neo4j nodes and edges they write, so every graph fact points
    back to its sources. Only a short excerpt is kept here; the text stays in the library.
    """

    class Kind(models.TextChoices):
        TEXT = "text", "Document text"
        STRUCTURED = "structured", "Structured import"
        VISION = "vision", "Vision (VLM)"
        HUMAN = "human", "Human"
        OPC = "opc", "OPC UA"

    source_kind = models.CharField(max_length=16, choices=Kind.choices)
    document = models.ForeignKey("ingestion.Document", on_delete=models.CASCADE, related_name="evidence",
                                 null=True, blank=True)
    data_file = models.ForeignKey("DataFile", on_delete=models.CASCADE, related_name="evidence", null=True, blank=True)
    page_start = models.PositiveIntegerField(null=True, blank=True)
    page_end = models.PositiveIntegerField(null=True, blank=True)
    section_path = models.JSONField(default=list, blank=True)
    row_number = models.PositiveIntegerField(null=True, blank=True)
    region = models.JSONField(default=dict, blank=True)          # {page, l, t, r, b, coord_origin} (figures, VLM)
    figure_id = models.CharField(max_length=64, blank=True, default="")
    chunk_index = models.PositiveIntegerField(null=True, blank=True)   # extraction window
    window = models.JSONField(default=dict, blank=True)          # chunking strategy + params of that window
    extractor = models.CharField(max_length=32, blank=True, default="")   # llm-text, column-mapping, vlm, human
    model = models.CharField(max_length=100, blank=True, default="")
    prompt = models.CharField(max_length=120, blank=True, default="")      # preset name + version
    schema_version = models.PositiveIntegerField(null=True, blank=True)
    excerpt = models.TextField(blank=True, default="")
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def __str__(self) -> str:
        where = f"p.{self.page_start}" if self.page_start else f"row {self.row_number}" if self.row_number else ""
        return f"{self.source_kind} {where}".strip()


class CandidateTriple(models.Model):
    """A triple staged for review: extracted from a library document (LLM) or
    built from a structured data file (column mapping).

    pending -> approved (written to Neo4j: curated layer for schema triples,
    lab layer for free-form ones) or rejected. Deleting an approved triple
    removes what it wrote.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"

    class Layer(models.TextChoices):
        CURATED = "curated", "Curated"
        LAB = "lab", "Lab"

    class Source(models.TextChoices):
        TEXT = "text", "Extracted from text"
        STRUCTURED = "structured", "Structured import"

    # Exactly one of document / data_file is set.
    document = models.ForeignKey("ingestion.Document", on_delete=models.CASCADE, related_name="triples",
                                 null=True, blank=True)
    data_file = models.ForeignKey(DataFile, on_delete=models.CASCADE, related_name="triples", null=True, blank=True)
    source = models.CharField(max_length=16, choices=Source.choices, default=Source.TEXT)
    job = models.ForeignKey("ingestion.Job", on_delete=models.SET_NULL, null=True, blank=True, related_name="triples")
    mode = models.CharField(max_length=16, choices=PromptPreset.Mode.choices)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING, db_index=True)
    layer = models.CharField(max_length=16, choices=Layer.choices, blank=True, default="")

    subject_name = models.CharField(max_length=255)
    subject_type = models.CharField(max_length=100)
    subject_id = models.CharField(max_length=512, blank=True, default="")
    subject_existing = models.BooleanField(default=False)
    predicate = models.CharField(max_length=100)
    object_name = models.CharField(max_length=255)
    object_type = models.CharField(max_length=100)
    object_id = models.CharField(max_length=512, blank=True, default="")
    object_existing = models.BooleanField(default=False)
    # Node properties from structured imports (written when the entity is created; missing ones added otherwise).
    subject_props = models.JSONField(default=dict, blank=True)
    object_props = models.JSONField(default=dict, blank=True)
    applied_props = models.JSONField(default=dict, blank=True)   # {node id: [property names added to an existing node]}
    applied_aliases = models.JSONField(default=dict, blank=True)  # {node id: [names added as aliases]}
    # Every place this triple was found (all windows / rows); the evidence_* fields below keep the first one.
    evidence = models.ManyToManyField(Evidence, related_name="triples", blank=True)
    confidence = models.FloatField(null=True, blank=True)
    issue = models.CharField(max_length=255, blank=True, default="")  # e.g. pair not allowed by the schema

    # Evidence (text lives here, in the document library — never in Neo4j)
    chunk_index = models.PositiveIntegerField(default=0)
    page_start = models.PositiveIntegerField(null=True, blank=True)
    page_end = models.PositiveIntegerField(null=True, blank=True)
    section_path = models.JSONField(default=list, blank=True)
    evidence_text = models.TextField(blank=True, default="")
    row_number = models.PositiveIntegerField(null=True, blank=True)   # spreadsheet row (structured imports)
    occurrences = models.PositiveIntegerField(default=1)

    # Provenance
    model = models.CharField(max_length=100, blank=True, default="")
    preset_name = models.CharField(max_length=100, blank=True, default="")
    preset_version = models.PositiveIntegerField(null=True, blank=True)
    schema_version = models.PositiveIntegerField(null=True, blank=True)

    edited = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    committed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["document_id", "chunk_index", "id"]
        indexes = [models.Index(fields=["document", "status"]), models.Index(fields=["data_file", "status"])]

    def __str__(self) -> str:
        return f"({self.subject_name})-[{self.predicate}]->({self.object_name}) [{self.status}]"

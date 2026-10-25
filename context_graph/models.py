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

    def save(self, *args, **kwargs):
        """Every version the active copy reaches is kept (P23): a version number with no PromptVersion row gets one,
        whatever path saved it (the Prompts tab, the old preset PATCH, a built-in seeded from code). Callers may
        set `_version_note`, `_version_source`, `_version_author` first."""
        super().save(*args, **kwargs)
        if not self.versions.filter(version=self.version).exists():
            from . import registry
            try:
                schema_version = registry.active_schema().version
            except Exception:   # no schema yet (a fresh database)
                schema_version = None
            PromptVersion.objects.create(
                preset=self, version=self.version, template=self.template, schema_version=schema_version,
                note=getattr(self, "_version_note", "")[:255], source=getattr(self, "_version_source", "edit"),
                author=getattr(self, "_version_author", "")[:100])


class PromptVersion(models.Model):
    """Every saved text of a prompt (P23, like GraphSchemaVersion): triples record name + version, which points here."""

    preset = models.ForeignKey(PromptPreset, on_delete=models.CASCADE, related_name="versions")
    version = models.PositiveIntegerField()
    template = models.TextField()
    note = models.CharField(max_length=255, blank=True, default="")
    schema_version = models.PositiveIntegerField(null=True, blank=True)   # the active schema when it was saved
    source = models.CharField(max_length=16, default="edit")   # built-in | edit | import | adopt | migrated
    author = models.CharField(max_length=100, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["preset", "-version"]
        constraints = [models.UniqueConstraint(fields=["preset", "version"], name="prompt_version_unique")]

    def __str__(self) -> str:
        return f"{self.preset.name} v{self.version}"


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
    # P26f: a newer version of the same file (same filename, same asset) was staged. Kept with its triples and their
    # evidence (deleting it would take its approved triples back out of the graph); hidden in the Import list.
    superseded_by = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="supersedes")
    superseded_at = models.DateTimeField(null=True, blank=True)

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
    # P25b: where the window sits in the document's text snapshot (ingestion/spans.py); P25e: the supporting quote
    # inside it. All empty for rows written before P25 until `manage.py evidence_spans` fills them.
    text_hash = models.CharField(max_length=64, blank=True, default="")
    char_start = models.PositiveIntegerField(null=True, blank=True)
    char_end = models.PositiveIntegerField(null=True, blank=True)
    span_status = models.CharField(max_length=16, blank=True, default="")    # exact | partial | ambiguous | not_found
    row_start = models.PositiveIntegerField(null=True, blank=True)           # a table part's rows (1-based)
    row_end = models.PositiveIntegerField(null=True, blank=True)
    quote_start = models.PositiveIntegerField(null=True, blank=True)
    quote_end = models.PositiveIntegerField(null=True, blank=True)
    quote_status = models.CharField(max_length=16, blank=True, default="")

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
    # Identity resolution (P8b): how each end matched — id | tag | alias | name | possible | new — and,
    # for "possible", the suggested entities [{id, name, score}] a reviewer picks from.
    subject_match = models.CharField(max_length=10, blank=True, default="")
    object_match = models.CharField(max_length=10, blank=True, default="")
    subject_candidates = models.JSONField(default=list, blank=True)
    object_candidates = models.JSONField(default=list, blank=True)
    # Node properties from structured imports (written when the entity is created; missing ones added otherwise).
    subject_props = models.JSONField(default=dict, blank=True)
    object_props = models.JSONField(default=dict, blank=True)
    applied_props = models.JSONField(default=dict, blank=True)   # {node id: [property names added to an existing node]}
    applied_aliases = models.JSONField(default=dict, blank=True)  # {node id: [names added as aliases]}
    replaces = models.JSONField(default=dict, blank=True)         # {node id: {property: [graph value, new value]}}, staged (P14b)
    replaced_props = models.JSONField(default=dict, blank=True)   # {node id: {property: value before approval}}
    # P26b: what the run was anchored on (an installed part, a device model or a product family), the reviewer's move
    # to another level ("moved from <id>"), and the link's own properties (`instance`: the installed part a family's
    # MAPS_TO / CARRIES / INDICATED_BY is about).
    anchor_id = models.CharField(max_length=512, blank=True, default="", db_index=True)
    moved_from = models.CharField(max_length=512, blank=True, default="")
    rel_props = models.JSONField(default=dict, blank=True)
    # Every place this triple was found (all windows / rows); the evidence_* fields below keep the first one.
    evidence = models.ManyToManyField(Evidence, related_name="triples", blank=True)
    confidence = models.FloatField(null=True, blank=True)
    issue = models.CharField(max_length=255, blank=True, default="")  # e.g. pair not allowed by the schema
    # A change made when staging, with the link as the model wrote it ("flipped from Alarm -MITIGATED_BY-> Procedure").
    auto_fix = models.CharField(max_length=255, blank=True, default="")
    # Knowledge facets (P22b): derived from the active schema's mapping (context_graph/facets.py), never typed
    facets = models.JSONField(default=list, blank=True)

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


class FactMoveBatch(models.Model):
    """One apply of P26c "Move to families": a family's (or a family-less model's) manual facts moved off its installed
    parts. `before` holds every node and link the apply touched, as they were, so `undo` can put them back."""

    class Status(models.TextChoices):
        APPLIED = "applied"
        UNDONE = "undone"

    target_id = models.CharField(max_length=512, db_index=True)        # the family (or model) the facts went to
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.APPLIED)
    choices = models.JSONField(default=dict, blank=True)               # {conflict key: models | a | b | both}, stays
    summary = models.JSONField(default=dict, blank=True)               # counts: moved, merged, conflicts, ...
    before = models.JSONField(default=dict, blank=True)                # {nodes: [...], rels: [...], triples: [...]}
    created_links = models.JSONField(default=list, blank=True)         # links the apply added outside the moved nodes
    author = models.CharField(max_length=150, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    undone_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]


class FactMove(models.Model):
    """One fact's move in a batch: where it was, where it went, and how (moved, merged into another, kept per model)."""

    class Kind(models.TextChoices):
        MOVE = "move"
        MERGE = "merge"          # merged into `new_id` (the survivor); the node `old_id` no longer exists
        MODEL = "model"          # a conflict kept on its own model
        STAY = "stay"            # left on the installed part by the reviewer

    batch = models.ForeignKey(FactMoveBatch, on_delete=models.CASCADE, related_name="moves")
    kind = models.CharField(max_length=10, choices=Kind.choices)
    label = models.CharField(max_length=40, blank=True, default="")
    old_id = models.CharField(max_length=512, db_index=True)
    new_id = models.CharField(max_length=512, blank=True, default="", db_index=True)
    owner_before = models.CharField(max_length=512, blank=True, default="")
    owner_after = models.CharField(max_length=512, blank=True, default="")

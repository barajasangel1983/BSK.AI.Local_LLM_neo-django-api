"""P29: problem-solving tickets (5-Why), owned by the Studio and keyed by machine.

A ticket opens automatically after a long downtime or a long SPC violation (problems.triggers, run by the monitor's
minute pass), or by hand from EDGE. A Manufacturing Engineer fills the 5-Why on the EDGE tablet; a second engineer
approves it; an approved report becomes a library document (P29d). See claude/P29_problem_solving_plan.md.
"""

from __future__ import annotations

from django.db import models
from django.utils import timezone


class ProblemSettings(models.Model):
    """One row: the triggers and the knowledge check. Switched off until an admin turns it on (no tickets for the
    past: only problems whose threshold is crossed after `enabled_at`)."""

    class CheckMode(models.TextChoices):
        ON_SUBMIT = "on_submit", "Right after submit (the model that answers)"
        DGX = "dgx", "When the 27B (DGX) answers"
        SCHEDULE = "schedule", "On a schedule"

    enabled = models.BooleanField(default=False)
    enabled_at = models.DateTimeField(null=True, blank=True)
    downtime_hours = models.FloatField(default=3.0)
    downtime_states = models.JSONField(default=list)                 # empty → DEFAULT_DOWNTIME_STATES
    spc_control_hours = models.FloatField(default=3.0)
    spc_spec_hours = models.FloatField(default=2.0)
    check_mode = models.CharField(max_length=16, choices=CheckMode.choices, default=CheckMode.DGX)
    check_day = models.CharField(max_length=3, default="sun")         # mon … sun, for the schedule
    check_time = models.CharField(max_length=5, default="22:00")      # HH:MM, the plant's time (UTC without a calendar)
    publish_folder = models.CharField(max_length=200, default="Problem solving")
    updated_by = models.CharField(max_length=200, blank=True, default="")
    updated_at = models.DateTimeField(auto_now=True)

    DEFAULT_DOWNTIME_STATES = ["STOPPED", "ALARMED"]

    class Meta:
        db_table = "problems_settings"

    @classmethod
    def get(cls) -> "ProblemSettings":
        return cls.objects.get_or_create(pk=1)[0]

    @property
    def states(self) -> list[str]:
        return [s.upper() for s in (self.downtime_states or self.DEFAULT_DOWNTIME_STATES)]


class Ticket(models.Model):
    class Trigger(models.TextChoices):
        DOWNTIME = "downtime", "Downtime"
        SPC_CONTROL = "spc_control", "Outside SPC control limits"
        SPC_SPEC = "spc_spec", "Outside specification limits"
        MANUAL = "manual", "Opened by hand"

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        DRAFT = "draft", "Draft"
        SUBMITTED = "submitted", "Submitted"
        RETURNED = "returned", "Returned"
        APPROVED = "approved", "Approved"            # the analysis and the action plan are accepted; actions are followed
        COMPLETED = "completed", "Completed"         # every action is done and validated
        CLOSED = "closed", "Closed without analysis"

    asset_id = models.CharField(max_length=512, db_index=True)
    asset_name = models.CharField(max_length=200, blank=True, default="")
    trigger = models.CharField(max_length=16, choices=Trigger.choices)
    source_key = models.CharField(max_length=300, blank=True, default="")   # what opened it (no ticket twice)
    title = models.CharField(max_length=300)
    window_start = models.DateTimeField()
    window_end = models.DateTimeField(null=True, blank=True)                 # empty: still going
    context = models.JSONField(default=dict, blank=True)                     # the snapshot (problems.context)
    recommendations = models.JSONField(default=dict, blank=True)             # P29b: what to check {items, model, at}
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.OPEN, db_index=True)

    # The 5-Why, as the engineer leaves it.
    notes = models.TextField(blank=True, default="")             # free notes, typed or spoken (P29b structures them)
    problem = models.TextField(blank=True, default="")
    whys = models.JSONField(default=list, blank=True)            # up to five strings
    root_cause = models.TextField(blank=True, default="")
    containment = models.TextField(blank=True, default="")
    corrective_actions = models.TextField(blank=True, default="")   # the actions as text (kept in step with `actions`)
    # The corrective actions: [{id, text, responsible, due_date (YYYY-MM-DD), status (in_progress | done), done_at,
    # validated_by {username, name}, validated_at}]. "Delayed" is not stored: a due date passed and not done.
    actions = models.JSONField(default=list, blank=True)
    final_report = models.TextField(blank=True, default="")       # P29b's final version, edited by the engineer

    knowledge_check = models.JSONField(default=dict, blank=True)            # P29c: {status, verdict, points, model, at, error}
    author = models.JSONField(default=dict, blank=True)           # {username, name, role, source} of who submitted
    approver = models.JSONField(default=dict, blank=True)
    close_reason = models.TextField(blank=True, default="")
    return_comment = models.TextField(blank=True, default="")
    history = models.JSONField(default=list, blank=True)          # [{at, action, by: {username, name, role}, comment}]
    document_id = models.CharField(max_length=64, blank=True, default="")   # P29d: the library document
    publish_error = models.TextField(blank=True, default="")

    opened_by = models.JSONField(default=dict, blank=True)        # empty: the system (a trigger)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    submitted_at = models.DateTimeField(null=True, blank=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "problems_ticket"
        ordering = ["-created_at"]
        constraints = [models.UniqueConstraint(fields=["source_key"], condition=~models.Q(source_key=""),
                                               name="problems_one_ticket_per_source")]

    @property
    def number(self) -> str:
        return f"PS-{self.pk:04d}"


class TicketPicture(models.Model):
    """A picture of the problem, taken on the tablet. Files live under PROBLEM_FILES_DIR/<ticket id>/."""

    ticket = models.ForeignKey(Ticket, related_name="pictures", on_delete=models.CASCADE)
    class Phase(models.TextChoices):
        PROBLEM = "problem", "The problem"
        AFTER = "after", "After the fix"

    phase = models.CharField(max_length=8, choices=Phase.choices, default=Phase.PROBLEM)
    filename = models.CharField(max_length=255)
    stored = models.CharField(max_length=255)                     # the file's name on disk
    width = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    caption = models.CharField(max_length=300, blank=True, default="")
    description = models.TextField(blank=True, default="")      # what the vision model read (for the report)
    uploaded_by = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "problems_picture"
        ordering = ["id"]


class ProblemCursor(models.Model):
    """How far the downtime trigger has read an asset's rows, and the downtime stretch it is in."""

    asset_id = models.CharField(max_length=512, primary_key=True)
    ts = models.DateTimeField()
    state = models.JSONField(default=dict, blank=True)            # {"stretch_start": iso | None, "ticket_id": int | None}

    class Meta:
        db_table = "problems_cursor"

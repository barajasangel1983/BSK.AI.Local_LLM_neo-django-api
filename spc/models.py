"""SPC tools and their control limits (the Studio's own database).

One tool monitors one signal of one asset. Everything about it is configuration: nothing
about a particular machine is in code. Specification: claude/P12_SPCimplementation.md, section 4.
"""

import uuid

from django.db import models
from django.utils import timezone

from . import engine


def default_states():
    return ["RUNNING"]


def default_rules():
    return list(engine.DEFAULT_RULES)


def default_event_actions():
    return {event_type: {"log": True, "agent": False} for event_type in (engine.WARNING, engine.CONTROL, engine.SPEC)}


class SpcTool(models.Model):
    METHODS = [(engine.IMR, "I-MR"), (engine.XBAR_R, "X-bar/R")]
    STRATEGIES = [("baseline", "From a baseline"), ("fixed", "Fixed")]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    asset_id = models.CharField(max_length=512)          # graph id, e.g. bsk:asset:EXTR01
    signal_id = models.CharField(max_length=512)         # graph id of the signal
    column = models.CharField(max_length=128)            # the historian column, from the signal's historian_ref
    name = models.CharField(max_length=200)
    unit = models.CharField(max_length=40, blank=True)
    enabled = models.BooleanField(default=True)          # False = paused

    states = models.JSONField(default=default_states)    # machine states in which the tool applies
    ignore_seconds = models.PositiveIntegerField(default=120)    # dropped at the start of every segment
    method = models.CharField(max_length=10, choices=METHODS, default=engine.IMR)
    subgroup_size = models.PositiveSmallIntegerField(default=3)  # X-bar/R only
    group_by = models.CharField(max_length=128, blank=True)      # a column whose value selects the limits

    lsl = models.FloatField(null=True, blank=True)       # specification: what the product must meet
    target = models.FloatField(null=True, blank=True)
    usl = models.FloatField(null=True, blank=True)

    limit_strategy = models.CharField(max_length=10, choices=STRATEGIES, default="baseline")
    sigma_multiplier = models.FloatField(default=3.0)
    baseline_start = models.DateTimeField(null=True, blank=True)
    baseline_end = models.DateTimeField(null=True, blank=True)

    rules = models.JSONField(default=default_rules)
    no_variation_count = models.PositiveSmallIntegerField(default=9)
    event_actions = models.JSONField(default=default_event_actions)
    window_hours = models.PositiveSmallIntegerField(default=8)

    version = models.PositiveIntegerField(default=1)     # raised by every saved change
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at"]
        constraints = [models.UniqueConstraint(fields=["asset_id", "signal_id"], name="spc_one_tool_per_signal")]

    def __str__(self) -> str:
        return f"{self.name} ({self.asset_id})"


class SpcLimits(models.Model):
    """The control limits of a tool for one value of its `group_by` column ("" without one)."""

    tool = models.ForeignKey(SpcTool, on_delete=models.CASCADE, related_name="limits")
    group = models.CharField(max_length=128, blank=True)
    source = models.CharField(max_length=10, choices=SpcTool.STRATEGIES)
    n_samples = models.PositiveIntegerField(default=0)
    mean = models.FloatField()
    sigma_within = models.FloatField()
    sigma_overall = models.FloatField(null=True)
    range_bar = models.FloatField()
    cl = models.FloatField()
    lcl = models.FloatField()
    ucl = models.FloatField()
    r_cl = models.FloatField()
    r_lcl = models.FloatField()
    r_ucl = models.FloatField()
    window_start = models.DateTimeField(null=True)
    window_end = models.DateTimeField(null=True)
    computed_at = models.DateTimeField(default=timezone.now)
    tool_version = models.PositiveIntegerField(default=1)

    class Meta:
        ordering = ["group"]
        constraints = [models.UniqueConstraint(fields=["tool", "group"], name="spc_one_limits_per_group")]


class SpcCursor(models.Model):
    """How far the evaluator has judged a tool's rows (the `ts` of the newest row judged)."""

    tool = models.OneToOneField(SpcTool, on_delete=models.CASCADE, related_name="cursor", primary_key=True)
    judged_until = models.DateTimeField()
    updated_at = models.DateTimeField(auto_now=True)

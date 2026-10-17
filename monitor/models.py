"""The monitoring agent's posts on Home, how far it has read each asset, and its switch (P15a)."""

from django.db import models


class MonitorPost(models.Model):
    """A post of the monitoring agent: an hourly summary, an alert, a state change or a data gap."""

    asset_id = models.CharField(max_length=512, db_index=True)
    kind = models.CharField(max_length=16)                      # summary | alert | state | data
    severity = models.CharField(max_length=16, default="info")  # info | warning
    title = models.CharField(max_length=200)
    text = models.TextField(blank=True, default="")
    data = models.JSONField(default=dict, blank=True)           # the numbers behind the post
    ts = models.DateTimeField()                                 # the time in the data the post is about
    model = models.CharField(max_length=120, blank=True, default="")   # empty = written without a model
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "monitor_post"
        ordering = ["-id"]


class MonitorCursor(models.Model):
    """How far the agent has read an asset's rows, and what it is tracking (states, signals out of range)."""

    asset_id = models.CharField(max_length=512, primary_key=True)
    ts = models.DateTimeField()
    state = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "monitor_cursor"


class MonitorSettings(models.Model):
    """One row: whether the agent writes posts."""

    agent_enabled = models.BooleanField(default=True)

    class Meta:
        db_table = "monitor_settings"

    @classmethod
    def get(cls) -> "MonitorSettings":
        return cls.objects.get_or_create(pk=1)[0]


class PlantCalendar(models.Model):
    """A plant's shifts, working days and time zone (P18a0); the chat, Home and reports read the same one.

    `working_days` are "mon" … "sun"; `shifts` are [{name, start, end}] with "HH:MM" times. A shift may cross
    midnight and belongs to the day it starts. Validation is in monitor.calendar.
    """

    plant_id = models.CharField(max_length=512, unique=True)   # the Plant node's id in the graph
    time_zone = models.CharField(max_length=64)                 # IANA, e.g. America/Chicago
    working_days = models.JSONField(default=list, blank=True)
    shifts = models.JSONField(default=list, blank=True)
    updated_by = models.CharField(max_length=200, blank=True, default="")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "monitor_plant_calendar"

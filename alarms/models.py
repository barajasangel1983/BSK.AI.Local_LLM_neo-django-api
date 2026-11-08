"""Alarm episodes (P31): every alarm the machine raised, from the historian's rows, kept by the Studio.

An episode is one alarm code from its first row to its last; the same code coming back within MERGE_MINUTES extends the
episode and counts a repeat (a 1-minute quality alarm that chatters is one row, not dozens). Built by the monitor's
minute pass from a cursor; a 30-day backfill with `manage.py alarms_update --rebuild --days 30`.
"""

from django.db import models


class AlarmEpisode(models.Model):
    asset_id = models.CharField(max_length=512, db_index=True)
    code = models.CharField(max_length=120, db_index=True)
    started_at = models.DateTimeField(db_index=True)
    ended_at = models.DateTimeField(db_index=True)          # the last row with the code + one period
    alarm_seconds = models.FloatField(default=0)           # time with the code on (the repeats' gaps not counted)
    repeats = models.PositiveIntegerField(default=1)       # how many times the code came on in this episode
    state_at_start = models.CharField(max_length=40, blank=True)

    class Meta:
        ordering = ["-started_at"]
        indexes = [models.Index(fields=["asset_id", "code", "ended_at"])]


class AlarmCode(models.Model):
    """A code's category, set in the Studio (defaults from the code's name when no row exists)."""
    CATEGORIES = [("equipment", "Equipment"), ("process", "Process"), ("quality", "Quality"), ("sensor", "Sensor")]
    asset_id = models.CharField(max_length=512)
    code = models.CharField(max_length=120)
    category = models.CharField(max_length=16, choices=CATEGORIES)
    updated_by = models.CharField(max_length=200, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["asset_id", "code"], name="alarm_code_once_per_asset")]


class AlarmCursor(models.Model):
    """How far an asset's historian rows have been read into episodes."""
    asset_id = models.CharField(max_length=512, unique=True)
    read_until = models.DateTimeField()

"""The record of every Start or Pause asked of the PLC from the Studio, done or refused (P13f)."""

from django.db import models


class PlcCommand(models.Model):
    OUTCOMES = [("done", "Done"), ("failed", "Failed"), ("refused", "Refused")]

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    command = models.CharField(max_length=16)                    # Start or Pause (anything else is refused, and recorded)
    requested_by = models.CharField(max_length=120, blank=True)  # the name the page sends; roles are not enforced by the backend
    outcome = models.CharField(max_length=10, choices=OUTCOMES)
    reason = models.CharField(max_length=255, blank=True)        # why refused or failed
    running_before = models.BooleanField(null=True)              # ExtruderSim.Running before and after
    running_after = models.BooleanField(null=True)
    period_before = models.BigIntegerField(null=True)            # Sample.PeriodCounter before and after
    period_after = models.BigIntegerField(null=True)
    seconds = models.FloatField(null=True)                       # from asking to the check of the result

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.created_at:%Y-%m-%d %H:%M:%S} {self.command} {self.outcome}"

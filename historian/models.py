from django.db import models


class PlcSignal(models.Model):
    """Simple PLC historian signal record.

    This lives in the dedicated Postgres historian database, separate from the
    main SQLite DB used for chat. It is intended for synthetic PLC data now
    and real PLC/Jetson feeds later.
    """

    tag_name = models.CharField(max_length=100)
    ts = models.DateTimeField()
    value = models.FloatField(null=True, blank=True)
    unit = models.CharField(max_length=32, blank=True)
    quality = models.CharField(max_length=32, blank=True)

    class Meta:
        db_table = "plc_signals"
        ordering = ["-ts"]

    def __str__(self) -> str:
        return f"{self.tag_name}@{self.ts.isoformat()}={self.value} {self.unit}"
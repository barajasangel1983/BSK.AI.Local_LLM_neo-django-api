"""Which historian a request reads.

"march": the simulated month (plc_1_historian.extruder_samples, one row a minute).
"live":  the PLC's rows (TimescaleDB historian_rows, one row every 20 s).

Both tables have the same 32 columns, so the callers only need the model.
"""

from django.conf import settings

from .models import ExtruderSample, LiveRow

SOURCES = ("march", "live")


def current() -> str:
    """The source the Studio serves (settings.HISTORIAN_SOURCE)."""
    source = getattr(settings, "HISTORIAN_SOURCE", "march")
    return source if source in SOURCES else "march"


def samples(source: str | None = None):
    """The model holding the historian rows of `source` (default: the one the Studio serves)."""
    return LiveRow if (source or current()) == "live" else ExtruderSample

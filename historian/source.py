"""Which historian a request reads.

"march": the simulated month (plc_1_historian.extruder_samples, one row a minute).
"live":  the PLC's rows (TimescaleDB historian_rows, one row every 20 s).

Both tables have the same 32 columns, so the callers only need the model.
"""

from django.conf import settings

from .models import ExtruderSample, LiveRow

SOURCES = ("march", "live")
# What a client is told its values come from, and how far apart the rows are.
LABEL = {"march": "plc_1_historian", "live": "historian_live"}
PERIOD_SECONDS = {"march": 60, "live": 20}
STALE_SECONDS = 90          # live only: with no row for this long, the newest row is not the present state


def current() -> str:
    """The source the Studio serves (settings.HISTORIAN_SOURCE)."""
    source = getattr(settings, "HISTORIAN_SOURCE", "march")
    return source if source in SOURCES else "march"


def samples(source: str | None = None):
    """The model holding the historian rows of `source` (default: the one the Studio serves)."""
    return LiveRow if (source or current()) == "live" else ExtruderSample


def label(source: str | None = None) -> str:
    return LABEL[source or current()]


def period_seconds(source: str | None = None) -> int:
    """Seconds between two rows of `source`."""
    return PERIOD_SECONDS[source or current()]

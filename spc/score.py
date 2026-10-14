"""The score of SPC against the injected anomalies (P12d): found, missed, delay and false alarms.

Pure functions on plain records: the anomaly rows and the SPC events are given to them, nothing is
read here. The command `spc_score` is the only code that reads the injected-anomaly record (FR-26).
Specification: claude/P12_SPCimplementation.md (Studio web repository), "P12d" in section 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median

from . import engine

MARGIN = timedelta(minutes=5)       # an event this long after an anomaly's last row still finds it
TYPES = {1: "Step", 2: "Drift", 3: "Spike", 4: "Stuck value", 5: "More noise", 6: "Broken relation"}


@dataclass
class Anomaly:
    event_id: int
    type: str
    signal: str                     # the historian column
    size: float
    start: datetime                 # its first row
    end: datetime                   # its last row
    rows: int


@dataclass
class Event:
    """An SPC event as the score needs it."""
    tool: str                       # the tool's name
    column: str
    type: str                       # engine.CONTROL, engine.SPEC or engine.WARNING
    rule: str
    started_at: datetime
    last_at: datetime


@dataclass
class Finding:
    anomaly: Anomaly
    monitored: bool
    event: Event | None = None      # the first event of a tool on the anomaly's signal that found it

    @property
    def delay_minutes(self) -> float | None:
        return None if self.event is None else (self.event.started_at - self.anomaly.start).total_seconds() / 60


@dataclass
class ToolFalseAlarms:
    tool: str
    control: int = 0
    spec: int = 0
    warnings: int = 0               # every statistical warning, on an anomaly or not
    false_warnings: int = 0         # statistical warnings that touch no anomaly


@dataclass
class Score:
    findings: list[Finding]
    false_alarms: list[ToolFalseAlarms] = field(default_factory=list)

    @property
    def monitored(self) -> list[Finding]:
        return [f for f in self.findings if f.monitored]

    @property
    def found(self) -> list[Finding]:
        return [f for f in self.findings if f.event is not None]

    @property
    def median_delay(self) -> float | None:
        delays = [f.delay_minutes for f in self.found]
        return median(delays) if delays else None


def anomalies(rows: list[dict]) -> list[Anomaly]:
    """One anomaly per event_id from the record's rows ({ts, event_id, event_type, event_signal, event_size}), oldest first."""
    by_id: dict[int, list[dict]] = {}
    for row in rows:
        by_id.setdefault(row["event_id"], []).append(row)
    out = []
    for event_id, own in by_id.items():
        times = [r["ts"] for r in own]
        first = own[0]
        out.append(Anomaly(event_id=event_id, type=TYPES.get(first["event_type"], str(first["event_type"])),
                           signal=first["event_signal"], size=first["event_size"], start=min(times), end=max(times),
                           rows=len(own)))
    return sorted(out, key=lambda a: a.start)


def _touches(event: Event, anomaly: Anomaly) -> bool:
    return event.started_at <= anomaly.end + MARGIN and event.last_at >= anomaly.start - MARGIN


def score(found_anomalies: list[Anomaly], events: list[Event], tools: dict[str, str]) -> Score:
    """Score the events of the tools ({tool name: column}) against the anomalies.

    Found: an event of a tool on the anomaly's signal starts between its first row and MARGIN after its
    last; the first such event counts. An anomaly on a signal with no tool is not monitored, not missed.
    False alarm: a control or specification violation that touches no anomaly on its signal (MARGIN
    either side). Statistical warnings are counted apart."""
    columns = set(tools.values())
    findings = []
    for anomaly in found_anomalies:
        hits = [e for e in events if e.column == anomaly.signal and anomaly.start <= e.started_at <= anomaly.end + MARGIN]
        findings.append(Finding(anomaly=anomaly, monitored=anomaly.signal in columns,
                                event=min(hits, key=lambda e: e.started_at) if hits else None))
    alarms = {name: ToolFalseAlarms(tool=name) for name in tools}
    for event in events:
        own = alarms[event.tool]
        false = not any(_touches(event, a) for a in found_anomalies if a.signal == event.column)
        if event.type == engine.WARNING:
            own.warnings += 1
            own.false_warnings += false
        elif false:
            if event.type == engine.SPEC:
                own.spec += 1
            else:
                own.control += 1
    return Score(findings=findings, false_alarms=list(alarms.values()))

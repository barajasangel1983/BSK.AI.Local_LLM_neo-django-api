"""The triggers (P29a): open a ticket after a long downtime or a long SPC violation. Run by the monitor's minute pass.

- **Downtime:** the machine stays in a downtime state (settings: STOPPED, ALARMED) for longer than
  `downtime_hours`, counting only working time when the plant has a calendar.
- **SPC:** an SPC event (one continuing violation) lasts longer than `spc_control_hours` (control limits) or
  `spc_spec_hours` (specification). Every event that crosses its threshold opens its own ticket.

Only problems whose threshold is crossed after the triggers were switched on (`enabled_at`) open a ticket. Like the SPC
posts, the triggers follow the live historian only (a replayed month is not judged again). When a problem ends, its
ticket's window closes and the context is read again.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from .models import ProblemCursor, ProblemSettings, Ticket
from . import service

logger = logging.getLogger("chat")

LOOKBACK_HOURS = 24           # the first pass reads this far back, to know when a downtime that is going on began
SPC_TYPES = {"control_violation": (Ticket.Trigger.SPC_CONTROL, "spc_control_hours", "outside its control limits"),
             "spec_violation": (Ticket.Trigger.SPC_SPEC, "spc_spec_hours", "outside its specification limits")}


def _when(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def working_minutes(cal, start: datetime, end: datetime) -> float:
    """Minutes of (start, end] inside scheduled shifts; every minute counts when there is no calendar."""
    if end <= start:
        return 0.0
    if cal is None:
        return (end - start).total_seconds() / 60
    total = 0.0
    for o in cal.occurrences(start, end):
        if o.scheduled:
            overlap = (min(end, o.end) - max(start, o.start)).total_seconds()
            total += max(0.0, overlap) / 60
    return total


def _hours(hours: float) -> str:
    return f"{hours:g} h"


def tick(asset_id: str, real: datetime | None = None) -> list[Ticket]:
    """One pass for one asset. Returns the tickets opened."""
    settings = ProblemSettings.get()
    if not settings.enabled or settings.enabled_at is None:
        return []
    from monitor import agent
    clk = agent.clock(asset_id, real)
    if clk is None or not clk["live"]:
        return []
    opened = _spc(asset_id, settings, clk["now"])
    opened += _downtime(asset_id, settings, clk["now"])
    for ticket in opened:             # P29b: what to check first (the ticket is already open when this fails)
        try:
            from . import llm
            llm.recommend(ticket)
        except Exception as exc:
            logger.warning("problems: no recommendations for %s (%s)", ticket.number, exc)
    return opened


# --- SPC ------------------------------------------------------------------------------------------------

def _spc(asset_id: str, settings: ProblemSettings, now: datetime) -> list[Ticket]:
    from spc import service as spc_service
    from spc.models import SpcEvent
    opened = []
    events = (SpcEvent.objects.filter(tool__asset_id=asset_id, tool__enabled=True, type__in=list(SPC_TYPES),
                                      last_at__gte=settings.enabled_at).select_related("tool"))
    for event in events:
        trigger, field, words = SPC_TYPES[event.type]
        hours = getattr(settings, field)
        crossed = event.started_at + timedelta(hours=hours)
        if event.last_at < crossed or crossed < settings.enabled_at:
            continue
        unit = f" {event.tool.unit}" if event.tool.unit else ""
        title = f"{event.tool.name} {words} for more than {_hours(hours)} (since {event.started_at:%H:%M} UTC)"
        if event.type == "control_violation" and _spec_overlaps(event):
            continue                      # Angel, 2026-11-08: a specification violation replaces the control one
        if event.type == "spec_violation" and _promote(event, title):
            continue
        ticket = service.open_ticket(asset_id, trigger, title, event.started_at, None, source_key=f"spc:{event.id}")
        if ticket:
            ticket.context["trigger_event"] = {"tool": event.tool.name, "type": event.type, "rule": event.rule,
                                               "value": event.value, "extreme": event.extreme, "unit": unit.strip(),
                                               "limits": event.limits}
            ticket.save(update_fields=["context"])
            opened.append(ticket)

    # A violation that ended: its ticket's window closes.
    for ticket in Ticket.objects.filter(asset_id=asset_id, window_end__isnull=True,
                                        trigger__in=[Ticket.Trigger.SPC_CONTROL, Ticket.Trigger.SPC_SPEC]):
        event = SpcEvent.objects.filter(pk=ticket.source_key.removeprefix("spc:")).select_related("tool").first()
        if event is None:
            continue
        active_since = spc_service._judged_until(event.tool) - timedelta(seconds=_active_seconds())
        if event.last_at < active_since:
            service.close_window(ticket, event.last_at)
    return opened


def _spec_overlaps(event) -> bool:
    """A specification violation on the same tool at the same time (the control one is then not ticketed)."""
    from spc.models import SpcEvent
    return SpcEvent.objects.filter(tool_id=event.tool_id, type="spec_violation", started_at__lte=event.last_at,
                                   last_at__gte=event.started_at).exists()


def _promote(event, title: str) -> bool:
    """A control-limit ticket already open for the same tool and time becomes this specification ticket."""
    from spc.models import SpcEvent
    if Ticket.objects.filter(source_key=f"spc:{event.id}").exists():
        return True
    controls = SpcEvent.objects.filter(tool_id=event.tool_id, type="control_violation", started_at__lte=event.last_at,
                                       last_at__gte=event.started_at).values_list("id", flat=True)
    ticket = (Ticket.objects.filter(source_key__in=[f"spc:{i}" for i in controls], trigger=Ticket.Trigger.SPC_CONTROL)
              .exclude(status__in=[Ticket.Status.APPROVED, Ticket.Status.COMPLETED, Ticket.Status.CLOSED]).first())
    if ticket is None:
        return False
    ticket.trigger, ticket.title, ticket.source_key = Ticket.Trigger.SPC_SPEC, title[:300], f"spc:{event.id}"
    ticket.window_start = min(ticket.window_start, event.started_at)
    ticket.window_end = None
    ticket.context = {**ticket.context, "trigger_event": {"tool": event.tool.name, "type": event.type, "rule": event.rule,
                                                          "value": event.value, "extreme": event.extreme,
                                                          "unit": event.tool.unit, "limits": event.limits}}
    ticket.context["analysis"] = service.summary(ticket.asset_id, ticket.window_start, None, ticket.source_key)
    service._log(ticket, "became a specification ticket", {"username": "system", "name": "Problem-solving trigger"})
    ticket.save()
    return True


def _active_seconds() -> int:
    from django.conf import settings as django_settings
    return int(django_settings.SPC_EVENT_ACTIVE_SECONDS)


# --- downtime -----------------------------------------------------------------------------------------

def _downtime(asset_id: str, settings: ProblemSettings, now: datetime) -> list[Ticket]:
    from monitor import agent, plant_calendar
    cursor = ProblemCursor.objects.filter(asset_id=asset_id).first()
    state = dict(cursor.state or {}) if cursor else {}
    start = cursor.ts if cursor else max(settings.enabled_at, now) - timedelta(hours=LOOKBACK_HOURS)
    start = max(start, now - timedelta(hours=LOOKBACK_HOURS))
    if now - start < timedelta(seconds=30):
        return []
    rows = agent._rows(agent.history(asset_id, start, now))
    if not rows:
        return []
    states = set(settings.states)
    threshold_min = settings.downtime_hours * 60
    cal = plant_calendar.for_asset(asset_id)
    stretch = _when(state["stretch_start"]) if state.get("stretch_start") else None
    ticket_id = state.get("ticket_id")
    opened = []
    for row in rows:
        down = (row.get("machine_state") or "").upper() in states
        if down:
            if stretch is None:
                stretch, ticket_id = row["ts"], None
            if (ticket_id is None and row["ts"] >= settings.enabled_at
                    and (row["ts"] - stretch).total_seconds() / 60 >= threshold_min
                    and working_minutes(cal, stretch, row["ts"]) >= threshold_min):
                title = (f"Downtime longer than {_hours(settings.downtime_hours)} "
                         f"({row.get('machine_state')} since {stretch:%H:%M} UTC)")
                ticket = service.open_ticket(asset_id, Ticket.Trigger.DOWNTIME, title, stretch, None,
                                             source_key=f"downtime:{asset_id}:{stretch.isoformat()}")
                existing = ticket or Ticket.objects.filter(source_key=f"downtime:{asset_id}:{stretch.isoformat()}").first()
                ticket_id = existing.id if existing else -1
                if ticket:
                    opened.append(ticket)
        else:
            if stretch is not None and ticket_id and ticket_id > 0:
                ticket = Ticket.objects.filter(pk=ticket_id).first()
                if ticket:
                    service.close_window(ticket, row["ts"])
            stretch, ticket_id = None, None
    ProblemCursor.objects.update_or_create(asset_id=asset_id, defaults={
        "ts": rows[-1]["ts"], "state": {"stretch_start": stretch.isoformat() if stretch else None, "ticket_id": ticket_id}})
    return opened

"""Problem-solving tickets (P29): opening, the 5-Why, submit / approve / return / close, pictures, settings.

Every action names who does it: `user` is {username, name, role, source}, passed by the client (BSKLAB EDGE through
the MCP token, or the Studio). The rules are checked here, whichever client calls:

- everybody may read;
- only engineers and admins (`WRITERS`) fill, submit, approve, return, close and open tickets;
- the approver is not the author (the engineer who submitted).

Status:  open → draft (saved) → submitted → approved → completed
                                          ↘ returned (comment) → draft …
         open / draft / returned → closed (with a reason; never published)

Approved: the analysis and the action plan are accepted; the report goes into the library. The corrective actions are
then followed (done, validated by an engineer who is not the author) and "after" pictures added. When every action is
done and validated the ticket is completed and the library document is replaced with the final report.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from django.conf import settings
from django.db import IntegrityError, transaction

from context_service.service import ContextError

from . import context
from .models import ProblemSettings, Ticket, TicketPicture

logger = logging.getLogger("chat")

WRITERS = ("engineer", "admin")
MAX_WHYS = 5
TEXT_LIMIT = 8000
EDITABLE = ("notes", "problem", "root_cause", "containment", "final_report")
MAX_ACTIONS = 20
ACTION_STATUSES = ("in_progress", "done")
OPEN_STATES = (Ticket.Status.OPEN, Ticket.Status.DRAFT, Ticket.Status.RETURNED)
LIST_LIMIT = 100
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class TicketError(ContextError):
    """An action the rules don't allow, or a bad request: the text is shown to the user."""


def now() -> datetime:
    return datetime.now(timezone.utc)


# --- who ------------------------------------------------------------------------------------------

def _who(user: dict | None) -> dict:
    user = user or {}
    username = str(user.get("username") or "").strip()
    if not username:
        raise TicketError("who is acting is not known (username is required)")
    return {"username": username[:64], "name": str(user.get("name") or username)[:120],
            "role": str(user.get("role") or "")[:32], "source": str(user.get("source") or "")[:32]}


def _writer(user: dict | None) -> dict:
    who = _who(user)
    if who["role"] not in WRITERS:
        raise TicketError("only engineers and admins can work on problem-solving tickets")
    return who


def _log(ticket: Ticket, action: str, who: dict | None, comment: str = "") -> None:
    ticket.history = [*(ticket.history or []), {"at": now().isoformat(), "action": action,
                                                "by": {k: (who or {}).get(k, "") for k in ("username", "name", "role")},
                                                "comment": comment[:1000]}]


# --- reading ----------------------------------------------------------------------------------------

def picture_json(p: TicketPicture) -> dict:
    return {"id": p.id, "phase": p.phase, "filename": p.filename, "width": p.width, "height": p.height, "caption": p.caption,
            "description": p.description, "uploaded_by": p.uploaded_by, "created_at": p.created_at.isoformat()}


def action_state(a: dict, today: date | None = None) -> str:
    """done (green), delayed (red: the due date passed and it isn't done), in_progress (yellow)."""
    if a.get("status") == "done":
        return "done"
    today = today or now().date()
    try:
        return "delayed" if a.get("due_date") and date.fromisoformat(a["due_date"]) < today else "in_progress"
    except ValueError:
        return "in_progress"


def actions_json(t: Ticket) -> list[dict]:
    return [{**a, "state": action_state(a)} for a in t.actions or []]


def actions_text(actions: list[dict]) -> str:
    return "\n".join(f"- {a['text']}" + (f" (responsible: {a['responsible']}" if a.get("responsible") else "")
                     + (f", due {a['due_date']}" if a.get("due_date") else "") + (")" if a.get("responsible") else "")
                     for a in actions)


def ticket_json(t: Ticket, full: bool = True) -> dict:
    out = {"id": t.id, "number": t.number, "asset_id": t.asset_id, "asset_name": t.asset_name, "trigger": t.trigger,
           "trigger_label": t.get_trigger_display(), "title": t.title, "status": t.status,
           "status_label": t.get_status_display(), "window_start": t.window_start.isoformat(),
           "window_end": t.window_end.isoformat() if t.window_end else None, "author": t.author,
           "approver": t.approver, "opened_by": t.opened_by, "created_at": t.created_at.isoformat(),
           "updated_at": t.updated_at.isoformat(), "submitted_at": t.submitted_at.isoformat() if t.submitted_at else None,
           "approved_at": t.approved_at.isoformat() if t.approved_at else None,
           "check_status": (t.knowledge_check or {}).get("status", "none"), "document_id": t.document_id,
           "completed_at": t.completed_at.isoformat() if t.completed_at else None,
           "actions_summary": {s: sum(1 for a in t.actions or [] if action_state(a) == s) for s in ("done", "in_progress", "delayed")}
                              | {"validated": sum(1 for a in t.actions or [] if a.get("validated_by")), "total": len(t.actions or [])}}
    if full:
        out.update({"context": t.context, "recommendations": t.recommendations, "notes": t.notes,
                    "problem": t.problem, "whys": t.whys, "root_cause": t.root_cause, "containment": t.containment,
                    "corrective_actions": t.corrective_actions, "actions": actions_json(t), "final_report": t.final_report, "check": t.knowledge_check,
                    "close_reason": t.close_reason, "return_comment": t.return_comment, "history": t.history,
                    "publish_error": t.publish_error, "pictures": [picture_json(p) for p in t.pictures.all()]})
    return out


def get(ticket_id) -> Ticket:
    try:
        return Ticket.objects.get(pk=int(ticket_id))
    except (Ticket.DoesNotExist, TypeError, ValueError):
        raise TicketError(f"no ticket {ticket_id}") from None


def list_tickets(asset_id: str = "", status: str = "", limit: int = 50, before_id: int = 0) -> dict:
    """Newest first. `status`: one status, "active" (open, draft, submitted, returned) or empty for all."""
    found = Ticket.objects.all()
    if asset_id:
        found = found.filter(asset_id=asset_id)
    if status == "active":
        found = found.filter(status__in=[*OPEN_STATES, Ticket.Status.SUBMITTED, Ticket.Status.APPROVED])
    elif status:
        found = found.filter(status=status)
    if before_id:
        found = found.filter(pk__lt=before_id)
    limit = max(1, min(int(limit or 50), LIST_LIMIT))
    page = list(found.order_by("-id")[:limit + 1])
    counts = {s: Ticket.objects.filter(status=s, **({"asset_id": asset_id} if asset_id else {})).count()
              for s, _ in Ticket.Status.choices}
    return {"tickets": [ticket_json(t, full=False) for t in page[:limit]], "more": len(page) > limit,
            "counts": counts}


# --- opening -------------------------------------------------------------------------------------------

def asset_name(asset_id: str) -> str:
    try:
        from context_service import service
        return next((a["name"] for a in service.list_assets() if a["id"] == asset_id), "")
    except Exception:
        return ""


def open_ticket(asset_id: str, trigger: str, title: str, start: datetime, end: datetime | None = None,
                source_key: str = "", opened_by: dict | None = None, with_context: bool = True) -> Ticket | None:
    """A new ticket, or None when one was already opened for `source_key`."""
    if source_key and Ticket.objects.filter(source_key=source_key).exists():
        return None
    ticket = Ticket(asset_id=asset_id, asset_name=asset_name(asset_id), trigger=trigger, source_key=source_key,
                    title=title[:300], window_start=start, window_end=end, opened_by=opened_by or {})
    if with_context:
        ticket.context = context.build(asset_id, start, end)
        ticket.context["analysis"] = summary(asset_id, start, end, source_key)
    _log(ticket, "opened", opened_by or {"username": "system", "name": "Problem-solving trigger"})
    try:
        with transaction.atomic():
            ticket.save()
    except IntegrityError:            # the same source opened it a moment ago
        return None
    _post(ticket)
    logger.info("problems: %s opened (%s) for %s", ticket.number, trigger, asset_id)
    return ticket


def _post(ticket: Ticket) -> None:
    """A post on Home (the Studio's and, through P24-4, EDGE's)."""
    try:
        from monitor.models import MonitorPost
        MonitorPost.objects.create(
            asset_id=ticket.asset_id, kind="ticket", severity="warning", title=f"Problem solving {ticket.number}: {ticket.title}"[:200],
            text=(f"A problem-solving ticket was opened: {ticket.title}. A Manufacturing Engineer fills the 5-Why on "
                  f"EDGE (Problem solving tab)."), ts=ticket.window_start,
            data={"ticket_id": ticket.id, "number": ticket.number, "trigger": ticket.trigger})
    except Exception:
        logger.exception("problems: the Home post could not be written")


def open_manual(asset_id: str, title: str, start: datetime, end: datetime | None, user: dict) -> Ticket:
    who = _writer(user)
    title = (title or "").strip()
    if not asset_id or not title:
        raise TicketError("a machine and a short title are required")
    if end and end <= start:
        raise TicketError("the end of the window must be after its start")
    if start > now():
        raise TicketError("the window can't start in the future")
    ticket = open_ticket(asset_id, Ticket.Trigger.MANUAL, title, start, end, opened_by=who)
    ticket.recommendations = {"status": "queued"}         # written by the next minute pass (a model call)
    ticket.save(update_fields=["recommendations", "updated_at"])
    return ticket


def summary(asset_id: str, start, end, source_key: str = "") -> dict:
    """The data summary (P29e); never blocks a ticket."""
    from . import analysis
    try:
        return analysis.build(asset_id, start, end, analysis.trigger_for(source_key or ""))
    except Exception as exc:
        logger.exception("problems: the data summary failed for %s", asset_id)
        return {"version": analysis.VERSION, "error": f"The data summary could not be made: {type(exc).__name__}"}


def refresh_context(ticket: Ticket) -> Ticket:
    """Read the window again (when it closes, or by hand). Keeps who was signed in (from EDGE) and the trigger's event."""
    kept = {k: v for k, v in (ticket.context or {}).items() if k in ("edge_users", "edge_users_at", "trigger_event")}
    ticket.context = {**context.build(ticket.asset_id, ticket.window_start, ticket.window_end), **kept,
                      "analysis": summary(ticket.asset_id, ticket.window_start, ticket.window_end, ticket.source_key)}
    ticket.save(update_fields=["context", "updated_at"])
    return ticket


def close_window(ticket: Ticket, end: datetime) -> None:
    if ticket.window_end is None:
        ticket.window_end = end
        ticket.save(update_fields=["window_end", "updated_at"])
        refresh_context(ticket)


def add_edge_users(ticket_id, users: list[dict], edge: str = "") -> Ticket:
    """Who was signed in on an EDGE during the window (EDGE knows its sign-ins; the Studio doesn't)."""
    ticket = get(ticket_id)
    known = {(u.get("edge", ""), u.get("username"), u.get("signed_in_at")) for u in ticket.context.get("edge_users", [])}
    merged = list(ticket.context.get("edge_users", []))
    for u in users or []:
        row = {"edge": edge[:64], "username": str(u.get("username", ""))[:64], "name": str(u.get("name", ""))[:120],
               "role": str(u.get("role", ""))[:32], "signed_in_at": str(u.get("signed_in_at", ""))[:40],
               "signed_out_at": str(u.get("signed_out_at", "") or "")[:40]}
        if row["username"] and (row["edge"], row["username"], row["signed_in_at"]) not in known:
            merged.append(row)
    ticket.context = {**ticket.context, "edge_users": merged, "edge_users_at": now().isoformat()}
    ticket.save(update_fields=["context", "updated_at"])
    return ticket


# --- the 5-Why -------------------------------------------------------------------------------------------

def save(ticket_id, fields: dict, user: dict) -> Ticket:
    """Save the engineer's work (a draft). Only the fields given change."""
    who = _writer(user)
    ticket = get(ticket_id)
    if ticket.status not in OPEN_STATES:
        raise TicketError(f"{ticket.number} is {ticket.get_status_display().lower()}: it can't be changed")
    for name in EDITABLE:
        if name in fields and fields[name] is not None:
            setattr(ticket, name, str(fields[name])[:TEXT_LIMIT])
    if fields.get("actions") is not None:
        ticket.actions = _clean_actions(fields["actions"], ticket.actions or [])
        ticket.corrective_actions = actions_text(ticket.actions)
    elif fields.get("corrective_actions") is not None and not ticket.actions:      # an old client: one action per line
        lines = [l.strip(" -•\t") for l in str(fields["corrective_actions"]).splitlines() if l.strip(" -•\t")]
        ticket.actions = _clean_actions([{"text": l} for l in lines], [])
        ticket.corrective_actions = actions_text(ticket.actions)
    if fields.get("whys") is not None:
        whys = [str(w)[:2000] for w in list(fields["whys"])[:MAX_WHYS]]
        while whys and not whys[-1].strip():
            whys.pop()
        ticket.whys = whys
    if ticket.status == Ticket.Status.OPEN:
        _log(ticket, "started", who)
    ticket.status = Ticket.Status.DRAFT if ticket.status != Ticket.Status.RETURNED else ticket.status
    ticket.save()
    return ticket


def _clean_actions(rows, old: list[dict]) -> list[dict]:
    """The action rows as the engineer left them; ids are kept so a row's status and validation stay with it."""
    known = {a["id"]: a for a in old}
    out = []
    for row in list(rows or [])[:MAX_ACTIONS]:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()[:1000]
        if not text:
            continue
        due = str(row.get("due_date") or "").strip()[:10]
        if due and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", due):
            raise TicketError(f"the due date of “{text[:40]}” must be a date (YYYY-MM-DD)")
        status = row.get("status") if row.get("status") in ACTION_STATUSES else "in_progress"
        before = known.get(str(row.get("id") or ""), {})
        action = {"id": before.get("id") or uuid.uuid4().hex[:8], "text": text,
                  "responsible": str(row.get("responsible") or "").strip()[:120], "due_date": due, "status": status,
                  "done_at": before.get("done_at") if status == "done" else None,
                  "validated_by": None, "validated_at": None}
        if status == "done" and not action["done_at"]:
            action["done_at"] = now().isoformat()
        out.append(action)
    return out


def _missing(ticket: Ticket) -> list[str]:
    missing = []
    if not ticket.problem.strip():
        missing.append("the problem statement")
    if not any(w.strip() for w in ticket.whys or []):
        missing.append("at least one why")
    if not ticket.root_cause.strip():
        missing.append("the root cause")
    if not ticket.actions:
        missing.append("at least one corrective action")
    elif any(not a.get("responsible") or not a.get("due_date") for a in ticket.actions):
        missing.append("a responsible and a due date for every corrective action")
    if not ticket.final_report.strip():
        missing.append("the final report (write it, or press “Clean up and write the final report”)")
    return missing


def submit(ticket_id, user: dict) -> Ticket:
    who = _writer(user)
    ticket = get(ticket_id)
    if ticket.status not in OPEN_STATES:
        raise TicketError(f"{ticket.number} is {ticket.get_status_display().lower()}: it can't be submitted")
    missing = _missing(ticket)
    if missing:
        raise TicketError("before submitting, fill in " + ", ".join(missing))
    ticket.status, ticket.author, ticket.submitted_at = Ticket.Status.SUBMITTED, who, now()
    ticket.return_comment = ""
    ticket.knowledge_check = {"status": "queued", "queued_at": now().isoformat()}       # P29c: the knowledge check
    _log(ticket, "submitted", who)
    ticket.save()
    return ticket


def approve(ticket_id, user: dict, comment: str = "") -> Ticket:
    who = _writer(user)
    ticket = get(ticket_id)
    if ticket.status != Ticket.Status.SUBMITTED:
        raise TicketError(f"{ticket.number} is not waiting for approval")
    if who["username"] == (ticket.author or {}).get("username"):
        raise TicketError("a second engineer approves: the author can't approve their own report")
    ticket.status, ticket.approver, ticket.approved_at = Ticket.Status.APPROVED, who, now()
    _log(ticket, "approved", who, comment)
    ticket.save()
    from . import publish
    publish.publish(ticket)            # P29d: the report goes into the library (errors are kept on the ticket)
    return ticket


def return_ticket(ticket_id, user: dict, comment: str) -> Ticket:
    who = _writer(user)
    ticket = get(ticket_id)
    if ticket.status != Ticket.Status.SUBMITTED:
        raise TicketError(f"{ticket.number} is not waiting for approval")
    if who["username"] == (ticket.author or {}).get("username"):
        raise TicketError("a second engineer reviews: the author can't return their own report")
    if not (comment or "").strip():
        raise TicketError("say what needs to change (a comment is required)")
    ticket.status, ticket.return_comment = Ticket.Status.RETURNED, comment.strip()[:2000]
    _log(ticket, "returned", who, comment)
    ticket.save()
    return ticket


def close(ticket_id, user: dict, reason: str) -> Ticket:
    """Closed without a 5-Why (e.g. a planned stop the calendar didn't know about). Never published."""
    who = _writer(user)
    ticket = get(ticket_id)
    if ticket.status not in OPEN_STATES:
        raise TicketError(f"{ticket.number} is {ticket.get_status_display().lower()}: it can't be closed")
    if not (reason or "").strip():
        raise TicketError("a reason is required to close a ticket without analysis")
    ticket.status, ticket.close_reason = Ticket.Status.CLOSED, reason.strip()[:2000]
    _log(ticket, "closed", who, reason)
    ticket.save()
    return ticket


# --- following the actions (after approval) ---------------------------------------------------------------

def _action(ticket: Ticket, action_id: str) -> dict:
    found = next((a for a in ticket.actions or [] if a.get("id") == action_id), None)
    if found is None:
        raise TicketError(f"no action {action_id} on {ticket.number}")
    return found


def set_action_status(ticket_id, action_id: str, status: str, user: dict) -> Ticket:
    """Done or in progress. After approval this is how the plan is followed; a validated action can't change."""
    who = _writer(user)
    ticket = get(ticket_id)
    if ticket.status not in (*OPEN_STATES, Ticket.Status.APPROVED):
        raise TicketError(f"{ticket.number} is {ticket.get_status_display().lower()}: its actions can't change")
    if status not in ACTION_STATUSES:
        raise TicketError("the status is done or in_progress")
    action = _action(ticket, action_id)
    if action.get("validated_by"):
        raise TicketError("this action is already validated")
    action["status"] = status
    action["done_at"] = now().isoformat() if status == "done" else None
    ticket.actions = [*ticket.actions]
    _log(ticket, f"action {'done' if status == 'done' else 'in progress'}", who, action["text"][:200])
    ticket.save()
    return ticket


def validate_action(ticket_id, action_id: str, user: dict) -> Ticket:
    """The second engineer confirms a done action. When every action is validated the ticket is completed."""
    who = _writer(user)
    ticket = get(ticket_id)
    if ticket.status != Ticket.Status.APPROVED:
        raise TicketError("actions are validated after the report is approved")
    if who["username"] == (ticket.author or {}).get("username"):
        raise TicketError("a second engineer validates: the author can't validate their own actions")
    action = _action(ticket, action_id)
    if action.get("status") != "done":
        raise TicketError("only a done action can be validated")
    action["validated_by"] = {"username": who["username"], "name": who["name"]}
    action["validated_at"] = now().isoformat()
    ticket.actions = [*ticket.actions]
    _log(ticket, "action validated", who, action["text"][:200])
    if all(a.get("validated_by") for a in ticket.actions):
        ticket.status, ticket.completed_at = Ticket.Status.COMPLETED, now()
        _log(ticket, "completed", who)
    ticket.save()
    if ticket.status == Ticket.Status.COMPLETED:
        from . import publish
        publish.publish(ticket)         # the library document is replaced with the final report
    return ticket


# --- pictures ---------------------------------------------------------------------------------------------

def _files_dir(ticket: Ticket) -> Path:
    return Path(settings.PROBLEM_FILES_DIR) / str(ticket.pk)


def _picture_allowed(ticket: Ticket, phase: str) -> None:
    if phase not in TicketPicture.Phase.values:
        raise TicketError("a picture is of the problem or after the fix")
    if ticket.status in OPEN_STATES:
        return
    if ticket.status == Ticket.Status.APPROVED and phase == TicketPicture.Phase.AFTER:
        return
    raise TicketError(f"{ticket.number} is {ticket.get_status_display().lower()}: "
                      + ("only pictures after the fix can be added now" if ticket.status == Ticket.Status.APPROVED
                         else "pictures can't be changed"))


def add_picture(ticket_id, data: bytes, filename: str, user: dict, caption: str = "",
                phase: str = TicketPicture.Phase.PROBLEM) -> TicketPicture:
    from gpu import images
    who = _writer(user)
    ticket = get(ticket_id)
    _picture_allowed(ticket, phase)
    if not data or len(data) > settings.PROBLEM_PICTURE_MAX_BYTES:
        raise TicketError(f"a picture must be at most {settings.PROBLEM_PICTURE_MAX_BYTES // (1024 * 1024)} MB")
    try:
        jpeg, width, height = images.to_jpeg(data, 2048)
    except Exception as exc:
        raise TicketError(f"the picture could not be read: {exc}") from None
    folder = _files_dir(ticket)
    folder.mkdir(parents=True, exist_ok=True)
    stored = f"{uuid.uuid4().hex}.jpg"
    (folder / stored).write_bytes(jpeg)
    return TicketPicture.objects.create(ticket=ticket, phase=phase, filename=Path(filename or "picture.jpg").name[:255], stored=stored,
                                        width=width, height=height, caption=(caption or "")[:300], uploaded_by=who)


def picture_bytes(ticket_id, picture_id) -> bytes:
    ticket = get(ticket_id)
    pic = ticket.pictures.filter(pk=picture_id).first()
    if pic is None:
        raise TicketError(f"no picture {picture_id} on {ticket.number}")
    return (_files_dir(ticket) / pic.stored).read_bytes()


def remove_picture(ticket_id, picture_id, user: dict) -> None:
    _writer(user)
    ticket = get(ticket_id)
    pic = ticket.pictures.filter(pk=picture_id).first()
    if pic is None:
        raise TicketError(f"no picture {picture_id} on {ticket.number}")
    _picture_allowed(ticket, pic.phase)
    (_files_dir(ticket) / pic.stored).unlink(missing_ok=True)
    pic.delete()


# --- settings ---------------------------------------------------------------------------------------------

def settings_json(s: ProblemSettings | None = None) -> dict:
    s = s or ProblemSettings.get()
    return {"enabled": s.enabled, "enabled_at": s.enabled_at.isoformat() if s.enabled_at else None,
            "downtime_hours": s.downtime_hours, "downtime_states": s.states, "spc_control_hours": s.spc_control_hours,
            "spc_spec_hours": s.spc_spec_hours, "check_mode": s.check_mode, "check_modes": ProblemSettings.CheckMode.choices,
            "check_day": s.check_day, "check_time": s.check_time, "publish_folder": s.publish_folder,
            "updated_by": s.updated_by, "updated_at": s.updated_at.isoformat() if s.updated_at else None}


def update_settings(data: dict, by: str = "") -> dict:
    import re
    s = ProblemSettings.get()
    errors = {}
    for name in ("downtime_hours", "spc_control_hours", "spc_spec_hours"):
        if name in data:
            try:
                value = float(data[name])
                if not 0.25 <= value <= 72:
                    raise ValueError
                setattr(s, name, value)
            except (TypeError, ValueError):
                errors[name] = "hours between 0.25 and 72"
    if "downtime_states" in data:
        states = [str(x).strip().upper() for x in data["downtime_states"] or [] if str(x).strip()]
        if not states:
            errors["downtime_states"] = "at least one state"
        s.downtime_states = states
    if "check_mode" in data:
        if data["check_mode"] not in ProblemSettings.CheckMode.values:
            errors["check_mode"] = "unknown mode"
        s.check_mode = data["check_mode"]
    if "check_day" in data:
        if data["check_day"] not in DAYS:
            errors["check_day"] = "mon … sun"
        s.check_day = data["check_day"]
    if "check_time" in data:
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(data["check_time"])):
            errors["check_time"] = "HH:MM"
        s.check_time = str(data["check_time"])
    if "publish_folder" in data:
        folder = str(data["publish_folder"] or "").strip().strip("/")
        if not folder:
            errors["publish_folder"] = "a library folder"
        s.publish_folder = folder[:200]
    if errors:
        raise TicketError("; ".join(f"{k}: {v}" for k, v in errors.items()))
    if "enabled" in data:
        enabled = bool(data["enabled"])
        if enabled and not s.enabled:
            s.enabled_at = now()          # only problems from now on
        s.enabled = enabled
    s.updated_by = (by or "")[:200]
    s.save()
    return settings_json(s)

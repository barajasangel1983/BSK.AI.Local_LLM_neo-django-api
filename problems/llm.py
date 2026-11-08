"""The models' part of problem solving.

- **P29b, recommendations** (when a ticket opens, or again by hand): what to check, from the ticket's context and
  what the Studio knows (graph facts [G] and numbered manual excerpts, through the Context Service's `assemble`).
- **P29b, clean-up and final report** ("Clean up and write the final report" on EDGE, just above the Final report): the
  fields the engineer filled are cleaned up (same content), empty ones are filled from the notes, and the final report is
  written. The engineer can undo it and edits it before submitting; nothing is saved by the model alone.
- **P29c, knowledge check** (after submit): the report is compared with the graph facts and manuals; the approver sees
  what agrees, what conflicts, what is missing. It never blocks an approval. It runs right after submit, when the 27B
  (DGX) answers, or on a schedule (settings), and at once with "Run now".

The 27B is used when it answers, else the 8B. The output is JSON; a reply that can't be read is an error kept on the
ticket, never a guess.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import requests
from django.conf import settings

from usage import recorder as usage

from . import context
from .models import ProblemSettings, Ticket
from .service import TicketError, now

logger = logging.getLogger("chat")

DGX = "dgx-qwen38-27b-fp8"
SMALL = "ollama-qwen3-8b"
KNOWLEDGE_CHARS = 9000
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

RECOMMEND_PROMPT = """You support a Manufacturing Engineer who starts a 5-Why problem solving on a machine.
You get WHAT HAPPENED (from the historian and SPC) and KNOWLEDGE (graph facts and manual excerpts, numbered).
Suggest what the engineer should check first to find the root cause: 3 to 6 short, concrete checks, most likely first.
Use the knowledge when it applies and cite it with its number, e.g. [2]; graph facts are [G]. Do not invent part numbers or values.
Only the alarms and values in WHAT HAPPENED occurred. The knowledge lists everything the machine can have (all its alarms,
procedures and parts): never say an alarm occurred unless WHAT HAPPENED lists it.
Reply with one JSON object and nothing else:
{"items": [{"text": "...", "sources": [1, 2]}]}  ("sources": the excerpt numbers, empty when none)"""

STRUCTURE_PROMPT = """You help a Manufacturing Engineer finish a 5-Why problem-solving report: a final clean-up.
You get WHAT HAPPENED (from the historian), the FIELDS the engineer filled, their NOTES (a scratch pad) and what the
PICTURES show. The FIELDS are what the engineer means: they come first; the NOTES only fill what the fields leave empty.
Write in {language}:
- a field the engineer filled: clean it up only (spelling, clear complete sentences); keep its content and meaning;
  do not add or remove facts, causes, actions or values;
- keep the plant's words, technical terms, codes and English loan words exactly as the engineer wrote them (e.g.
  "scrap", "OEE", "die", alarm codes): never translate them;
- "whys": when the engineer wrote whys, keep the same number, in the same order, each cleaned up. When there are none,
  build the chain from the notes: only the whys the notes support (often 3 or 4, at most 5), each answering the one
  before;
- an empty field: fill it from the notes, only with what the notes say; leave it empty when they say nothing;
- "actions": the corrective actions' texts only, in the same order as the engineer's, cleaned up; never put who or
  when in the text (they are kept apart); when there are none, take them from the notes;
- "report": the final report, a short summary (Markdown, 120 to 300 words) with: Problem, What happened, 5-Why,
  Root cause, Containment, Corrective actions;
- "missing": what the report still lacks (e.g. no corrective action, no containment).
Reply with one JSON object and nothing else:
{"problem": "...", "whys": ["...", "..."], "root_cause": "...", "containment": "...", "actions": ["...", "..."],
 "report": "...", "missing": ["..."]}"""

CHECK_PROMPT = """You review a 5-Why problem-solving report before a second engineer approves it.
Compare the REPORT with the KNOWLEDGE (graph facts and manual excerpts, numbered) and WHAT HAPPENED (historian data).
Only the alarms and values in WHAT HAPPENED occurred; the knowledge lists everything the machine can have.
List the points that matter:
- "conflicts": only a direct contradiction: the report states something the knowledge or the data says is not so
  (a value, a limit, a procedure step). A doubt, an alternative cause or a check not done is "missing", not a conflict;
- "agrees": the knowledge supports the root cause or the actions;
- "missing": something the knowledge says should be checked or done that the report does not mention.
Cite manual excerpts by their number in "sources"; graph facts are [G] in the text. Do not judge style. When the knowledge says nothing about the topic, say so.
verdict: "conflicts" when there is at least one conflict, "consistent" when the knowledge supports it and nothing
conflicts, "unclear" when the knowledge says little about it.
Reply with one JSON object and nothing else:
{"verdict": "consistent" | "conflicts" | "unclear", "points": [{"kind": "agrees" | "conflicts" | "missing",
 "text": "...", "sources": [1]}]}"""


def dgx_answers() -> bool:
    try:
        return requests.get(f"{settings.DGX_API_BASE.rstrip('/')}/v1/models", timeout=3).status_code == 200
    except (requests.RequestException, AttributeError):
        return False


def pick() -> str:
    return DGX if dgx_answers() else SMALL


def _complete(model_id: str, system: str, body: str, purpose: str) -> dict:
    from chat.question_agent import _parse
    from chat.views import _backend_stream
    with usage.scope(purpose=purpose):
        raw = "".join(_backend_stream(model_id, [{"role": "system", "content": system}, {"role": "user", "content": body}]))
    try:
        return _parse(raw)
    except (ValueError, TypeError) as exc:
        raise TicketError(f"the model's reply could not be read ({exc})") from None


def knowledge(ticket: Ticket, query: str) -> tuple[str, list[dict]]:
    """Numbered graph facts and manual excerpts for a query about the ticket's machine: (text, sources)."""
    from context_service import service
    try:
        packet = service.assemble(query[:1500], asset_id=ticket.asset_id, include_documents=True,
                                  budget_chars=KNOWLEDGE_CHARS)
    except Exception as exc:
        logger.warning("problems: no knowledge for %s (%s)", ticket.number, exc)
        return "(the knowledge could not be read)", []
    sources = [{"n": n, "document": d.get("document", ""), "page": d.get("page"),
                "section": (d.get("section_path") or [""])[-1], "text": (d.get("text") or "")[:300]}
               for n, d in enumerate(packet.get("document_evidence") or [], 1)]
    return packet.get("context_text") or "(nothing found)", sources


def _pictures(ticket: Ticket) -> str:
    lines = [f"- {p.caption or p.filename}: {p.description}" for p in ticket.pictures.all() if p.description or p.caption]
    return "\n".join(lines) or "(none described)"


def recommend(ticket: Ticket) -> Ticket:
    """P29b: what to check first."""
    model = pick()
    text, sources = knowledge(ticket, f"{ticket.title}. Causes and what to check. " + _focus(ticket))
    data = _complete(model, RECOMMEND_PROMPT, f"WHAT HAPPENED\n{context.as_text(ticket)}\n\nKNOWLEDGE\n{text}",
                     "problem-recommend")
    items = [{"text": str(i.get("text", ""))[:500], "sources": [s for s in i.get("sources", []) if isinstance(s, int)]}
             for i in data.get("items", []) if isinstance(i, dict) and i.get("text")][:8]
    ticket.recommendations = {"items": items, "sources": sources, "model": model, "at": now().isoformat()}
    ticket.save(update_fields=["recommendations", "updated_at"])
    return ticket


def _focus(ticket: Ticket) -> str:
    c = ticket.context or {}
    words = list((c.get("alarms") or {}).keys())
    words += [o["name"] for o in c.get("out_of_range") or []]
    words += [e["tool"] for e in c.get("spc") or []]
    return " ".join(dict.fromkeys(words))[:500]


def structure(ticket: Ticket) -> dict:
    """P29b: the final version, returned for the engineer to edit (nothing saved)."""
    fields = (f"Problem: {ticket.problem}\n" + "".join(f"Why {i}: {w}\n" for i, w in enumerate(ticket.whys or [], 1))
              + f"Root cause: {ticket.root_cause}\nContainment: {ticket.containment}\n"
              + "Corrective actions:\n" + "".join(f"{i}. {a['text']}" + (" (who and when: set)" if a.get("responsible") and a.get("due_date")
                                                       else " (who or when: not set yet)") + "\n"
                                                       for i, a in enumerate(ticket.actions or [], 1)))
    if not (ticket.notes.strip() or ticket.problem.strip() or ticket.root_cause.strip() or any(w.strip() for w in ticket.whys or [])):
        raise TicketError("fill the fields (or write some notes) first")
    from .publish import describe_pictures
    describe_pictures(ticket)
    from chat.question_agent import _language
    language = "Spanish" if _language(f"{ticket.notes} {ticket.problem} {' '.join(ticket.whys or [])}") == "es" else "English"
    model = pick()
    data = _complete(model, STRUCTURE_PROMPT.replace("{language}", language),
                     f"WHAT HAPPENED\n{context.as_text(ticket)}\n\nFIELDS\n{fields}\n\nNOTES\n{ticket.notes or '(none)'}"
                     f"\n\nPICTURES\n{_pictures(ticket)}", "problem-structure")
    whys = [str(w).strip()[:2000] for w in data.get("whys") or [] if str(w).strip()][:5]
    return {"problem": str(data.get("problem") or "")[:8000], "whys": whys,
            "root_cause": str(data.get("root_cause") or "")[:8000], "containment": str(data.get("containment") or "")[:8000],
            "actions": [str(a).strip()[:1000] for a in (data.get("actions") or [])
                        if isinstance(a, (str, int, float)) and str(a).strip()][:20]
                       or [l.strip(" -•") for l in str(data.get("corrective_actions") or "").splitlines() if l.strip(" -•")][:20],
            "final_report": str(data.get("report") or "")[:8000],
            "missing": [str(m)[:200] for m in data.get("missing") or []][:8], "model": model}


def report_text(ticket: Ticket) -> str:
    """The report as written (the final version, else the fields)."""
    if ticket.final_report.strip():
        return ticket.final_report
    whys = "\n".join(f"{i}. {w}" for i, w in enumerate(ticket.whys or [], 1))
    return (f"Problem: {ticket.problem}\n\n5-Why:\n{whys}\n\nRoot cause: {ticket.root_cause}\n\n"
            f"Containment: {ticket.containment}\n\nCorrective actions:\n{ticket.corrective_actions}")


def check(ticket: Ticket) -> Ticket:
    """P29c: the knowledge check. Its result is kept on the ticket for the approver."""
    model = pick()
    ticket.knowledge_check = {**(ticket.knowledge_check or {}), "status": "running", "started_at": now().isoformat(), "model": model}
    ticket.save(update_fields=["knowledge_check", "updated_at"])
    try:
        text, sources = knowledge(ticket, f"{ticket.problem} {ticket.root_cause} {ticket.corrective_actions}")
        data = _complete(model, CHECK_PROMPT, f"REPORT\n{report_text(ticket)}\n\nWHAT HAPPENED\n{context.as_text(ticket)}"
                                              f"\n\nKNOWLEDGE\n{text}", "problem-check")
        verdict = data.get("verdict") if data.get("verdict") in ("consistent", "conflicts", "unclear") else "unclear"
        points = [{"kind": p.get("kind") if p.get("kind") in ("agrees", "conflicts", "missing") else "missing",
                   "text": str(p.get("text", ""))[:900],
                   "sources": [s for s in p.get("sources", []) if isinstance(s, int)]}
                  for p in data.get("points", []) if isinstance(p, dict) and p.get("text")][:12]
        if any(p["kind"] == "conflicts" for p in points):
            verdict = "conflicts"
        ticket.knowledge_check = {"status": "done", "verdict": verdict, "points": points, "sources": sources, "model": model,
                        "at": now().isoformat()}
    except Exception as exc:
        logger.warning("problems: check of %s failed (%s)", ticket.number, exc)
        ticket.knowledge_check = {**ticket.knowledge_check, "status": "failed", "error": str(exc)[:300], "at": now().isoformat()}
    ticket.save(update_fields=["knowledge_check", "updated_at"])
    return ticket


def request_check(ticket: Ticket) -> Ticket:
    """"Run now": the next minute pass runs it, whatever the mode."""
    if ticket.status != Ticket.Status.SUBMITTED:
        raise TicketError(f"{ticket.number} is not waiting for approval")
    ticket.knowledge_check = {**(ticket.knowledge_check or {}), "status": "queued", "now": True, "queued_at": now().isoformat()}
    ticket.save(update_fields=["knowledge_check", "updated_at"])
    return ticket


def _last_scheduled(s: ProblemSettings, at: datetime) -> datetime:
    """The most recent scheduled moment (the plant's time when it has a calendar, else UTC) at or before `at`."""
    tz = timezone.utc
    try:
        from monitor import plant_calendar
        cal = next((c for c in (plant_calendar.for_plant(p["id"]) for p in plant_calendar.plants()) if c), None)
        tz = cal.tz if cal else tz
    except Exception:
        pass
    local = at.astimezone(tz)
    hh, mm = (int(x) for x in s.check_time.split(":"))
    day = DAYS.index(s.check_day)
    moment = local.replace(hour=hh, minute=mm, second=0, microsecond=0) - timedelta(days=(local.weekday() - day) % 7)
    if moment > local:
        moment -= timedelta(days=7)
    return moment.astimezone(timezone.utc)


def due(s: ProblemSettings, ticket: Ticket, at: datetime, dgx_up) -> bool:
    c = ticket.knowledge_check or {}
    if c.get("status") != "queued":
        return False
    if c.get("now") or s.check_mode == ProblemSettings.CheckMode.ON_SUBMIT:
        return True
    if s.check_mode == ProblemSettings.CheckMode.DGX:
        return dgx_up()
    queued = datetime.fromisoformat(c["queued_at"]) if c.get("queued_at") else at
    return queued <= _last_scheduled(s, at) <= at


def recommend_queued(limit: int = 1) -> int:
    """"What to check first" for tickets opened by hand (the triggers write it as they open a ticket)."""
    done = 0
    for ticket in Ticket.objects.filter(recommendations__status="queued").order_by("id")[:limit]:
        try:
            recommend(ticket)
        except Exception as exc:
            logger.warning("problems: no recommendations for %s (%s)", ticket.number, exc)
            ticket.recommendations = {"status": "failed", "error": str(exc)[:300], "at": now().isoformat()}
            ticket.save(update_fields=["recommendations", "updated_at"])
        done += 1
    return done


def run_due(limit: int = 1, at: datetime | None = None) -> int:
    """One minute pass: the queued "what to check first", then the knowledge checks that are due (one of each, so the
    monitor isn't held up). Returns how many checks ran."""
    recommend_queued(limit)
    at = at or now()
    s = ProblemSettings.get()
    cache: dict = {}

    def dgx_up() -> bool:
        if "dgx" not in cache:
            cache["dgx"] = dgx_answers()
        return cache["dgx"]

    done = 0
    for ticket in Ticket.objects.filter(status=Ticket.Status.SUBMITTED).order_by("submitted_at"):
        if done >= limit:
            break
        if due(s, ticket, at, dgx_up):
            check(ticket)
            done += 1
    return done

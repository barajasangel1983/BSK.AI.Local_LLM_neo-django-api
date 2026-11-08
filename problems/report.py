"""The problem-solving report: one template, two forms (P29).

- `markdown(ticket)`: for the document library (text only; the pictures by what they show), so the chat finds it.
- `html(ticket)`: the printable report with the pictures, for "Download report" (the browser saves it as PDF).

Sections, in order: header (machine, ticket, trigger, window, author, approver, dates, status) · Problem · What
happened · 5-Why · Root cause · Pictures of the problem · Containment · Corrective actions (table) · Final report ·
Pictures after the fix · Validation.
"""

from __future__ import annotations

import base64
import html as h
import logging
from datetime import datetime

from . import context
from .models import Ticket

logger = logging.getLogger("chat")

STATE_LABEL = {"done": "Done", "in_progress": "In progress", "delayed": "Delayed"}
STATE_COLOR = {"done": "#16a34a", "in_progress": "#ca8a04", "delayed": "#dc2626"}


def _when(value) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return f"{value:%Y-%m-%d %H:%M} UTC"


def _person(p: dict) -> str:
    return f"{p.get('name') or p.get('username', '')} ({p.get('role', '')})".replace(" ()", "") if p else ""


def header(t: Ticket) -> list[tuple[str, str]]:
    return [("Machine", f"{t.asset_name} ({t.asset_id})"), ("Ticket", t.number), ("Trigger", t.get_trigger_display()),
            ("Window", f"{_when(t.window_start)} to " + (_when(t.window_end) if t.window_end else "still going")),
            ("Written by", _person(t.author)), ("Approved by", _person(t.approver)),
            ("Approved", _when(t.approved_at)), ("Completed", _when(t.completed_at)), ("Status", t.get_status_display())]


def _actions(t: Ticket) -> list[dict]:
    from .service import actions_json
    return actions_json(t)


def markdown(t: Ticket) -> str:
    lines = [f"# {t.number}: {t.title}", "",
             f"Problem-solving report (5-Why) for {t.asset_name or t.asset_id}.", ""]
    lines += [f"- {k}: {v}" for k, v in header(t) if v] + [""]
    lines += ["## Problem", "", t.problem or "(not written)", "",
              "## What happened (historian and SPC)", "", context.as_text(t), "",
              "## 5-Why", ""] + [f"{i}. Why? {w}" for i, w in enumerate(t.whys or [], 1)] + [""]
    lines += ["## Root cause", "", t.root_cause or "(not written)", ""]
    lines += _md_pictures(t, "problem", "Pictures of the problem")
    lines += ["## Containment", "", t.containment or "(none)", "", "## Corrective actions", "",
              "| Action | Responsible | Due date | Status | Validated by |", "|---|---|---|---|---|"]
    for a in _actions(t):
        v = (a.get("validated_by") or {}).get("name", "")
        lines.append(f"| {a['text']} | {a.get('responsible', '')} | {a.get('due_date', '')} | {STATE_LABEL[a['state']]} | {v} |")
    lines.append("")
    if t.final_report.strip():
        lines += ["## Final report", "", t.final_report, ""]
    lines += _md_pictures(t, "after", "Pictures after the fix")
    return "\n".join(lines)


def _md_pictures(t: Ticket, phase: str, title: str) -> list[str]:
    pics = [p for p in t.pictures.all() if p.phase == phase]
    if not pics:
        return []
    return [f"## {title}", ""] + [f"- {p.caption or p.filename}: {p.description or '(not described)'}" for p in pics] + [""]


CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#111;max-width:900px;margin:24px auto;padding:0 20px;font-size:14px;line-height:1.45}
h1{font-size:22px;margin:0 0 4px} h2{font-size:16px;border-bottom:1px solid #ddd;padding-bottom:3px;margin:22px 0 8px}
.meta{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;font-size:13px;margin:10px 0}
.meta dt{color:#555} table{border-collapse:collapse;width:100%;font-size:13px} th,td{border:1px solid #ccc;padding:5px 7px;text-align:left;vertical-align:top}
th{background:#f3f4f6} .state{font-weight:600} .pics{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
figure{margin:0} figure img{width:100%;border:1px solid #ccc;border-radius:4px} figcaption{font-size:12px;color:#444;margin-top:3px}
pre{white-space:pre-wrap;font-family:inherit;margin:0} .muted{color:#666} ol{margin:0;padding-left:22px}
.bar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #eee;margin-bottom:12px}
@media print{.bar{display:none} body{margin:0}}
"""


def html(t: Ticket) -> str:
    from .service import picture_bytes
    e = h.escape
    meta = "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in header(t) if v)
    rows = "".join(
        f"<tr><td>{e(a['text'])}</td><td>{e(a.get('responsible', ''))}</td><td>{e(a.get('due_date', ''))}</td>"
        f"<td class='state' style='color:{STATE_COLOR[a['state']]}'>● {STATE_LABEL[a['state']]}</td>"
        f"<td>{e((a.get('validated_by') or {}).get('name', ''))}{(' · ' + e(a['validated_at'][:10])) if a.get('validated_at') else ''}</td></tr>"
        for a in _actions(t))

    def pictures(phase: str, title: str) -> str:
        figs = []
        for p in t.pictures.all():
            if p.phase != phase:
                continue
            try:
                data = base64.b64encode(picture_bytes(t.pk, p.pk)).decode()
                img = f"<img src='data:image/jpeg;base64,{data}' alt='{e(p.caption or p.filename)}'>"
            except Exception as exc:
                logger.warning("problems: picture %s missing from the report (%s)", p.pk, exc)
                img = "<p class='muted'>(picture missing)</p>"
            figs.append(f"<figure>{img}<figcaption>{e(p.caption or p.filename)}"
                        f"{(' — ' + e(p.description)) if p.description else ''}</figcaption></figure>")
        return f"<h2>{e(title)}</h2><div class='pics'>{''.join(figs)}</div>" if figs else ""

    whys = "".join(f"<li>{e(w)}</li>" for w in t.whys or [])
    validation = "".join(f"<li>{e(_when(x['at']))}: {e(x['action'])} by {e(x['by'].get('name') or x['by'].get('username', ''))}"
                         f"{(': ' + e(x['comment'])) if x.get('comment') else ''}</li>" for x in t.history or [])
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{e(t.number)} {e(t.title)}</title>
<style>{CSS}</style></head><body>
<div class="bar"><button onclick="window.print()">Print / save as PDF</button></div>
<h1>{e(t.number)}: {e(t.title)}</h1><p class="muted">Problem-solving report (5-Why)</p>
<dl class="meta">{meta}</dl>
<h2>Problem</h2><pre>{e(t.problem or '(not written)')}</pre>
<h2>What happened (historian and SPC)</h2><pre>{e(context.as_text(t))}</pre>
<h2>5-Why</h2><ol>{whys}</ol>
<h2>Root cause</h2><pre>{e(t.root_cause or '(not written)')}</pre>
{pictures('problem', 'Pictures of the problem')}
<h2>Containment</h2><pre>{e(t.containment or '(none)')}</pre>
<h2>Corrective actions</h2><table><tr><th>Action</th><th>Responsible</th><th>Due date</th><th>Status</th><th>Validated</th></tr>{rows}</table>
{('<h2>Final report</h2><pre>' + e(t.final_report) + '</pre>') if t.final_report.strip() else ''}
{pictures('after', 'Pictures after the fix')}
<h2>Validation and history</h2><ul>{validation}</ul>
</body></html>"""

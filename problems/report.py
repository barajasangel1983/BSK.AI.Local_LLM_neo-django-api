"""The problem-solving report: one template, two forms (P29).

- `markdown(ticket)`: for the document library (text only; the pictures by what they show), so the chat finds it.
- `html(ticket)`: the printable report with the pictures, for "Download report" (the browser saves it as PDF).

Sections, in order: header (machine, ticket, trigger, window, author, approver, dates, status) · Problem · What
happened · Data summary (P29e) · 5-Why · Root cause · Pictures of the problem · Containment · Corrective actions (table) · Final report ·
Pictures after the fix · Validation.
"""

from __future__ import annotations

import base64
import html as h
import logging
from datetime import datetime, timezone

from . import analysis, context
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
{data_summary_html(t)}
<h2>5-Why</h2><ol>{whys}</ol>
<h2>Root cause</h2><pre>{e(t.root_cause or '(not written)')}</pre>
{pictures('problem', 'Pictures of the problem')}
<h2>Containment</h2><pre>{e(t.containment or '(none)')}</pre>
<h2>Corrective actions</h2><table><tr><th>Action</th><th>Responsible</th><th>Due date</th><th>Status</th><th>Validated</th></tr>{rows}</table>
{('<h2>Final report</h2><pre>' + e(t.final_report) + '</pre>') if t.final_report.strip() else ''}
{pictures('after', 'Pictures after the fix')}
<h2>Validation and history</h2><ul>{validation}</ul>
</body></html>"""


# --- the data summary as inline SVG (P29e) ------------------------------------------------------------------------

STATE_FILL = {"RUNNING": "#dcfce7", "STOPPED": "#fee2e2", "ALARMED": "#fecaca"}


def _svg_trend(a: dict, series: dict, height: int, caption: str) -> str:
    """One trend: the line, the normal range or limits, the onset band and the start line."""
    e = h.escape
    pts = series.get("points") or []
    if not pts:
        return f"<p class='muted'>{e(caption)}: no values.</p>"
    x0 = datetime.fromisoformat(a["from"]).timestamp()
    x1 = max(datetime.fromisoformat(a["until"]).timestamp(), x0 + 60)
    lim = series.get("limits") or {}
    refs = [(k.upper(), lim[k]) for k in ("lsl", "lcl", "cl", "ucl", "usl") if isinstance(lim.get(k), (int, float))]
    if not refs:
        refs = [(n, series[k]) for n, k in (("low", "low"), ("high", "high")) if isinstance(series.get(k), (int, float))]
    ys = [p[1] for p in pts] + [v for _, v in refs]
    y0, y1 = min(ys), max(ys)
    pad = (y1 - y0) * 0.08 or 1
    y0, y1 = y0 - pad, y1 + pad
    W, L, R, T, B = 860, 48, 8, 6, 18
    pw, ph = W - L - R, height - T - B

    def X(t):
        return L + (t - x0) / (x1 - x0) * pw

    def Y(v):
        return T + (y1 - v) / (y1 - y0) * ph

    parts = []
    start = datetime.fromisoformat(a["start"]).timestamp()
    onset = datetime.fromisoformat(a["onset"]).timestamp()
    if onset < start:
        parts.append(f"<rect x='{X(onset):.1f}' y='{T}' width='{X(start) - X(onset):.1f}' height='{ph}' fill='#fef3c7'/>")
    for name, v in refs:
        dash = "" if name in ("LSL", "USL") else " stroke-dasharray='4 3'"
        color = "#dc2626" if name in ("LSL", "USL") else "#94a3b8"
        parts.append(f"<line x1='{L}' x2='{L + pw}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='{color}' stroke-width='1'{dash}/>"
                     f"<text x='{L + pw - 2}' y='{Y(v) - 2:.1f}' font-size='9' text-anchor='end' fill='{color}'>{name} {v:.4g}</text>")
    line = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in pts)
    parts.append(f"<polyline points='{line}' fill='none' stroke='#2563eb' stroke-width='1.3'/>")
    parts.append(f"<line x1='{X(start):.1f}' x2='{X(start):.1f}' y1='{T}' y2='{T + ph}' stroke='#111' stroke-width='1'/>")
    parts.append(f"<text x='{L - 4}' y='{T + 9}' font-size='9' text-anchor='end'>{y1:.4g}</text>"
                 f"<text x='{L - 4}' y='{T + ph}' font-size='9' text-anchor='end'>{y0:.4g}</text>")
    for frac in (0, 0.5, 1):
        t = x0 + frac * (x1 - x0)
        anchor = "start" if frac == 0 else "end" if frac == 1 else "middle"
        parts.append(f"<text x='{X(t):.1f}' y='{height - 4}' font-size='9' text-anchor='{anchor}'>"
                     f"{datetime.fromtimestamp(t, tz=timezone.utc):%H:%M}</text>")
    return (f"<figure><figcaption><b>{e(caption)}</b></figcaption><svg viewBox='0 0 {W} {height}' width='100%' "
            f"role='img' aria-label='{e(caption)}'>{''.join(parts)}</svg></figure>")


def _svg_states(a: dict) -> str:
    st = a.get("states") or []
    if not st:
        return ""
    x0 = datetime.fromisoformat(a["from"]).timestamp()
    x1 = max(datetime.fromisoformat(a["until"]).timestamp(), x0 + 60)
    W, L, R, H = 860, 48, 8, 14
    pw = W - L - R
    rects = []
    for i, (t, s) in enumerate(st):
        t_end = st[i + 1][0] if i + 1 < len(st) else x1
        a_, b_ = max(t, x0), min(t_end, x1)
        if b_ <= a_:
            continue
        x = L + (a_ - x0) / (x1 - x0) * pw
        w = (b_ - a_) / (x1 - x0) * pw
        rects.append(f"<rect x='{x:.1f}' y='0' width='{w:.1f}' height='{H}' fill='{STATE_FILL.get(s, '#e5e7eb')}'>"
                     f"<title>{h.escape(s)}</title></rect>")
    return (f"<figure><figcaption><b>Machine state</b> (green running, red stopped or alarmed, grey other)</figcaption>"
            f"<svg viewBox='0 0 {W} {H}' width='100%'>{''.join(rects)}</svg></figure>")


def data_summary_html(t: Ticket) -> str:
    a = (t.context or {}).get("analysis") or {}
    if not a or a.get("error") and not a.get("from"):
        return f"<h2>Data summary</h2><p class='muted'>{h.escape(a.get('error', 'No data summary.'))}</p>" if a else ""
    blocks = [_svg_states(a)]
    if a.get("trigger"):
        tr = a["trigger"]
        blocks.append(_svg_trend(a, tr, 150, f"Trigger: {tr['name']} ({tr.get('unit') or '-'})"))
    for c in a.get("candidates") or []:
        follow = f"; with {', '.join(c['with'])}" if c.get("with") else ""
        blocks.append(_svg_trend(a, c, 90, f"{c['name']} ({c.get('unit') or '-'}): moved {analysis._minutes(c['lead_min'])} "
                                           f"the start, {c['direction']} {c['peak_sigma']} σ{follow}"))
    words = "".join(f"<li>{h.escape(x)}</li>" for x in analysis.as_lines(a))
    if a.get("error"):
        words += f"<li>{h.escape(a['error'])}</li>"
    return ("<h2>Data summary</h2><p class='muted'>The black line is the start of the problem; the yellow band is from "
            "the first signal that moved to the start. \"Moved\" means more than 3 σ from the running baseline 2 h to "
            "1 h before the start; it is not a cause.</p>" + "".join(blocks) + (f"<ul>{words}</ul>" if words else ""))

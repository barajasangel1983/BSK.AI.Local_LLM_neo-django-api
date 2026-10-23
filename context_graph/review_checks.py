"""Staging checks from the QUINT4-PS review (P26b-1, 2026-10-23): a backstop to the prompts.

The review rejected 50 of 271 triples that had no issue. The prompts now avoid those mistakes; these checks flag the
ones that still come through, as an `issue` the review filters on ("Issues only"). They never drop a triple: the
reviewer decides.

- `triple_issue`: one triple, at staging: a rating with no real value (empty cell, a standard's name, only its
  condition), a "Row 3" name. (A "name not in the text" check was tried on the QUINT4 triples and dropped: a table
  window doesn't carry its row labels in a form it can match, so it flagged 22 good ratings.)
- `flag_job`: after a job, across its facet runs: the same name staged under two types, a rating that repeats a
  parameter, likely duplicates (same values, one name contained in the other).
"""

from __future__ import annotations

import re

from .models import CandidateTriple

ISSUE_MAX = 255
_EMPTY = {"", "-", "–", "—", "--", "---", "n/a", "na", "none", "not required", "not applicable", "no data", "x"}
_STANDARD = re.compile(r"^(?:EN|IEC|ISO|UL|CSA|DIN|VDE|IEEE|NEMA|ANSI|GB|JIS)\b[\s\dA-Z\-–./:,]*$")
_ROW_NAME = re.compile(r"\b(?:row|column|col\.?)\s*\d+\b", re.I)
_PRIVATE = re.compile("[\\ue000-\\uf8ff]")       # icon-font glyphs a parser leaves for a symbol
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
STOPWORDS = {"the", "a", "an", "of", "for", "to", "and", "or", "in", "on", "at", "by", "from", "typ", "max",
             "min", "nominal"}
SIMILAR = 0.75         # word overlap (Jaccard) at which two components or procedures are likely the same
PHYSICAL = ("Component", "IOPoint", "Interface")
CHECKED = ("Component", "IOPoint", "Parameter", "Interface", "Specification", "Procedure", "Alarm")


def _norm(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _words(name: str) -> set[str]:
    return {w for w in _TOKEN.findall(_norm(name)) if w not in STOPWORDS}


def _numbers(props: dict) -> tuple:
    return tuple(sorted(n.replace(",", ".") for k in ("value", "min", "max") for n in _NUMBER.findall(str(props.get(k) or ""))))


def _empty(value) -> bool:
    v = _norm(_PRIVATE.sub("", str(value or "")))
    return v.strip(" .") in _EMPTY


def triple_issue(object_type: str, object_name: str, props: dict) -> str:
    """The issue for one staged rating (Specification), or "". Other types: ""."""
    if object_type != "Specification":
        return ""
    values = {k: props.get(k) for k in ("value", "min", "max") if str(props.get(k) or "").strip()}
    name = _norm(object_name)
    if not values or all(_empty(v) for v in values.values()):
        return "a rating with no value (an empty cell, or a column the parser dropped)"
    if all(_STANDARD.match(str(v).strip()) for v in values.values()):
        return "the value is only a standard's name; it belongs in the condition of the level it defines"
    condition = _norm(props.get("condition"))
    if all(_norm(v) in (condition, "") or (len(_norm(v)) > 2 and _norm(v) in name) for v in values.values()):
        return "the value is only what the name or condition already says (a test condition, not a rating)"
    if _ROW_NAME.search(object_name):
        return "named after a table row or column number; name it after what the row states"
    return ""


def flag_job(job) -> int:
    """Flag this job's pending triples against each other and the document's other kept triples. Returns how many."""
    mine = list(CandidateTriple.objects.filter(job=job, status=CandidateTriple.Status.PENDING, mode="schema",
                                               object_type__in=CHECKED))
    if not mine:
        return 0
    others = list(CandidateTriple.objects.filter(document_id=job.document_id, mode="schema", object_type__in=CHECKED)
                  .exclude(status=CandidateTriple.Status.REJECTED).exclude(job=job))
    issues: dict[int, str] = {}

    def flag(t, message):
        if not t.issue and t.pk not in issues:
            issues[t.pk] = message[:ISSUE_MAX]

    # The same thing under two types (the QUINT4 LEDs: Component from Maintenance, IOPoint from Controls). Only
    # physical things: a Signal and a Parameter may share a name ("Output power") and be two facts.
    by_name: dict[tuple, set] = {}
    for t in mine + others:
        if t.object_type in PHYSICAL:
            by_name.setdefault((t.anchor_id, _norm(t.object_name)), set()).add(t.object_type)
    for t in mine:
        types = by_name.get((t.anchor_id, _norm(t.object_name)), set()) - {t.object_type}
        if types:
            flag(t, f"also staged as {', '.join(sorted(types))} from this document; keep one type")

    # A rating that repeats a parameter of the same document ("Default setting: Output current")
    params = {_norm(t.object_name) for t in mine + others if t.object_type == "Parameter"}
    for t in mine:
        if t.object_type == "Specification":
            bare = _norm(re.sub(r"^(?:default|factory)\s+settings?\s*[:\-]?\s*|^setting\s+range\s+(?:of\s+)?(?:the\s+)?",
                                "", t.object_name, flags=re.I))
            if bare in params:
                flag(t, f"repeats the parameter '{bare}': a value the user sets is a parameter, not a rating")

    # Likely duplicates on the same anchor and type
    pool = mine + others
    for t in mine:
        w = _words(t.object_name)
        if not w:
            continue
        for o in pool:
            if o.pk == t.pk or o.object_type != t.object_type or o.anchor_id != t.anchor_id or o.subject_id != t.subject_id:
                continue
            ow = _words(o.object_name)
            if not ow or o.object_id == t.object_id:
                continue        # the same node: one fact read twice is already one
            if t.object_type == "Specification":
                same = w < ow and _numbers(t.object_props or {}) == _numbers(o.object_props or {}) != ()
            else:      # of a pair in this run, the later one is flagged
                same = ((w <= ow or w >= ow or len(w & ow) / len(w | ow) >= SIMILAR)
                        and not (o.job_id == t.job_id and o.pk > t.pk))
            if same:
                flag(t, f"likely the same as '{o.object_name}' (p{o.page_start}); keep one")
                break

    for pk, message in issues.items():
        CandidateTriple.objects.filter(pk=pk, issue="").update(issue=message)
    return len(issues)

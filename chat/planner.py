"""Retrieval planner for asset chats (P17b, 2026-10-17): rules, no model call.

For a question about an asset it decides
- which parts the question is about, and so which manuals the document search tries first: the parts in focus
  (P16a) or named in the question; a group without a manual (the "Cutter") gives its sub-parts' manuals, best
  matching first ("cutter drive" → the SINAMICS G120C manual); a sub-part without one (a fan) its parent's;
- the kind of question, which sets the split of the prompt between graph facts and excerpts: a parts-list
  question ("which motors …", "how many zones …") is answered from the graph, so the facts get more room.

The search still falls back to the asset's documents, then the library, as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from django.conf import settings

from .asset_context import STOPWORDS

PARTS_QUESTION = re.compile(r"\b(which|how many|what model|made by|manufacturer|what (kind|type|make) of|list)\b")
# … unless it asks about maintenance, a fault or a value ("which parts need replacing at 9 years").
NOT_PARTS = re.compile(r"\b(replac\w*|interval|how often|maintenance|lubric\w*|clean\w*|calibrat\w*|fault|alarm|"
                       r"warning|years?|hours?|torque)\b")
CODE = re.compile(r"\b(?:[A-F]?\d{4}|[A-F][0-9A-F]{3}|[A-Z]\d{4,5}|\d{2}\.\d{1,2})\b")


@dataclass
class Plan:
    kind: str = "general"               # parts | code | general
    parts: list[str] = field(default_factory=list)      # the parts the question is about (ids)
    doc_keys: list[str] = field(default_factory=list)   # their manuals, searched first
    facts_share: float = 0.0            # 0 = the setting's share
    excerpts_share: float = 0.0
    # Added to the search query: the parts with their make and model. The reranker scores "clean the lens of the
    # infrared thermometer" ~0.01 against the optris CT manual's lens cleaning, and 1.0 with "(… optris CT LT15 …)".
    query_note: str = ""

    def describe(self) -> str:
        return f"kind={self.kind} parts={len(self.parts)} manuals={len(self.doc_keys)} note={self.query_note!r}"

    def query(self, question: str) -> str:
        return f"{question} ({self.query_note})" if self.query_note else question


def _words(text: str) -> set[str]:
    """Lower-case words without stop words, plural "s" dropped ("drives" = "drive")."""
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOPWORDS or len(w) < 2:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def _named(question_words: set[str], parts: list[dict]) -> list[str]:
    """Parts named in the question: their whole name ("feed screw drive"), longest first; when none is, those
    sharing at least two of their words and half of them ("infrared thermometer at the product discharge" →
    "Product discharge temperature (infrared)"), most shared first. A part named whole drops the near ones
    ("water injection flow meter" is not also the steam flow meter)."""
    whole, most = [], []
    for p in parts:
        w = _words(p["name"])
        shared = len(w & question_words)
        if w and shared == len(w):
            whole.append((-len(w), p["id"]))
        elif shared >= 2 and shared * 2 >= len(w):
            most.append((-shared / len(w), -shared, p["id"]))
    return [i for *_, i in sorted(whole)] if whole else [i for *_, i in sorted(most)]


def question_kind(question: str) -> Plan:
    """The kind of question and its prompt split (known before the fact sheet is built)."""
    out = Plan()
    if PARTS_QUESTION.search(question.lower()) and not NOT_PARTS.search(question.lower()) and not CODE.search(question):
        out.kind, out.facts_share, out.excerpts_share = "parts", settings.PLANNER_PARTS_FACTS_SHARE, \
            settings.PLANNER_PARTS_EXCERPTS_SHARE
    elif CODE.search(question):
        out.kind = "code"
    return out


def plan(question: str, parts: list[dict], focus_ids: set[str]) -> Plan:
    out = question_kind(question)
    by_id = {p["id"]: p for p in parts}
    children: dict[str, list[dict]] = {}
    for p in parts:
        children.setdefault(p["parent"], []).append(p)
    q = _words(question)
    chosen = [i for i in [*_named(q, parts), *sorted(focus_ids)] if i in by_id]
    out.parts = list(dict.fromkeys(chosen))
    keys: list[str] = []
    noted: list[dict] = []
    for pid in out.parts:
        part = by_id[pid]
        if part["doc_keys"]:
            keys += part["doc_keys"]
            noted.append(part)
            continue
        # A group: its sub-parts' manuals, those sharing the question's words first.
        subs = sorted((c for c in children.get(pid, []) if c["doc_keys"]),
                      key=lambda c: -len(_words(c["name"]) & q))
        if subs:
            best = len(_words(subs[0]["name"]) & q)
            best_subs = [c for c in subs if len(_words(c["name"]) & q) == best]
            keys += [k for c in best_subs for k in c["doc_keys"]]
            noted += best_subs
            continue
        # A sub-part: the nearest ancestor's manuals.
        up = by_id.get(part["parent"])
        while up is not None and not up["doc_keys"]:
            up = by_id.get(up["parent"])
        if up is not None:
            keys += up["doc_keys"]
            noted.append(up)
    out.doc_keys = list(dict.fromkeys(keys))[: settings.PLANNER_MAX_MANUALS]
    notes = [f"{p['name']}: {p['model']}" if p.get("model") else p["name"]
             for p in {p["id"]: p for p in noted}.values()]
    out.query_note = "; ".join(notes[: settings.PLANNER_QUERY_PARTS])
    return out

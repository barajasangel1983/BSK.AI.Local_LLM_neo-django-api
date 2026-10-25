"""Quicker review (2026-10-18): the same thing staged under different names in one document, and terminals that may
not be signal channels.

Each window is read on its own, so one manual gives "DI 0" and "Digital input 0", "+24V IN" and "24 V IN",
"PROFIBUS" and "PROFIBUS DP". Groups are found by rules that hold for any manufacturer — same type, same part, and
the same name once normalised, one name the initials of the other, very close names, or the same key (a parameter's
code, a terminal's connector + terminal) — and never across different numbers ("DI 0" and "DI 1" stay apart).
Merging renames the staged ends to one name and id and folds triples that became identical. Nothing is approved.
"""

from __future__ import annotations

import re
from django.db import transaction

from .extraction import differing, merge_properties
from .identity import normalize
from .models import CandidateTriple

ENDS = ("subject", "object")


def _numbers(name: str) -> tuple:
    return tuple(re.findall(r"\d+", name))


def _words(name: str) -> list[str]:
    return re.findall(r"[a-z]+", name.lower())


def same_name(a: str, b: str) -> bool:
    """One thing under two names? Never when the numbers differ; never by similar spelling alone."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na.replace(" ", "") == nb.replace(" ", ""):
        return True
    if _numbers(a) != _numbers(b):
        return False
    wa, wb = _words(a), _words(b)
    if wa and wb and ("".join(wa) == "".join(w[0] for w in wb) or "".join(wb) == "".join(w[0] for w in wa)):
        return True                                       # DI 0 / Digital input 0
    # One name holds the other's words ("PROFIBUS" / "PROFIBUS DP", "Jogging bit 0" / "BI: Jogging bit 0"). Not mere
    # similar spelling: "Minimum speed" / "Maximum speed" and "input" / "output" are different things (2026-10-18).
    sa, sb = set(wa), set(wb)
    return bool(sa) and bool(sb) and (sa < sb or sb < sa)


def closeness(a: dict, b: dict) -> int | None:
    """How close two staged ends are (lower is closer), or None when they are not one thing."""
    if (a["id"] and a["id"] == b["id"]) or (a["keys"] & b["keys"]):
        return 0
    if not same_name(a["name"], b["name"]):
        return None
    wa, wb = set(_words(a["name"])), set(_words(b["name"]))
    return 1 if wa == wb or not (wa < wb or wb < wa) else 2 + len(wa ^ wb)


def _part(node_id: str) -> str:
    return node_id.rsplit("/", 1)[0] if node_id else ""


def _is_code_name(name: str, props: dict) -> bool:
    code = str((props or {}).get("code") or "")
    return bool(code) and normalize(name) == normalize(code)


def _key(typ: str, props: dict) -> str:
    """What makes it one thing whatever its name: a parameter's code. (A terminal's connector and terminal are not
    used: finds on several pages join them, "5; DI 0", "-X138; -X133", so they can't tell two terminals apart.)"""
    p = props or {}
    if typ == "Parameter" and p.get("code"):
        return normalize(str(p["code"]))
    return ""


def _ends(triples):
    """Each new (not yet in the graph) staged end: (type, part, name, id, props, [(triple, end)])."""
    items: dict[tuple, dict] = {}
    for t in triples:
        for end in ENDS:
            if getattr(t, f"{end}_existing"):
                continue
            typ, name, node_id = getattr(t, f"{end}_type"), getattr(t, f"{end}_name"), getattr(t, f"{end}_id")
            key = (typ, _part(node_id), name)
            item = items.setdefault(key, {"type": typ, "part": _part(node_id), "name": name, "id": node_id,
                                          "props": getattr(t, f"{end}_props") or {}, "uses": [], "keys": set()})
            item["uses"].append((t.pk, end))
            k = _key(typ, getattr(t, f"{end}_props"))
            if k:
                item["keys"].add(k)
    return list(items.values())


def find(triples) -> dict:
    """{groups: [{type, into, names: [{name, id, triples}]}], checks: [...]} for these staged triples."""
    items = _ends(triples)
    parent = list(range(len(items)))
    keys = [set(item["keys"]) for item in items]       # per cluster root: the keys it holds
    members = [[i] for i in range(len(items))]         # per cluster root: its items

    def fits(x, y):
        return closeness(items[x], items[y]) is not None

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # The closest pairs join first (the same code or key, then the same name or its initials, then the fewest extra
    # words), so a loose pair can't pull a group together before a close one has formed.
    pairs = []
    for i, a in enumerate(items):
        for j in range(i + 1, len(items)):
            b = items[j]
            if a["type"] == b["type"] and a["part"] == b["part"]:
                score = closeness(a, b)
                if score is not None:
                    pairs.append((score, i, j))
    for _, i, j in sorted(pairs):
        ri, rj = root(i), root(j)
        if ri == rj:
            continue
        if keys[ri] and keys[rj] and not keys[ri] & keys[rj]:
            continue                                      # two different codes or terminals: never one group
        # Every name of one group fits every name of the other: "PROFIBUS or PROFINET" holds both names but must not
        # join PROFIBUS and PROFINET into one group (2026-10-18).
        if all(fits(x, y) for x in members[ri] for y in members[rj]):
            parent[ri] = rj
            keys[rj] |= keys[ri]
            members[rj] += members[ri]
    clusters: dict[int, list] = {}
    for i, item in enumerate(items):
        clusters.setdefault(root(i), []).append(item)
    groups = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        # The name to keep: not a bare code, then the one used most, then the shortest.
        members.sort(key=lambda m: (_is_code_name(m["name"], m["props"]), -len(m["uses"]), len(m["name"])))
        groups.append({"type": members[0]["type"], "into": members[0]["name"],
                       "names": [{"name": m["name"], "id": m["id"], "triples": sorted({tid for tid, _ in m["uses"]})}
                                 for m in members]})
    groups.sort(key=lambda g: (g["type"], g["into"].lower()))
    return {"groups": groups, "checks": checks(triples)}


def checks(triples) -> list[dict]:
    """Terminals with no signal type, no direction and no terminal number: perhaps not a signal channel (a whole
    terminal strip, a power terminal, a control-word bit). Flagged for a look, never rejected by themselves."""
    out: dict[str, list[int]] = {}
    for t in triples:
        for end in ENDS:
            if getattr(t, f"{end}_type") != "IOPoint":
                continue
            p = getattr(t, f"{end}_props") or {}
            if not any(p.get(k) for k in ("signal_type", "direction", "terminal")):
                out.setdefault(getattr(t, f"{end}_name"), []).append(t.pk)
    return [{"type": "IOPoint", "name": n, "reason": "no signal type, direction or terminal given",
             "triples": sorted(set(ids))} for n, ids in sorted(out.items())]


@transaction.atomic
def merge(triples, type_: str, names: list[str], into: str) -> dict:
    """Rename the staged ends of `type_` called one of `names` to `into` (its id), then fold the triples that became
    identical (evidence, occurrences and properties kept). Only pending triples change."""
    pending = [t for t in triples if t.status == CandidateTriple.Status.PENDING]
    target = next(((getattr(t, f"{e}_id"), getattr(t, f"{e}_props")) for t in pending for e in ENDS
                   if getattr(t, f"{e}_type") == type_ and getattr(t, f"{e}_name") == into), None)
    if target is None:
        raise ValueError(f"no staged {type_} named {into!r}")
    into_id, into_props = target
    wanted, renamed = set(names) | {into}, 0
    for t in pending:
        changed = False
        for end in ENDS:
            if getattr(t, f"{end}_type") == type_ and getattr(t, f"{end}_name") in wanted \
                    and _part(getattr(t, f"{end}_id")) == _part(into_id):
                if getattr(t, f"{end}_name") != into or getattr(t, f"{end}_id") != into_id:
                    setattr(t, f"{end}_props", merge_properties(into_props, getattr(t, f"{end}_props")))
                    setattr(t, f"{end}_name", into)
                    setattr(t, f"{end}_id", into_id)
                    setattr(t, f"{end}_match", "new")
                    changed = True
        if changed:
            t.edited = True
            t.save()
            renamed += 1
    # Fold triples that are now the same (subject, link, object).
    seen: dict[tuple, CandidateTriple] = {}
    folded = 0
    for t in pending:
        if not any(getattr(t, f"{e}_type") == type_ and getattr(t, f"{e}_id") == into_id for e in ENDS):
            continue                      # only the triples of this merge are folded
        key = (t.mode, t.subject_id, t.predicate, t.object_id)
        if key not in seen:
            seen[key] = t
            continue
        keep = seen[key]
        keep.occurrences = (keep.occurrences or 1) + (t.occurrences or 1)
        keep.confidence = max(keep.confidence or 0, t.confidence or 0) or keep.confidence
        notes = differing(keep.subject_props, t.subject_props) + differing(keep.object_props, t.object_props)
        if notes and not keep.issue:      # P26b-2: the kept triple's value stays; the reviewer sees the other
            keep.issue = "; ".join(notes)[:255]
        keep.subject_props = merge_properties(keep.subject_props, t.subject_props)
        keep.object_props = merge_properties(keep.object_props, t.object_props)
        keep.evidence.add(*t.evidence.all())
        keep.save()
        t.delete()
        folded += 1
    return {"renamed": renamed, "folded": folded}

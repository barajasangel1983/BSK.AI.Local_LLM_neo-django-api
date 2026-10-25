"""P26b-2: a facet run on the real path (page choice, tables by row, the same windows and calls) that stages nothing,
and a score of what it found against the document's review. Test on a page reads text windows only and once, so it
can't tell whether a prompt change helps the table rows most ratings come from (2026-10-23).

    manage.py extract_dry_run --doc QUINT4-PS-... --anchor bsk:product-family:... --facet controls --runs 2
"""

from __future__ import annotations

import re
from collections import Counter

from django.test.utils import override_settings

from . import extraction, page_suggest
from .identity import normalize
from .models import CandidateTriple, PromptPreset

# What the G120C cleanup removed from the terminals (2026-10-23): kept general, no vendor words.
_BIT = re.compile(r"\bbit\b|\b(control|status) word\b", re.I)
_STRIP = re.compile(r"\bterminal strip\b|^\s*-?x\d+\s*$|\bconnector\b$", re.I)
_GENERIC = re.compile(r"^\s*(a\s+)?(digital|analog(ue)?)\s+(input|output)s?\s*(for .*)?$", re.I)
_CONTACT = re.compile(r"\b(nc|no|com|pos|neg)\s*$|[+-]\s*$", re.I)


def preset_for(facet: str, template: str | None = None) -> PromptPreset:
    run = extraction.FACET_RUN_BY_KEY[facet]
    live = PromptPreset.objects.get(mode="schema", name=run["preset"])
    if template is None:
        return live
    return PromptPreset(name=live.name, mode="schema", template=template, version=live.version + 1)   # never saved


def run(doc, *, facet: str, anchor_id: str, asset_id: str = "", pages: str = "", template: str | None = None,
        old_code: bool = False) -> dict:
    """One dry facet run: {pages, staged, failed, disagreed, triples: [...]}."""
    preset = preset_for(facet, template)
    spec = extraction.FACET_RUN_BY_KEY[facet]
    params = extraction.validate_params({"asset_id": asset_id, "anchor_id": anchor_id, "anchor_level": "part"})
    if not pages:
        pages = page_suggest.suggest(doc, preset=preset, asset_id=asset_id, anchor_id=anchor_id,
                                     follow_languages=params["follow_languages"])["pages"]
    params.update(mode="schema", presets={}, preset_objects={"schema": preset}, pages=extraction.parse_pages(pages),
                  facet_runs=[], leave_types=[], keep_links=list(spec.get("keeps") or ()), dry_run=True)
    if spec.get("chunking"):
        from ingestion.chunking import resolve_params
        params["chunking"] = {"strategy": spec["chunking"]["strategy"],
                              "params": resolve_params(spec["chunking"]["strategy"], spec["chunking"]["params"])}
    with override_settings(GRAPH_SINGLE_VALUES=not old_code):
        out = extraction._extract(None, doc, params, label=facet)
    out["pages"] = pages
    out["triples"] = [_plain(t) for t in out["triples"]]
    return out


def _plain(t: CandidateTriple) -> dict:
    return {k: getattr(t, k) for k in ("subject_name", "subject_type", "subject_id", "predicate", "object_name",
                                       "object_type", "object_id", "subject_props", "object_props", "issue",
                                       "page_start", "page_end", "confidence")}


def _key(t: dict) -> tuple:
    return (t["predicate"], t["object_type"], normalize(t["object_name"]))


def labels(doc, preset_name: str) -> dict:
    """The document's reviewed triples of this prompt: {key: status}; rejected wins over approved for one key."""
    out = {}
    for t in CandidateTriple.objects.filter(document=doc, preset_name=preset_name).exclude(status="pending"):
        k = (t.predicate, t.object_type, normalize(t.object_name))
        if out.get(k) != "rejected":
            out[k] = t.status
    return out


def score(result: dict, review: dict) -> dict:
    found = {_key(t) for t in result["triples"]}
    approved = {k for k, s in review.items() if s == "approved"}
    rejected = {k for k, s in review.items() if s == "rejected"}
    io = [t for t in result["triples"] if t["object_type"] == "IOPoint" and t["predicate"] == "HAS_IO"]
    io_names = sorted({t["object_name"] for t in io})
    joined = [f"{t['object_name']}.{k}" for t in result["triples"] for k, v in (t["object_props"] or {}).items()
              if isinstance(v, str) and "; " in v and k not in extraction.JOINED_PROPS]
    return {
        "triples": len(found),
        "approved_found": len(found & approved), "approved_total": len(approved),
        "rejected_back": len(found & rejected), "rejected_total": len(rejected),
        "rejected_back_names": sorted(k[2] for k in found & rejected),
        "approved_missed_names": sorted(k[2] for k in approved - found),
        "new": len(found - approved - rejected),
        "iopoints": len(io_names), "iopoint_ids": len({t["object_id"] for t in io}),
        "bits": [n for n in io_names if _BIT.search(n)],
        "strips": [n for n in io_names if _STRIP.search(n)],
        "generic": [n for n in io_names if _GENERIC.match(n)],
        "contacts": [n for n in io_names if _CONTACT.search(n)],
        "joined_values": joined,
        "disagreed": result.get("disagreed", 0),
        "failed_calls": result.get("failed", 0),
        "strips_dropped": result.get("strips_dropped", []),
    }


def summary(scores: list[dict]) -> dict:
    """The runs of one side, side by side (min / max of each number)."""
    out = {}
    for k in ("triples", "approved_found", "rejected_back", "new", "iopoints", "iopoint_ids", "disagreed", "failed_calls"):
        values = [s[k] for s in scores]
        out[k] = values
    for k in ("bits", "strips", "generic", "contacts", "joined_values", "strips_dropped"):
        out[k] = [len(s[k]) for s in scores]
    out["approved_total"] = scores[0]["approved_total"] if scores else 0
    out["rejected_total"] = scores[0]["rejected_total"] if scores else 0
    out["rejected_back_names"] = Counter(n for s in scores for n in s["rejected_back_names"]).most_common()
    return out

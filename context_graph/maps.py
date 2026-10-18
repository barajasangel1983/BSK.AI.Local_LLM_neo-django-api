"""Which parameter of a device shows what a PLC signal measures (P22c-3c, `manage.py suggest_maps`).

A drive's manual lists hundreds of readouts (the ACS880's parameter reference is 436 pages); extracting them all
would bury the graph. Instead, for each PLC signal of the device's assembly (the device, its parent and the parent's
other parts: the drive, the motor it drives, the gearbox) with a unit, the device's manuals are searched for the
readout of that quantity, the model is asked one question over the excerpts, and the answer is checked (its number in
the excerpts, its unit the signal's) before anything is staged:

    <device> -HAS_PARAMETER-> <parameter> (Readout, code, unit)      and      <parameter> -MAPS_TO-> <signal>

with the page as evidence, pending Angel's review. No answer, a unit that disagrees, or a number not in the text:
nothing staged, the reason reported. Rules for any maker whose manual lists its readouts.
"""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace

from django.conf import settings

from . import extraction, facets, registry
from .driver import session
from .identity import EntityIndex
from .models import CandidateTriple, Evidence

logger = logging.getLogger("chat")

PROMPT_NAME = "suggest_maps"
PROMPT_VERSION = 1
EXCERPTS = 10
UNITS = {"rpm": "rpm", "1/min": "rpm", "min-1": "rpm", "r/min": "rpm", "%": "%", "kw": "kw", "w": "w", "a": "a",
         "v": "v", "hz": "hz", "nm": "nm", "°c": "°c", "degc": "°c", "bar": "bar", "kg/h": "kg/h", "l/h": "l/h"}

SYSTEM = """You find, in a device's manual, the parameter that shows a measured quantity (a readout, an actual value).
Reply with one JSON object and nothing else:
{"code": "...", "name": "...", "unit": "...", "page": number, "quote": "..."}
- "code" and "name" exactly as the excerpts write them; "unit" as the excerpts give it for that parameter.
- A readout of the measured value (an actual value, a signal shown), never a setting, a limit, a reference or a
  scaling; the one closest to what the PLC signal measures.
- "quote": the few words of the excerpt that show it.
- If no excerpt names such a parameter with its number, reply {"code": "", "reason": "..."}."""


NOT_A_READOUT = re.compile(r"\b(ref|reference|setpoint|set point|limit|scaling|max|min|maximum|minimum)\b", re.I)


def code_in(code: str, text: str) -> str:
    """The code as the text prints it, or "": "1.1" is found as "01.01", "p310" as "p0310" (numbers compared, not
    characters — "1.1" inside "01.10" is not a match)."""
    parts = re.findall(r"[A-Za-z]+|\d+", code)
    if not parts:
        return ""
    pattern = r"(?<![\w.])" + r"[.\s]?".join(re.escape(p) if p.isalpha() else rf"0*{int(p)}" for p in parts) + r"(?![\w]|\.\d)"
    m = re.search(pattern, text, re.I)
    return m.group(0) if m else ""


def unit_key(unit: str) -> str:
    u = re.sub(r"\s+", "", str(unit or "")).lower()
    return UNITS.get(u, u)


def assembly_signals(part_id: str) -> list[dict]:
    """The PLC signals of a device's assembly: the device, its parent, the parent's other parts — with a unit."""
    with session() as s:
        rows = s.run(
            "MATCH (d:Component {id: $id}) OPTIONAL MATCH (p:Component)-[:HAS_COMPONENT]->(d) "
            "OPTIONAL MATCH (p)-[:HAS_COMPONENT]->(sib:Component) "
            "WITH d, collect(DISTINCT p) + collect(DISTINCT sib) AS more "
            "UNWIND [d] + more AS c "
            "MATCH (c)-[:MONITORED_BY]->(sg:Signal) WHERE sg.unit IS NOT NULL AND sg.unit <> '' "
            "WITH sg, collect(DISTINCT c.name) AS on_parts "
            "RETURN sg.id AS id, sg.name AS name, sg.subtype AS subtype, sg.unit AS unit, on_parts ORDER BY name",
            id=part_id).data()
    return rows


def device(part_id: str) -> dict:
    with session() as s:
        rec = s.run("MATCH (d:Component {id: $id}) OPTIONAL MATCH (d)-[:DOCUMENTED_BY]->(doc:Document) "
                    "RETURN d.name AS name, d.model AS model, d.manufacturer AS maker, "
                    "collect(DISTINCT doc.doc_key) AS doc_keys", id=part_id).single()
    if rec is None:
        raise ValueError(f"unknown part {part_id!r}")
    return dict(rec)


def _excerpts(signal: dict, doc_keys: list[str]) -> list:
    from chat.retrieval import search_v2
    quantity = (signal.get("subtype") or signal["name"]).lower()
    queries = [f"{quantity} actual value {signal['unit']}", f"actual values {quantity}", f"actual {quantity} parameter",
               signal["name"]]
    found, seen = [], set()
    for q in queries:
        for c in search_v2(q, top_n=5, doc_keys=doc_keys).chunks:
            if c.id not in seen:
                seen.add(c.id)
                found.append(c)
    return found[:EXCERPTS]


def look_up(dev: dict, signal: dict) -> dict:
    """{state: found | none, code, name, unit, page, quote, chunk, reason} for one signal."""
    chunks = _excerpts(signal, dev["doc_keys"])
    if not chunks:
        return {"state": "none", "reason": "no excerpt found in the device's manuals"}
    text = "\n\n".join(f"[p.{c.page_start}] {c.text[:1500]}" for c in chunks)
    user = (f"Device: {dev['name']} ({', '.join(x for x in (dev.get('maker'), dev.get('model')) if x)}).\n"
            f"PLC signal: \"{signal['name']}\" — {signal.get('subtype') or 'a quantity'} in {signal['unit']}.\n"
            f"Which parameter of this device shows it?\n\nExcerpts:\n{text}")
    try:
        answer = json.loads(extraction.call_llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]) or "{}")
    except (ValueError, TypeError):
        return {"state": "none", "reason": "the model's answer was not JSON"}
    code = str(answer.get("code") or "").strip()
    if not code:
        return {"state": "none", "reason": str(answer.get("reason") or "no parameter named")[:200]}
    found = next(((c, code_in(code, c.text)) for c in chunks if code_in(code, c.text)), None)
    if found is None:
        return {"state": "none", "reason": f"{code} is not in the excerpts (not staged)", "code": code}
    chunk, code = found
    code = extraction.normalize_code(code)[0]             # one form: a parsed "1.1" is ABB's 01.01
    name = str(answer.get("name") or code)
    if NOT_A_READOUT.search(name):
        return {"state": "none", "reason": f"{code} {name} is a reference, setpoint or limit, not a readout (not staged)",
                "code": code}
    if unit_key(answer.get("unit")) != unit_key(signal["unit"]):
        return {"state": "none", "reason": f"unit {answer.get('unit')!r} is not the signal's {signal['unit']!r} (not staged)",
                "code": code}
    return {"state": "found", "code": code, "name": name[:200], "unit": signal["unit"],
            "page": chunk.page_start, "quote": str(answer.get("quote") or "")[:300], "chunk": chunk}


def suggest(part_id: str, dry_run: bool = False) -> list[dict]:
    """Look up each signal of the device's assembly; stage the found ones (unless dry_run). One result per signal."""
    from ingestion.models import Document

    dev = device(part_id)
    if not dev["doc_keys"]:
        raise ValueError(f"{dev['name']} has no manual (DOCUMENTED_BY)")
    schema = registry.active_schema()
    if "MAPS_TO" not in schema.relationship_types:
        raise ValueError("the active schema has no MAPS_TO (schema v6)")
    index = EntityIndex.load()
    scope_key = part_id.split(":", 2)[2].split("/")[0]
    part = index.part_scope(part_id, scope_key)
    preset = SimpleNamespace(name=PROMPT_NAME, version=PROMPT_VERSION)
    results = []
    for sig in assembly_signals(part_id):
        r = {"signal": sig["name"], "unit": sig["unit"], **look_up(dev, sig)}
        r.pop("chunk", None) if r["state"] != "found" else None
        results.append(r)
        if r["state"] != "found" or dry_run:
            continue
        chunk = r.pop("chunk")
        doc = Document.objects.filter(doc_key=chunk.asset_id).first() or Document.objects.filter(filename=chunk.source).first()
        window = SimpleNamespace(page_start=chunk.page_start, page_end=chunk.page_end, section_path=chunk.section_path,
                                 text=chunk.text)
        has = extraction._candidate(
            {"subject": dev["name"], "subject_type": "Component", "predicate": "HAS_PARAMETER", "object": r["name"],
             "object_type": "Parameter", "object_subtype": "Readout",
             "object_properties": {"code": r["code"], "unit": r["unit"], "access": "read"}, "confidence": 0.9},
            "schema", schema, index, scope_key, doc, None, window, 0, preset, part=part)
        maps = CandidateTriple(
            document=doc, mode="schema", source="text", subject_name=has.object_name, subject_type="Parameter",
            subject_id=has.object_id, subject_existing=has.object_existing, subject_props=has.object_props,
            predicate="MAPS_TO", object_name=sig["name"], object_type="Signal", object_id=sig["id"], object_existing=True,
            object_match="id", subject_match=has.object_match or "new", confidence=0.9,
            page_start=chunk.page_start, page_end=chunk.page_end, section_path=chunk.section_path,
            evidence_text=chunk.text[:extraction.EVIDENCE_CHARS], model=settings.DGX_CHAT_MODEL,
            preset_name=PROMPT_NAME, preset_version=PROMPT_VERSION, schema_version=schema.version,
            auto_fix=f"suggested: {r['code']} shows {sig['name']} ({sig['unit']})")
        if CandidateTriple.objects.filter(subject_id=maps.subject_id, predicate="MAPS_TO", object_id=sig["id"]).exists():
            r["state"], r["reason"] = "already", "already staged or reviewed"
            continue
        if has.object_id != part_id and not CandidateTriple.objects.filter(
                subject_id=has.subject_id, predicate="HAS_PARAMETER", object_id=has.object_id).exists():
            facets.stamp([has], schema)
            has.save()
            _evidence(has, doc, chunk, r)
        facets.stamp([maps], schema)
        maps.save()
        _evidence(maps, doc, chunk, r)
        r["staged"] = [has.pk, maps.pk] if has.pk else [maps.pk]
    return results


def _evidence(t: CandidateTriple, doc, chunk, r: dict) -> None:
    t.evidence.add(Evidence.objects.create(
        source_kind="text", document=doc, page_start=chunk.page_start, page_end=chunk.page_end,
        section_path=chunk.section_path or [], extractor="suggest-maps", model=settings.DGX_CHAT_MODEL,
        prompt=f"{PROMPT_NAME} v{PROMPT_VERSION}", excerpt=(r.get("quote") or chunk.text)[:1500], confidence=0.9))

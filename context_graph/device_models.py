"""Device models and product families proposed from an asset's components (P26a, schema v7).

The proposal is a CSV for the "Device models" import template: one row per installed part with a manufacturer,
its model (the Component's `model`, as the BOM wrote it) and, when the model names a variant ("SINAMICS G120C,
7.5 kW, 400 V, PROFINET"), its family (the text before the first comma). Parts with the same manufacturer and
model share one model (the twins); models with the same family text share one family (the two G120C drives).
Angel confirms or edits the names and the grouping before the file is imported and its triples approved.
"""

from __future__ import annotations

import csv
import io
import re

COLUMNS = ["Component", "Name", "Manufacturer", "Family Key", "Family", "Model Key", "Model", "Order Code"]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", str(text).lower()).strip("-.")


def order_code(part_number: str) -> str:
    """The part number when it identifies the model; "PM6 (configure)" and "7F2C (product root)" do not."""
    pn = str(part_number or "").strip()
    return "" if "(" in pn else pn


def propose(components: list[dict], asset_key: str) -> list[dict]:
    """One row per component that has a manufacturer and a model, sorted by family, model and part."""
    rows = []
    for c in components:
        p = c.get("properties") or {}
        manufacturer, model = str(p.get("manufacturer") or "").strip(), str(p.get("model") or "").strip()
        if not manufacturer or not model:
            continue        # OEM parts without a manufacturer's model are not device models
        prefix = f"{asset_key}/"
        key = c["id"].split(":", 2)[2]
        if "/" in key[len(prefix):]:
            continue        # a manual's sub-part (water-flowmeter/Seals): moved to its family in P26c (D5)
        family = model.split(",", 1)[0].strip() if "," in model else ""
        rows.append({
            "Component": key[len(prefix):] if key.startswith(prefix) else key,
            "Name": c.get("name", ""),
            "Manufacturer": manufacturer,
            "Family Key": f"{slug(manufacturer)}/{slug(family)}" if family else "",
            "Family": family,
            "Model Key": f"{slug(manufacturer)}/{slug(model)}",
            "Model": model,
            "Order Code": order_code(p.get("part_number")),
        })
    return sorted(rows, key=lambda r: (r["Family Key"] or r["Model Key"], r["Model Key"], r["Component"]))


def to_csv(rows: list[dict]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def summary(rows: list[dict]) -> str:
    """Families, their models and the parts of each, for the person who confirms the list."""
    families: dict[str, dict[str, list[str]]] = {}
    names = {}
    for r in rows:
        fam = r["Family Key"] or "(no family: a single model)"
        names[fam] = r["Family"] or fam
        families.setdefault(fam, {}).setdefault(r["Model"], []).append(r["Component"])
    lines = [f"{len(families)} families or single models, {len({r['Model Key'] for r in rows})} models, "
             f"{len(rows)} installed parts"]
    for fam in sorted(families, key=lambda f: (f.startswith("("), f)):
        lines.append(f"- {names[fam]}")
        for model, parts in sorted(families[fam].items()):
            lines.append(f"    - {model}: {', '.join(sorted(parts))}")
    return "\n".join(lines)


# --- levels (P26b) -------------------------------------------------------------------------------------------------

KINDS = {"component": "part", "device-model": "model", "product-family": "family"}
LABELS = {"part": "Component", "model": "DeviceModel", "family": "ProductFamily"}


def kind_of(node_id: str) -> str:
    """"part", "model" or "family" for a component, device model or product family id; "" otherwise."""
    parts = str(node_id or "").split(":", 2)
    return KINDS.get(parts[1], "") if len(parts) == 3 else ""


def levels(node_id: str) -> dict:
    """The levels around an installed part, a model or a family: {part, model, family, models, parts}, each node
    {id, name, label, properties} (None or [] where there is none). `models` are the family's (or the one model),
    `parts` the installed parts of those models. Raises ValueError for an unknown node."""
    from . import repository
    from .driver import session

    kind = kind_of(node_id)
    if not kind:
        raise ValueError(f"{node_id!r} is not a component, device model or product family id")

    def node(n):
        return repository.serialize_node(n) if n is not None else None

    with session() as s:
        row = s.run(f"MATCH (n:{LABELS[kind]} {{id: $id}}) "
                    "OPTIONAL MATCH (n)-[:INSTANCE_OF]->(m:DeviceModel) "
                    "WITH n, m, CASE WHEN m IS NULL THEN n ELSE m END AS v "
                    "OPTIONAL MATCH (v)-[:VARIANT_OF]->(f:ProductFamily) "
                    "RETURN n, m, f LIMIT 1", id=node_id).single()
        if row is None:
            raise ValueError(f"{node_id} is not in the graph")
        n, m, f = row["n"], row["m"], row["f"]
        part = n if kind == "part" else None
        model = n if kind == "model" else m
        family = n if kind == "family" else f
        if family is not None:
            models = [r["m"] for r in s.run("MATCH (m:DeviceModel)-[:VARIANT_OF]->(:ProductFamily {id: $id}) "
                                            "RETURN m ORDER BY m.name", id=family["id"])]
        else:
            models = [model] if model is not None else []
        parts = [r["c"] for r in s.run("MATCH (c:Component)-[:INSTANCE_OF]->(m:DeviceModel) WHERE m.id IN $ids "
                                       "RETURN c ORDER BY c.name", ids=[x["id"] for x in models])]
    return {"part": node(part), "model": node(model), "family": node(family),
            "models": [node(x) for x in models], "parts": [node(x) for x in parts]}


def pick(found: dict, want: str) -> dict:
    """The node of level `want` ("part" | "model" | "family"), falling back to the next lower level that exists (a
    model without a family is run on the model, a part without a model on the part), then to the higher ones (a
    family's ratings, asked for its model, stay on the family)."""
    order = ["family", "model", "part"]
    i = order.index(want)
    for level in order[i:] + order[:i][::-1]:
        if found.get(level):
            return found[level]
    raise ValueError("no part, model or family to run on")

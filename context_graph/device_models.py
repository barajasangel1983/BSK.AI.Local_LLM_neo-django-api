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

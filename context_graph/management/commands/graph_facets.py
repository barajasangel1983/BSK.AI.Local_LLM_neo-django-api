"""Tag the graph's nodes and the staged triples with their knowledge facets (P22b).

The facets come from the active schema's `facets:` mapping (schema v5). The default is a dry run: counts per
facet and type, nodes with two or more facets, untagged nodes, samples; nothing is written. `--apply` sets
`facets` on the nodes (no other property, node or link changes: the node and link counts are checked before
and after) and on every schema-mode CandidateTriple. Idempotent.

    manage.py graph_facets [--schema-file industrial_v5.yaml] [--report path]   # dry run
    manage.py graph_facets --apply

`--schema-file` previews a mapping that isn't active yet (dry run only).
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import yaml
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from context_graph import facets, registry
from context_graph.driver import session
from context_graph.models import CandidateTriple
from context_graph.schema import Schema

COUNTS = "MATCH (n:Entity) WITH count(n) AS nodes OPTIONAL MATCH (:Entity)-[r]->(:Entity) RETURN nodes, count(r) AS links"


class Command(BaseCommand):
    help = "Compute (dry run) or write (--apply) the knowledge facets of the graph's nodes and the staged triples."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true", help="write the facets (default: dry run)")
        parser.add_argument("--schema-file", default="", help="preview this schema file's mapping (dry run only)")
        parser.add_argument("--report", default="", help="also write the report to this file")
        parser.add_argument("--samples", type=int, default=8, help="sample nodes per facet in the report")

    def handle(self, *args, apply=False, schema_file="", report="", samples=8, **options):
        if schema_file:
            if apply:
                raise CommandError("--schema-file is for a dry run; activate the schema, then --apply")
            schema = Schema.from_definition(yaml.safe_load(Path(schema_file).read_text()))
            source = f"file {schema_file}"
        else:
            schema = registry.active_schema()
            source = f"active schema v{schema.version}"
        mapping = facets.mapping_of(schema)
        if mapping is None:
            raise CommandError(f"{source} has no facets: mapping (P22b needs schema v5)")

        with session() as s:
            before = s.execute_read(lambda tx: tx.run(COUNTS).single().data())
            rows = s.execute_read(facets.plan, mapping)
        triples = list(CandidateTriple.objects.filter(mode="schema"))
        new_triple_facets = {t.pk: facets.for_triple(mapping, t) for t in triples}

        lines = self.report(source, before, rows, triples, new_triple_facets, samples)
        if apply:
            with session() as s:
                changed = s.execute_write(facets.write, rows)
                after = s.execute_read(lambda tx: tx.run(COUNTS).single().data())
            if after != before:
                raise CommandError(f"node/link counts changed during the write: {before} -> {after} (investigate)")
            stale = [t for t in triples if (t.facets or []) != new_triple_facets[t.pk]]
            for t in stale:
                t.facets = new_triple_facets[t.pk]
            CandidateTriple.objects.bulk_update(stale, ["facets"], batch_size=500)
            lines += ["", f"APPLIED: {changed} node(s) and {len(stale)} staged triple(s) updated; "
                          f"nodes {after['nodes']}, links {after['links']} (unchanged)."]
        else:
            lines += ["", "Dry run: nothing written. Write with: manage.py graph_facets --apply"]

        text = "\n".join(lines)
        self.stdout.write(text)
        if report:
            Path(report).write_text(text + "\n")

    def report(self, source, counts, rows, triples, new_triple_facets, samples) -> list[str]:
        per_facet, per_type, untagged, multi, combos = Counter(), {}, Counter(), 0, Counter()
        by_facet = {f: [] for f in facets.FACETS}
        for r in rows:
            fs = r["facets"]
            if not fs:
                untagged[r["label"]] += 1
                continue
            multi += len(fs) > 1
            combos[" + ".join(fs)] += 1
            for f in fs:
                per_facet[f] += 1
                per_type.setdefault(f, Counter())[r["label"]] += 1
                by_facet[f].append(r)
        changed = sum(1 for r in rows if r["facets"] != r["current"])
        tagged = len(rows) - sum(untagged.values())
        out = [f"P22b knowledge facets — {timezone.now():%Y-%m-%d %H:%M} UTC — mapping from {source}",
               f"Graph: {counts['nodes']} nodes, {counts['links']} links.",
               f"Tagged: {tagged} nodes ({multi} with 2+ facets); untagged (core): {sum(untagged.values())}; "
               f"nodes whose stored facets would change: {changed}.",
               "", "Per facet (nodes, by type):"]
        for f in facets.FACETS:
            types = ", ".join(f"{label} {n}" for label, n in per_type.get(f, Counter()).most_common())
            out.append(f"  {f:<16}{per_facet[f]:>6}   {types}")
        out += ["", "Combinations:"] + [f"  {n:>6}  {c}" for c, n in combos.most_common()]
        out += ["", "Untagged (core) by type: " + ", ".join(f"{label} {n}" for label, n in untagged.most_common())]
        rng = random.Random(22)
        out += ["", f"Samples ({samples} per facet):"]
        for f in facets.FACETS:
            picked = rng.sample(by_facet[f], min(samples, len(by_facet[f])))
            for r in picked:
                out.append(f"  [{f}] {r['label']}{' / ' + str(r['subtype']) if r['subtype'] else ''}: "
                           f"{r['name']}  -> {', '.join(r['facets'])}")
        t_counter, t_none, t_changed = Counter(), 0, 0
        for t in triples:
            fs = new_triple_facets[t.pk]
            t_none += not fs
            t_changed += (t.facets or []) != fs
            for f in fs:
                t_counter[t.status, f] += 1
        out += ["", f"Staged triples (schema mode): {len(triples)}; with no facet: {t_none}; would change: {t_changed}."]
        for status in CandidateTriple.Status.values:
            row = ", ".join(f"{f} {t_counter[status, f]}" for f in facets.FACETS if t_counter[status, f])
            if row:
                out.append(f"  {status:<9} {row}")
        return out

"""Write evidence IDs and aliases of already-approved triples into Neo4j (P8a, run once after migrating).

Approving now stores each triple's evidence IDs on the nodes and edges it writes, and
adds a differing name as an alias. This applies the same to triples approved before P8a.
Idempotent: running it twice changes nothing.

    manage.py graph_backfill_evidence [--dry-run]
"""

from django.core.management.base import BaseCommand

from context_graph.driver import session
from context_graph.identity import normalize
from context_graph.models import CandidateTriple
from context_graph.triples import _evidence_ids, apply_alias


class Command(BaseCommand):
    help = "Copy evidence IDs and aliases of approved triples onto their Neo4j nodes / edges."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, dry_run=False, **options):
        approved = list(CandidateTriple.objects.filter(status=CandidateTriple.Status.APPROVED)
                        .prefetch_related("evidence").order_by("id"))
        stats = {"triples": len(approved), "edges": 0, "nodes": 0, "aliases": 0}

        def apply(tx, t):
            ev = _evidence_ids(t)
            if t.layer == CandidateTriple.Layer.LAB:
                stats["edges"] += tx.run(
                    "MATCH ()-[r:LAB_RELATION {triple_id: $tid}]->() SET r.evidence_ids = $ev RETURN count(r) AS n",
                    tid=t.pk, ev=ev).single()["n"]
                return
            stats["edges"] += tx.run(
                "MATCH (:Entity)-[r]->(:Entity) WHERE $tid IN coalesce(r.triple_ids, []) "
                "SET r.evidence_ids = [x IN coalesce(r.evidence_ids, []) WHERE NOT x IN $ev] + $ev RETURN count(r) AS n",
                tid=t.pk, ev=ev).single()["n"]
            aliased = dict(t.applied_aliases or {})
            for node_id, name in ((t.subject_id, t.subject_name), (t.object_id, t.object_name)):
                row = tx.run("MATCH (n:Entity {id: $id}) "
                             "SET n.evidence_ids = [x IN coalesce(n.evidence_ids, []) WHERE NOT x IN $ev] + $ev "
                             "RETURN n.name AS name, coalesce(n.aliases, []) AS aliases", id=node_id, ev=ev).single()
                if row is None:
                    continue
                stats["nodes"] += 1
                if any(normalize(a) == normalize(name or "") for a in aliased.get(node_id, [])):
                    continue                                   # already claimed (idempotent)
                added = not any(normalize(a) == normalize(name or "") for a in row["aliases"])
                claimed = apply_alias(tx, node_id, name, row["name"], row["aliases"])
                if claimed:
                    aliased.setdefault(node_id, []).append(claimed)
                    stats["aliases"] += int(added)
            t.applied_aliases = aliased
            # Save now: the next triple's alias claim checks the approved triples' claims.
            CandidateTriple.objects.filter(pk=t.pk).update(applied_aliases=aliased)

        if dry_run:
            self.stdout.write(f"would update {len(approved)} approved triples")
            return
        with session() as s:
            for t in approved:
                s.execute_write(apply, t)
        self.stdout.write(self.style.SUCCESS(
            f"{stats['triples']} approved triples: evidence on {stats['edges']} edges / {stats['nodes']} node ends, "
            f"{stats['aliases']} aliases added"))

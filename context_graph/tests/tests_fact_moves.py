"""P26c: moving installed parts' manual facts to their family or model.

Without a graph: the plan on a small in-memory graph shaped like the two G120C drives (levels, merges, joined values,
conflicts and their choices, sub-facts, shared facts, machine links, stays). Against a throwaway Neo4j
(GRAPH_TEST_URI): apply (ids, merged_ids, links, the triples' anchors), old ids resolving (get_node, identity,
approval) and undo putting every node and link back."""

import os
import unittest
from pathlib import Path

import yaml
from django.test import TestCase, override_settings

from context_graph import driver, extraction, fact_moves, registry, repository, triples
from context_graph.fact_moves import Graph
from context_graph.identity import EntityIndex
from context_graph.models import CandidateTriple, FactMove, FactMoveBatch
from ingestion.models import Document

V7 = yaml.safe_load((Path(extraction.__file__).parent / "schemas" / "industrial_v7.yaml").read_text())
ASSET = "bsk:asset:EXTR01"
FAMILY = "bsk:product-family:siemens/sinamics-g120c"
M75 = "bsk:device-model:siemens/g120c-7.5"
M55 = "bsk:device-model:siemens/g120c-5.5"
CUTTER = "bsk:component:EXTR01/cutter-drive"
FEED = "bsk:component:EXTR01/feeder-screw-drive"


def text(**props):
    return {"source": "text", "doc_key": "G120C", "page_start": 7, **props}


def g120c() -> Graph:
    """Both drives read from one manual, as in the live graph before P26c."""
    g = Graph()

    def node(i, label, **props):
        g.nodes[i] = {"label": label, "labels": ["Entity", label], "props": {"id": i, **props}}

    def rel(a, t, b, **props):
        g.rels.append({"eid": f"r{len(g.rels)}", "type": t, "a": a, "b": b, "props": props})

    node(ASSET, "Asset", name="Extruder", source="seed:extr01-v1")
    node(FAMILY, "ProductFamily", name="SINAMICS G120C")
    node(M75, "DeviceModel", name="G120C 7.5 kW")
    node(M55, "DeviceModel", name="G120C 5.5 kW")
    for part, name, model in ((CUTTER, "Cutter drive", M75), (FEED, "Feed screw drive", M55)):
        node(part, "Component", name=name, source="structured")
        rel(ASSET, "HAS_COMPONENT", part, source="structured")
        rel(part, "INSTANCE_OF", model)
    rel(M75, "VARIANT_OF", FAMILY)
    rel(M55, "VARIANT_OF", FAMILY)
    for part in (CUTTER, FEED):
        key = part.split(":", 2)[2]
        node(f"bsk:parameter:{key}/p1082", "Parameter", name="Maximum speed", code="p1082", default="1500")
        rel(part, "HAS_PARAMETER", f"bsk:parameter:{key}/p1082", **text(triple_ids=[1 if part == CUTTER else 2]))
        node(f"bsk:specification:{key}/Ambient-temperature", "Specification", name="Ambient temperature", max="40", unit="°C")
        rel(part, "HAS_SPEC", f"bsk:specification:{key}/Ambient-temperature", **text())
        node(f"bsk:specification:{key}/Rated-power", "Specification", name="Rated power",
             value="7.5" if part == CUTTER else "5.5", unit="kW")
        rel(part, "HAS_SPEC", f"bsk:specification:{key}/Rated-power", **text())
    node(f"bsk:parameter:EXTR01/cutter-drive/p1058", "Parameter", name="Jog 1 speed setpoint", code="p1058", default="150")
    node(f"bsk:parameter:EXTR01/feeder-screw-drive/p1058", "Parameter", name="Jogging 1 speed setpoint", code="p1058",
         default="150 rpm", basis="Table 7")
    node(f"bsk:parameter:EXTR01/cutter-drive/p0756", "Parameter", name="Analog input type", code="p0756", default="3")
    node(f"bsk:parameter:EXTR01/feeder-screw-drive/p0756", "Parameter", name="Analog input type", code="p0756", default="4")
    for i in ("p1058", "p0756"):
        rel(CUTTER, "HAS_PARAMETER", f"bsk:parameter:EXTR01/cutter-drive/{i}", **text())
        rel(FEED, "HAS_PARAMETER", f"bsk:parameter:EXTR01/feeder-screw-drive/{i}", **text())
    # an alarm with its procedure (reached through ADDRESSES), a manual sub-part with its rating
    node("bsk:alarm:EXTR01/F30001", "Alarm", name="Overcurrent", code="F30001")
    node("bsk:procedure:EXTR01/cutter-drive/F30001-what-to-do", "Procedure", name="Overcurrent: what to do")
    rel(CUTTER, "HAS_ALARM", "bsk:alarm:EXTR01/F30001", **text())
    rel("bsk:procedure:EXTR01/cutter-drive/F30001-what-to-do", "ADDRESSES", "bsk:alarm:EXTR01/F30001", **text())
    node("bsk:component:EXTR01/Fan-unit", "Component", name="Fan unit", source="text")
    node("bsk:specification:EXTR01/Fan-unit/Air-flow", "Specification", name="Air flow", value="20", unit="m3/h")
    rel(CUTTER, "HAS_COMPONENT", "bsk:component:EXTR01/Fan-unit", **text())
    rel("bsk:component:EXTR01/Fan-unit", "HAS_SPEC", "bsk:specification:EXTR01/Fan-unit/Air-flow", **text())
    # a machine fact: stays; a family parameter's PLC link: gets `instance`
    node("bsk:signal:EXTR01/cutter-speed", "Signal", name="Cutter speed")
    rel(CUTTER, "MONITORED_BY", "bsk:signal:EXTR01/cutter-speed", source="structured")
    rel("bsk:parameter:EXTR01/cutter-drive/p1082", "MAPS_TO", "bsk:signal:EXTR01/cutter-speed", **text())
    # a procedure the asset also owns: shared, stays
    node("bsk:procedure:EXTR01/Lockout", "Procedure", name="Lockout")
    rel(CUTTER, "HAS_PROCEDURE", "bsk:procedure:EXTR01/Lockout", **text())
    rel(ASSET, "HAS_PROCEDURE", "bsk:procedure:EXTR01/Lockout", source="structured")
    node("bsk:document:G120C", "Document", name="g120c.pdf", doc_key="G120C")
    return g.index()


def by_old(p):
    return {m["old_id"]: m for m in p["moves"]}


class PlanTests(TestCase):
    def setUp(self):
        self.g = g120c()
        self.p = fact_moves.plan(self.g, FAMILY)
        self.m = by_old(self.p)

    def test_targets(self):
        t = next(t for t in fact_moves.targets(self.g) if t["id"] == FAMILY)
        self.assertEqual((t["level"], t["models"], t["parts"]), ("family", [M55, M75], [CUTTER, FEED]))

    def test_levels_and_new_ids(self):
        self.assertEqual(self.m["bsk:alarm:EXTR01/F30001"]["new_id"], "bsk:alarm:siemens/sinamics-g120c/F30001")
        self.assertEqual(self.m["bsk:procedure:EXTR01/cutter-drive/F30001-what-to-do"]["new_id"],
                         "bsk:procedure:siemens/sinamics-g120c/F30001-what-to-do")       # follows its alarm
        self.assertEqual(self.m["bsk:specification:EXTR01/cutter-drive/Rated-power"]["new_id"],
                         "bsk:specification:siemens/g120c-7.5/Rated-power")              # a rating: the model
        self.assertEqual(self.m["bsk:specification:EXTR01/Fan-unit/Air-flow"]["new_id"],
                         "bsk:specification:siemens/sinamics-g120c/Fan-unit/Air-flow")  # a sub-part's: the family
        for machine in (CUTTER, FEED, ASSET, "bsk:signal:EXTR01/cutter-speed"):
            self.assertNotIn(machine, self.m)

    def test_copies_merge_and_identical_ratings_go_to_the_family(self):
        groups = {x["id"]: x for x in self.p["merges"]}
        self.assertIn("bsk:parameter:siemens/sinamics-g120c/p1082", groups)
        both = "bsk:specification:siemens/sinamics-g120c/Ambient-temperature"
        self.assertEqual(len(groups[both]["members"]), 2)
        jog = groups["bsk:parameter:siemens/sinamics-g120c/p1058"]       # "150" and "150 rpm" agree
        self.assertEqual(jog["agreed"]["default"], "150 rpm")

    def test_a_conflict_stays_on_each_model_by_default(self):
        [c] = self.p["conflicts"]
        self.assertEqual((c["id"], c["properties"], c["choice"]), ("bsk:parameter:siemens/sinamics-g120c/p0756", ["default"], "models"))
        self.assertEqual(self.m["bsk:parameter:EXTR01/cutter-drive/p0756"]["new_id"], "bsk:parameter:siemens/g120c-7.5/p0756")
        self.assertEqual(self.m["bsk:parameter:EXTR01/feeder-screw-drive/p0756"]["new_id"], "bsk:parameter:siemens/g120c-5.5/p0756")
        keep = fact_moves.plan(self.g, FAMILY, choices={c["id"]: "bsk:parameter:EXTR01/feeder-screw-drive/p0756"})
        self.assertTrue(any(x["id"] == c["id"] for x in keep["merges"]))
        with self.assertRaisesMessage(fact_moves.MoveError, "must be models, both or one of its members"):
            fact_moves.plan(self.g, FAMILY, choices={c["id"]: "x"})

    def test_shared_machine_links_documents_and_stays(self):
        self.assertEqual([s["id"] for s in self.p["shared"]], ["bsk:procedure:EXTR01/Lockout"])
        self.assertEqual(self.p["machine_links"], [{"type": "MAPS_TO", "from": "bsk:parameter:EXTR01/cutter-drive/p1082",
                                                    "to": "bsk:signal:EXTR01/cutter-speed", "instance": CUTTER}])
        self.assertIn({"owner": FAMILY, "doc_key": "G120C", "document": "bsk:document:G120C"}, self.p["documents"])
        stay = fact_moves.plan(self.g, FAMILY, stays=["bsk:component:EXTR01/Fan-unit"])
        self.assertNotIn("bsk:specification:EXTR01/Fan-unit/Air-flow", by_old(stay))   # stays with its sub-part
        self.assertEqual(len(stay["stays"]), 2)

    def test_an_unknown_target(self):
        with self.assertRaisesMessage(fact_moves.MoveError, "is not a product family"):
            fact_moves.plan(self.g, "bsk:product-family:x/y")


TEST_URI = os.getenv("GRAPH_TEST_URI")


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI (see tests_integration.py) to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class ApplyUndoTests(TestCase):
    def setUp(self):
        registry.save_version(V7, note="v7")
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        g = g120c()
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
            for n in g.nodes.values():
                s.run(f"CREATE (n:{':'.join(n['labels'])}) SET n = $p", p=n["props"]).consume()
            for r in g.rels:
                s.run(f"MATCH (a {{id: $a}}), (b {{id: $b}}) CREATE (a)-[r:{r['type']}]->(b) SET r = $p",
                      a=r["a"], b=r["b"], p=r["props"]).consume()
        self.doc = Document.objects.create(filename="g120c.pdf", doc_key="G120C", sha256="3" * 64, size=1)
        self.approved = CandidateTriple.objects.create(
            document=self.doc, mode="schema", status="approved", anchor_id=CUTTER, subject_name="Cutter drive",
            subject_type="Component", subject_id=CUTTER, predicate="HAS_PARAMETER", object_name="Maximum speed",
            object_type="Parameter", object_id="bsk:parameter:EXTR01/cutter-drive/p1082")

    def state(self):
        with driver.session() as s:
            nodes = s.run("MATCH (n) RETURN n.id AS id, labels(n) AS l, properties(n) AS p ORDER BY id").data()
            rels = s.run("MATCH (a)-[r]->(b) RETURN a.id AS a, type(r) AS t, b.id AS b, properties(r) AS p "
                         "ORDER BY a, t, b").data()
        clean = lambda p: {k: str(v) for k, v in p.items()}
        return [(n["id"], sorted(n["l"]), clean(n["p"])) for n in nodes], [(r["a"], r["t"], r["b"], clean(r["p"])) for r in rels]

    def test_apply_then_undo(self):
        before = self.state()
        batch = fact_moves.apply(FAMILY, author="test")
        with driver.session() as s:
            fam = s.run("MATCH (f {id: $f})-[r]->(x) RETURN type(r) AS t, x.id AS x", f=FAMILY).data()
            left = s.run("MATCH (n) WHERE n.id STARTS WITH 'bsk:parameter:EXTR01/' RETURN count(n) AS n").single()["n"]
            p1082 = s.run("MATCH (n {id: 'bsk:parameter:siemens/sinamics-g120c/p1082'}) RETURN n").single()["n"]
            maps = s.run("MATCH (:Parameter)-[r:MAPS_TO]->() RETURN r.instance AS i").single()["i"]
            rel = s.run("MATCH (f {id: $f})-[r:HAS_PARAMETER]->({id: 'bsk:parameter:siemens/sinamics-g120c/p1082'}) "
                        "RETURN r.triple_ids AS t", f=FAMILY).single()["t"]
            self.assertEqual(repository.get_node(s, "bsk:parameter:EXTR01/feeder-screw-drive/p1082")["id"],
                             "bsk:parameter:siemens/sinamics-g120c/p1082")         # an old id still resolves
        self.assertIn(("DOCUMENTED_BY", "bsk:document:G120C"), {(r["t"], r["x"]) for r in fam})
        self.assertEqual(left, 0)
        self.assertEqual(sorted(p1082["merged_ids"]), ["bsk:parameter:EXTR01/cutter-drive/p1082",
                                                       "bsk:parameter:EXTR01/feeder-screw-drive/p1082"])
        self.assertEqual(sorted(rel), [1, 2])                                        # both runs' support kept
        self.assertEqual(maps, CUTTER)
        self.assertEqual(EntityIndex.load().resolve("Parameter", "x", "EXTR01",
                                                    proposed="bsk:parameter:EXTR01/cutter-drive/p1082").id,
                         "bsk:parameter:siemens/sinamics-g120c/p1082")
        self.approved.refresh_from_db()
        self.assertEqual((self.approved.anchor_id, self.approved.moved_from), (FAMILY, CUTTER))
        self.assertEqual(batch.moves.filter(kind=FactMove.Kind.MERGE).count(), 3)   # p1082, p1058, the ambient rating
        with self.assertRaisesMessage(fact_moves.MoveError, "nothing to move"):
            fact_moves.apply(FAMILY)

        fact_moves.undo(batch)
        self.assertEqual(self.state(), before)
        self.approved.refresh_from_db()
        self.assertEqual((self.approved.anchor_id, self.approved.moved_from), (CUTTER, ""))
        self.assertEqual(FactMoveBatch.objects.get().status, "undone")
        with self.assertRaisesMessage(fact_moves.MoveError, "not applied"):
            fact_moves.undo(batch)

    def test_a_triple_staged_before_the_move_is_approved_onto_the_moved_fact(self):
        fact_moves.apply(FAMILY)
        pending = CandidateTriple.objects.create(
            document=self.doc, mode="schema", status="pending", anchor_id=CUTTER, subject_name="Reset the drive",
            subject_type="Procedure", subject_id="bsk:procedure:EXTR01/cutter-drive/Reset-the-drive", predicate="ADDRESSES",
            object_name="Overcurrent", object_type="Alarm", object_id="bsk:alarm:EXTR01/F30001",
            subject_match="new", object_match="id")
        result = triples.approve([pending])
        self.assertEqual(result["errors"], {})
        pending.refresh_from_db()
        self.assertEqual(pending.object_id, "bsk:alarm:siemens/sinamics-g120c/F30001")
        with driver.session() as s:
            self.assertEqual(s.run("MATCH (n) WHERE n.id = 'bsk:alarm:EXTR01/F30001' RETURN count(n) AS n").single()["n"], 0)

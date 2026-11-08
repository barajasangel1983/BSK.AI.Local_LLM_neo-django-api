"""P22f: the coverage grid (parts × facets) and its "not applicable" overrides. The graph is mocked."""

from unittest.mock import patch

from django.test import TestCase
from rest_framework.test import APIClient

from context_graph import coverage
from context_graph.models import CandidateTriple, CoverageOverride

ASSET = "bsk:asset:T1"
DRIVE, HEATER, PC = "bsk:component:T1/drive", "bsk:component:T1/heater", "bsk:component:T1/pc"
MODEL, FAMILY = "bsk:device-model:acme/d1", "bsk:product-family:acme/d"


def _node(id_, name, **props):
    return {"id": id_, "name": name, "label": "Entity", "properties": props}


def context():
    family = _node(FAMILY, "Acme D")
    return {
        "asset": _node(ASSET, "Test line"),
        "components": [
            {**_node(DRIVE, "Drive", component_type="Drive"), "children": [],
             "device_model": {**_node(MODEL, "D1"), "family": family}},
            {**_node(HEATER, "Heater", component_type="Heater"), "children": [], "device_model": None},
            {**_node(PC, "PC", component_type="Controller"), "children": [], "device_model": None},
        ],
        "hazards": [{**_node("bsk:hazard:acme/d/shock", "Electric shock", facets=["Safety"], severity="DANGER"),
                     "components": [DRIVE], "owners": [FAMILY], "measures": []}],
        "parameters": [{**_node("bsk:parameter:acme/d/p1", "Ramp time", facets=["Setup"]),
                        "components": [DRIVE], "owners": [FAMILY]},
                       {**_node("bsk:parameter:T1/drive/p2", "Local setting", facets=["Setup", "Controls"]),
                        "components": [DRIVE], "owners": [DRIVE]}],
        "documents": [{**_node("bsk:document:d", "Acme D manual", doc_key="ACME-D"), "describes": [ASSET, DRIVE, HEATER],
                       "sections": []}],
    }


def grid():
    with patch("context_graph.services.get_asset_context", return_value=context()):
        return {r["id"]: r for r in coverage.grid(ASSET)["rows"]}


class CoverageTests(TestCase):
    def test_counts_through_model_and_family(self):
        rows = grid()
        self.assertEqual(rows[DRIVE]["cells"]["Safety"], {"approved": 1, "pending": 0, "status": "covered", "override": None})
        self.assertEqual(rows[DRIVE]["cells"]["Setup"]["approved"], 2)
        self.assertEqual(rows[DRIVE]["cells"]["Controls"]["approved"], 1)

    def test_none_na_and_no_manual(self):
        rows = grid()
        self.assertEqual(rows[DRIVE]["cells"]["Maintenance"]["status"], "none")
        self.assertEqual(rows[HEATER]["cells"]["Controls"]["status"], "na")      # a heater has no wiring or settings
        self.assertEqual(rows[HEATER]["cells"]["Safety"]["status"], "none")
        self.assertEqual(rows[PC]["cells"]["Safety"]["status"], "no_manual")

    def test_pending_by_anchor(self):
        CandidateTriple.objects.create(status="pending", anchor_id=FAMILY, subject_name="D", predicate="HAS_HAZARD",
                                       object_name="Burns", facets=["Safety"])
        CandidateTriple.objects.create(status="pending", anchor_id="", subject_id=HEATER, subject_name="Heater",
                                       predicate="HAS_SPEC", object_name="Power", facets=["Installation"])
        rows = grid()
        self.assertEqual(rows[DRIVE]["cells"]["Safety"]["pending"], 1)
        self.assertEqual(rows[HEATER]["cells"]["Installation"], {"approved": 0, "pending": 1, "status": "pending",
                                                                 "override": None})

    def test_override_wins_and_resets(self):
        coverage.set_applicable(DRIVE, "Maintenance", False, by="angel")
        coverage.set_applicable(HEATER, "Controls", True)
        rows = grid()
        self.assertEqual(rows[DRIVE]["cells"]["Maintenance"]["status"], "na")
        self.assertEqual(rows[HEATER]["cells"]["Controls"]["status"], "none")
        coverage.set_applicable(DRIVE, "Maintenance", None)
        self.assertFalse(CoverageOverride.objects.filter(component_id=DRIVE).exists())
        self.assertEqual(grid()[DRIVE]["cells"]["Maintenance"]["status"], "none")

    def test_cell_says_where_each_fact_comes_from(self):
        with patch("context_graph.services.get_asset_context", return_value=context()):
            c = coverage.cell(ASSET, DRIVE, "Setup")
        self.assertEqual([(f["name"], f["level"], f["owner"]) for f in c["facts"]],
                         [("Local setting", "part", "Drive"), ("Ramp time", "family", "Acme D")])
        self.assertEqual(c["manuals"], [{"name": "Acme D manual", "doc_key": "ACME-D"}])
        self.assertTrue(c["applicable"])
        with self.assertRaises(ValueError):
            coverage.cell(ASSET, DRIVE, "Cooking")

    def test_api(self):
        client = APIClient()
        with patch("context_graph.services.get_asset_context", return_value=context()):
            r = client.get("/api/graph/coverage/", {"asset": ASSET})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["facets"][0], "Maintenance")
            self.assertEqual(client.get("/api/graph/coverage/").status_code, 400)
        r = client.post("/api/graph/coverage/override/", {"component_id": DRIVE, "facet": "Quality", "applicable": False},
                        format="json")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(CoverageOverride.objects.get(component_id=DRIVE, facet="Quality").applicable)
        bad = client.post("/api/graph/coverage/override/", {"component_id": "bsk:asset:T1", "facet": "Quality",
                                                             "applicable": False}, format="json")
        self.assertEqual(bad.status_code, 400)

"""Quicker review (2026-10-18): one thing staged under several names, grouped and merged; terminals that may not be
signal channels. Rules for any manufacturer."""

from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIClient

from context_graph import duplicates
from context_graph.models import CandidateTriple, Evidence
from ingestion.tests.tests_library import LibraryTestCase

DRIVE = "bsk:component:EXTR01/feeder-screw-drive"
P = "bsk:io-point:EXTR01/feeder-screw-drive/"


class SameNameTests(SimpleTestCase):
    def test_rules(self):
        same = duplicates.same_name
        for a, b in (("DI 0", "Digital input 0"), ("AO 0", "Analog output 0"), ("+24V IN", "24 V IN"),
                     ("PROFIBUS", "PROFIBUS DP"), ("Current output 1", "current output 1"), ("DI COM1", "DI COM 1"),
                     ("Jogging bit 0", "BI: Jogging bit 0"), ("Analog input supply", "Analog input power supply")):
            self.assertTrue(same(a, b), (a, b))
        for a, b in (("DI 0", "DI 1"), ("Digital input 0", "Digital input 2"), ("24 V IN", "24 V OUT"),
                     ("PROFIBUS", "PROFINET"), ("Relay 1", "Relay 2"), ("Minimum speed", "Maximum speed"),
                     ("Ramp-up time", "Ramp-down time"), ("Analog input 0", "Analog output 0"), ("IOP-2", "BOP-2"),
                     ("Inhibit negative direction", "Inhibit positive direction")):
            self.assertFalse(same(a, b), (a, b))


@override_settings(GRAPH_ENABLED=False)
class MergeTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        self.doc, _ = self.upload()
        self.client = APIClient()

    def stage(self, name, node_id, props=None, typ="IOPoint", predicate="HAS_IO", page=1):
        t = CandidateTriple.objects.create(document=self.doc, mode="schema", subject_name="Feed screw drive",
                                           subject_type="Component", subject_id=DRIVE, subject_existing=True,
                                           predicate=predicate, object_name=name, object_type=typ, object_id=node_id,
                                           object_props=props or {}, page_start=page, confidence=0.8)
        t.evidence.add(Evidence.objects.create(source_kind="text", document=self.doc, page_start=page, excerpt=name))
        return t

    def test_groups_merge_and_fold(self):
        a = self.stage("DI 0", P + "DI-0", {"terminal": "5", "signal_type": "24 V DC"}, page=92)
        b = self.stage("DI 0", P + "DI-0", {"terminal": "5"}, page=93)
        c = self.stage("Digital input 0", P + "Digital-input-0", {"direction": "in"}, page=110)
        d = self.stage("DI 1", P + "DI-1", {"terminal": "6", "signal_type": "24 V DC"})
        e = self.stage("Terminal strip -X138", P + "Terminal-strip-X138")
        p1 = self.stage("p2172", "bsk:parameter:EXTR01/feeder-screw-drive/p2172", {"code": "p2172"}, "Parameter", "HAS_PARAMETER")
        p2 = self.stage("Speed threshold 3", "bsk:parameter:EXTR01/feeder-screw-drive/p2172", {"code": "p2172"},
                        "Parameter", "HAS_PARAMETER")
        r = self.client.get("/api/graph/triples/duplicates/", {"document": str(self.doc.id)}).json()
        groups = {(g["type"], g["into"]): [n["name"] for n in g["names"]] for g in r["groups"]}
        self.assertEqual(groups[("IOPoint", "DI 0")], ["DI 0", "Digital input 0"])
        self.assertEqual(groups[("Parameter", "Speed threshold 3")], ["Speed threshold 3", "p2172"])   # not the bare code
        self.assertNotIn("DI 1", [n for names in groups.values() for n in names])
        self.assertEqual([c_["name"] for c_ in r["checks"]], ["Terminal strip -X138"])

        r = self.client.post("/api/graph/triples/merge/", {"document": str(self.doc.id), "type": "IOPoint",
                                                           "names": ["Digital input 0"], "into": "DI 0"}, format="json")
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json(), {"renamed": 1, "folded": 2})              # c renamed; b and c folded into a
        left = CandidateTriple.objects.filter(document=self.doc, object_type="IOPoint")
        di0 = left.get(object_name="DI 0")
        self.assertEqual((di0.pk, di0.object_id, di0.occurrences, di0.evidence.count()), (a.pk, P + "DI-0", 3, 3))
        self.assertEqual(di0.object_props, {"terminal": "5", "signal_type": "24 V DC", "direction": "in"})
        self.assertTrue(left.filter(object_name="DI 1").exists())
        self.client.post("/api/graph/triples/merge/", {"document": str(self.doc.id), "type": "Parameter",
                                                       "names": ["p2172"], "into": "Speed threshold 3"}, format="json")
        self.assertEqual(list(CandidateTriple.objects.filter(object_type="Parameter").values_list("object_name", flat=True)),
                         ["Speed threshold 3"])                                  # renamed, then folded into one
        self.assertEqual(self.client.post("/api/graph/triples/merge/", {"document": str(self.doc.id), "type": "IOPoint",
                                                                        "names": ["x"], "into": "Nope"}, format="json").status_code, 400)

    def test_two_codes_are_never_one_group(self):
        base = "bsk:parameter:EXTR01/feeder-screw-drive/"
        self.stage("DO 0 function", base + "p0730", {"code": "p0730"}, "Parameter", "HAS_PARAMETER")
        self.stage("DO 0 signal source", base + "p0730", {"code": "p0730"}, "Parameter", "HAS_PARAMETER")
        self.stage("DO 0 function", base + "p0731-x", {"code": "p0731"}, "Parameter", "HAS_PARAMETER")   # same name, other code
        groups = duplicates.find(list(CandidateTriple.objects.all()))["groups"]
        self.assertTrue(all(len({n["id"] for n in g["names"]}) == 1 or g["type"] != "Parameter" for g in groups), groups)

    def test_a_name_holding_two_names_joins_neither(self):
        for name in ("PROFIBUS", "PROFINET", "PROFIBUS or PROFINET", "PROFIBUS DP"):
            self.stage(name, "bsk:interface:EXTR01/feeder-screw-drive/" + name.replace(" ", "-"), typ="Interface",
                       predicate="COMMUNICATES_VIA")
        groups = [sorted(n["name"] for n in g["names"]) for g in duplicates.find(list(CandidateTriple.objects.all()))["groups"]]
        self.assertNotIn(True, [("PROFIBUS" in g and "PROFINET" in g) for g in groups], groups)
        self.assertIn(["PROFIBUS", "PROFIBUS DP"], groups)

    def test_approved_triples_are_not_touched(self):
        a = self.stage("DI 0", P + "DI-0")
        b = self.stage("Digital input 0", P + "Digital-input-0")
        CandidateTriple.objects.filter(pk=b.pk).update(status="approved")
        r = self.client.get("/api/graph/triples/duplicates/", {"document": str(self.doc.id)}).json()
        self.assertEqual(r["groups"], [])

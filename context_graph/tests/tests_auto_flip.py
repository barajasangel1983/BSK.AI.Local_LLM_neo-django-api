"""Links written the wrong way round are flipped when staged (2026-10-16): the ACS880 fault list came as
"Alarm -MITIGATED_BY-> Procedure", which schema v3 allows only as "Procedure -ADDRESSES-> Alarm".
The DGX is mocked; the graph is disabled."""

import json
from pathlib import Path
from unittest.mock import patch

import yaml
from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIClient

from context_graph import extraction, registry
from context_graph.models import CandidateTriple
from context_graph.schema import Schema
from ingestion import library
from ingestion.models import Job
from ingestion.tests.tests_library import LibraryTestCase

V3 = yaml.safe_load((Path(extraction.__file__).parent / "schemas" / "industrial_v3.yaml").read_text())


def triple(subject, subject_type, predicate, obj, object_type, **more):
    return {"subject": subject, "subject_type": subject_type, "predicate": predicate, "object": obj,
            "object_type": object_type, "confidence": 0.9, **more}


class ReversedRelationshipTests(SimpleTestCase):
    def test_only_a_link_allowed_the_other_way_round(self):
        v3 = Schema.from_definition(V3)
        self.assertEqual(v3.reversed_relationship("Alarm", "MITIGATED_BY", "Procedure"), "ADDRESSES")
        self.assertEqual(v3.reversed_relationship("Alarm", "HAS_PROCEDURE", "Procedure"), "ADDRESSES")
        self.assertEqual(v3.reversed_relationship("Procedure", "fixes", "FailureMode"), "MITIGATED_BY")
        self.assertEqual(v3.reversed_relationship("Component", "PART_OF", "Asset"), "HAS_COMPONENT")
        # Allowed as written (only the relationship is wrong): not a reversal.
        self.assertIsNone(v3.reversed_relationship("FailureMode", "ADDRESSES", "Procedure"))
        # Not allowed either way.
        self.assertIsNone(v3.reversed_relationship("Alarm", "HAS_ALARM", "Alarm"))
        # Two relationships allowed reversed: only the model's own one.
        self.assertIsNone(v3.reversed_relationship("Signal", "RELATED_TO", "Component"))
        self.assertEqual(v3.reversed_relationship("Signal", "CONTROLS", "Component"), "CONTROLS")
        self.assertEqual(v3.reversed_relationship("Alarm", "ASSOCIATED_WITH", "Component"), "ASSOCIATED_WITH")


@override_settings(GRAPH_ENABLED=False)
class FlipStagingTests(LibraryTestCase):
    REPLY = [
        triple("Short circuit", "Alarm", "MITIGATED_BY", "Short circuit: what to do", "Procedure",
               subject_subtype="Device fault", object_properties={"summary": "Check the motor cable."}),
        triple("Short circuit: what to do", "Procedure", "ADDRESSES", "Short circuit", "Alarm"),   # the same, as allowed
        triple("ID run", "Alarm", "HAS_ALARM", "ID run aux 0001", "Alarm"),                        # no flip possible
        triple("Drive", "Component", "HAS_ALARM", "Short circuit", "Alarm"),                       # allowed: untouched
    ]

    def setUp(self):
        super().setUp()
        patcher = patch("context_graph.extraction.call_llm", return_value=json.dumps({"triples": self.REPLY}))
        patcher.start()
        self.addCleanup(patcher.stop)
        registry.save_version(V3, note="v3")
        self.doc, _ = self.upload()
        self.drain()
        library.enqueue(self.doc, Job.Kind.EXTRACT, {"mode": "schema", "asset_id": "bsk:asset:EXTR01"})
        self.drain()

    def test_flipped_with_its_ends_and_merged_with_the_same_fact(self):
        staged = list(CandidateTriple.objects.filter(document=self.doc).order_by("id"))
        self.assertEqual(len(staged), 3)
        flipped, kept, allowed = staged
        self.assertEqual((flipped.subject_name, flipped.subject_type, flipped.predicate, flipped.object_name,
                          flipped.object_type), ("Short circuit: what to do", "Procedure", "ADDRESSES", "Short circuit", "Alarm"))
        self.assertEqual((flipped.issue, flipped.auto_fix), ("", "flipped from Alarm -MITIGATED_BY-> Procedure"))
        self.assertEqual(flipped.subject_props, {"summary": "Check the motor cable."})
        self.assertEqual(flipped.object_props, {"subtype": "Device fault"})
        self.assertTrue(flipped.subject_id.startswith("bsk:procedure:") and flipped.object_id.startswith("bsk:alarm:"))
        self.assertIn("not allowed by the schema", kept.issue)
        self.assertEqual((kept.predicate, kept.auto_fix), ("HAS_ALARM", ""))
        self.assertEqual((allowed.issue, allowed.auto_fix), ("", ""))

        body = APIClient().get(f"/api/graph/triples/?document={self.doc.id}").json()
        rows = body["triples"]
        self.assertEqual({r["id"]: r["auto_fix"] for r in rows}[flipped.id], "flipped from Alarm -MITIGATED_BY-> Procedure")

    def test_a_flip_by_hand_keeps_each_ends_properties(self):
        t = CandidateTriple.objects.get(document=self.doc, auto_fix__startswith="flipped")
        r = APIClient().patch(f"/api/graph/triples/{t.id}/", {
            "subject_name": t.object_name, "subject_type": t.object_type, "predicate": "MITIGATED_BY",
            "object_name": t.subject_name, "object_type": t.subject_type}, format="json")
        self.assertEqual(r.status_code, 200, r.content)
        t.refresh_from_db()
        self.assertEqual((t.subject_type, t.subject_props), ("Alarm", {"subtype": "Device fault"}))
        self.assertEqual((t.object_type, t.object_props), ("Procedure", {"summary": "Check the motor cable."}))
        # A rename that is not a swap leaves the properties where they are.
        APIClient().patch(f"/api/graph/triples/{t.id}/", {"subject_name": "Short circuit (2340)"}, format="json")
        t.refresh_from_db()
        self.assertEqual(t.subject_props, {"subtype": "Device fault"})

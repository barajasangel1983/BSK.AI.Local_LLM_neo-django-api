"""P26b-1: the staging checks from the QUINT4-PS review, and "Back to pending".

The triples below are the ones Angel rejected by hand on the QUINT4-PS datasheet (2026-10-23), and some kept next to
them that must not be flagged. No graph and no DGX."""

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from context_graph import review_checks, triples
from context_graph.models import CandidateTriple
from ingestion.models import Document, Job

FAMILY = "bsk:product-family:phoenix-contact/quint4-ps"
MODEL = "bsk:device-model:phoenix-contact/quint4-ps-1ac-24dc-20-480-w"


def issue(name, **props):
    return review_checks.triple_issue("Specification", name, props)


class RatingIssueTests(TestCase):
    def test_a_rating_needs_a_real_value(self):
        self.assertIn("no value", issue("Power frequency magnetic field (Minimum normative requirements)", value="not required"))
        self.assertIn("no value", issue("Hold-up time", value=""))
        self.assertIn("no value", issue("Power dissipation", condition="120 V AC"))
        self.assertIn("standard's name", issue("Voltage dips (CE basic standard)", value="EN 61000-4-11"))
        self.assertIn("already says", issue("Power dissipation at 120 V AC", value="120 V AC"))
        self.assertIn("already says", issue("Frequency range (80 MHz ... 1 GHz)", min="80 MHz", max="1 GHz"))
        self.assertIn("already says", issue("Input voltage (230 V AC, 50 Hz)", value="230 V AC, 50 Hz"))
        self.assertIn("row or column", issue("Power frequency magnetic field (Higher requirements, Row 3)",
                                             value="50 Hz, 60 Hz (1 kA/m, 3 s)"))

    def test_good_ratings_pass(self):
        for name, props in (("Crest factor at 230 V AC", {"value": "1.65", "condition": "230 V AC"}),
                            ("Nominal output current (IN)", {"value": "20 A", "unit": "A"}),
                            ("Setting range of the output voltage (USet)", {"min": "24", "max": "29.5", "unit": "V DC"}),
                            ("Degree of protection", {"value": "IP20"}),
                            ("Test field strength (80 MHz ... 1 GHz)", {"value": "10 V/m (Test Level 3)",
                                                                        "condition": "80 MHz ... 1 GHz"})):
            self.assertEqual(issue(name, **props), "", name)
        self.assertEqual(review_checks.triple_issue("Procedure", "Fuse", {}), "")    # ratings only


@override_settings(GRAPH_ENABLED=False)
class JobFlagTests(TestCase):
    def setUp(self):
        self.doc = Document.objects.create(filename="quint4.pdf", doc_key="QUINT4", sha256="1" * 64, size=1)
        self.job = Job.objects.create(document=self.doc, kind=Job.Kind.EXTRACT, params={})

    def stage(self, object_name, object_type, anchor=FAMILY, job=None, status="pending", page=1, object_id=None, **props):
        subject = anchor
        return CandidateTriple.objects.create(
            document=self.doc, job=job or self.job, mode="schema", status=status, anchor_id=anchor,
            subject_name="QUINT4-PS", subject_type="ProductFamily", subject_id=subject, predicate="HAS",
            object_name=object_name, object_type=object_type,
            object_id=object_id or f"bsk:{object_type.lower()}:x/{object_name}", object_props=props, page_start=page)

    def test_the_same_led_under_two_types_is_flagged(self):
        led = self.stage("LED status indicator 1", "Component")
        io = self.stage("LED status indicator 1", "IOPoint")
        signal = self.stage("Output power", "Signal")
        param = self.stage("Output power", "Parameter")
        review_checks.flag_job(self.job)
        for t in (led, io):
            t.refresh_from_db()
            self.assertIn("also staged as", t.issue)
        for t in (signal, param):     # a signal and a parameter may share a name
            t.refresh_from_db()
            self.assertEqual(t.issue, "")

    def test_a_rating_that_repeats_a_parameter(self):
        self.stage("Output current", "Parameter", default="5 ... 150 %")
        spec = self.stage("Default setting: Output current", "Specification", anchor=MODEL, value="5 ... 150 %")
        review_checks.flag_job(self.job)
        spec.refresh_from_db()
        self.assertIn("repeats the parameter 'output current'", spec.issue)

    def test_likely_duplicates(self):
        short = self.stage("Mains buffering", "Specification", anchor=MODEL, value="≥ 20", page=1)
        full = self.stage("Mains buffering (120 V AC)", "Specification", anchor=MODEL, value="≥ 20", page=5)
        other = self.stage("Mains buffering (230 V AC)", "Specification", anchor=MODEL, value="≥ 20", page=5)
        a = self.stage("Gas-filled surge arrester", "Component", page=18)
        b = self.stage("Gas-filled surge arrester for surge protection", "Component", page=19)
        out = self.stage("Out 1", "IOPoint")
        out2 = self.stage("Out 2", "IOPoint")
        review_checks.flag_job(self.job)
        for t in (short, full, other, a, b, out, out2):
            t.refresh_from_db()
        self.assertIn("likely the same as 'Mains buffering (120 V AC)'", short.issue)
        self.assertEqual((full.issue, other.issue), ("", ""))      # two voltages: two ratings
        self.assertEqual(sum(bool(t.issue) for t in (a, b)), 1)     # one of the pair
        self.assertEqual((out.issue, out2.issue), ("", ""))

    def test_one_node_read_twice_and_reviewed_triples(self):
        self.stage("Rem", "IOPoint", object_id="bsk:io-point:x/Rem-Rem")
        same = self.stage("Remote input", "IOPoint", object_id="bsk:io-point:x/Rem-Rem")
        old = Job.objects.create(document=self.doc, kind=Job.Kind.EXTRACT, params={})
        self.stage("DC OK LED", "Component", job=old, status="rejected")
        new = self.stage("DC OK LED", "IOPoint")
        flagged_before = self.stage("Fuse", "Component")
        CandidateTriple.objects.filter(pk=flagged_before.pk).update(issue="type not in schema")
        review_checks.flag_job(self.job)
        for t in (same, new, flagged_before):
            t.refresh_from_db()
        self.assertEqual(same.issue, "")                         # the same node
        self.assertEqual(new.issue, "")                          # the other one was rejected
        self.assertEqual(flagged_before.issue, "type not in schema")   # an existing issue is kept


@override_settings(GRAPH_ENABLED=False)
class BackToPendingTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.doc = Document.objects.create(filename="q.pdf", doc_key="Q", sha256="2" * 64, size=1)
        base = dict(document=self.doc, mode="schema", subject_name="A", subject_type="Component", predicate="HAS_SPEC",
                    object_name="B", object_type="Specification", anchor_id=MODEL)
        self.rejected = CandidateTriple.objects.create(**base, status="rejected", edited=True, moved_from=FAMILY)
        self.pending = CandidateTriple.objects.create(**base, status="pending")
        self.approved = CandidateTriple.objects.create(**base, status="approved")

    def test_rejected_back_to_pending_keeps_its_edits(self):
        r = self.client.post("/api/graph/triples/unreject/",
                             {"ids": [self.rejected.pk, self.pending.pk, self.approved.pk]}, format="json")
        self.assertEqual(r.json(), {"pending": 1, "skipped": 2})
        self.rejected.refresh_from_db()
        self.approved.refresh_from_db()
        self.assertEqual((self.rejected.status, self.rejected.edited, self.rejected.moved_from, self.rejected.anchor_id),
                         ("pending", True, FAMILY, MODEL))
        self.assertEqual(self.approved.status, "approved")
        self.assertEqual(triples.unreject([self.rejected]), 0)     # already pending
        self.assertEqual(self.client.post("/api/graph/triples/unreject/", {"ids": [99999]}, format="json").status_code, 404)

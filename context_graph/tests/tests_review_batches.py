"""P27c: apply the automated review as a batch, and undo it. The graph is mocked here (approve / edit / _uncommit);
`tests_integration.py`'s test Neo4j covers what approve and _uncommit write."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from django.test import override_settings
from rest_framework.test import APIClient

from context_graph import review_batches as rb, triples
from context_graph.models import CandidateTriple as T, Evidence, ReviewBatch
from ingestion.tests.tests_library import LibraryTestCase

FAM = "bsk:product-family:acme/drive"


@override_settings(GRAPH_ENABLED=False)
class BatchTestCase(LibraryTestCase):
    def setUp(self):
        super().setUp()
        self.doc, _ = self.upload()
        p = patch("context_graph.review_batches.ra.source_of", return_value=("", ""))
        p.start()
        self.addCleanup(p.stop)
        self.failing = set()      # approvals the fake graph refuses

    def stage(self, subject, s_type, predicate, obj, o_type, **fields):
        slug = lambda name: name.lower().replace(" ", "-")   # noqa: E731
        sid = FAM if s_type == "ProductFamily" else f"bsk:{s_type.lower()}:acme/drive/{slug(subject)}"
        t = T.objects.create(document=self.doc, mode="schema", subject_name=subject, subject_type=s_type, subject_id=sid,
                             predicate=predicate, object_name=obj, object_type=o_type, anchor_id=FAM,
                             object_id=f"bsk:{o_type.lower()}:acme/drive/{slug(obj)}", facets=["Safety"], **fields)
        t.evidence.add(Evidence.objects.create(source_kind="text", document=self.doc, quote_status="exact"))
        return t

    def hazard(self, name, **kw):
        return self.stage("Drive", "ProductFamily", "HAS_HAZARD", name, "Hazard", **kw)

    def measure(self, hazard, name, **kw):
        return self.stage(hazard, "Hazard", "PROTECTED_BY", name, "SafetyMeasure", **kw)

    @staticmethod
    def row(t, tier, reason="", **kw):
        return {"id": t.pk, "fact": rb._fact(t), "quote": "", "tier": tier, "reason": reason, "edited": t.edited,
                "rename_subject_to": None, **kw}

    def fake_approve(self, ts, automatic=False):
        assert automatic
        ok = [t.pk for t in ts if t.pk not in self.failing]
        T.objects.filter(pk__in=ok).update(status=T.Status.APPROVED, layer="curated")
        for t in ts:
            t.refresh_from_db()
        return {"approved": ok, "errors": {i: "refused" for i in self.failing if i in [t.pk for t in ts]}}

    def fake_edit(self, t, changes):
        for k, v in changes.items():
            setattr(t, k, v)
        if "object_name" in changes:
            t.object_id = f"bsk:hazard:acme/drive/{changes['object_name'].lower().replace(' ', '-')}"
        t.edited = True
        t.save()
        return t

    def apply_rows(self, rows, sheet=None, doc=""):
        p = rb.plan({"rows": rows}, sheet, doc)
        with patch.object(triples, "approve", side_effect=self.fake_approve), \
                patch.object(triples, "edit", side_effect=self.fake_edit):
            return p, rb.apply(p, {"facet": "Safety"}, started_by="angel")


class PlanTests(BatchTestCase):
    def test_rows_changed_since_the_report_are_skipped(self):
        a, b, c, d = (self.hazard(n) for n in ("Hot surface", "Noise", "Electric shock", "Crushing"))
        rows = [self.row(t, "auto") for t in (a, b, c, d)]
        T.objects.filter(pk=b.pk).update(status=T.Status.APPROVED)
        T.objects.filter(pk=c.pk).update(edited=True)
        T.objects.filter(pk=d.pk).update(object_name="Crushing hazard")
        rows.append({"id": 999999, "fact": "x", "tier": "auto"})
        p = rb.plan({"rows": rows})
        self.assertEqual([s.triple.pk for s in p.steps], [a.pk])
        self.assertEqual(p.skipped, {b.pk: "now approved", c.pk: "edited since the report",
                                     d.pk: "another fact than in the report", 999999: "no longer staged"})

    def test_the_sheet_wins_over_the_tier(self):
        a, b, c = (self.hazard(n) for n in ("Hot surface", "Noise", "Laser"))
        p = rb.plan({"rows": [self.row(a, "reviewer"), self.row(b, "auto"), self.row(c, "reviewer")]},
                    {str(a.pk): {"verdict": "approve", "reason": "stated"}, str(b.pk): {"verdict": "reject", "reason": "x"},
                     str(c.pk): {"verdict": "ask", "reason": "Q1"}})
        self.assertEqual([(s.action, s.reason) for s in p.steps],
                         [("approve", "sheet: stated"), ("reject", "sheet: x"), ("doubtful", "sheet: Q1")])
        self.assertEqual(rb.plan({"rows": [self.row(a, "reviewer")]}, doc="Other manual").steps, [])


class ApplyTests(BatchTestCase):
    def test_hazards_first_measures_follow_and_every_triple_is_marked(self):
        hot, noise = self.hazard("Hot surface"), self.hazard("Noise")
        gloves, plugs = self.measure("Hot surface", "Wear gloves"), self.measure("Noise", "Wear ear plugs")
        bad, doubt = self.hazard("Minor injury"), self.hazard("Sharp edges")
        self.failing = {noise.pk}
        _, batch = self.apply_rows([self.row(gloves, "auto"), self.row(plugs, "auto"), self.row(hot, "auto"),
                             self.row(noise, "auto"), self.row(bad, "reject", "repeat"), self.row(doubt, "reviewer", "judge doubts")])
        got = {t.pk: (t.status, t.review_tier, t.review_reason) for t in T.objects.filter(review_batch=batch)}
        self.assertEqual(got[hot.pk][:2], ("approved", "auto"))
        self.assertEqual(got[gloves.pk][:2], ("approved", "auto"))
        self.assertEqual(got[noise.pk], ("pending", "doubtful", "approval failed: refused"))
        self.assertEqual(got[plugs.pk][:2], ("pending", "leave"))             # its hazard didn't commit
        self.assertEqual(got[bad.pk], ("rejected", "reject", "repeat"))
        self.assertEqual(got[doubt.pk], ("pending", "doubtful", "judge doubts"))
        self.assertEqual((batch.summary["approved"], batch.summary["rejected"], batch.author, batch.started_by),
                         (2, 1, "auto:p27-v1", "angel"))

    def test_sheet_fixes_move_a_measure_and_rename_a_hazard_with_its_measures(self):
        shock = self.hazard("Electric shock from capacitors")
        wait = self.measure("Electric shock", "Wait 5 minutes")
        injury = self.hazard("Injury or death")
        ground = self.measure("Injury or death", "Ground the drive")
        sheet = {str(wait.pk): {"verdict": "fix", "reason": "belongs to the capacitors", "subject": "Electric shock from capacitors"},
                 str(injury.pk): {"verdict": "fix", "reason": "named after the consequence", "hazard": "Improper grounding"},
                 str(ground.pk): {"verdict": "fix", "reason": "named after the consequence", "hazard": "Improper grounding"}}
        _, batch = self.apply_rows([self.row(shock, "auto")] + [self.row(t, "reviewer") for t in (wait, injury, ground)], sheet)
        wait.refresh_from_db(), injury.refresh_from_db(), ground.refresh_from_db()
        self.assertEqual((wait.status, wait.subject_name, wait.subject_id), ("approved", shock.object_name, shock.object_id))
        self.assertEqual((injury.status, injury.object_name), ("approved", "Improper grounding"))
        self.assertEqual((ground.status, ground.subject_id), ("approved", injury.object_id))
        self.assertEqual(batch.before[str(wait.pk)]["subject_name"], "Electric shock")
        self.assertEqual(batch.before[str(injury.pk)]["object_name"], "Injury or death")

    def test_a_measure_of_a_repeated_hazard_goes_under_the_kept_one(self):
        kept = self.hazard("Hot parts")
        T.objects.filter(pk=kept.pk).update(status=T.Status.APPROVED)       # approved before
        repeat = self.hazard("Hot parts again")
        m = self.measure("Hot parts again", "Check the temperature")
        self.apply_rows([self.row(repeat, "reject", f"repeat of {kept.pk}"), self.row(m, "auto", rename_subject_to="hot parts")])
        m.refresh_from_db()
        self.assertEqual((m.status, m.subject_id), ("approved", kept.object_id))


class AutomaticApprovalTests(BatchTestCase):
    def test_automatic_skips_only_the_one_at_a_time_rule(self):
        a, b = self.hazard("Hot surface"), self.hazard("Noise")
        weak = self.hazard("Laser")
        weak.evidence.update(quote_status="not_found")

        @contextmanager
        def fake_session():
            yield MagicMock()
        with patch.object(triples, "session", fake_session), patch.object(triples, "_follow_moves"):
            result = triples.approve([a, b, weak], automatic=True)
        self.assertEqual(sorted(result["approved"]), sorted([a.pk, b.pk]))
        self.assertEqual(result["errors"], {weak.pk: triples.SAFETY_QUOTE})
        c = self.hazard("Crushing")
        self.assertEqual(triples.approve([c, weak])["errors"], {c.pk: triples.SAFETY_ALONE, weak.pk: triples.SAFETY_ALONE})


class UndoTests(BatchTestCase):
    def test_undo_takes_the_batch_back_and_keeps_what_a_person_decided_since(self):
        hot, noise, sharp, laser = (self.hazard(n) for n in ("Hot surface", "Noise", "Sharp edges", "Laser"))
        injury = self.hazard("Injury or death")
        _, batch = self.apply_rows([self.row(hot, "auto"), self.row(noise, "reject"), self.row(sharp, "reviewer"),
                             self.row(laser, "reject"), self.row(injury, "reviewer")],
                            {str(injury.pk): {"verdict": "fix", "reason": "x", "hazard": "Improper grounding"}})
        T.objects.filter(pk=laser.pk).update(status=T.Status.PENDING)        # a person put it back since
        uncommitted = []

        @contextmanager
        def fake_session():
            s = MagicMock()
            s.execute_write.side_effect = lambda fn, t: uncommitted.append(t.pk)
            yield s
        with patch.object(rb, "session", fake_session), patch.object(triples, "_follow_moves"):
            result = rb.undo(batch)
        self.assertEqual(sorted(uncommitted), sorted([hot.pk, injury.pk]))
        self.assertEqual(result["kept"], [laser.pk])
        for t in (hot, noise, sharp, injury):
            t.refresh_from_db()
            self.assertEqual((t.status, t.review_tier, t.review_batch_id), ("pending", "", None))
        self.assertEqual((injury.object_name, injury.edited), ("Injury or death", False))   # the batch's edit put back
        laser.refresh_from_db()
        self.assertEqual(laser.review_tier, "reject")
        batch.refresh_from_db()
        self.assertEqual(batch.status, ReviewBatch.Status.UNDONE)
        with self.assertRaises(rb.ReviewError):
            rb.undo(batch)


class ListFilterTests(BatchTestCase):
    def test_the_doubtful_filter_and_count(self):
        a, b = self.hazard("Hot surface"), self.hazard("Noise")
        self.apply_rows([self.row(a, "reviewer", "judge doubts (1 of 3): inferred"), self.row(b, "reject")])
        body = APIClient().get(f"/api/graph/triples/?document={self.doc.id}&review_tier=doubtful&facet=Safety").json()
        self.assertEqual([t["id"] for t in body["triples"]], [a.pk])
        self.assertEqual(body["doubtful"], 1)
        self.assertEqual(body["triples"][0]["review_reason"], "judge doubts (1 of 3): inferred")
        self.assertEqual(APIClient().get("/api/graph/triples/?review_tier=maybe").status_code, 400)

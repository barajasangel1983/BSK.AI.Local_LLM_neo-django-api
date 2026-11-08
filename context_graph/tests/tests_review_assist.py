"""P27: assisted review (report only): code checks, repeats, the judge's votes, tiers."""

from types import SimpleNamespace

from django.test import SimpleTestCase

from context_graph import review_assist as ra


class _Ev(SimpleNamespace):
    pass


class _Many:
    def __init__(self, items):
        self.items = items

    def all(self):
        return self.items


_ids = iter(range(1, 10_000))


def fact(subject, s_type, predicate, obj, o_type, quote_status="exact", start=0, page=1, issue="", safety=True):
    ev = _Ev(quote_status=quote_status, quote_start=start if quote_status == "exact" else None,
             quote_end=start + 10 if quote_status == "exact" else None, text_hash="h", document_id="d")
    return SimpleNamespace(id=next(_ids), subject_name=subject, subject_type=s_type, predicate=predicate,
                           object_name=obj, object_type=o_type, evidence=_Many([ev]), document_id="d",
                           anchor_id="fam", issue=issue, object_props={}, facets=["Safety"] if safety else ["Setup"], page_start=page)


YES = {"stated": "yes", "specific": "yes", "applies": "yes", "reason": ""}


class CodeCheckTests(SimpleTestCase):
    def test_rules(self):
        no_quote = fact("Fam", "ProductFamily", "HAS_HAZARD", "Hot surface", "Hazard", quote_status="not_found")
        under = fact("Hot surface", "Hazard", "PROTECTED_BY", "Wear gloves", "SafetyMeasure", start=20)
        schema = fact("Fam", "ProductFamily", "PROTECTED_BY", "Gloves", "SafetyMeasure", issue="x is not allowed by the schema")
        word = fact("Noise", "Hazard", "PROTECTED_BY", "Covers", "SafetyMeasure", start=40)
        split_a = fact("Fam", "ProductFamily", "HAS_HAZARD", "Burns", "Hazard", start=60)
        split_b = fact("Fam", "ProductFamily", "HAS_HAZARD", "Property damage", "Hazard", start=60)
        older = fact("P1", "Parameter", "HAS_RANGE", "0-10", "Specification", quote_status="none", safety=False)
        out = ra.code_checks([no_quote, under, schema, word, split_a, split_b, older])
        self.assertEqual(out[no_quote.id][0], "reject")
        self.assertEqual(out[under.id], ("leave", "its hazard has no exact quote"))
        self.assertEqual(out[schema.id][0], "reject")
        self.assertEqual(out[word.id], ("reviewer", "one-word measure"))
        self.assertTrue(out[split_a.id][1].startswith("one notice split into 2"))
        self.assertNotIn(older.id, out)   # no quote is no reason to reject outside Safety


class RepeatTests(SimpleTestCase):
    def test_repeats_and_merged_owner(self):
        h1 = fact("Fam", "ProductFamily", "HAS_HAZARD", "Hot parts", "Hazard", start=0, page=18)
        m1 = fact("Hot parts", "Hazard", "PROTECTED_BY", "Never touch parts", "SafetyMeasure", start=20, page=18)
        h2 = fact("Fam", "ProductFamily", "HAS_HAZARD", "Hot parts again", "Hazard", start=100, page=89)
        m2 = fact("Hot parts again", "Hazard", "PROTECTED_BY", "Never touch parts", "SafetyMeasure", start=120, page=89)
        m3 = fact("Hot parts again", "Hazard", "PROTECTED_BY", "Check temperature", "SafetyMeasure", start=140, page=89)
        vec = {"Hot parts": [1, 0, 0], "Hot parts again": [0.99, 0.1, 0], "Never touch parts": [0, 1, 0],
               "Check temperature": [0, 0, 1]}
        out, alias = ra.duplicate_groups([h1, m1, h2, m2, m3], {}, embed=lambda texts: [vec[t] for t in texts])
        self.assertEqual(set(out), {h2.id, m2.id})
        self.assertEqual(alias, {"hot parts again": "hot parts"})

    def test_one_sentence_several_measures_is_no_repeat(self):
        a = fact("H", "Hazard", "PROTECTED_BY", "Barriers", "SafetyMeasure", start=5)
        b = fact("H", "Hazard", "PROTECTED_BY", "Safety talks", "SafetyMeasure", start=5)
        q = "Protect personnel by erecting barriers and giving safety talks."
        out, _ = ra.duplicate_groups([a, b], {a.id: q, b.id: q},
                                     embed=lambda texts: [[1, 0] if t == q else ([0, 1] if t == "Barriers" else [0.6, 0.8]) for t in texts])
        self.assertEqual(out, {})


class JudgeTests(SimpleTestCase):
    def test_votes_and_tiers(self):
        t = fact("Fam", "ProductFamily", "HAS_HAZARD", "Hot surface", "Hazard")
        replies = iter(['{"stated": "yes", "specific": "yes", "applies": "yes", "reason": "ok"}'] * 2
                       + ['no json here'])
        votes = ra.judge(t, "The surface is hot.", "...", post=lambda m: next(replies))
        self.assertEqual(len(votes), 3)
        self.assertEqual(votes[2]["stated"], "unclear")
        self.assertFalse(ra.passes(votes))
        self.assertTrue(ra.passes([YES] * 3))
        self.assertEqual(ra.tier(None, None, [YES] * 3)[0], "auto")
        self.assertEqual(ra.tier(None, None, votes)[0], "reviewer")
        self.assertEqual(ra.tier(("reject", "x"), None, [YES] * 3)[0], "reject")
        self.assertEqual(ra.tier(None, ("reject", "repeat"), [YES] * 3)[0], "reject")
        self.assertEqual(ra.tier(("reviewer", "flag"), None, [YES] * 3)[0], "reviewer")


class CrossPageRepeatTests(SimpleTestCase):
    """2026-11-06: the repeats the 10-27 test missed (SIMOTICS p.43/p.90, p.86/p.84)."""

    def test_a_stock_step_under_two_different_hazards_is_no_repeat(self):
        h1 = fact("Fam", "ProductFamily", "HAS_HAZARD", "Rotating parts", "Hazard", start=0, page=18)
        m1 = fact("Rotating parts", "Hazard", "PROTECTED_BY", "Switch off and disconnect", "SafetyMeasure", start=20, page=18)
        h2 = fact("Fam", "ProductFamily", "HAS_HAZARD", "Electric shock", "Hazard", start=100, page=17)
        m2 = fact("Electric shock", "Hazard", "PROTECTED_BY", "Switch off and disconnect", "SafetyMeasure", start=120, page=17)
        q = {m1.id: "Switch off and disconnect the machine first.", m2.id: "Switch off and disconnect the machine first!"}
        vec = {"Rotating parts": [1, 0, 0], "Electric shock": [0, 1, 0], "Switch off and disconnect": [0, 0, 1],
               q[m1.id]: [0, 0, 1], q[m2.id]: [0, 0, 1]}
        out, alias = ra.duplicate_groups([h1, m1, h2, m2], q, embed=lambda texts: [vec[t] for t in texts])
        self.assertEqual((out, alias), ({}, {}))

    def test_name_and_quote_both_close(self):
        h1 = fact("Fam", "ProductFamily", "HAS_HAZARD", "Destruction from vibration", "Hazard", start=0, page=84)
        h2 = fact("Fam", "ProductFamily", "HAS_HAZARD", "Mechanical destruction from high vibration", "Hazard", start=50, page=86)
        q = {h1.id: "The machine can be destroyed if vibration values are exceeded.", h2.id: "If vibration values are not kept, the machine can be destroyed."}
        vec = {h1.object_name: [1, 0], h2.object_name: [0.83, 0.558], q[h1.id]: [1, 0], q[h2.id]: [0.82, 0.572]}
        out, alias = ra.duplicate_groups([h1, h2], q, embed=lambda texts: [vec[t] for t in texts])
        # P27c C9: in this band a repeat is only possible (gear units "improper startup" / "improper foundation" scored
        # the same): the reviewer decides, and the later hazard keeps its own measures
        self.assertEqual(out[h2.id][0], "reviewer")
        self.assertTrue(out[h2.id][1].startswith(f"possible repeat of {h1.id}"))
        self.assertEqual(alias, {})

    def test_a_repeat_of_an_approved_fact_is_rejected_and_the_approved_one_kept(self):
        approved = fact("Fam", "ProductFamily", "HAS_HAZARD", "Hot parts", "Hazard", start=0, page=89)
        pending = fact("Fam", "ProductFamily", "HAS_HAZARD", "Hot parts", "Hazard", start=50, page=18)
        out, _ = ra.duplicate_groups([pending, approved], {}, embed=lambda texts: [[1, 0] for _ in texts],
                                     kept_first={approved.id})
        self.assertEqual(set(out), {pending.id})


class NoticeTests(SimpleTestCase):
    TEXT = ("## WARNING\n\n## Sparks at rotating parts\n\nSparks can ignite an explosive atmosphere.\n\n- Keep covers closed.\n\n"
            "## 7 Maintenance\n\nGrease the bearings every 4000 h.\n\n**CAUTION**\n\nHot surface. Let it cool.\n")

    def at(self, s):
        return self.TEXT.index(s)

    def test_a_measure_outside_its_hazards_notice_goes_to_the_reviewer(self):
        from unittest.mock import patch
        h = fact("Fam", "ProductFamily", "HAS_HAZARD", "Sparks", "Hazard", start=self.at("Sparks can"))
        inside = fact("Sparks", "Hazard", "PROTECTED_BY", "Keep covers closed", "SafetyMeasure", start=self.at("Keep covers"))
        outside = fact("Sparks", "Hazard", "PROTECTED_BY", "Let it cool", "SafetyMeasure", start=self.at("Let it cool"))
        with patch("context_graph.quotes.snapshot_text", return_value=self.TEXT):
            out = ra.notice_checks([h, inside, outside])
        self.assertEqual(out, {outside.id: ("reviewer", "its quote is outside its hazard's notice")})

    def test_the_title_heading_after_a_signal_word_is_the_same_notice(self):
        lo, hi = ra._block(self.TEXT, self.at("Sparks can"), self.at("Sparks can") + 10)
        self.assertEqual((lo, hi), (0, self.at("## 7 Maintenance")))


class FollowHazardTests(SimpleTestCase):
    def test_a_measure_is_auto_only_with_its_hazard(self):
        h = fact("Fam", "ProductFamily", "HAS_HAZARD", "Rotating parts", "Hazard")
        m = fact("Rotating parts", "Hazard", "PROTECTED_BY", "Close covers", "SafetyMeasure")
        h_ok = fact("Fam", "ProductFamily", "HAS_HAZARD", "Hot parts", "Hazard")
        m_ok = fact("Hot parts", "Hazard", "PROTECTED_BY", "Let cool", "SafetyMeasure")
        tiers = {h.id: ("reviewer", "judge doubts"), m.id: ("auto", "judge: 3 of 3"),
                 h_ok.id: ("approved", "already approved"), m_ok.id: ("auto", "judge: 3 of 3")}
        ra.follow_hazards([h, m, h_ok, m_ok], tiers, {})
        self.assertEqual(tiers[m.id], ("reviewer", "its hazard is reviewer"))
        self.assertEqual(tiers[m_ok.id][0], "auto")

    def test_a_measure_without_a_live_hazard_is_left(self):
        gone = fact("Lifting", "Hazard", "PROTECTED_BY", "Use hoists", "SafetyMeasure")
        h_rej = fact("Fam", "ProductFamily", "HAS_HAZARD", "Hot parts", "Hazard")
        m_rej = fact("Hot parts", "Hazard", "PROTECTED_BY", "Let cool", "SafetyMeasure")
        tiers = {gone.id: ("auto", "judge: 3 of 3"), h_rej.id: ("reject", "no exact quote"),
                 m_rej.id: ("auto", "judge: 3 of 3")}
        ra.follow_hazards([gone, h_rej, m_rej], tiers, {})
        self.assertEqual(tiers[gone.id], ("leave", "its hazard is not pending or approved"))
        self.assertEqual(tiers[m_rej.id], ("leave", "its hazard is reject"))

    def test_a_hazard_outside_the_run_counts_by_its_status(self):
        m_pend = fact("Hot parts", "Hazard", "PROTECTED_BY", "Let cool", "SafetyMeasure")
        m_rej = fact("Lifting", "Hazard", "PROTECTED_BY", "Use hoists", "SafetyMeasure")
        tiers = {m_pend.id: ("auto", "judge: 3 of 3"), m_rej.id: ("auto", "judge: 3 of 3")}
        outside = {(m_pend.document_id, m_pend.anchor_id, ra._norm("Hot parts")): "pending",
                   (m_rej.document_id, m_rej.anchor_id, ra._norm("Lifting")): "rejected"}
        ra.follow_hazards([m_pend, m_rej], tiers, {}, outside)
        self.assertEqual(tiers[m_pend.id][0], "auto")
        self.assertEqual(tiers[m_rej.id], ("leave", "its hazard is rejected"))

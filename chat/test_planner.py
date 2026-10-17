"""P17b: the retrieval planner picks the manuals of the parts a question is about, and the prompt split."""

from django.test import SimpleTestCase, override_settings

from chat.planner import plan, question_kind

PARTS = [
    {"id": "cutter", "name": "Cutter", "parent": "asset", "doc_keys": []},
    {"id": "cutter-drive", "name": "Cutter drive", "parent": "cutter", "doc_keys": ["G120C"], "model": "Siemens SINAMICS G120C"},
    {"id": "cutter-motor", "name": "Cutter motor", "parent": "cutter", "doc_keys": ["1LE1"]},
    {"id": "knives", "name": "Cutter knife assembly", "parent": "cutter", "doc_keys": []},
    {"id": "main", "name": "Main Drive Motor", "parent": "asset", "doc_keys": []},
    {"id": "vfd", "name": "Main drive frequency converter", "parent": "main", "doc_keys": ["ACS880-HW", "ACS880-FW"]},
    {"id": "fan", "name": "Cooling fan", "parent": "vfd", "doc_keys": []},
    {"id": "ir", "name": "Product discharge temperature (infrared)", "parent": "asset", "doc_keys": ["OPTRIS"]},
    {"id": "die-plate", "name": "Die plate", "parent": "asset", "doc_keys": []},
    {"id": "water-fm", "name": "Water injection flow meter", "parent": "asset", "doc_keys": ["PROMAG"]},
    {"id": "steam-fm", "name": "Steam flow meter", "parent": "asset", "doc_keys": ["PROWIRL"]},
]


@override_settings(PLANNER_MAX_MANUALS=3, PLANNER_QUERY_PARTS=2, PLANNER_PARTS_FACTS_SHARE=0.45, PLANNER_PARTS_EXCERPTS_SHARE=0.1)
class PlannerTests(SimpleTestCase):
    def test_a_named_part_gives_its_manual(self):
        self.assertEqual(plan("The cutter drive trips with F07801", PARTS, set()).doc_keys, ["G120C"])

    def test_a_group_gives_the_best_matching_sub_parts_manual(self):
        self.assertEqual(plan("What drives the cutter?", PARTS, {"cutter"}).doc_keys, ["G120C"])
        self.assertEqual(plan("How often should the cutter be serviced?", PARTS, {"cutter"}).doc_keys, ["G120C", "1LE1"])

    def test_a_sub_part_gives_its_parents_manuals(self):
        self.assertEqual(plan("The cooling fan is noisy", PARTS, set()).doc_keys, ["ACS880-HW", "ACS880-FW"])

    def test_most_of_a_name_is_enough(self):
        p = plan("How do I clean the lens of the infrared thermometer at the product discharge?", PARTS, set())
        self.assertEqual(p.doc_keys, ["OPTRIS"])

    def test_no_manual_no_keys(self):
        self.assertEqual(plan("What torque for the die plate bolts?", PARTS, set()).doc_keys, [])

    def test_kinds_and_shares(self):
        self.assertEqual(question_kind("Which motors are made by Siemens?").kind, "parts")
        self.assertEqual(question_kind("Which motors are made by Siemens?").facts_share, 0.45)
        self.assertEqual(question_kind("How many barrel zones are there, and what controls the temperature?").kind, "parts")
        self.assertEqual(question_kind("Which parts need replacing 9 years after start-up?").kind, "general")
        self.assertEqual(question_kind("The main drive shows fault 2340").kind, "code")
        self.assertEqual(question_kind("The cutter drive shows F07801").kind, "code")
        g = question_kind("How should the flow meter be cleaned?")
        self.assertEqual((g.kind, g.facts_share, g.excerpts_share), ("general", 0.0, 0.0))

    def test_the_query_names_the_parts_make_and_model(self):
        p = plan("The cutter drive trips with F07801", PARTS, set())
        self.assertEqual(p.query_note, "Cutter drive: Siemens SINAMICS G120C")
        self.assertEqual(p.query("Q?"), "Q? (Cutter drive: Siemens SINAMICS G120C)")
        self.assertEqual(plan("What torque for the die plate bolts?", PARTS, set()).query("Q?"), "Q?")

    def test_a_part_named_whole_drops_the_near_ones(self):
        self.assertEqual(plan("How should the water injection flow meter be cleaned?", PARTS, set()).doc_keys, ["PROMAG"])
        self.assertEqual(plan("How do I clean the flow meter of the steam line?", PARTS, set()).doc_keys, ["PROWIRL"])

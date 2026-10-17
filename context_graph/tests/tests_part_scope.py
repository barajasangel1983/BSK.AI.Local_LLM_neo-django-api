"""P21: a manual run with a part chosen gives part-scoped ids and matches within the part's family."""

from django.test import SimpleTestCase, override_settings

from context_graph.extraction import part_scope_for
from context_graph.identity import EntityIndex, score, normalize, Node


def index() -> EntityIndex:
    i = EntityIndex()
    # The parts list (BOM) and a manual sub-part of another motor.
    for cid, name in (("cutter", "Cutter"), ("cutter-motor", "Cutter motor"), ("cutter-drive", "Cutter drive"),
                      ("main-motor", "Extruder motor"), ("feeder-screw-motor", "Feed screw motor")):
        i.add(f"bsk:component:EXTR01/{cid}", "Component", name)
        i.bom.add(f"bsk:component:EXTR01/{cid}")
    i.add("bsk:component:EXTR01/Bearings", "Component", "Bearings")                      # the ABB motor's
    i.parents = {"bsk:component:EXTR01/cutter-motor": "bsk:component:EXTR01/cutter",
                 "bsk:component:EXTR01/cutter-drive": "bsk:component:EXTR01/cutter",
                 "bsk:component:EXTR01/Bearings": "bsk:component:EXTR01/main-motor"}
    # Facts: the ABB motor's inspection, the drive's fault and its procedure, an asset-level procedure.
    i.add("bsk:procedure:EXTR01/General-inspection", "Procedure", "General inspection")
    i.add("bsk:alarm:EXTR01/0001", "Alarm", "0001")
    i.add("bsk:procedure:EXTR01/A798-what-to-do", "Procedure", "A798: what to do")
    i.add("bsk:procedure:EXTR01/Daily-walkdown", "Procedure", "Daily walkdown")
    i.owners = {"bsk:procedure:EXTR01/General-inspection": {"bsk:component:EXTR01/main-motor"},
                "bsk:alarm:EXTR01/0001": {"bsk:component:EXTR01/main-drive-vfd"},
                "bsk:procedure:EXTR01/A798-what-to-do": {"bsk:component:EXTR01/main-drive-vfd"}}
    return i


@override_settings(GRAPH_PART_SCOPED_IDS=True)
class PartScopeTests(SimpleTestCase):
    def setUp(self):
        self.i = index()
        self.part = part_scope_for({"anchor_id": "bsk:component:EXTR01/cutter-motor"}, self.i, "EXTR01")

    def r(self, label, name, part=True):
        return self.i.resolve(label, name, "EXTR01", part=self.part if part else None)

    def test_the_family_and_the_key(self):
        self.assertEqual(self.part.key, "EXTR01/cutter-motor")
        self.assertEqual(self.part.family, {"bsk:component:EXTR01/cutter-motor", "bsk:component:EXTR01/cutter"})

    def test_new_facts_get_ids_under_the_part(self):
        m = self.r("Procedure", "Plastic fan removal")
        self.assertEqual((m.kind, m.id), ("new", "bsk:procedure:EXTR01/cutter-motor/Plastic-fan-removal"))
        self.assertEqual(self.r("Component", "Bearing").id, "bsk:component:EXTR01/cutter-motor/Bearing")
        # Without a part: as before.
        self.assertEqual(self.r("Procedure", "Plastic fan removal", part=False).id,
                         "bsk:procedure:EXTR01/Plastic-fan-removal")

    def test_another_parts_facts_are_never_matched(self):
        self.assertEqual(self.r("Procedure", "General inspection").kind, "new")             # the ABB motor's
        self.assertTrue(self.r("Procedure", "General inspection", part=False).existing)     # as before
        self.assertEqual(self.r("Component", "Bearings").kind, "new")                        # the ABB motor's sub-part
        self.assertEqual(self.r("Component", "Rolling bearings").kind, "new")                # no "possible"

    def test_the_parts_list_and_unowned_facts_still_match(self):
        self.assertEqual(self.r("Component", "Cutter drive").kind, "name")                   # BOM, outside the family
        self.assertEqual(self.r("Component", "Cutter motor").kind, "name")
        self.assertEqual(self.r("Procedure", "Daily walkdown").kind, "name")                 # asset-level

    def test_differing_codes_are_never_suggested(self):
        self.assertEqual(self.r("Procedure", "F01000: what to do", part=False).kind, "new")  # vs A798: what to do
        self.assertEqual(self.r("Alarm", "F30001", part=False).kind, "new")                  # vs 0001
        self.assertEqual(score(normalize("barrel zone 3"), Node("x", "Component", "Barrel zone 2")), 0.0)

    def test_an_explicit_id_matches_anywhere(self):
        self.assertEqual(self.r("Procedure", "bsk:procedure:EXTR01/General-inspection").kind, "id")

    @override_settings(GRAPH_PART_SCOPED_IDS=False)
    def test_the_setting_off(self):
        self.assertIsNone(part_scope_for({"anchor_id": "bsk:component:EXTR01/cutter-motor"}, self.i, "EXTR01"))
        self.assertIsNone(part_scope_for({}, self.i, "EXTR01"))

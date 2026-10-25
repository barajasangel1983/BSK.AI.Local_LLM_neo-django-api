"""P26e (2026-10-25): a part question gets the part's own, model's and family's facts. Parameters, terminals and
interfaces are fact lines; a family's or model's fact says where it comes from; a part in focus gives its matching
facts a bonus; a part is found by its model or family name and by all but one word of its name; a generic rating
name ("Output") is no focus."""

from unittest.mock import patch

from django.test import SimpleTestCase

from chat import asset_context as ac
from context_service import service

DRIVE, FEED = "bsk:component:EXTR01/cutter-drive", "bsk:component:EXTR01/feeder-screw-drive"
FAMILY, MODEL = "bsk:product-family:siemens/sinamics-g120c", "bsk:device-model:siemens/sinamics-g120c-7.5"
PRODUCTS = {FAMILY: {"level": "family", "name": "SINAMICS G120C"}, MODEL: {"level": "model", "name": "G120C 7.5 kW"}}


def node(nid, name, owners, parts=(DRIVE,), **props):
    return {"id": nid, "name": name, "properties": props, "components": list(parts), "owners": list(owners)}


class LevelTests(SimpleTestCase):
    def test_a_family_or_model_fact_says_so_a_parts_own_does_not(self):
        self.assertEqual(ac._level([FAMILY], PRODUCTS), ("family", FAMILY, " (SINAMICS G120C family manual)"))
        self.assertEqual(ac._level([MODEL, FAMILY], PRODUCTS)[0], "family")
        self.assertEqual(ac._level([MODEL], PRODUCTS), ("model", MODEL, " (G120C 7.5 kW model data)"))
        self.assertEqual(ac._level([DRIVE], PRODUCTS), ("part", "", ""))

    def test_parameters_terminals_and_interfaces_are_fact_lines(self):
        ctx = {"products": PRODUCTS, "alarms": [],
               "parameters": [node("bsk:parameter:siemens/sinamics-g120c/p1082", "Maximum speed", [FAMILY], (DRIVE, FEED),
                                   code="p1082", subtype="Setting", default=1500, min=0, max=210000, unit="rpm")],
               "io_points": [node("bsk:io-point:siemens/sinamics-g120c/X138-5", "DI 0", [FAMILY], terminal="5",
                                  connector="-X138", subtype="Digital input")],
               "interfaces": [node("bsk:interface:siemens/sinamics-g120c/PROFINET", "PROFINET", [FAMILY], telegram="1, 20")]}
        lines, _ = ac._part_lines(ctx, {DRIVE: "Cutter drive", FEED: "Feed screw drive"}, set(), set())
        texts = {ln.section: ln.text for ln in lines}
        self.assertEqual(texts["PARAMETERS"], "- p1082 Maximum speed of Cutter drive, Feed screw drive: Setting, default 1500, "
                                              "range 0 to 210000, unit rpm (SINAMICS G120C family manual)")
        self.assertEqual(texts["TERMINALS"], "- DI 0 of Cutter drive: Digital input, terminal 5, connector -X138 "
                                             "(SINAMICS G120C family manual)")
        self.assertEqual(texts["INTERFACES"], "- PROFINET of Cutter drive: telegram 1, 20 (SINAMICS G120C family manual)")
        p1082 = next(ln for ln in lines if ln.section == "PARAMETERS")
        self.assertEqual((p1082.parts, p1082.level, p1082.owner), ((DRIVE, FEED), "family", FAMILY))


class RankingTests(SimpleTestCase):
    def test_a_part_in_focus_gives_its_matching_facts_a_bonus(self):
        lines = [ac.Line("SPECIFICATIONS", "- Maximum speed of Feed screw motor", node_id="a", parts=("motor",)),
                 ac.Line("PARAMETERS", "- p1082 Maximum speed of Cutter drive", node_id="b", parts=(DRIVE,)),
                 ac.Line("PARAMETERS", "- p1120 Ramp-up time of Cutter drive", node_id="c", parts=(DRIVE,)),
                 ac.Line("PART_PROCEDURES", "- Maximum speed limit too low: check the maximum speed", node_id="d")]
        order = [lines[i].node_id for i, _ in ac._ranked(lines, "maximum speed", {DRIVE})]
        self.assertEqual(order[0], "b")                 # the part's matching fact, before an equal one of another part
        self.assertEqual(order[-1], "c")                # the part's non-matching facts after every matching one
        self.assertLess(order.index("d"), order.index("c"))

    def test_a_plural_finds_its_singular(self):
        self.assertEqual(ac._tokens("Which motors are made by Siemens"), {"motor", "siemen"})   # facts fold the same way
        self.assertEqual(ac._tokens("pressure status bus"), {"pressure", "status", "bus"})


class FocusTests(SimpleTestCase):
    def entity(self, nid, label, name, keys=(), group_keys=()):
        return {"id": nid, "label": label, "name": name, "keys": [name, *keys], "group_keys": list(group_keys)}

    def resolve(self, question, known):
        with patch.object(ac, "check_asset", return_value="Extruder EXTR01"), patch.object(ac, "entities", return_value=known):
            return service.resolve(question, "bsk:asset:EXTR01")

    def test_generic_rating_names_are_no_focus_codes_and_long_names_are(self):
        self.assertFalse(service._focus_key("Specification", "Output"))
        self.assertFalse(service._focus_key("Specification", "Measuring range"))
        self.assertTrue(service._focus_key("Specification", "Nominal output current"))
        self.assertTrue(service._focus_key("Parameter", "p1082"))
        self.assertTrue(service._focus_key("Component", "PLC"))

    def test_a_part_named_with_all_its_words_but_one(self):
        psu = self.entity("bsk:component:EXTR01/psu", "Component", "24 V DC power supply", keys=("power supply",))
        terminals = [self.entity(f"bsk:io-point:x/{n}", "IOPoint", n, keys=("24 V power supply",)) for n in ("+24 V IN", "+24 V OUT")]
        zone2 = self.entity("bsk:component:EXTR01/zone-2", "Component", "Barrel zone 2 temperature sensor")
        out = self.resolve("24 V power supply output current rating", [psu, *terminals, zone2])
        self.assertEqual(([f["name"] for f in out["focus"]], out["ambiguous"]), (["24 V DC power supply"], []))
        out = self.resolve("barrel zone 1 temperature sensor range", [zone2])
        self.assertEqual(out["focus"], [])              # the missing word is a number: another zone

    def test_a_family_name_focuses_every_part_of_it(self):
        drive = self.entity(DRIVE, "Component", "Cutter drive", group_keys=("SINAMICS G120C",))
        feed = self.entity(FEED, "Component", "Feed screw drive", group_keys=("SINAMICS G120C",))
        out = self.resolve("SINAMICS G120C maximum speed", [drive, feed])
        self.assertEqual(({f["id"] for f in out["focus"]}, out["ambiguous"]), ({DRIVE, FEED}, []))

"""P16a (2026-10-16): the chat reads the parts' alarms, procedures and failure modes.

Schema v3 puts what the manuals say on the components (Component -HAS_ALARM-> / -HAS_PROCEDURE-> /
-HAS_FAILURE_MODE->); the chat's fact sheet read only the asset's own alarms and procedures. Unit tests
mock the graph; the integration test reads the new keys from the throwaway Neo4j:

    GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test chat.test_part_facts
"""

import copy
import os
import unittest
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase, override_settings

from chat import asset_context, views
from chat.test_asset_chat import ASSET, CONTEXT, SEED, node
from context_graph import driver, services

TEST_URI = os.getenv("GRAPH_TEST_URI")
DRIVE = "bsk:component:EXTR01/main-drive-vfd"
MOTOR = "bsk:component:EXTR01/main-motor"
A581 = "bsk:alarm:EXTR01/A581"
FAN_FIX = "bsk:procedure:EXTR01/A581-what-to-do"
REGREASE = "bsk:procedure:EXTR01/Regreasing-ball-bearings"
STALL = "bsk:failure-mode:EXTR01/Motor-stalls"
STALL_FIX = "bsk:procedure:EXTR01/Motor-stalls-what-to-do"


def part_context(filler: int = 0) -> dict:
    """CONTEXT with a drive and a motor, their alarms, procedures and a failure mode; `filler` more drive alarms."""
    ctx = copy.deepcopy(CONTEXT)
    ctx["components"] += [
        {**node(DRIVE, "Component", "Main drive frequency converter", model="ACS880-07"), "parent": ASSET,
         "alarms": [], "children": [], "signals": []},
        {**node(MOTOR, "Component", "Extruder motor", model="M3BP 355MLA 4"), "parent": ASSET,
         "alarms": [], "children": [], "signals": []},
    ]
    fan_fix = node(FAN_FIX, "Procedure", "A581: what to do", subtype="Corrective",
                   summary="Check parameter 95.20 HW options word 1, bit 14; identify the fan by the auxiliary code.")
    ctx["part_alarms"] = [
        {**node(A581, "Alarm", "A581", subtype="Device fault", description="Fan"), "components": [DRIVE],
         "procedures": [fan_fix]},
        *[{**node(f"bsk:alarm:EXTR01/F{n:03d}", "Alarm", f"F{n:03d}", subtype="Device fault",
                  description=f"Main drive internal condition {n}"), "components": [DRIVE], "procedures": []}
          for n in range(filler)],
    ]
    ctx["part_procedures"] = [
        {**node(REGREASE, "Procedure", "Regreasing (ball bearings)", subtype="Lubrication", interval="5,600 h",
                amount="70 g/bearing", summary="From Table 7.2, frame size 355, 1500 r/min."),
         "components": [MOTOR], "addresses": []},
    ]
    ctx["failure_modes"] = [
        {**node(STALL, "FailureMode", "Motor stalls", subtype="Electrical",
                causes=["One phase may be open", "Wrong application", "Overload"]),
         "components": [MOTOR], "indicated_by": [],
         "fixes": [node(STALL_FIX, "Procedure", "Motor stalls: what to do", subtype="Corrective",
                        summary="One phase may be open: check lines for open phase; Overload: reduce load.")]},
    ]
    return ctx


def build(ctx, question="x", budget=10**6, focus=None):
    with patch("context_graph.services.get_asset_context", return_value=ctx), \
         patch("chat.asset_context.historian_snapshot", return_value=None):
        return asset_context.build(ASSET, question, budget, focus)


class PartFactsTests(SimpleTestCase):
    def test_the_parts_alarms_procedures_and_failure_modes(self):
        text = build(part_context()).text
        self.assertIn("Alarms on parts:\n- A581 (Device fault) on Main drive frequency converter: Fan; "
                      "procedure: A581: what to do", text)
        self.assertIn("- Regreasing (ball bearings) (Lubrication) for Extruder motor: interval 5,600 h; amount 70 g/bearing: "
                      "From Table 7.2, frame size 355, 1500 r/min", text)
        # A procedure reached from its alarm is listed once, with the part it is for and the alarm it addresses.
        self.assertEqual(text.count("- A581: what to do"), 1)
        self.assertIn("- A581: what to do (Corrective) for Main drive frequency converter: Check parameter 95.20", text)
        self.assertIn("; addresses: A581", text)
        self.assertIn("Failure modes:\n- Motor stalls (Electrical) on Extruder motor: causes: One phase may be open; "
                      "Wrong application; Overload; fix: Motor stalls: what to do", text)
        self.assertIn("- Motor stalls: what to do (Corrective) for Extruder motor", text)

    def test_without_part_facts_the_sheet_is_as_before(self):
        empty = {**copy.deepcopy(CONTEXT), "part_alarms": [], "part_procedures": [], "failure_modes": []}
        self.assertEqual(build(empty).text, build(copy.deepcopy(CONTEXT)).text)

    def test_a_long_summary_is_cut(self):
        ctx = part_context()
        ctx["part_procedures"][0]["properties"]["summary"] = "Step. " * 300
        line = next(ln for ln in build(ctx).text.splitlines() if ln.startswith("- Regreasing"))
        self.assertLess(len(line), asset_context.SUMMARY_CHARS + 150)
        self.assertTrue(line.endswith("…"))

    def test_focus_keeps_an_alarm_with_its_procedure_in_a_small_budget(self):
        ctx = part_context(filler=80)
        small = build(ctx, "The main drive shows warning A581 Fan", 1800, {A581}).text
        self.assertIn("- A581 (Device fault)", small)
        self.assertIn("- A581: what to do", small)            # brought by its alarm

    def test_a_word_on_every_alarm_counts_less(self):
        # "main drive" is on each of the 80 filler alarms; "regreasing" is on one line only.
        small = build(part_context(filler=80), "Regreasing interval for the main drive motor", 1800).text
        self.assertIn("- Regreasing (ball bearings)", small)

    def test_entities_include_the_parts_facts(self):
        with patch("context_graph.services.get_asset_context", return_value=part_context()):
            found = {e["id"]: e for e in asset_context.entities(ASSET)}
        self.assertEqual(found[A581]["label"], "Alarm")
        self.assertEqual(found[STALL]["label"], "FailureMode")
        self.assertEqual(found[FAN_FIX]["label"], "Procedure")
        self.assertIn(REGREASE, found)
        self.assertEqual(len([e for e in found if e == FAN_FIX]), 1)

    def test_the_chat_focus_names_every_candidate_of_an_ambiguous_phrase(self):
        resolved = {"asset": {"id": ASSET}, "focus": [{"id": A581}],
                    "ambiguous": [{"phrase": "motor stalls", "candidates": [{"id": STALL}, {"id": "bsk:failure-mode:x"}]}]}
        with patch("context_service.service.resolve", return_value=resolved):
            self.assertEqual(views._focus(ASSET, "q"), {A581, STALL, "bsk:failure-mode:x"})


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class PartFactsQueryTests(TestCase):
    def setUp(self):
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
        services.init_graph()
        services.load_seed(SEED)
        drive = "bsk:component:EXTR01/main-drive"
        with driver.session() as s:
            s.run(
                "MATCH (c:Entity {id: $drive}) "
                "CREATE (c)-[:HAS_ALARM]->(al:Entity:Alarm {id: $alarm, name: 'A581'}) "
                "CREATE (p:Entity:Procedure {id: $fix, name: 'A581: what to do'})-[:ADDRESSES]->(al) "
                "CREATE (c)-[:HAS_PROCEDURE]->(:Entity:Procedure {id: $grease, name: 'Regreasing', interval: '5,600 h'}) "
                "CREATE (c)-[:HAS_FAILURE_MODE]->(f:Entity:FailureMode {id: $fm, name: 'Motor stalls'}) "
                "CREATE (f)-[:MITIGATED_BY]->(:Entity:Procedure {id: $fmfix, name: 'Motor stalls: what to do'}) "
                "CREATE (f)-[:INDICATED_BY]->(al)",
                drive=drive, alarm=A581, fix=FAN_FIX, grease=REGREASE, fm=STALL, fmfix=STALL_FIX).consume()

    def test_the_new_keys(self):
        ctx = services.get_asset_context(ASSET)
        drive = "bsk:component:EXTR01/main-drive"
        self.assertEqual([(a["id"], a["components"], [p["id"] for p in a["procedures"]]) for a in ctx["part_alarms"]],
                         [(A581, [drive], [FAN_FIX])])
        self.assertEqual([(p["id"], p["components"], p["properties"]["interval"]) for p in ctx["part_procedures"]],
                         [(REGREASE, [drive], "5,600 h")])
        self.assertEqual([(f["id"], f["components"], [p["id"] for p in f["fixes"]], f["indicated_by"])
                          for f in ctx["failure_modes"]], [(STALL, [drive], [STALL_FIX], [A581])])
        self.assertEqual((ctx["counts"]["part_alarms"], ctx["counts"]["part_procedures"], ctx["counts"]["failure_modes"]),
                         (1, 1, 1))
        self.assertEqual((ctx["counts"]["alarms"], ctx["counts"]["procedures"]), (20, 8))     # the asset's own: unchanged

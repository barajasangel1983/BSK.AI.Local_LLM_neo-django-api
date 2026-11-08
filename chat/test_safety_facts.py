"""P22g (2026-11-08): the parts' safety notices in the asset chat, and the question's facet.

The approved Safety facts (Hazard -PROTECTED_BY-> SafetyMeasure, ProductFamily -PROVIDES-> SafetyFunction,
Procedure -REQUIRES-> SafetyMeasure) never reached the chat. With CHAT_SAFETY_FACTS on they join the fact sheet, and a
safety question puts the facts of the parts it is about first. Unit tests mock the graph.
"""

from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from chat import asset_context, planner
from chat.test_asset_chat import ASSET, node
from chat.test_part_facts import DRIVE, MOTOR, build, part_context

SHOCK = "bsk:hazard:EXTR01/Electric-shock-from-residual-charges"
HOT = "bsk:hazard:EXTR01/Hot-surfaces"
LIFT = "bsk:hazard:EXTR01/Falling-motor"
STO = "bsk:safety-function:EXTR01/STO"


def safety_context(filler: int = 0) -> dict:
    """part_context() with hazards on the drive and the motor, a safety function and a procedure's measure;
    `filler` more drive parameters that share the question's words."""
    ctx = part_context()
    ctx["hazards"] = [
        {**node(SHOCK, "Hazard", "Electric shock from residual charges", severity="WARNING", facets=["Safety"],
                consequence="death or serious injury"),
         "components": [DRIVE], "owners": [DRIVE],
         "measures": [node("bsk:safety-measure:EXTR01/wait", "SafetyMeasure", "Wait 5 minutes"),
                      node("bsk:safety-measure:EXTR01/check", "SafetyMeasure", "Check voltage at converter connections")]},
        {**node(HOT, "Hazard", "Burns from hot surfaces", severity="CAUTION", facets=["Safety"]),
         "components": [DRIVE], "owners": [DRIVE], "measures": []},
        {**node(LIFT, "Hazard", "Falling motor during lifting", severity="DANGER", facets=["Safety"]),
         "components": [MOTOR], "owners": [MOTOR],
         "measures": [node("bsk:safety-measure:EXTR01/eyes", "SafetyMeasure", "Use all the lifting eyes")]},
    ]
    ctx["safety_functions"] = [{**node(STO, "SafetyFunction", "Safe Torque Off (STO)", facets=["Safety"]),
                                "components": [DRIVE], "owners": [DRIVE]}]
    ctx["part_procedures"][0]["requires"] = ["Motor at standstill"]
    ctx["parameters"] = [
        {**node(f"bsk:parameter:EXTR01/p{n}", "Parameter", f"Drive open loop setting {n}", facets=["Setup"]),
         "components": [DRIVE], "owners": [DRIVE]} for n in range(filler)]
    return ctx


class SafetyFactsTests(SimpleTestCase):
    def test_off_leaves_the_fact_sheet_as_before(self):
        with override_settings(CHAT_SAFETY_FACTS=False):
            self.assertEqual(build(safety_context()).text, build(part_context()).text)
            self.assertEqual(planner.question_kind("Is it safe to open the drive?").facet, "")

    @override_settings(CHAT_SAFETY_FACTS=True)
    def test_hazards_with_their_measures_and_the_safety_functions(self):
        text = build(safety_context()).text
        self.assertIn("Safety notices of parts", text)
        self.assertIn("- Electric shock from residual charges (WARNING) on Main drive frequency converter: "
                      "consequence: death or serious injury; measures: Wait 5 minutes; "
                      "Check voltage at converter connections", text)
        self.assertIn("- Burns from hot surfaces (CAUTION) on Main drive frequency converter", text)
        self.assertIn("Safety functions of parts:\n- Safe Torque Off (STO) (safety function) of Main drive", text)
        # A procedure's required measures only in a safety question (they lengthen every procedure line).
        self.assertNotIn("; requires:", text)
        safety_text = "\n".join(x["text"] for x in _items(safety_context(), "x", 10**6, set(), "Safety"))
        self.assertIn("; requires: Motor at standstill", safety_text)

    @override_settings(CHAT_SAFETY_FACTS=True)
    def test_another_question_ranks_as_without_the_safety_facts(self):
        """In a question of another facet the safety notices only use room that is left, and don't change the
        other facts' word weights: the same facts are kept as with the switch off."""
        q = "What is the open loop setting of the main drive?"
        on = [x["entity_id"] for x in _items(safety_context(filler=40), q, 2600, {DRIVE}, "")]
        with override_settings(CHAT_SAFETY_FACTS=False):
            off = [x["entity_id"] for x in _items(safety_context(filler=40), q, 2600, {DRIVE}, "")]
        self.assertEqual([i for i in on if i not in (SHOCK, HOT, LIFT, STO)], off)

    @override_settings(CHAT_SAFETY_FACTS=True)
    def test_the_question_facet(self):
        for q in ("Is it safe to open the cutter drive?", "What PPE do I need for the pump?",
                  "Before I work on the gearbox heater, what do I do?", "How do I lock out the main drive?"):
            self.assertEqual(planner.question_kind(q).facet, "Safety", q)
        for q in ("The drive shows warning A581, what does it mean?", "What is the rated current of the motor?"):
            self.assertEqual(planner.question_kind(q).facet, "", q)
        self.assertEqual(planner.question_kind("Is it safe to open the drive?").facts_share, 0.4)

    @override_settings(CHAT_SAFETY_FACTS=True)
    def test_a_safety_question_puts_the_parts_safety_facts_first(self):
        ctx = safety_context(filler=40)
        q = "Is it safe to open the main drive?"
        with_facet = [x["entity_id"] for x in _items(ctx, q, budget=2600, focus={DRIVE}, facet="Safety")]
        self.assertIn(SHOCK, with_facet)
        self.assertIn(HOT, with_facet)
        # Another part's hazard is not pulled in by the facet; without the facet the drive's parameters win the room.
        self.assertNotIn(LIFT, with_facet)
        without = [x["entity_id"] for x in _items(ctx, q, budget=2600, focus={DRIVE}, facet="")]
        self.assertNotIn(SHOCK, without)

    @override_settings(CHAT_SAFETY_FACTS=True)
    def test_the_stronger_signal_word_first_on_a_tie(self):
        ctx = safety_context()
        lines, _ = asset_context._part_lines(ctx, {DRIVE: "Main drive frequency converter", MOTOR: "Extruder motor"},
                                             set(), set())
        for ln in lines:
            ln.facets = ("Safety",) if ln.section in ("SAFETY", "SAFETY_FUNCTIONS") else ()
        ranked = [lines[i].node_id for i, _ in asset_context._ranked(lines, "x", {DRIVE, MOTOR}, "Safety")]
        safety = [n for n in ranked if n in (SHOCK, HOT, LIFT)]
        self.assertEqual(safety, [LIFT, SHOCK, HOT])     # DANGER, WARNING, CAUTION

    @override_settings(CHAT_SAFETY_FACTS=True)
    def test_the_answer_rule_only_when_on(self):
        ctx = build(safety_context())
        self.assertIn("give its measures first", asset_context.system_prompt(ctx))
        with override_settings(CHAT_SAFETY_FACTS=False):
            self.assertNotIn("give its measures first", asset_context.system_prompt(ctx))


def _items(ctx, question, budget, focus, facet):
    with patch("context_graph.services.get_asset_context", return_value=ctx), \
         patch("chat.asset_context.historian_snapshot", return_value=None):
        return asset_context.build(ASSET, question, budget, focus, facet=facet).items

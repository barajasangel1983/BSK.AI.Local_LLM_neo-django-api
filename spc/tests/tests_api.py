"""The SPC endpoints (specification section 6). The graph and the historian are replaced by fakes."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from chat.test_asset_chat import ASSET, node, signal
from spc.models import SpcCursor, SpcLimits, SpcTool

NOW = datetime(2026, 10, 12, 16, 0, tzinfo=timezone.utc)
MOISTURE, PRESSURE, TORQUE = (f"bsk:signal:EXTR01/{k}" for k in ("product_moisture_pct", "die_pressure_bar", "screw_torque_pct"))
CONTEXT = {
    "asset": node(ASSET, "Asset", "Extruder EXTR01"),
    "components": [{**node("bsk:component:EXTR01/die-head", "Component", "Die Head"), "children": [],
                    "signals": [signal("die_pressure_bar", "Die pressure", "bar", 84.3, 99.2)]}],
    "signals": [signal("product_moisture_pct", "Product moisture", "%", 8.15, 10.85), signal("screw_torque_pct", "Screw torque", "%"),
                signal("recipe_id", "Recipe", ""), signal("machine_state", "Machine state", ""), signal("shift_code", "Shift", "")],
}


def row(ts):
    """The plant: stopped from minute 30 to 35 of every hour; recipe RC-A in even hours, RC-B in odd ones."""
    i = int(ts.timestamp()) // 20
    swing = 1 if i % 2 else -1
    recipe = "RC-A" if ts.hour % 2 == 0 else "RC-B"
    return {"ts": ts, "machine_state": "STOPPED" if 30 <= ts.minute < 35 else "RUNNING", "recipe_id": recipe,
            "product_moisture_pct": 9.5 + 0.4 * swing, "die_pressure_bar": (94 if recipe == "RC-A" else 90) + swing,
            "screw_torque_pct": 95 + swing}


def historian_rows(asset_key, columns, start, end, limit):
    first = start.replace(second=start.second // 20 * 20, microsecond=0) + timedelta(seconds=20)
    stamps = [first + timedelta(seconds=20 * i) for i in range(int((end - first).total_seconds() // 20) + 1)]
    return [{k: v for k, v in row(ts).items() if k == "ts" or k in columns} for ts in stamps]


@override_settings(HISTORIAN_SOURCE="live", SPC_MAX_TOOLS_PER_ASSET=3)
class SpcApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.reads = patch("chat.asset_context.historian_rows", side_effect=historian_rows).start()
        for target, value in (("context_graph.services.get_asset_context", CONTEXT),
                              ("spc.service.newest_row", {"ts": NOW, "machine_state": "RUNNING"}),
                              ("spc.service.recent_states", ["RUNNING", "STOPPED"])):
            patch(target, return_value=value).start()
        self.addCleanup(patch.stopall)

    def add(self, signal_id=MOISTURE, **fields):
        return self.client.post("/api/spc/tools/", {"asset_id": ASSET, "signal_id": signal_id, **fields}, format="json")

    def baseline(self, tool_id, preview=False, hours=8):
        return self.client.post(f"/api/spc/tools/{tool_id}/baseline/", {
            "start": (NOW - timedelta(hours=hours)).isoformat(), "end": NOW.isoformat(), "preview": preview}, format="json")

    def screening(self, **params):
        return self.client.get("/api/spc/screening/", {"asset_id": ASSET, **params}).json()

    # --- 6.1 ---

    def test_the_signals_a_tool_can_be_made_for_come_from_the_graph_with_what_is_proposed(self):
        body = self.client.get("/api/spc/signals/", {"asset_id": ASSET}).json()
        self.assertEqual((body["max_tools"], body["tools"], body["states"]), (3, 0, ["RUNNING", "STOPPED"]))
        self.assertEqual([s["name"] for s in body["signals"]], ["Die pressure", "Product moisture", "Screw torque"])   # numeric only
        self.assertEqual(body["group_columns"], ["recipe_id", "shift_code"])        # text signals; the state is a filter of its own
        moisture = next(s for s in body["signals"] if s["signal_id"] == MOISTURE)
        self.assertEqual((moisture["column"], moisture["unit"], moisture["normal_low"], moisture["has_tool"]),
                         ("product_moisture_pct", "%", 8.15, False))
        self.assertEqual(moisture["proposed"], {"lsl": 8.0, "target": 9.5, "usl": 11.0, "group_by": "", "method": "imr",
                                                "states": ["RUNNING"], "ignore_seconds": 120, "sigma_multiplier": 3, "baseline_days": 7})
        pressure = next(s for s in body["signals"] if s["signal_id"] == PRESSURE)
        self.assertEqual((pressure["proposed"]["group_by"], pressure["proposed"]["lsl"]), ("recipe_id", None))
        # what a new tool's block opens with, and the rules it can switch on, come from here and not from the page
        self.assertEqual((body["defaults"]["rules"], body["defaults"]["states"], body["defaults"]["method"], body["defaults"]["window_hours"]),
                         (["beyond_limits", "spec", "no_variation"], ["RUNNING"], "imr", 8))
        self.assertEqual(len(body["rules"]), 8)
        self.assertIn({"id": "range_beyond_limits", "type": "control_violation"}, body["rules"])
        self.assertEqual(self.client.get("/api/spc/signals/").status_code, 400)

    # --- 6.2 ---

    def test_the_page_starts_with_no_tool_and_a_tool_is_added_changed_paused_and_deleted(self):
        self.assertEqual(self.client.get("/api/spc/tools/", {"asset_id": ASSET}).json(), {"tools": []})
        self.assertEqual(self.screening()["summary"]["selected_tools"], 0)

        made = self.add(lsl=8.0, target=9.5, usl=11.0)
        self.assertEqual(made.status_code, 201)
        tool = made.json()
        self.assertEqual((tool["name"], tool["unit"], tool["column"], tool["version"], tool["limits"]),
                         ("Product moisture", "%", "product_moisture_pct", 1, []))        # name and unit from the graph
        self.assertEqual((tool["states"], tool["ignore_seconds"], tool["method"], tool["window_hours"]), (["RUNNING"], 120, "imr", 8))
        self.assertEqual(tool["rules"], ["beyond_limits", "spec", "no_variation"])   # the range chart and the run rules are off
        self.assertTrue(SpcCursor.objects.filter(tool_id=tool["id"]).exists())
        self.assertTrue(self.client.get("/api/spc/signals/", {"asset_id": ASSET}).json()["signals"][1]["has_tool"])

        url = f"/api/spc/tools/{tool['id']}/"
        changed = self.client.patch(url, {"usl": 10.9, "name": "Moisture at the dryer", "rules": ["spec", "beyond_limits", "no_variation"]},
                                    format="json").json()
        self.assertEqual((changed["usl"], changed["name"], changed["version"], changed["limits_cleared"]),
                         (10.9, "Moisture at the dryer", 2, False))
        self.assertEqual(changed["rules"], ["beyond_limits", "spec", "no_variation"])
        self.assertEqual(self.client.patch(url, {"usl": 10.9}, format="json").json()["version"], 2)   # nothing changed: same version
        self.assertEqual(self.client.patch(url, {"enabled": False}, format="json").json()["version"], 3)
        self.assertEqual(self.screening()["tools"][0]["status"], "paused")

        self.assertEqual(self.client.delete(url).status_code, 204)
        self.assertEqual(self.client.get(url).status_code, 404)
        self.assertEqual(self.client.get("/api/spc/tools/not-a-tool/").status_code, 404)
        self.assertEqual(SpcCursor.objects.count(), 0)

    def test_what_is_refused_and_why(self):
        self.assertEqual(self.add().status_code, 201)
        again = self.add()
        self.assertEqual((again.status_code, again.json()["error"]), (409, "this signal already has an SPC tool"))
        self.assertEqual(self.add(PRESSURE).status_code, 201)
        self.assertEqual(self.add(TORQUE).status_code, 201)
        tool_id = SpcTool.objects.get(signal_id=TORQUE).id
        SpcTool.objects.filter(signal_id=MOISTURE).delete()
        self.assertEqual(self.add().status_code, 201)                                # three again
        SpcTool.objects.filter(signal_id=MOISTURE).update(signal_id="bsk:signal:EXTR01/other")
        full = self.add()
        self.assertEqual(full.status_code, 409)
        self.assertIn("already has 3 SPC tools, the most allowed", full.json()["error"])

        SpcTool.objects.exclude(id=tool_id).delete()
        for fields, why in (({"signal_id": "bsk:signal:EXTR01/recipe_id"}, "no numeric historian column"),
                            ({"signal_id": "bsk:signal:EXTR01/nope"}, "no numeric historian column"),
                            ({"lsl": 11, "usl": 8}, "LSL must be below USL"), ({"lsl": 8, "usl": 11, "target": 12}, "between LSL and USL"),
                            ({"method": "ewma"}, "imr or xbar_r"), ({"subgroup_size": 11}, "between 2 and 10"),
                            ({"rules": ["nelson_9"]}, "rules must be among"), ({"states": []}, "at least one machine state"),
                            ({"group_by": "no_such_column"}, "no text column"), ({"group_by": "die_pressure_bar"}, "no text column"),
                            ({"colour": "red"}, "unknown field: colour")):
            answer = self.add(**{"signal_id": MOISTURE, **fields})
            self.assertEqual(answer.status_code, 400, fields)
            self.assertIn(why, answer.json()["error"])
        moved = self.client.patch(f"/api/spc/tools/{tool_id}/", {"signal_id": MOISTURE}, format="json")
        self.assertEqual(moved.status_code, 400)
        self.assertEqual(self.client.post("/api/spc/tools/", {"asset_id": ASSET}, format="json").status_code, 400)

    # --- 6.3 ---

    def test_a_block_previews_its_baseline_before_the_tool_is_saved(self):
        window = {"start": (NOW - timedelta(hours=8)).isoformat(), "end": NOW.isoformat()}
        body = {"asset_id": ASSET, "signal_id": MOISTURE, "sigma_multiplier": 2, **window}
        preview = self.client.post("/api/spc/baseline/preview/", body, format="json")
        self.assertEqual(preview.status_code, 200)
        only = preview.json()["groups"][0]
        self.assertEqual((preview.json()["saved"], only["n_samples"]), (False, 1320 - 9 * 6))
        self.assertAlmostEqual(only["ucl"] - only["cl"], 2 * 0.8 / 1.128, places=6)       # the block's values, not the defaults
        self.assertEqual((SpcTool.objects.count(), SpcLimits.objects.count()), (0, 0))     # nothing is stored
        for bad, text in (({**body, "signal_id": "bsk:signal:EXTR01/none"}, "no numeric historian column"),
                          ({**body, "method": "ewma"}, "method must be"), ({**window}, "asset_id and signal_id")):
            answer = self.client.post("/api/spc/baseline/preview/", bad, format="json")
            self.assertEqual(answer.status_code, 400)
            self.assertIn(text, answer.json()["error"])

    def test_a_baseline_is_previewed_then_saved_and_frozen(self):
        tool = self.add(lsl=8.0, usl=11.0).json()
        self.assertEqual(self.screening()["tools"][0]["status"], "configuration_required")     # no limits yet
        preview = self.baseline(tool["id"], preview=True).json()
        self.assertEqual((preview["saved"], preview["skipped"], len(preview["groups"])), (False, [], 1))
        self.assertEqual(SpcLimits.objects.count(), 0)                               # a preview stores nothing
        only = preview["groups"][0]
        # 8 hours: 165 rows an hour while running, 1,320 in all, in 9 stretches (a stop every hour); 6 rows ignored at the start of each
        self.assertEqual((only["group"], only["n_samples"], round(only["mean"], 2), round(only["range_bar"], 3)), ("", 1320 - 9 * 6, 9.5, 0.8))
        self.assertAlmostEqual(only["sigma_within"], 0.8 / 1.128, places=6)
        self.assertAlmostEqual(only["ucl"] - only["cl"], 3 * 0.8 / 1.128, places=6)

        saved = self.baseline(tool["id"]).json()
        self.assertTrue(saved["saved"])
        stored = self.client.get(f"/api/spc/tools/{tool['id']}/").json()
        self.assertEqual((stored["version"], stored["limit_strategy"], len(stored["limits"])), (2, "baseline", 1))
        self.assertEqual((stored["baseline_end"][:19], stored["limits"][0]["source"], stored["limits"][0]["tool_version"]),
                         ("2026-10-12T16:00:00", "baseline", 2))
        view = self.screening()["tools"][0]
        self.assertEqual((view["status"], view["current_value"], view["machine_state"]), ("stable", 9.1, "RUNNING"))
        self.assertAlmostEqual(view["cpk"], 1.5 / (3 * 0.8 / 1.128), delta=0.005)    # the nearer limit is 1.5 away
        self.assertAlmostEqual(view["cp"], 3.0 / (6 * 0.8 / 1.128), places=3)
        self.assertAlmostEqual(view["sigma_shift"], 0.0, places=2)
        self.assertEqual((view["lsl"], view["usl"], view["violation_count"], len(view["trend"]) <= 60, sum(view["distribution"]["counts"])),
                         (8.0, 11.0, 0, True, 1266))

        # A change of how samples are made deletes the limits: they no longer fit.
        changed = self.client.patch(f"/api/spc/tools/{tool['id']}/", {"method": "xbar_r"}, format="json").json()
        self.assertEqual((changed["limits_cleared"], changed["limits"], changed["baseline_start"]), (True, [], None))
        self.assertEqual(self.screening()["tools"][0]["status"], "configuration_required")
        for body, why in (({"start": "yesterday", "end": NOW.isoformat()}, "ISO date"),
                          ({"start": NOW.isoformat(), "end": (NOW - timedelta(hours=1)).isoformat()}, "before its end"),
                          ({"start": (NOW - timedelta(days=40)).isoformat(), "end": NOW.isoformat()}, "at most 31 days")):
            answer = self.client.post(f"/api/spc/tools/{tool['id']}/baseline/", body, format="json")
            self.assertEqual(answer.status_code, 400)
            self.assertIn(why, answer.json()["error"])

    def test_limits_by_recipe_one_set_for_each_and_none_for_a_recipe_with_too_few_rows(self):
        tool = self.add(PRESSURE, group_by="recipe_id").json()
        found = self.baseline(tool["id"], preview=True, hours=3).json()             # 13:00 to 16:00: two hours of RC-B, one of RC-A
        # RC-A: two stretches (84 + 69 samples). RC-B: four (83 + 69 + 84 + 69). A recipe change starts a stretch, with its ignore period.
        self.assertEqual([(g["group"], g["n_samples"], round(g["mean"], 1)) for g in found["groups"]],
                         [("RC-A", 153, 94.0), ("RC-B", 305, 90.0)])
        with override_settings(SPC_MIN_BASELINE_SAMPLES=200):
            few = self.baseline(tool["id"], hours=3).json()
        self.assertEqual(few["skipped"], [{"group": "RC-A", "n_samples": 153, "reason": "fewer than 200 samples"}])
        self.assertEqual([g["group"] for g in few["groups"]], ["RC-B"])
        view = self.screening()["tools"][0]                                          # the newest row (16:00:00) is the first of RC-A
        self.assertEqual((view["current_group"], view["status"], view["status_reason"], view["cl"]),
                         ("RC-A", "configuration_required", "no control limits for RC-A", None))
        with patch("spc.service.newest_row", return_value={"ts": NOW - timedelta(seconds=20), "machine_state": "RUNNING"}):
            view = self.screening()["tools"][0]                                      # one row earlier the recipe was RC-B
        self.assertEqual((view["current_group"], view["status"], round(view["cl"], 1)), ("RC-B", "stable", 90.0))
        self.assertEqual((view["cpk"], view["capability_reason"]), (None, "no specification limit"))
        detail = self.client.get(f"/api/spc/tools/{tool['id']}/detail/").json()
        self.assertEqual(detail["groups_without_limits"], ["RC-A"])                  # its rows are shown, not judged
        self.assertTrue(all(s["rules"] == [] for s in detail["samples"] if s["group"] == "RC-A"))
        with override_settings(SPC_MIN_BASELINE_SAMPLES=500):
            none = self.baseline(tool["id"], hours=3)
        self.assertEqual((none.status_code, none.json()["error"]), (400, "no group has enough samples in this window: nothing was saved"))

    def test_fixed_limits_are_typed_by_the_user(self):
        tool = self.add().json()
        url = f"/api/spc/tools/{tool['id']}/limits/"
        done = self.client.put(url, {"groups": [{"group": "", "cl": 9.5, "lcl": 9.2, "ucl": 9.8}]}, format="json").json()
        self.assertEqual((done["saved"], done["groups"][0]["lcl"], round(done["groups"][0]["sigma_within"], 3)), (True, 9.2, 0.1))
        stored = self.client.get(f"/api/spc/tools/{tool['id']}/").json()
        self.assertEqual((stored["limit_strategy"], stored["limits"][0]["source"], stored["baseline_start"]), ("fixed", "fixed", None))
        detail = self.client.get(f"/api/spc/tools/{tool['id']}/detail/", {"window_hours": 1}).json()
        self.assertEqual((detail["window_hours"], len(detail["samples"])), (1, 153))
        self.assertTrue(all("beyond_limits" in s["rules"] for s in detail["samples"]))          # 9.1 and 9.9 against 9.2 to 9.8
        self.assertEqual([len(lim["spans"]) for lim in detail["limits"]], [1])                   # one group, so one span of limits
        self.assertEqual((detail["capability"]["reason"], detail["capability"]["target"]), ("no specification limit", None))
        self.assertLessEqual(detail["box"]["q1"], detail["box"]["median"])
        self.assertEqual((len(detail["box"]["quantiles"]), detail["box"]["quantiles"][5]["value"]), (11, detail["box"]["median"]))
        for groups, why in (([{"cl": 9.5, "lcl": 9.8, "ucl": 9.2}], "LCL < CL < UCL"), ([{"cl": 9.5}], "cl, lcl and ucl"),
                            ([], "at least one group"), ([{"group": "RC-A", "cl": 9.5, "lcl": 9.2, "ucl": 9.8}], "needs none")):
            answer = self.client.put(url, {"groups": groups}, format="json")
            self.assertEqual(answer.status_code, 400)
            self.assertIn(why, answer.json()["error"])

    # --- 6.4 and 6.5 ---

    def test_summary_and_screening_are_one_computation_from_one_read_for_all_tools(self):
        for signal_id in (MOISTURE, PRESSURE, TORQUE):
            self.baseline(self.add(signal_id, **({"lsl": 8.0, "usl": 11.0} if signal_id == MOISTURE else {})).json()["id"])
        paused = SpcTool.objects.get(signal_id=TORQUE)
        self.client.patch(f"/api/spc/tools/{paused.id}/", {"enabled": False}, format="json")

        self.reads.reset_mock()
        body = self.screening()
        self.assertEqual(self.reads.call_count, 2)                                   # 8 hours in two pieces, whatever the number of tools
        self.assertEqual(set(self.reads.call_args.args[1]), {"machine_state", "product_moisture_pct", "die_pressure_bar", "screw_torque_pct"})
        self.assertEqual((body["source"], body["period_seconds"], body["machine_state"], body["stale"]), ("historian_live", 20, "RUNNING", True))
        self.assertEqual(body["summary"], {
            "selected_tools": 3, "stable": 2, "warning": 0, "trending": 0, "out_of_control": 0, "specification_violations": 0,
            "paused": 1, "configuration_required": 0, "average_cpk": body["tools"][0]["cpk"], "lowest_cpk": body["tools"][0]["cpk"],
            "active_violations": 0})
        self.assertEqual(self.client.get("/api/spc/summary/", {"asset_id": ASSET}).json()["summary"], body["summary"])
        self.assertNotIn("tools", self.client.get("/api/spc/summary/", {"asset_id": ASSET}).json())

        one = self.screening(tools=str(paused.id))
        self.assertEqual((one["summary"]["selected_tools"], one["summary"]["paused"], one["summary"]["average_cpk"]), (1, 1, None))
        self.assertEqual(self.screening(tools="")["summary"]["selected_tools"], 0)
        stopped = self.screening(state="STOPPED")                                    # the tools apply while RUNNING only
        self.assertEqual([t["sample_count"] for t in stopped["tools"]], [0, 0, 0])
        self.assertEqual(self.screening(window_hours=1)["tools"][0]["sample_count"], 153)
        self.assertEqual(self.client.get("/api/spc/screening/", {"asset_id": ASSET, "window_hours": 99}).status_code, 400)

    def test_the_historian_or_the_graph_being_down_is_said(self):
        tool = self.add().json()
        from django.db import OperationalError
        self.reads.side_effect = OperationalError("timescale down")
        down = self.baseline(tool["id"], preview=True)
        self.assertEqual((down.status_code, down.json()["error"]), (503, "the historian is not available"))
        from context_graph.driver import GraphUnavailable
        with patch("context_graph.services.get_asset_context", side_effect=GraphUnavailable("neo4j down")):
            self.assertEqual(self.client.get("/api/spc/signals/", {"asset_id": ASSET}).status_code, 503)


class NoGroundTruthTests(TestCase):
    def test_the_injected_anomaly_record_is_never_read_by_the_spc_code(self):
        folder = Path(__file__).resolve().parent.parent
        for name in ("engine.py", "service.py", "views.py", "models.py"):
            text = (folder / name).read_text()
            self.assertNotIn("LiveSimEvent", text, name)
            self.assertNotIn("sim_events", text, name)

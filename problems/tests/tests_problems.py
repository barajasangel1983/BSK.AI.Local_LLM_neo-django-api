"""P29: problem-solving tickets — triggers, the 5-Why workflow and its rules, the models' steps, publishing, the API
and the MCP tools. The historian, the graph and the models are fakes."""

import asyncio
import base64
import io
import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from django.test import TestCase, TransactionTestCase, override_settings
from rest_framework.test import APIClient

from monitor import plant_calendar
from problems import llm, publish, service, triggers
from problems.models import ProblemSettings, Ticket
from problems.service import TicketError

ASSET = "bsk:asset:EXTR01"
T0 = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)
ENG = {"username": "maria", "name": "Maria", "role": "engineer", "source": "edge"}
ENG2 = {"username": "luis", "name": "Luis", "role": "engineer", "source": "edge"}
ASSOC = {"username": "pat", "name": "Pat", "role": "production_associate", "source": "edge"}
SIGNALS = [{"key": "machine_state", "name": "Machine state", "unit": "", "low": None, "high": None},
           {"key": "alarm_code", "name": "Active alarm code", "unit": "", "low": None, "high": None},
           {"key": "die_pressure_bar", "name": "Die pressure", "unit": "bar", "low": 84.3, "high": 99.2}]


class Historian:
    """The live historian: one row a minute; `state(ts)` says the machine state."""

    def __init__(self, test, now, state):
        self.now, self.state = now, state
        for target, fake in (("monitor.agent.clock", self.clock), ("context_service.service.signal_history", self.history),
                             ("problems.service.asset_name", lambda a: "EXTR01"),
                             ("monitor.plant_calendar.for_asset", lambda a: None),
                             ("problems.llm.recommend", lambda t: t)):
            p = patch(target, side_effect=fake)
            p.start()
            test.addCleanup(p.stop)

    def clock(self, asset_id, real=None):
        return {"now": self.now, "live": True, "replay": False, "latest": self.now, "data": {"stale": False}}

    def history(self, asset_id, start, end, signals=None, bucket_minutes=None):
        start, end = datetime.fromisoformat(start), datetime.fromisoformat(end)
        stamps = [start.replace(second=0) + timedelta(minutes=i) for i in range(1, int((end - start).total_seconds() // 60) + 1)]
        rows = [{"machine_state": self.state(t), "alarm_code": "E42" if self.state(t) == "ALARMED" else "",
                 "die_pressure_bar": 90.0} for t in stamps]
        return {"source": "historian_live", "period_seconds": 60, "signals": SIGNALS, "ts": [t.isoformat() for t in stamps],
                "series": {s["key"]: [r[s["key"]] for r in rows] for s in SIGNALS}}


def _enable(at: datetime, **kw):
    s = ProblemSettings.get()
    s.enabled, s.enabled_at = True, at
    for k, v in kw.items():
        setattr(s, k, v)
    s.save()
    return s


class DowntimeTriggerTests(TestCase):
    def stopped(self, begin, end):
        return lambda t: "STOPPED" if begin < t <= end else "RUNNING"

    def test_a_downtime_over_3_hours_opens_one_ticket_and_its_window_closes(self):
        h = Historian(self, T0 + timedelta(hours=12), self.stopped(T0 + timedelta(hours=8), T0 + timedelta(hours=11, minutes=30)))
        _enable(T0 + timedelta(hours=7))
        opened = triggers.tick(ASSET)
        self.assertEqual(len(opened), 1)
        t = Ticket.objects.get()
        self.assertEqual(t.trigger, "downtime")
        self.assertEqual(t.window_start, T0 + timedelta(hours=8, minutes=1))
        self.assertEqual(t.window_end, T0 + timedelta(hours=11, minutes=31))
        self.assertIn("STOPPED", t.title)
        self.assertEqual(t.context["states"][0][0], "STOPPED")
        # A Home post says so.
        from monitor.models import MonitorPost
        self.assertTrue(MonitorPost.objects.filter(kind="ticket", data__ticket_id=t.id).exists())
        # The next passes open nothing more.
        h.now += timedelta(minutes=5)
        self.assertEqual(triggers.tick(ASSET), [])
        self.assertEqual(Ticket.objects.count(), 1)

    def test_a_downtime_still_going_opens_a_ticket_with_an_open_window(self):
        Historian(self, T0 + timedelta(hours=12), self.stopped(T0 + timedelta(hours=8), T0 + timedelta(hours=20)))
        _enable(T0)
        triggers.tick(ASSET)
        self.assertIsNone(Ticket.objects.get().window_end)

    def test_shorter_downtimes_and_planned_states_open_nothing(self):
        state = lambda t: ("STOPPED" if T0 + timedelta(hours=1) < t <= T0 + timedelta(hours=3) else
                           "CHANGEOVER" if T0 + timedelta(hours=4) < t <= T0 + timedelta(hours=9) else "RUNNING")
        Historian(self, T0 + timedelta(hours=12), state)
        _enable(T0)
        self.assertEqual(triggers.tick(ASSET), [])

    def test_alarmed_then_stopped_counts_as_one_downtime(self):
        state = lambda t: ("ALARMED" if T0 + timedelta(hours=1) < t <= T0 + timedelta(hours=2) else
                           "STOPPED" if T0 + timedelta(hours=2) < t <= T0 + timedelta(hours=5) else "RUNNING")
        Historian(self, T0 + timedelta(hours=6), state)
        _enable(T0)
        self.assertEqual(len(triggers.tick(ASSET)), 1)

    def test_only_problems_crossing_the_threshold_after_switching_on(self):
        Historian(self, T0 + timedelta(hours=12), self.stopped(T0 + timedelta(hours=2), T0 + timedelta(hours=6)))
        _enable(T0 + timedelta(hours=7))
        self.assertEqual(triggers.tick(ASSET), [])

    def test_switched_off_opens_nothing(self):
        Historian(self, T0 + timedelta(hours=12), self.stopped(T0 + timedelta(hours=2), T0 + timedelta(hours=11)))
        self.assertEqual(triggers.tick(ASSET), [])

    def test_only_working_time_counts(self):
        cal = plant_calendar.Calendar.from_dict("bsk:plant:P1", {"time_zone": "UTC", "working_days": ["mon", "tue"],
                                                                 "shifts": [{"name": "A", "start": "06:00", "end": "14:00"}]})
        # Monday 2026-11-02: stopped 12:00 to 17:00 → 2 h inside the shift.
        h = Historian(self, T0 + timedelta(hours=18), self.stopped(T0 + timedelta(hours=12), T0 + timedelta(hours=17)))
        _enable(T0)
        with patch("monitor.plant_calendar.for_asset", return_value=cal):
            self.assertEqual(triggers.tick(ASSET), [])
        self.assertEqual(round(triggers.working_minutes(cal, T0 + timedelta(hours=12), T0 + timedelta(hours=17))), 120)


@override_settings(SPC_EVENT_ACTIVE_SECONDS=120)
class SpcTriggerTests(TestCase):
    def setUp(self):
        from spc.models import SpcCursor, SpcTool
        self.now = T0 + timedelta(hours=14)
        Historian(self, self.now, lambda t: "RUNNING")
        self.tool = SpcTool.objects.create(asset_id=ASSET, signal_id="bsk:signal:EXTR01/scrap_output_kg",
                                           column="scrap_output_kg", name="Scrap output", unit="kg")
        SpcCursor.objects.create(tool=self.tool, judged_until=self.now)

    def event(self, kind, start, hours):
        from spc.models import SpcEvent
        return SpcEvent.objects.create(tool=self.tool, tool_version=1, type=kind, rule="spec", started_at=start,
                                       last_at=start + timedelta(hours=hours), value=4.6, extreme=5.2,
                                       limits={"lsl": 0.5, "usl": 4.0})

    def test_spec_over_2_hours_and_control_over_3_hours_each_open_a_ticket(self):
        _enable(T0)
        self.event("spec_violation", T0 + timedelta(hours=11, minutes=36), 2.4)   # → ticket, still going
        self.event("spec_violation", T0 + timedelta(hours=6), 1.5)        # too short
        self.event("control_violation", T0 + timedelta(hours=9), 2.5)     # too short for control
        self.event("control_violation", T0 + timedelta(hours=1), 3.5)     # → ticket
        opened = triggers.tick(ASSET)
        self.assertEqual(sorted(t.trigger for t in opened), ["spc_control", "spc_spec"])
        spec = Ticket.objects.get(trigger="spc_spec")
        self.assertIn("Scrap output", spec.title)
        self.assertEqual(spec.context["trigger_event"]["extreme"], 5.2)
        self.assertIsNone(spec.window_end)                                  # still active at the evaluator's newest row
        control = Ticket.objects.get(trigger="spc_control")
        self.assertEqual(control.window_end, T0 + timedelta(hours=4, minutes=30))     # ended long ago: closed
        self.assertEqual(triggers.tick(ASSET), [])                          # never twice

    def test_events_before_switching_on_open_nothing(self):
        self.event("spec_violation", T0 + timedelta(hours=2), 3)
        _enable(T0 + timedelta(hours=6))
        self.assertEqual(triggers.tick(ASSET), [])

    def test_thresholds_are_settings(self):
        self.event("spec_violation", T0 + timedelta(hours=10), 1.5)
        _enable(T0, spc_spec_hours=1.0)
        self.assertEqual(len(triggers.tick(ASSET)), 1)


def _ticket(**kw) -> Ticket:
    with patch("problems.service.asset_name", return_value="EXTR01"):
        return service.open_ticket(ASSET, kw.pop("trigger", "downtime"), "Downtime longer than 3 h", T0, T0 + timedelta(hours=4),
                                   with_context=False, **kw)


def _fill(t: Ticket, user=ENG) -> Ticket:
    return service.save(t.id, {"problem": "Extruder stopped 4 h", "whys": ["Die plugged", "Melt too cold", "Heater zone 3 failed", ""],
                               "root_cause": "Heater zone 3 contactor worn", "containment": "Replaced contactor",
                               "actions": [{"text": "Add contactor to PM list", "responsible": "Maria", "due_date": "2026-11-20"},
                                           {"text": "Train shift on zone alarms", "responsible": "Luis", "due_date": "2026-11-01"}],
                               "final_report": "Zone 3 contactor wore out; it is now on the PM list."}, user)


class WorkflowTests(TestCase):
    def test_draft_submit_approve_by_a_second_engineer(self):
        t = _fill(_ticket())
        self.assertEqual(t.status, "draft")
        self.assertEqual(t.whys, ["Die plugged", "Melt too cold", "Heater zone 3 failed"])
        t = service.submit(t.id, ENG)
        self.assertEqual((t.status, t.author["username"], t.knowledge_check["status"]), ("submitted", "maria", "queued"))
        with self.assertRaisesMessage(TicketError, "can't approve their own"):
            service.approve(t.id, ENG)
        with patch("problems.publish.publish") as pub:
            t = service.approve(t.id, ENG2, "Good")
        pub.assert_called_once()
        self.assertEqual((t.status, t.approver["username"]), ("approved", "luis"))
        self.assertEqual([h["action"] for h in t.history], ["opened", "started", "submitted", "approved"])
        with self.assertRaises(TicketError):
            service.save(t.id, {"problem": "x"}, ENG)

    def test_associates_read_but_do_not_write(self):
        t = _ticket()
        for action in (lambda: service.save(t.id, {"problem": "x"}, ASSOC), lambda: service.close(t.id, ASSOC, "planned"),
                       lambda: service.open_manual(ASSET, "x", T0, None, ASSOC)):
            with self.assertRaisesMessage(TicketError, "only engineers and admins"):
                action()
        self.assertEqual(service.ticket_json(t)["status"], "open")

    def test_submit_needs_the_essentials(self):
        t = service.save(_ticket().id, {"problem": "Stopped"}, ENG)
        with self.assertRaisesMessage(TicketError, "at least one why, the root cause, at least one corrective action, the final report"):
            service.submit(t.id, ENG)

    def test_return_needs_a_comment_and_goes_back_to_the_author(self):
        t = service.submit(_fill(_ticket()).id, ENG)
        with self.assertRaises(TicketError):
            service.return_ticket(t.id, ENG2, "  ")
        t = service.return_ticket(t.id, ENG2, "Why 3 is not proven")
        self.assertEqual((t.status, t.return_comment), ("returned", "Why 3 is not proven"))
        t = service.save(t.id, {"whys": ["Die plugged", "Melt too cold", "Thermocouple read low"]}, ENG)
        self.assertEqual(t.status, "returned")
        self.assertEqual(service.submit(t.id, ENG).status, "submitted")

    def test_close_without_analysis_needs_a_reason_and_is_never_published(self):
        t = _ticket()
        with self.assertRaises(TicketError):
            service.close(t.id, ENG, "")
        t = service.close(t.id, ENG, "Planned stop for cleaning, not in the calendar")
        self.assertEqual(t.status, "closed")
        with patch("ingestion.library.add_document") as add:
            publish.publish(t)
        add.assert_not_called()

    def test_manual_ticket(self):
        with patch("problems.service.asset_name", return_value="EXTR01"), \
                patch("problems.context.build", return_value={"states": []}):
            t = service.open_manual(ASSET, "Scrap spikes after restart", T0, T0 + timedelta(hours=1), ENG)
        self.assertEqual((t.trigger, t.opened_by["username"]), ("manual", "maria"))
        self.assertEqual(t.recommendations, {"status": "queued"})
        with patch("problems.llm.recommend") as rec:
            llm.recommend_queued()
        rec.assert_called_once()
        with self.assertRaises(TicketError):
            service.open_manual(ASSET, "x", T0 + timedelta(hours=2), T0 + timedelta(hours=1), ENG)

    def test_edge_users_merge(self):
        t = _ticket()
        u = {"username": "pat", "name": "Pat", "role": "production_associate", "signed_in_at": "2026-11-02T06:01:00Z"}
        service.add_edge_users(t.id, [u], "edge-1")
        t = service.add_edge_users(t.id, [u, {**u, "username": "maria"}], "edge-1")
        self.assertEqual([x["username"] for x in t.context["edge_users"]], ["pat", "maria"])

    def test_settings(self):
        out = service.update_settings({"enabled": True, "spc_spec_hours": 2, "downtime_states": ["stopped"]}, "angel")
        self.assertTrue(out["enabled"])
        self.assertIsNotNone(out["enabled_at"])
        self.assertEqual(out["downtime_states"], ["STOPPED"])
        with self.assertRaises(TicketError):
            service.update_settings({"downtime_hours": 0})
        with self.assertRaises(TicketError):
            service.update_settings({"check_time": "25:00"})


@override_settings(PROBLEM_FILES_DIR="/tmp/claude-1001/p29-test-files")
class PictureTests(TestCase):
    def test_a_picture_is_stored_as_jpeg(self):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (40, 30), "red").save(buf, format="PNG")
        t = _ticket()
        pic = service.add_picture(t.id, buf.getvalue(), "leak.png", ENG, "Oil under the gearbox")
        self.assertEqual((pic.width, pic.height), (40, 30))
        self.assertTrue(service.picture_bytes(t.id, pic.id).startswith(b"\xff\xd8"))
        with self.assertRaises(TicketError):
            service.add_picture(t.id, b"not an image", "x.jpg", ENG)
        service.remove_picture(t.id, pic.id, ENG)
        self.assertEqual(t.pictures.count(), 0)


class ModelStepTests(TestCase):
    def test_final_version_is_returned_not_saved(self):
        t = service.save(_ticket().id, {"notes": "se paro el extrusor, dado tapado, zona 3 fria"}, ENG)
        reply = {"problem": "El extrusor se detuvo", "whys": ["Dado tapado", "Zona 3 fría", ""], "root_cause": "Contactor",
                 "containment": "", "actions": ["Cambiar contactor", ""], "report": "## Problema", "missing": ["responsable"]}
        with patch("problems.llm._complete", return_value=reply) as c, patch("problems.llm.pick", return_value="ollama-qwen3-8b"):
            out = llm.structure(t)
        body = c.call_args.args[2]
        self.assertIn("NOTES\nse paro el extrusor", body)
        self.assertLess(body.index("FIELDS"), body.index("NOTES"))            # the fields come first
        self.assertIn("The FIELDS are what the engineer means", c.call_args.args[1])
        self.assertEqual(out["whys"], ["Dado tapado", "Zona 3 fría"])
        self.assertEqual(out["final_report"], "## Problema")
        self.assertEqual(out["actions"], ["Cambiar contactor"])
        self.assertIn("Write in Spanish:", c.call_args.args[1])
        self.assertEqual(Ticket.objects.get(pk=t.pk).problem, "")

    def test_final_version_needs_something_written(self):
        with self.assertRaises(TicketError):
            llm.structure(_ticket())

    def test_recommendations_cite_the_knowledge(self):
        t = _ticket()
        packet = {"context_text": "DOCUMENT EXCERPTS:\n[1] (Manual, p.4)\nCheck heater zones",
                  "document_evidence": [{"document": "Manual.pdf", "page": 4, "text": "Check heater zones"}]}
        with patch("context_service.service.assemble", return_value=packet), patch("problems.llm.pick", return_value="m"), \
                patch("problems.llm._complete", return_value={"items": [{"text": "Check heater zone 3", "sources": [1]}]}):
            t = llm.recommend(t)
        self.assertEqual(t.recommendations["items"][0]["sources"], [1])
        self.assertEqual(t.recommendations["sources"][0]["document"], "Manual.pdf")

    def test_check_keeps_conflicts_for_the_approver(self):
        t = service.submit(_fill(_ticket()).id, ENG)
        reply = {"verdict": "consistent", "points": [{"kind": "conflicts", "text": "Manual says zone 3 max 230 °C", "sources": [2]}]}
        with patch("problems.llm.knowledge", return_value=("K", [])), patch("problems.llm.pick", return_value="m"), \
                patch("problems.llm._complete", return_value=reply):
            t = llm.check(t)
        self.assertEqual((t.knowledge_check["status"], t.knowledge_check["verdict"]), ("done", "conflicts"))

    def test_a_failed_check_is_kept_as_failed(self):
        t = service.submit(_fill(_ticket()).id, ENG)
        with patch("problems.llm.knowledge", side_effect=RuntimeError("down")), patch("problems.llm.pick", return_value="m"):
            t = llm.check(t)
        self.assertEqual(t.knowledge_check["status"], "failed")

    def test_when_a_check_is_due(self):
        s = ProblemSettings.get()
        t = service.submit(_fill(_ticket()).id, ENG)
        at = datetime.fromisoformat(t.knowledge_check["queued_at"])
        s.check_mode = "on_submit"
        self.assertTrue(llm.due(s, t, at, lambda: False))
        s.check_mode = "dgx"
        self.assertFalse(llm.due(s, t, at, lambda: False))
        self.assertTrue(llm.due(s, t, at, lambda: True))
        s.check_mode, s.check_day, s.check_time = "schedule", "sun", "22:00"
        with patch("monitor.plant_calendar.plants", return_value=[]):
            self.assertFalse(llm.due(s, t, at + timedelta(minutes=1), lambda: True))
            self.assertTrue(llm.due(s, t, at + timedelta(days=8), lambda: True))
        t = llm.request_check(t)
        self.assertTrue(llm.due(s, t, at, lambda: False))

    def test_run_due_runs_one(self):
        s = ProblemSettings.get()
        s.check_mode = "on_submit"
        s.save()
        for _ in range(2):
            service.submit(_fill(_ticket()).id, ENG)
        with patch("problems.llm.check") as check:
            self.assertEqual(llm.run_due(limit=1), 1)
        check.assert_called_once()


class PublishTests(TestCase):
    def approved(self):
        t = service.submit(_fill(_ticket()).id, ENG)
        with patch("problems.publish.publish"):
            return service.approve(t.id, ENG2)

    def test_the_report_goes_into_the_library(self):
        t = self.approved()
        with patch.dict(os.environ, {"PROBLEMS_PUBLISH": "1"}), patch("ingestion.library.add_document") as add, \
                patch("problems.publish.describe_pictures"):
            add.return_value = (type("Doc", (), {"pk": 77})(), True)
            t = publish.publish(t)
        self.assertEqual(t.document_id, "77")
        upload = add.call_args.args[0]
        text = upload.read().decode()
        self.assertIn("Heater zone 3 contactor worn", text)
        self.assertIn("| Add contactor to PM list | Maria | 2026-11-20 |", text)
        self.assertIn("- Written by: Maria (engineer)\n- Approved by: Luis (engineer)", text)
        self.assertEqual(add.call_args.kwargs["plant_asset"], ASSET)
        self.assertTrue(add.call_args.kwargs["then_embed"])
        from ingestion.models import Folder
        self.assertTrue(Folder.objects.filter(name="EXTR01", parent__name="Problem solving").exists())

    def test_switched_off_in_a_preview(self):
        t = self.approved()
        with patch.dict(os.environ, {"PROBLEMS_PUBLISH": "0"}), patch("ingestion.library.add_document") as add:
            t = publish.publish(t)
        add.assert_not_called()
        self.assertIn("switched off", t.publish_error)

    def test_a_failure_is_kept_on_the_ticket(self):
        t = self.approved()
        with patch.dict(os.environ, {"PROBLEMS_PUBLISH": "1"}), patch("problems.publish.describe_pictures"), \
                patch("ingestion.library.add_document", side_effect=OSError("disk full")):
            t = publish.publish(t)
        self.assertIn("disk full", t.publish_error)
        self.assertEqual(t.document_id, "")


class ApiTests(TestCase):
    def test_list_detail_settings(self):
        t = _ticket()
        c = APIClient()
        r = c.get("/api/problems/tickets/", {"status": "active"})
        self.assertEqual(r.data["tickets"][0]["number"], t.number)
        self.assertEqual(r.data["counts"]["open"], 1)
        self.assertEqual(c.get(f"/api/problems/tickets/{t.id}/").data["title"], t.title)
        self.assertEqual(c.get("/api/problems/tickets/999/").status_code, 404)
        r = c.put("/api/problems/settings/", {"enabled": True, "spc_spec_hours": 2, "updated_by": "angel"}, format="json")
        self.assertTrue(r.data["enabled"])
        self.assertEqual(c.put("/api/problems/settings/", {"check_day": "xyz"}, format="json").status_code, 400)


class TicketToolTests(TransactionTestCase):
    """EDGE works on tickets through the MCP tools; the rules come back as the tool's error."""

    def call(self, tool, args):
        from context_service import mcp_server
        result = asyncio.run(mcp_server.build_server().call_tool(tool, args))
        content = result[0] if isinstance(result, tuple) else result
        return json.loads(content[0].text)

    def test_the_workflow_through_the_tools(self):
        t = _ticket()
        self.assertEqual(self.call("list_tickets", {"asset_id": ASSET})["tickets"][0]["id"], t.id)
        fields = {"problem": "Stopped", "whys": ["A"], "root_cause": "B", "final_report": "D",
                  "actions": [{"text": "C", "responsible": "Maria", "due_date": "2026-11-20"}]}
        self.assertEqual(self.call("save_ticket", {"ticket_id": t.id, "fields": fields, "user": ENG})["status"], "draft")
        self.assertEqual(self.call("submit_ticket", {"ticket_id": t.id, "user": ENG})["status"], "submitted")
        from mcp.server.fastmcp.exceptions import ToolError
        with self.assertRaisesRegex(ToolError, "can't approve their own"):
            self.call("approve_ticket", {"ticket_id": t.id, "user": ENG})
        with patch("problems.publish.publish"):
            self.assertEqual(self.call("approve_ticket", {"ticket_id": t.id, "user": ENG2})["status"], "approved")
        self.assertEqual(self.call("get_ticket", {"ticket_id": t.id})["approver"]["username"], "luis")
        self.assertIn("spc_spec_hours", self.call("get_problem_settings", {}))

    def test_manual_ticket_with_iso_times(self):
        with patch("problems.service.asset_name", return_value="EXTR01"), \
                patch("problems.context.build", return_value={}):
            out = self.call("open_ticket", {"asset_id": ASSET, "title": "Scrap after restart",
                                            "window_start": "2026-11-02T06:00:00Z", "user": ENG})
        self.assertEqual(out["window_start"], "2026-11-02T06:00:00+00:00")


class ActionTests(TestCase):
    def approved(self):
        t = service.submit(_fill(_ticket()).id, ENG)
        with patch("problems.publish.publish"):
            return service.approve(t.id, ENG2)

    def test_actions_need_a_responsible_and_a_due_date(self):
        t = service.save(_ticket().id, {"problem": "x", "whys": ["y"], "root_cause": "z",
                                        "actions": [{"text": "Fix it", "responsible": "", "due_date": ""}]}, ENG)
        with self.assertRaisesMessage(TicketError, "a responsible and a due date for every corrective action"):
            service.submit(t.id, ENG)
        with self.assertRaises(TicketError):
            service.save(t.id, {"actions": [{"text": "Fix it", "due_date": "next week"}]}, ENG)

    def test_states_green_yellow_red(self):
        from datetime import date
        self.assertEqual(service.action_state({"status": "done", "due_date": "2020-01-01"}), "done")
        self.assertEqual(service.action_state({"status": "in_progress", "due_date": "2026-11-01"}, date(2026, 11, 2)), "delayed")
        self.assertEqual(service.action_state({"status": "in_progress", "due_date": "2026-11-20"}, date(2026, 11, 2)), "in_progress")
        t = _fill(_ticket())
        self.assertEqual(service.ticket_json(t)["actions"][1]["state"], "delayed")      # due 2026-11-01
        self.assertEqual(t.corrective_actions.splitlines()[0], "- Add contactor to PM list (responsible: Maria, due 2026-11-20)")

    def test_ids_are_kept_when_rows_are_edited(self):
        t = _fill(_ticket())
        first = t.actions[0]["id"]
        t = service.save(t.id, {"actions": [{**t.actions[0], "text": "Add contactor to the weekly PM"}]}, ENG)
        self.assertEqual((t.actions[0]["id"], len(t.actions)), (first, 1))

    def test_follow_validate_and_complete(self):
        t = self.approved()
        a, b = (x["id"] for x in t.actions)
        with self.assertRaisesMessage(TicketError, "only a done action"):
            service.validate_action(t.id, a, ENG2)
        service.set_action_status(t.id, a, "done", ENG)
        with self.assertRaisesMessage(TicketError, "can't validate their own"):
            service.validate_action(t.id, a, ENG)
        t = service.validate_action(t.id, a, ENG2)
        self.assertEqual(t.status, "approved")
        with self.assertRaisesMessage(TicketError, "already validated"):
            service.set_action_status(t.id, a, "in_progress", ENG)
        service.set_action_status(t.id, b, "done", ENG2)
        with patch("problems.publish.publish") as pub:
            t = service.validate_action(t.id, b, ENG2)
        self.assertEqual(t.status, "completed")
        self.assertIsNotNone(t.completed_at)
        pub.assert_called_once()

    def test_completion_replaces_the_library_document(self):
        t = self.approved()
        t.document_id = "11"
        t.save()
        for x in t.actions:
            service.set_action_status(t.id, x["id"], "done", ENG)
        for x in t.actions[:-1]:
            service.validate_action(t.id, x["id"], ENG2)
        old = type("Doc", (), {"pk": 11})()
        with patch.dict(os.environ, {"PROBLEMS_PUBLISH": "1"}), patch("problems.publish.describe_pictures"), \
                patch("ingestion.library.add_document", return_value=(type("Doc", (), {"pk": 12})(), True)) as add, \
                patch("ingestion.models.Document.objects.filter") as find, patch("ingestion.library.delete_document") as delete:
            find.return_value.first.return_value = old
            t = service.validate_action(t.id, t.actions[-1]["id"], ENG2)
        self.assertEqual((t.status, t.document_id), ("completed", "12"))
        delete.assert_called_once_with(old)
        self.assertIn("completed", add.call_args.args[0].name)


@override_settings(PROBLEM_FILES_DIR="/tmp/claude-1001/p29-test-files")
class PicturePhaseTests(TestCase):
    def png(self):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (20, 20), "blue").save(buf, format="PNG")
        return buf.getvalue()

    def test_problem_pictures_before_after_pictures_also_after_approval(self):
        t = _fill(_ticket())
        service.add_picture(t.id, self.png(), "leak.png", ENG, "Leak", "problem")
        t = service.submit(t.id, ENG)
        with patch("problems.publish.publish"):
            t = service.approve(t.id, ENG2)
        with self.assertRaisesMessage(TicketError, "only pictures after the fix"):
            service.add_picture(t.id, self.png(), "x.png", ENG, "", "problem")
        pic = service.add_picture(t.id, self.png(), "fixed.png", ENG, "New seal", "after")
        self.assertEqual(pic.phase, "after")
        html = __import__("problems.report", fromlist=["html"]).html(t)
        self.assertIn("Pictures of the problem", html)
        self.assertIn("Pictures after the fix", html)
        self.assertIn("data:image/jpeg;base64,", html)
        self.assertLess(html.index("Pictures of the problem"), html.index("Containment"))
        self.assertLess(html.index("Corrective actions"), html.index("Pictures after the fix"))


@override_settings(SPC_EVENT_ACTIVE_SECONDS=120)
class SpecReplacesControlTests(TestCase):
    def setUp(self):
        from spc.models import SpcCursor, SpcTool
        self.now = T0 + timedelta(hours=14)
        Historian(self, self.now, lambda t: "RUNNING")
        self.tool = SpcTool.objects.create(asset_id=ASSET, signal_id="bsk:signal:EXTR01/scrap_output_kg",
                                           column="scrap_output_kg", name="Scrap output", unit="kg")
        SpcCursor.objects.create(tool=self.tool, judged_until=self.now)
        _enable(T0)

    def event(self, kind, start, hours):
        from spc.models import SpcEvent
        return SpcEvent.objects.create(tool=self.tool, tool_version=1, type=kind, rule="r", started_at=start,
                                       last_at=start + timedelta(hours=hours), value=9.0, extreme=10.0, limits={"usl": 4.0})

    def test_only_the_spec_ticket_opens_when_both_happen(self):
        self.event("control_violation", T0 + timedelta(hours=9), 5)
        self.event("spec_violation", T0 + timedelta(hours=10), 4)
        opened = triggers.tick(ASSET)
        self.assertEqual([t.trigger for t in opened], ["spc_spec"])
        self.assertEqual(triggers.tick(ASSET), [])

    def test_a_control_ticket_that_came_first_becomes_the_spec_ticket(self):
        control = self.event("control_violation", T0 + timedelta(hours=9), 5)
        opened = triggers.tick(ASSET)
        self.assertEqual([t.trigger for t in opened], ["spc_control"])
        self.event("spec_violation", T0 + timedelta(hours=11), 3)
        self.assertEqual(triggers.tick(ASSET), [])
        t = Ticket.objects.get()
        self.assertEqual(t.trigger, "spc_spec")
        self.assertIn("specification", t.title)
        self.assertIn("became a specification ticket", [h["action"] for h in t.history])
        self.assertEqual(triggers.tick(ASSET), [])
        self.assertEqual(Ticket.objects.count(), 1)
        self.assertTrue(control)

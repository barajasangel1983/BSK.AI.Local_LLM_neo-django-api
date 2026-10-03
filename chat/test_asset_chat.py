"""Asset-scoped chat (P7): graph fact sheet + historian values + asset-first documents.

Unit tests mock the graph and the historian. The integration test builds the
EXTR01 fact sheet from the seed in the throwaway Neo4j:

    GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test chat.test_asset_chat
"""

import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase, override_settings
from rest_framework.test import APIClient

from chat import asset_context
from chat.models import Conversation
from chat.retrieval import SearchResult, V2Chunk
from chat.tests import _ok_response
from context_graph.driver import GraphUnavailable

ASSET = "bsk:asset:EXTR01"
TEST_URI = os.getenv("GRAPH_TEST_URI")
SEED = Path(__file__).resolve().parent.parent / "context_graph" / "seeds" / "extr01.yaml"


def node(id_, label, name, **props):
    return {"id": id_, "label": label, "name": name, "properties": {"name": name, **props}}


def signal(key, name, unit, low=None, high=None):
    limits = [node(f"bsk:operating-limit:EXTR01/{key}/normal", "OperatingLimit", f"{name} normal range",
                   low=low, high=high)] if low is not None else []
    return {**node(f"bsk:signal:EXTR01/{key}", "Signal", name, unit=unit,
                   historian_ref=f"plc_1_historian.extruder_samples.{key}"), "limits": limits, "opc_nodes": []}


CONTEXT = {
    "asset": node(ASSET, "Asset", "Extruder EXTR01", asset_type="Extruder"),
    "hierarchy": [node("bsk:plant:demo", "Plant", "Demo Plant"), node("bsk:line:demo/1", "Line", "Line 1")],
    "components": [{
        **node("bsk:component:EXTR01/die-head", "Component", "Die Head", component_type="die"),
        "parent": ASSET, "alarms": ["bsk:alarm:EXTR01/DIE_PLUG"], "children": [],
        "signals": [signal("die_pressure_bar", "Die pressure", "bar", 84.3, 99.2),
                    signal("head_temp_c", "Die head temperature", "°C", 93.9, 115.9)],
    }],
    "connections": [],
    "signals": [signal("throughput_actual_kg_hr", "Throughput", "kg/h")],
    "alarms": [{**node("bsk:alarm:EXTR01/DIE_PLUG", "Alarm", "DIE_PLUG", code="DIE_PLUG", severity="high",
                       description="Die plugged"),
                "components": ["bsk:component:EXTR01/die-head"],
                "procedures": [node("bsk:procedure:EXTR01/die-plug-clearing", "Procedure", "Die plug clearing")]}],
    "procedures": [{**node("bsk:procedure:EXTR01/die-plug-clearing", "Procedure", "Die plug clearing",
                           summary="Stop feed, relieve die pressure, clear die plate."),
                    "applies_to": ["bsk:component:EXTR01/die-head"], "addresses": ["bsk:alarm:EXTR01/DIE_PLUG"]}],
    "documents": [{**node("bsk:document:MANUAL", "Document", "Manual.pdf", doc_key="MANUAL"), "describes": [ASSET], "sections": []}],
    "counts": {},
}

T = lambda h, m: datetime(2026, 3, 31, h, m, tzinfo=timezone.utc)  # noqa: E731
SNAPSHOT = {
    "latest": {"ts": T(23, 59), "machine_state": "STOPPED", "die_pressure_bar": 46.7},
    "running": {"ts": T(19, 13), "die_pressure_bar": 104.0, "head_temp_c": 96.4, "throughput_actual_kg_hr": 9100},
    "state_since": T(23, 6),
}


class FactSheetTests(SimpleTestCase):
    def setUp(self):
        for target, value in (("context_graph.services.get_asset_context", CONTEXT),
                              ("chat.asset_context.historian_snapshot", SNAPSHOT)):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_full_sheet(self):
        ctx = asset_context.build(ASSET, "die plug", 10000)
        self.assertIn("Location: Demo Plant > Line 1", ctx.text)
        self.assertIn("machine state STOPPED since 2026-03-31 23:06", ctx.text)
        self.assertIn("last RUNNING sample (2026-03-31 19:13 UTC)", ctx.text)
        self.assertIn("- Die pressure [bsk:signal:EXTR01/die_pressure_bar] on Die Head: unit bar; normal 84.3–99.2 bar; "
                      "last running value 104 bar (ABOVE normal range)", ctx.text)
        self.assertIn("Die head temperature", ctx.text)
        self.assertIn("(in range)", ctx.text)
        self.assertIn("- DIE_PLUG (severity high): Die plugged; components: Die Head; procedures: Die plug clearing", ctx.text)
        self.assertIn("Stop feed, relieve die pressure, clear die plate; applies to: Die Head; addresses: DIE_PLUG", ctx.text)
        self.assertEqual(ctx.historian["out_of_range"], ["Die pressure: 104 bar (ABOVE normal range)"])
        self.assertEqual(ctx.doc_keys, ["MANUAL"])
        self.assertEqual(len(ctx.facts), ctx.total_facts + 2)   # every fact, plus location and historian status

        prompt = asset_context.system_prompt(ctx)
        self.assertIn("assistant for Extruder EXTR01 (bsk:asset:EXTR01)", prompt)
        self.assertIn("Out of normal range in the last running sample: Die pressure: 104 bar", prompt)
        graph, historian = asset_context.citations(ctx)
        self.assertEqual((graph["kind"], graph["source"], historian["kind"]), ("graph", "Context Graph", "historian"))
        self.assertIn("out of range: Die pressure", historian["snippet"])

    def test_small_budget_keeps_header_and_what_matches_the_question(self):
        ctx = asset_context.build(ASSET, "die head temperature", 700)
        self.assertLessEqual(len(ctx.text), 700)
        self.assertIn("Asset bsk:asset:EXTR01", ctx.text)
        self.assertIn("Historian (plc_1_historian", ctx.text)
        self.assertIn("Die head temperature", ctx.text)
        self.assertNotIn("Throughput", ctx.text)
        self.assertLess(len(ctx.facts), ctx.total_facts)

    def test_without_historian(self):
        with patch("chat.asset_context.historian_snapshot", return_value=None):
            ctx = asset_context.build(ASSET, "x", 10000)
        self.assertIsNone(ctx.historian)
        self.assertNotIn("last running value", ctx.text)
        self.assertEqual(len(asset_context.citations(ctx)), 1)


@patch("chat.views.generate_title", return_value=("Title", "llm"))
@patch("chat.views.search_v2", return_value=SearchResult(chunks=[], candidates=0, reranker="ok", latency_ms=1))
@patch("chat.views.requests.post", return_value=_ok_response("Stop the feed [G]."))
class AssetChatViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        for target, value in (("context_graph.services.get_asset_context", CONTEXT),
                              ("chat.asset_context.historian_snapshot", SNAPSHOT),
                              ("context_graph.services.node_detail", CONTEXT["asset"])):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def chat(self, message, conversation_id=None, use_rag=False, **extra):
        return self.client.post("/api/chat/", {"conversation_id": conversation_id, "message": message,
                                               "model": "dgx-qwen38-27b-fp8", "use_rag": use_rag, **extra}, format="json")

    def test_scope_is_stored_used_and_sticky(self, post, search, title):
        r = self.chat("What should I do on DIE_PLUG?", asset_id=ASSET)
        self.assertEqual((r.status_code, r.data["asset_id"]), (200, ASSET))
        system = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("ASSET FACTS", system)
        self.assertIn("DIE_PLUG (severity high)", system)
        sources = r.data["messages"][-1]["sources"]
        self.assertEqual([s["kind"] for s in sources], ["graph", "historian"])   # RAG off: facts only
        search.assert_not_called()

        r = self.chat("And the die pressure?", conversation_id=r.data["id"])        # follow-up: no asset_id sent
        self.assertIn("ASSET FACTS", post.call_args.kwargs["json"]["messages"][0]["content"])
        r = self.chat("Unrelated", conversation_id=r.data["id"], asset_id="")       # cleared
        self.assertEqual(r.data["asset_id"], "")
        self.assertNotIn("ASSET FACTS", post.call_args.kwargs["json"]["messages"][0]["content"])
        listed = self.client.get("/api/conversations/").json()
        self.assertIn("asset_id", listed[0] if isinstance(listed, list) else listed["results"][0])

    def test_asset_documents_first_then_library(self, post, search, title):
        chunk = V2Chunk(id="c1", text="Clear the screen pack.", source="Manual.pdf", asset_id="MANUAL",
                        section_path=["Manual", "Die"], page_start=4, page_end=4, content_type="text", vector_score=0.8,
                        rerank_score=0.9)
        search.side_effect = [SearchResult(chunks=[chunk], candidates=1, reranker="ok", latency_ms=1)]
        r = self.chat("How do I clear the screen?", use_rag=True, asset_id=ASSET)
        self.assertEqual(search.call_args.kwargs["doc_keys"], ["MANUAL"])
        self.assertEqual([s["kind"] for s in r.data["messages"][-1]["sources"]], ["graph", "historian", "document"])
        system = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("DOCUMENT EXCERPTS:", system)
        self.assertIn("Clear the screen pack.", system)

        # Nothing relevant in the asset's documents -> the whole library.
        search.side_effect = [SearchResult(chunks=[], candidates=0, reranker="ok", latency_ms=1),
                              SearchResult(chunks=[chunk], candidates=1, reranker="ok", latency_ms=1)]
        self.chat("Something else", use_rag=True, asset_id=ASSET)
        self.assertEqual([c.kwargs.get("doc_keys") for c in search.call_args_list[-2:]], [["MANUAL"], None])

    def test_no_shift_summaries_in_asset_chats(self, post, search, title):
        with patch("chat.views.query_chunks") as shifts:
            shifts.return_value = []
            self.chat("Any extruder alarm today?", use_rag=True, asset_id=ASSET)        # factory keywords
            shifts.assert_not_called()
            self.assertEqual(search.call_args.kwargs["top_n"], 5)                     # full document share
            self.chat("Any extruder alarm today?", use_rag=True)                       # no asset: as before
            shifts.assert_called_once()
            self.assertEqual(search.call_args.kwargs["top_n"], 3)

    def test_invalid_scope(self, post, search, title):
        self.assertEqual(self.chat("hi", asset_id="EXTR01").status_code, 400)
        with patch("context_graph.services.node_detail", return_value={**CONTEXT["asset"], "label": "Component"}):
            self.assertEqual(self.chat("hi", asset_id="bsk:asset:X").status_code, 400)
        with patch("context_graph.services.node_detail", side_effect=GraphUnavailable("down")):
            self.assertEqual(self.chat("hi", asset_id=ASSET).status_code, 503)
        self.assertEqual(Conversation.objects.count(), 0)     # nothing saved
        post.assert_not_called()

    def test_graph_down_mid_conversation_still_answers(self, post, search, title):
        r = self.chat("hi", asset_id=ASSET)
        with patch("context_graph.services.get_asset_context", side_effect=GraphUnavailable("down")):
            r = self.chat("still there?", conversation_id=r.data["id"])
        self.assertEqual(r.status_code, 200)
        self.assertIn("facts are unavailable right now", post.call_args.kwargs["json"]["messages"][0]["content"])
        self.assertEqual(r.data["messages"][-1]["sources"], [])


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI (see module docstring) to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class SeedFactSheetIntegrationTests(TestCase):

    def setUp(self):
        from context_graph import driver, services
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
        services.init_graph()
        services.load_seed(SEED)

    @patch("chat.asset_context.historian_snapshot", return_value=None)
    def test_extr01_fact_sheet(self, _):
        self.assertEqual(asset_context.check_asset(ASSET), "Extruder EXTR01")
        with self.assertRaises(asset_context.AssetScopeError):
            asset_context.check_asset("bsk:asset:NOPE")
        ctx = asset_context.build(ASSET, "die plug", 100000)
        self.assertEqual(ctx.total_facts, 9 + 30 + 20 + 8 + len([1 for ln in ctx.text.splitlines() if "→" in ln]))
        self.assertIn("Die pressure [bsk:signal:EXTR01/die_pressure_bar] on Die Head: unit bar; normal 84.3–99.2 bar", ctx.text)
        self.assertIn("  - Barrel Zone 1 [bsk:component:EXTR01/barrel-zone-1]", ctx.text)   # nested under Screw & Barrel
        self.assertIn("- DIE_PLUG (severity high): Die plugged; components: Die Head; procedures: Die plug / screen block clearing",
                      ctx.text)
        self.assertNotIn("lab:", ctx.text)

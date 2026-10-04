"""Context Service (P10a): resolution, the Context Packet, fallbacks and REST. The graph,
the historian, document search and the models are mocked (fixtures from the asset-chat tests)."""

from unittest.mock import patch

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from chat.retrieval import SearchResult, V2Chunk
from chat.test_asset_chat import ASSET, CONTEXT, SNAPSHOT, node
from context_graph.driver import GraphUnavailable
from context_graph.repository import NodeNotFound
from context_service import service

ASSETS = [node(ASSET, "Asset", "Extruder EXTR01", asset_type="Extruder", description="Kibble extrusion cooker")]
DIE_HEAD = "bsk:component:EXTR01/die-head"
DIE_PRESSURE = "bsk:signal:EXTR01/die_pressure_bar"


def chunk(text, source="Manual.pdf", page=4, rerank=0.9, content_type="paragraph", key="MANUAL"):
    return V2Chunk(id=text[:8], text=text, source=source, asset_id=key, section_path=["Manual", "Die"], page_start=page,
                   page_end=page, content_type=content_type, vector_score=0.4, rerank_score=rerank)


def result(*chunks, reranker="ok"):
    return SearchResult(chunks=list(chunks), candidates=len(chunks), reranker=reranker, latency_ms=1)


class ContextTestCase(TestCase):
    def setUp(self):
        self.search = None
        for target, kwargs in (
            ("context_graph.services.get_asset_context", {"return_value": CONTEXT}),
            ("context_graph.services.list_assets", {"return_value": ASSETS}),
            ("context_graph.services.node_detail", {"return_value": {**ASSETS[0], "relationships": []}}),
            ("chat.asset_context.historian_snapshot", {"return_value": SNAPSHOT}),
            ("context_graph.provenance.fact_source_labels", {"side_effect": lambda props: {k: "Seed (extr01-v1)" for k in props}}),
            ("context_service.service.search_v2", {"return_value": result(
                chunk("Die pressure above 99 bar means the die is plugging."),
                chunk("[Figure p.7 (schematic): The die head with its pressure sensor PT-101.]", page=7, content_type="figure"),
                chunk("Barely related.", rerank=0.01))}),
        ):
            patcher = patch(target, **kwargs)
            mock = patcher.start()
            self.addCleanup(patcher.stop)
            if target.endswith("search_v2"):
                self.search = mock


class ResolveTests(ContextTestCase):
    def test_asset_named_in_the_question_and_focus_entities(self):
        r = service.resolve("Why is the die pressure high on EXTR01?")
        self.assertEqual(r["asset"], {"id": ASSET, "name": "Extruder EXTR01", "match": "named"})
        self.assertEqual([(f["id"], f["label"]) for f in r["focus"]], [(DIE_PRESSURE, "Signal")])
        self.assertEqual(r["ambiguous"], [])

    def test_longer_name_wins_and_codes_match(self):
        r = service.resolve("die head temperature after a DIE_PLUG alarm", ASSET)
        self.assertEqual(r["asset"]["match"], "given")
        self.assertEqual({f["name"] for f in r["focus"]}, {"Die head temperature", "DIE_PLUG"})   # not "Die Head" too

    def test_no_asset_named(self):
        self.assertEqual(service.resolve("What is attention?"), {"asset": None, "focus": [], "ambiguous": []})

    def test_two_assets_named_is_ambiguous_never_guessed(self):
        both = ASSETS + [node("bsk:asset:CX0", "Asset", "Mixer CX0")]
        with patch("context_graph.services.list_assets", return_value=both):
            r = service.resolve("Compare EXTR01 and CX0")
        self.assertIsNone(r["asset"])
        self.assertEqual([c["id"] for c in r["ambiguous"][0]["candidates"]], [ASSET, "bsk:asset:CX0"])

    def test_unknown_asset(self):
        with patch("context_graph.services.node_detail", side_effect=NodeNotFound("x")):
            with self.assertRaisesMessage(service.ContextError, "unknown asset"):
                service.resolve("anything", "bsk:asset:NOPE")


class PacketTests(ContextTestCase):
    def test_packet_has_facts_documents_state_and_provenance(self):
        p = service.assemble("Why is the die pressure high on EXTR01?")
        self.assertEqual(p["packet_version"], "1")
        self.assertEqual(p["resolved_scope"]["asset"]["id"], ASSET)
        fact = next(f for f in p["graph_facts"] if f["entity_id"] == DIE_PRESSURE)
        self.assertTrue(fact["focus"])
        self.assertIn("normal 84.3–99.2 bar", fact["text"])
        self.assertEqual(fact["source"], {"label": "Seed (extr01-v1)"})
        self.assertTrue(all(f["id"].startswith("f") for f in p["graph_facts"]))

        self.assertEqual([(d["id"], d["kind"], d["page"], d["scope"]) for d in p["document_evidence"]],
                         [("d1", "text", 4, "asset"), ("d2", "figure", 7, "asset")])      # below the rerank threshold: dropped
        self.assertEqual(self.search.call_args.kwargs["doc_keys"], ["MANUAL"])           # the asset's documents first

        state = p["operational_context"]
        self.assertEqual(state["current_state"]["machine_state"], "STOPPED")
        pressure = next(v for v in state["values"] if v["signal_id"] == DIE_PRESSURE)
        self.assertEqual((pressure["value"], pressure["out_of_range"], pressure["high"]), (104.0, True, 99.2))
        self.assertEqual(state["recent_events"], [])

        self.assertIn(DIE_HEAD, p["provenance"]["graph_entities"])
        self.assertEqual(p["provenance"]["documents"], ["Manual.pdf"])
        self.assertEqual(p["warnings"], [])
        self.assertLessEqual(p["limits"]["used_chars"], p["limits"]["budget_chars"])
        self.assertIn("total", p["timings_ms"])
        text = p["context_text"]
        self.assertIn("GRAPH FACTS [G]", text)
        self.assertIn("Out of normal range in the last running sample: Die pressure: 104.0 bar", text)
        self.assertIn("[2] (Manual.pdf, p.7 (figure))", text)

    def test_small_budget_keeps_the_focus_facts(self):
        p = service.assemble("What is the die pressure limit?", ASSET, include_documents=False, budget_chars=2000)
        self.assertLessEqual(p["limits"]["used_chars"], 2000)
        self.assertTrue(any(f["entity_id"] == DIE_PRESSURE for f in p["graph_facts"]))
        self.assertLess(p["limits"]["facts_included"], p["limits"]["facts_total"] + 1)
        self.assertEqual(p["document_evidence"], [])
        self.search.assert_not_called()

    def test_documents_fit_the_budget(self):
        self.search.return_value = result(chunk("A" * 3000), chunk("B" * 3000), chunk("short one"))
        p = service.assemble("die", ASSET, budget_chars=6000)
        self.assertLessEqual(p["limits"]["used_chars"], 6000)
        self.assertLessEqual(sum(len(d["text"]) for d in p["document_evidence"]), 6000)

    def test_no_asset_gives_a_documents_only_packet(self):
        p = service.assemble("What is attention?")
        self.assertIsNone(p["resolved_scope"]["asset"])
        self.assertEqual((p["graph_facts"], p["operational_context"]), ([], None))
        self.assertEqual(len(p["document_evidence"]), 2)
        self.assertEqual(p["document_evidence"][0]["scope"], "library")
        self.assertEqual(p["warnings"], ["no asset in scope: documents only"])

    def test_library_fallback_when_the_assets_documents_have_nothing(self):
        self.search.side_effect = [result(), result(chunk("From another manual.", source="Other.pdf", key="OTHER"))]
        p = service.assemble("die", ASSET)
        self.assertEqual([(d["document"], d["scope"]) for d in p["document_evidence"]], [("Other.pdf", "library")])

    def test_sources_down_become_warnings(self):
        self.search.side_effect = RuntimeError("chroma down")
        p = service.assemble("die pressure", ASSET)
        self.assertEqual(p["warnings"], ["documents unavailable"])
        self.assertTrue(p["graph_facts"])

        self.search.side_effect = None
        with patch("context_graph.services.get_asset_context", side_effect=GraphUnavailable("neo4j down")), \
                patch("context_graph.services.list_assets", side_effect=GraphUnavailable("neo4j down")):
            p = service.assemble("die pressure on EXTR01")
        self.assertEqual(p["warnings"], ["graph unavailable"])
        self.assertEqual(len(p["document_evidence"]), 2)

        self.search.return_value = result(chunk("x" * 50), reranker="fallback")
        self.assertIn("reranker unavailable", service.assemble("die", ASSET)["warnings"][0])

    def test_empty_question(self):
        with self.assertRaisesMessage(service.ContextError, "query is required"):
            service.assemble("  ")


class LookupTests(ContextTestCase):
    def test_list_assets_state_and_search(self):
        self.assertEqual(service.list_assets(), [{"id": ASSET, "name": "Extruder EXTR01", "asset_type": "Extruder",
                                                  "description": "Kibble extrusion cooker"}])
        state = service.operational_state(ASSET)
        self.assertEqual(state["source"], "plc_1_historian")
        self.assertTrue(any(v["out_of_range"] for v in state["values"]))
        found = service.search_documents("die pressure", ASSET, top_k=3)
        self.assertEqual((found["scope"], len(found["results"])), ("asset", 2))

    @patch("chat.views.requests.post")
    def test_ask_answers_from_the_packet(self, post):
        post.return_value.json.return_value = {"choices": [{"message": {"content": "It is above its range [G]."}}]}
        post.return_value.raise_for_status.return_value = None
        out = service.ask("Why is the die pressure high on EXTR01?")
        self.assertEqual(out["answer"], "It is above its range [G].")
        system = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("GRAPH FACTS [G]", system)
        self.assertIn("Do not invent values", system)
        self.assertEqual(out["packet"]["resolved_scope"]["asset"]["id"], ASSET)
        with self.assertRaisesMessage(service.ContextError, "unknown model"):
            service.ask("x", model="gpt-9")


class RestTests(ContextTestCase):
    def setUp(self):
        super().setUp()
        self.client = APIClient()

    def test_endpoints(self):
        self.assertEqual(self.client.get("/api/context/assets/").data["assets"][0]["id"], ASSET)
        r = self.client.post("/api/context/resolve/", {"text": "die pressure on EXTR01"}, format="json")
        self.assertEqual(r.data["focus"][0]["id"], DIE_PRESSURE)
        r = self.client.post("/api/context/assemble/", {"query": "die pressure", "asset_id": ASSET,
                                                        "include_documents": False, "budget_chars": 3000}, format="json")
        self.assertEqual((r.status_code, r.data["packet_version"], r.data["document_evidence"]), (200, "1", []))
        self.assertEqual(self.client.get(f"/api/context/state/{ASSET}/").data["current_state"]["machine_state"], "STOPPED")
        self.assertEqual(self.client.get(f"/api/context/entities/{ASSET}/").data["name"], "Extruder EXTR01")
        r = self.client.post("/api/context/search/", {"query": "die"}, format="json")
        self.assertEqual(r.data["scope"], "library")

    def test_errors(self):
        self.assertEqual(self.client.post("/api/context/assemble/", {"query": ""}, format="json").status_code, 400)
        self.assertEqual(self.client.post("/api/context/search/", {}, format="json").status_code, 400)
        with patch("context_graph.services.node_detail", side_effect=NodeNotFound("x")):
            self.assertEqual(self.client.post("/api/context/assemble/", {"query": "x", "asset_id": "bsk:asset:NOPE"},
                                              format="json").status_code, 404)
        with patch("context_graph.services.list_assets", side_effect=GraphUnavailable("down")):
            self.assertEqual(self.client.get("/api/context/assets/").status_code, 503)

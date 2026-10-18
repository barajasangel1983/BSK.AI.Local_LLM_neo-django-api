"""P23 (2026-10-18): prompts kept like schemas — every version, validate, export / import, built-in updates, a test
on a page. The DGX is mocked; the graph is disabled."""

import json
from pathlib import Path
from unittest.mock import patch

import yaml
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from rest_framework.test import APIClient

from context_graph import extraction, prompts, registry
from context_graph.models import CandidateTriple, PromptPreset, PromptVersion
from ingestion import library
from ingestion.chunker import PAGE_BREAK
from ingestion.tests.tests_library import LibraryTestCase

SCHEMAS = Path(extraction.__file__).parent / "schemas"
V3 = yaml.safe_load((SCHEMAS / "industrial_v3.yaml").read_text())
V6 = yaml.safe_load((SCHEMAS / "industrial_v6.yaml").read_text())
CONTROLS = "Controls & Setup (v6)"


@override_settings(GRAPH_ENABLED=False)
class PromptVersionTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        registry.save_version(V6, note="v6")
        self.client = APIClient()
        self.listed = self.client.get("/api/graph/prompts/").json()
        self.by_name = {p["name"]: p for p in self.listed["prompts"]}

    def detail(self, name):
        return self.client.get(f"/api/graph/prompts/{self.by_name[name]['id']}/").json()

    def save(self, name, template, note=""):
        return self.client.post(f"/api/graph/prompts/{self.by_name[name]['id']}/versions/",
                                {"template": template, "note": note, "author": "Angel"}, format="json")

    def test_listed_by_facet_with_their_built_in_first_version(self):
        self.assertEqual([f["key"] for f in self.listed["facets"]], ["maintenance", "ratings", "controls"])
        self.assertEqual([p["facet"]["label"] for p in self.listed["prompts"]][:3], ["Maintenance", "Ratings", "Controls & Setup"])
        self.assertEqual(self.by_name["Default schema-guided"]["facet"]["key"], "other")
        d = self.detail(CONTROLS)
        self.assertEqual([(v["version"], v["source"], v["active"], v["schema_version"]) for v in d["versions"]],
                         [(1, "built-in", True, registry.active_schema().version)])
        self.assertIsNone(d["builtin_update"])

    def test_every_version_is_kept_and_activate_rolls_back(self):
        base = self.by_name[CONTROLS]["template"]
        r = self.save(CONTROLS, base + "\n- At most 15 triples per text.", "cap per window")
        self.assertEqual(r.status_code, 201, r.content)
        self.assertEqual((r.json()["version"], r.json()["versions"][0]["note"], r.json()["versions"][0]["author"]),
                         (2, "cap per window", "Angel"))
        self.assertEqual(self.save(CONTROLS, base + "\n- At most 15 triples per text.").status_code, 400)  # no change
        pid = self.by_name[CONTROLS]["id"]
        r = self.client.post(f"/api/graph/prompts/{pid}/versions/1/activate/").json()
        self.assertEqual((r["version"], r["template"]), (1, base))
        self.assertEqual(PromptPreset.objects.get(pk=pid).template, base)        # what extraction reads
        r = self.save(CONTROLS, base + "\n- Skip the macros.", "macros").json()
        self.assertEqual(r["version"], 3)                                          # never reuses 2
        self.assertEqual([v["version"] for v in r["versions"]], [3, 2, 1])
        CandidateTriple.objects.create(mode="schema", subject_name="a", subject_type="Component", predicate="HAS_PARAMETER",
                                       object_name="b", object_type="Parameter", preset_name=CONTROLS, preset_version=2)
        self.assertEqual({v["version"]: v["triples"] for v in self.detail(CONTROLS)["versions"]}, {3: 0, 2: 1, 1: 0})
        self.assertEqual(self.client.post(f"/api/graph/prompts/{pid}/versions/9/activate/").status_code, 404)

    def test_the_old_preset_edit_also_keeps_versions(self):
        pid = self.by_name[CONTROLS]["id"]
        self.client.post(f"/api/graph/prompts/{pid}/versions/1/activate/")
        self.save(CONTROLS, "Types {entity_types} {relationships} {text} two")
        self.client.post(f"/api/graph/prompts/{pid}/versions/1/activate/")
        r = self.client.patch(f"/api/graph/presets/{pid}/", {"template": "Types {entity_types} {relationships} {text} three"},
                              format="json")
        self.assertEqual(r.json()["version"], 3)
        self.assertEqual(PromptVersion.objects.filter(preset_id=pid).count(), 3)

    def test_validate(self):
        ok = self.client.post("/api/graph/prompts/validate/", {"template": self.by_name[CONTROLS]["template"],
                                                                "mode": "schema"}, format="json").json()
        self.assertEqual((ok["valid"], ok["warnings"]), (True, []))
        registry.save_version(V3, note="v3")                                     # no Parameter, no HAS_PARAMETER
        r = self.client.post("/api/graph/prompts/validate/", {"template": self.by_name[CONTROLS]["template"],
                                                               "mode": "schema"}, format="json").json()
        self.assertTrue(any("HAS_PARAMETER" in w and "links not in the active schema (v" in w for w in r["warnings"]))
        self.assertTrue(any("Parameter" in w and "types not in the active schema" in w for w in r["warnings"]))
        bad = self.client.post("/api/graph/prompts/validate/", {"template": "  ", "mode": "schema"}, format="json").json()
        self.assertEqual(bad["valid"], False)
        odd = prompts.check("Read {txt} {text} {entity_types} {relationships}", "schema")
        self.assertTrue(any("{txt}" in w for w in odd["warnings"]))

    def test_export_and_import(self):
        pid = self.by_name[CONTROLS]["id"]
        text = self.client.get(f"/api/graph/prompts/{pid}/export/").content.decode()
        data = yaml.safe_load(text)
        self.assertEqual((data["name"], data["facet"], data["version"], data["schema_version"]), (CONTROLS, "Controls & Setup", 1, registry.active_schema().version))
        self.assertEqual(data["template"], self.by_name[CONTROLS]["template"])
        self.assertIn("template: |", text)                                         # readable block text
        r = self.client.post("/api/graph/prompts/import/", {"text": text}, format="json").json()
        self.assertEqual(r["results"], [{"name": CONTROLS, "result": "unchanged", "version": 1}])
        data["template"] += "\n- Imported change."
        data["note"] = "from the other Studio"
        r = self.client.post("/api/graph/prompts/import/", {"text": yaml.safe_dump(data)}, format="json").json()
        self.assertEqual(r["results"][0]["result"], "new version")
        self.assertEqual(self.detail(CONTROLS)["versions"][0]["source"], "import")
        bundle = yaml.safe_load(self.client.get("/api/graph/prompts/export/").content)
        self.assertEqual(len(bundle["prompts"]), len(self.listed["prompts"]))
        bundle["prompts"].append({"name": "Alarms only", "mode": "schema", "template": "T {entity_types} {relationships} {text}"})
        r = self.client.post("/api/graph/prompts/import/", {"text": yaml.safe_dump(bundle)}, format="json").json()
        self.assertEqual(r["results"][-1]["result"], "new prompt")
        self.assertEqual(self.client.post("/api/graph/prompts/import/", {"text": "{"}, format="json").status_code, 400)

    def test_a_changed_built_in_is_offered_not_applied(self):
        pid = self.by_name[CONTROLS]["id"]
        newer = self.by_name[CONTROLS]["template"] + "\n- A rule added in code."
        with patch.dict(extraction.EXTRA_PRESETS, {CONTROLS: ("schema", newer)}):
            d = self.detail(CONTROLS)
            self.assertEqual(d["builtin_update"], newer)
            self.assertNotEqual(d["template"], newer)                              # not applied by itself
            r = self.client.post(f"/api/graph/prompts/{pid}/adopt-builtin/").json()
            self.assertEqual((r["template"], r["versions"][0]["source"]), (newer, "adopt"))
            self.assertIsNone(r["builtin_update"])


@override_settings(GRAPH_ENABLED=False)
class TestOnAPageTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        self.mock_convert.return_value = {"status": "success", "errors": [], "document": {"md_content": f"\n\n{PAGE_BREAK}\n\n".join(
            ["# 4.10.3 Terminal strips\n\nTerminal 5 is DI 0 (24 V DC); its function is set in p0840. " * 3,
             "# 5.4 Quick commissioning\n\nSet p1082 to the maximum speed. " * 3])}}
        registry.save_version(V6, note="v6")
        self.doc, _ = library.add_document(SimpleUploadedFile("g120c.pdf", b"%PDF g120c"))
        self.drain()
        self.client = APIClient()
        self.calls = []

        def llm(messages):
            self.calls.append(messages)
            return json.dumps({"note": "terminals", "triples": [
                {"subject": "Drive", "subject_type": "Component", "predicate": "HAS_IO", "object": "DI 0",
                 "object_type": "IOPoint", "object_subtype": "Digital input",
                 "object_properties": {"connector": "-X138", "terminal": "5"}, "confidence": 0.9}]})
        p = patch("context_graph.extraction.call_llm", side_effect=llm)
        p.start()
        self.addCleanup(p.stop)

    def test_a_draft_runs_on_the_chosen_pages_and_nothing_is_stored(self):
        r = self.client.post("/api/graph/prompts/test/", {"document_id": str(self.doc.id), "pages": "1",
                                                          "template": "DRAFT {entity_types} {relationships} {text}"},
                             format="json")
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        self.assertEqual(body["windows"], 1)
        self.assertTrue(self.calls[0][0]["content"].startswith("DRAFT"))
        self.assertIn("Terminal strips", self.calls[0][0]["content"])
        self.assertNotIn("Quick commissioning", self.calls[0][0]["content"])
        t = body["triples"][0]
        self.assertEqual((t["object"]["type"], t["object"]["id"], t["facets"], t["issue"]),
                         ("IOPoint", "bsk:io-point:G120C/X138-5", ["Installation", "Controls"], ""))
        self.assertEqual(CandidateTriple.objects.count(), 0)
        self.assertEqual(self.client.post("/api/graph/prompts/test/", {"document_id": str(self.doc.id), "pages": "1-5",
                                                                       "template": "x"}, format="json").status_code, 400)

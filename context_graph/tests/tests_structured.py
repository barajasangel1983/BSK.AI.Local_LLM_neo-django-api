"""Structured data import (P6): CSV / Excel -> column mapping -> staged triples.

Unit tests run with the graph disabled (every entity is "new"). Integration
tests (matching the EXTR01 seed, approve / delete) need the throwaway Neo4j:

    GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test context_graph
"""

import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from openpyxl import Workbook
from rest_framework.test import APIClient

from context_graph import driver, registry, services, structured
from context_graph.models import CandidateTriple, DataFile

TEST_URI = os.getenv("GRAPH_TEST_URI")
ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "seeds" / "extr01.yaml"
SAMPLES = ROOT / "samples"
ASSET = "bsk:asset:EXTR01"


class ImportTestCase(TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.override = override_settings(GRAPH_DATA_BASE=self.tmp)
        self.override.enable()
        self.client = APIClient()

    def tearDown(self):
        self.override.disable()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def upload(self, name: str, content: bytes | None = None) -> dict:
        content = (SAMPLES / name).read_bytes() if content is None else content
        r = self.client.post("/api/graph/datafiles/", {"files": [SimpleUploadedFile(name, content)]}, format="multipart")
        self.assertEqual(r.status_code, 201, r.content)
        return r.json()["results"][0]

    def detail(self, file_id: str) -> dict:
        return self.client.get(f"/api/graph/datafiles/{file_id}/").json()

    def suggested(self, file_id: str, template: str) -> dict:
        return next(s for s in self.detail(file_id)["suggestions"] if s["key"] == template)["mapping"]

    def post(self, file_id: str, action: str, mapping: dict, asset_id: str = ASSET):
        return self.client.post(f"/api/graph/datafiles/{file_id}/{action}/", {"mapping": mapping, "asset_id": asset_id},
                                format="json")


@override_settings(GRAPH_ENABLED=False)
class StructuredImportTests(ImportTestCase):
    def test_upload_reads_columns_and_dedupes(self):
        result = self.upload("extr01_tag_list.csv")
        df = result["data_file"]
        self.assertTrue(result["created"])
        self.assertEqual((df["key"], df["kind"], df["row_count"]), ("EXTR01-TAG-LIST", "csv", 33))
        self.assertEqual(df["columns"], ["Tag", "Description", "Unit", "Data Type", "Component", "Low", "High"])
        self.assertFalse(self.upload("copy.csv", (SAMPLES / "extr01_tag_list.csv").read_bytes())["created"])
        self.assertIn("unsupported file type", self.upload("notes.pdf", b"%PDF")["error"])
        detail = self.detail(df["id"])
        self.assertEqual(detail["sample"][0], {"row": 2, "values": {
            "Tag": "feeder_rate_actual_kg_hr", "Description": "Feeder rate", "Unit": "kg/h", "Data Type": "float",
            "Component": "Feeder", "Low": "7959", "High": "10391"}})

    def test_templates_are_suggested_from_the_header(self):
        for name, template, relationships in [
            ("extr01_tag_list.csv", "tag_list", ["MONITORED_BY", "MONITORED_BY", "HAS_LIMIT"]),
            ("extr01_alarm_list.csv", "alarm_list", ["HAS_ALARM", "ASSOCIATED_WITH"]),
            ("extr01_bom.csv", "bom", ["HAS_COMPONENT", "HAS_COMPONENT"]),
        ]:
            best = self.detail(self.upload(name)["data_file"]["id"])["suggestions"][0]
            self.assertEqual((best["key"], best["complete"]), (template, True), name)
            self.assertEqual([r["type"] for r in best["mapping"]["relationships"]], relationships)
        signal = next(e for e in self.suggested(str(DataFile.objects.get(key="EXTR01-TAG-LIST").id), "tag_list")["entities"]
                      if e["key"] == "signal")
        self.assertEqual((signal["id_column"], signal["name_column"], signal["properties"]),
                         ("Tag", "Description", {"unit": "Unit", "data_type": "Data Type", "description": "Description"}))

    def test_mapping_validation(self):
        file_id = self.upload("extr01_tag_list.csv")["data_file"]["id"]
        mapping = self.suggested(file_id, "tag_list")
        self.assertEqual(self.post(file_id, "preview", mapping, asset_id="").json()["errors"],
                         ["asset: select an asset (the mapping uses the selected asset)"])
        bad = {"entities": [{"key": "a", "type": "Alarm", "id_column": "Nope", "properties": {"Bad Name": "Unit", "id": "Tag"}},
                            {"key": "s", "type": "Signal", "id_column": "Tag"}],
               "relationships": [{"from": "a", "type": "HAS_LIMIT", "to": "s"}, {"from": "x", "type": "HAS_ALARM", "to": "s"}]}
        body = self.post(file_id, "preview", bad).json()
        self.assertFalse(body["valid"])
        errors = " | ".join(body["errors"])
        for expected in ("key column 'Nope' is not in the file", "invalid property name 'Bad Name'", "invalid property name 'id'",
                         "Alarm -HAS_LIMIT-> Signal is not allowed", "unknown entity 'x'"):
            self.assertIn(expected, errors)
        self.assertEqual(self.post(file_id, "stage", bad).status_code, 400)

    def test_tag_list_preview_and_stage(self):
        file_id = self.upload("extr01_tag_list.csv")["data_file"]["id"]
        mapping = self.suggested(file_id, "tag_list")
        body = self.post(file_id, "preview", mapping).json()
        self.assertTrue(body["valid"], body)
        # 33 signals: 32 under a component, 1 under the asset; 18 with limits.
        self.assertEqual(body["stats"], {"rows": 33, "triples": 51, "skipped_rows": 0, "new_entities": 62,
                                         "existing_entities": 0, "row_errors": 0})
        first = body["triples"][0]
        self.assertEqual((first["subject"]["id"], first["predicate"], first["object"]["id"], first["row"]),
                         ("bsk:component:EXTR01/Feeder", "MONITORED_BY", "bsk:signal:EXTR01/feeder_rate_actual_kg_hr", 2))
        self.assertEqual(first["object"]["properties"], {"unit": "kg/h", "data_type": "float", "description": "Feeder rate"})
        self.assertEqual(CandidateTriple.objects.count(), 0)    # preview saves nothing

        staged = self.post(file_id, "stage", mapping).json()
        self.assertEqual((staged["staged"], staged["data_file"]["triples"]["pending"]), (51, 51))
        limit = CandidateTriple.objects.get(object_id="bsk:operating-limit:EXTR01/steam_flow_kg_hr/normal")
        self.assertEqual((limit.predicate, limit.object_name, limit.object_props, limit.source, limit.mode),
                         ("HAS_LIMIT", "Steam flow normal range", {"low": 96.6, "high": 132.9, "unit": "kg/h"}, "structured", "schema"))
        loose = CandidateTriple.objects.get(object_id="bsk:signal:EXTR01/line_speed_m_min", predicate="MONITORED_BY")
        self.assertEqual((loose.subject_id, loose.row_number), (ASSET, 34))    # no component -> the asset
        self.assertIn("Tag: line_speed_m_min", loose.evidence_text)

        listed = self.client.get(f"/api/graph/triples/?data_file={file_id}&status=pending&limit=5").json()
        self.assertEqual((listed["total"], listed["triples"][0]["source"], listed["triples"][0]["evidence"]["row"]),
                         (51, "structured", 2))
        ids = self.client.get(f"/api/graph/triples/?data_file={file_id}&fields=ids").json()
        self.assertEqual((ids["total"], len(ids["ids"])), (51, 51))

        # Re-staging replaces pending triples and keeps reviewed ones.
        self.client.post("/api/graph/triples/reject/", {"ids": ids["ids"][:2]}, format="json")
        again = self.post(file_id, "stage", mapping).json()
        self.assertEqual((again["staged"], again["already_reviewed"]), (49, 2))
        self.assertEqual(self.detail(file_id)["triples"], {"pending": 49, "approved": 0, "rejected": 2})

    def test_bom_merges_names_and_reports_row_errors(self):
        content = ("Item;Description;Parent\n"
                   "gearbox;Gearbox;main-drive\n"
                   "main-drive;Main Drive Motor;\n"        # parent seen first by key only, named here
                   ";Orphan row;\n"                        # no item
                   "loop;Loop;loop\n").encode()
        file_id = self.upload("bom.csv", content)["data_file"]["id"]
        body = self.post(file_id, "preview", self.suggested(file_id, "bom")).json()
        self.assertEqual((body["stats"]["rows"], body["stats"]["triples"], body["stats"]["skipped_rows"]), (4, 2, 2))
        nested = body["triples"][0]
        self.assertEqual((nested["subject"]["name"], nested["predicate"], nested["object"]["name"]),
                         ("Main Drive Motor", "HAS_COMPONENT", "Gearbox"))
        self.assertEqual(body["triples"][1]["subject"]["id"], ASSET)
        self.assertEqual(body["row_errors"], [{"row": 4, "message": "no relationship built (missing: component, parent)"},
                                              {"row": 5, "message": "HAS_COMPONENT: 'Loop' would point to itself"}])

    def test_excel_sheet_and_header_row(self):
        wb = Workbook()
        wb.active.title = "Cover"
        wb.active.append(["Plant export"])
        ws = wb.create_sheet("Alarms")
        ws.append(["Alarm list EXTR01"])
        ws.append([])
        ws.append(["Code", "Message", "Priority"])
        ws.append(["DIE_PLUG", "Die plugged", 1])
        ws.append([101, "Overload", 2.0])
        buffer = io.BytesIO()
        wb.save(buffer)
        df = self.upload("alarms.xlsx", buffer.getvalue())["data_file"]
        self.assertEqual((df["sheets"], df["sheet"]), (["Cover", "Alarms"], "Cover"))
        r = self.client.patch(f"/api/graph/datafiles/{df['id']}/", {"sheet": "Alarms", "header_row": 3}, format="json")
        self.assertEqual((r.json()["columns"], r.json()["row_count"]), (["Code", "Message", "Priority"], 2))
        self.assertEqual(self.client.patch(f"/api/graph/datafiles/{df['id']}/", {"sheet": "Nope"}, format="json").status_code, 400)
        body = self.post(df["id"], "preview", self.suggested(df["id"], "alarm_list")).json()
        self.assertEqual([(t["object"]["id"], t["object"]["properties"]) for t in body["triples"]], [
            ("bsk:alarm:EXTR01/DIE_PLUG", {"code": "DIE_PLUG", "severity": 1, "description": "Die plugged"}),
            ("bsk:alarm:EXTR01/101", {"code": 101, "severity": 2, "description": "Overload"})])

    @override_settings(GRAPH_IMPORT_MAX_ROWS=2)
    def test_row_limit(self):
        file_id = self.upload("extr01_alarm_list.csv")["data_file"]["id"]
        body = self.post(file_id, "preview", self.suggested(file_id, "alarm_list")).json()
        self.assertEqual((body["valid"], body["errors"]), (False, ["21 rows: the limit is 2 rows per file"]))

    def test_saved_mappings_and_delete(self):
        file_id = self.upload("extr01_alarm_list.csv")["data_file"]["id"]
        mapping = self.suggested(file_id, "alarm_list")
        r = self.client.post("/api/graph/mappings/", {"name": "Alarm export", "mapping": mapping}, format="json")
        self.assertEqual(r.status_code, 201)
        self.assertEqual(self.client.post("/api/graph/mappings/", {"name": "Alarm export", "mapping": mapping},
                                          format="json").status_code, 200)       # same name: replaced
        self.assertEqual([m["name"] for m in self.client.get("/api/graph/mappings/").json()["mappings"]], ["Alarm export"])
        self.assertEqual(self.client.delete(f"/api/graph/mappings/{r.json()['id']}/").status_code, 204)

        self.post(file_id, "stage", mapping)
        self.assertTrue((Path(self.tmp) / file_id).exists())
        self.assertEqual(self.client.delete(f"/api/graph/datafiles/{file_id}/").json()["triples_deleted"], 39)   # 21 HAS_ALARM + 18 ASSOCIATED_WITH
        self.assertFalse((Path(self.tmp) / file_id).exists())
        self.assertEqual(CandidateTriple.objects.count(), 0)


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI (see tests_integration.py) to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class StructuredImportIntegrationTests(ImportTestCase):
    def setUp(self):
        super().setUp()
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
        services.init_graph()
        services.load_seed(SEED)

    def q(self, cypher, **params):
        with driver.session() as s:
            return s.run(cypher, **params).data()

    def nodes(self):
        return self.q("MATCH (n:Entity) RETURN count(n) AS n")[0]["n"]

    def stage_all(self, name: str, template: str) -> tuple[str, list[int]]:
        file_id = self.upload(name)["data_file"]["id"]
        self.assertEqual(self.post(file_id, "stage", self.suggested(file_id, template)).status_code, 200)
        return file_id, self.client.get(f"/api/graph/triples/?data_file={file_id}&fields=ids").json()["ids"]

    def test_tag_list_matches_the_seed_and_adds_only_what_is_new(self):
        base = self.nodes()
        file_id = self.upload("extr01_tag_list.csv")["data_file"]["id"]
        mapping = self.suggested(file_id, "tag_list")
        stats = self.post(file_id, "preview", mapping).json()["stats"]
        # New: 3 signals, 2 limits, the Gearbox component. Everything else is already in the graph.
        self.assertEqual((stats["triples"], stats["new_entities"], stats["existing_entities"]), (51, 6, 56))

        self.post(file_id, "stage", mapping)
        matched = CandidateTriple.objects.get(object_id="bsk:signal:EXTR01/melt_temp_c", predicate="MONITORED_BY")
        self.assertEqual((matched.subject_id, matched.subject_existing, matched.object_existing),
                         ("bsk:component:EXTR01/die-head", True, False))     # "Die Head" matched by name
        self.assertEqual(self.client.get(f"/api/graph/triples/?data_file={file_id}&new=1").json()["total"], 5)   # 6 new entities across 5 triples
        ids = self.client.get(f"/api/graph/triples/?data_file={file_id}&fields=ids").json()["ids"]
        result = self.client.post("/api/graph/triples/approve/", {"ids": ids}, format="json").json()
        self.assertEqual((len(result["approved"]), result["errors"]), (51, {}))
        self.assertEqual(self.nodes(), base + 6)

        new = self.q("MATCH (c)-[r:MONITORED_BY]->(s:Signal {id: 'bsk:signal:EXTR01/melt_temp_c'}) "
                     "RETURN s.name AS name, s.unit AS unit, s.source AS source, r.source AS rel, r.row AS row, c.id AS c")[0]
        self.assertEqual(new, {"name": "Melt temperature", "unit": "°C", "source": "structured", "rel": "structured",
                               "row": 32, "c": "bsk:component:EXTR01/die-head"})
        seeded = self.q("MATCH (:Component)-[r:MONITORED_BY]->(s:Signal {id: 'bsk:signal:EXTR01/steam_flow_kg_hr'}) "
                        "RETURN s.name AS name, s.source AS source, r.source AS rel, size(r.triple_ids) AS n")[0]
        self.assertEqual((seeded["name"], seeded["n"]), ("Steam flow", 1))
        self.assertNotEqual(seeded["rel"], "structured")            # seed provenance untouched
        self.assertNotEqual(seeded["source"], "structured")
        self.assertEqual(self.q("MATCH (d:Document) RETURN count(d) AS n")[0]["n"], 0)   # no evidence nodes

        self.assertEqual(self.client.delete(f"/api/graph/datafiles/{file_id}/").status_code, 200)
        self.assertEqual(self.nodes(), base)
        self.assertEqual(self.q("MATCH ()-[r]->() WHERE size(coalesce(r.triple_ids, [])) > 0 RETURN count(r) AS n")[0]["n"], 0)

    def test_bom_adds_missing_properties_and_removes_them_on_delete(self):
        base = self.nodes()
        file_id, ids = self.stage_all("extr01_bom.csv", "bom")
        self.assertEqual(len(ids), 13)
        self.client.post("/api/graph/triples/approve/", {"ids": ids}, format="json")
        self.assertEqual(self.nodes(), base + 4)       # gearbox, die plate, screen pack, knife assembly
        die = self.q("MATCH (n {id: 'bsk:component:EXTR01/die-head'}) RETURN n.name AS name, n.quantity AS qty, "
                     "n.component_type AS type, n.source AS source")[0]
        self.assertEqual((die["name"], die["qty"], die["type"]), ("Die Head", 1, "die"))   # quantity was missing; type kept
        self.assertNotEqual(die["source"], "structured")
        gearbox = self.q("MATCH (p)-[:HAS_COMPONENT]->(n {id: 'bsk:component:EXTR01/gearbox'}) "
                         "RETURN p.id AS parent, n.manufacturer AS m, n.name AS name")[0]
        self.assertEqual(gearbox, {"parent": "bsk:component:EXTR01/main-drive", "m": "SEW", "name": "Gearbox"})

        self.client.post("/api/graph/triples/delete/", {"ids": ids}, format="json")
        self.assertEqual(self.nodes(), base)
        self.assertIsNone(self.q("MATCH (n {id: 'bsk:component:EXTR01/die-head'}) RETURN n.quantity AS qty")[0]["qty"])

    def test_alarm_list(self):
        base = self.nodes()
        file_id, ids = self.stage_all("extr01_alarm_list.csv", "alarm_list")
        self.assertEqual(self.detail(file_id)["asset_id"], ASSET)
        new = CandidateTriple.objects.get(object_id="bsk:alarm:EXTR01/GEARBOX_OIL_TEMP_HIGH", predicate="HAS_ALARM")
        self.assertEqual((new.subject_existing, new.object_existing), (True, False))
        self.assertEqual(CandidateTriple.objects.filter(data_file_id=file_id, object_existing=False).count(), 2)
        self.client.post("/api/graph/triples/approve/", {"ids": ids}, format="json")
        self.assertEqual(self.nodes(), base + 2)       # the new alarm and the Gearbox component

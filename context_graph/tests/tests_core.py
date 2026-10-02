"""Context Graph P1 unit tests: IDs, schema, seed planning, registry, API views.

No Neo4j needed (services are mocked in the view tests). Run with the Django
test runner: `manage.py test context_graph`.
"""

import copy
from pathlib import Path
from unittest.mock import patch

import yaml
from django.test import SimpleTestCase, TestCase
from rest_framework.test import APIClient

from context_graph import registry
from context_graph.driver import GraphUnavailable
from context_graph.ids import clean_part, is_valid_id, make_id, parse_id, type_slug
from context_graph.models import GraphSchemaVersion
from context_graph.repository import NodeNotFound
from context_graph.schema import Schema, SchemaError, validate_definition
from context_graph.services import plan_seed, resolve_ref

SEED = Path(__file__).resolve().parent.parent / "seeds" / "extr01.yaml"


def _default_definition():
    return copy.deepcopy(Schema.from_yaml().definition)


class IdsTests(SimpleTestCase):
    def test_type_slug(self):
        self.assertEqual(type_slug("Asset"), "asset")
        self.assertEqual(type_slug("OperatingLimit"), "operating-limit")
        self.assertEqual(type_slug("OPCNode"), "opc-node")
        self.assertEqual(type_slug("DocumentSection"), "document-section")

    def test_make_id_keeps_case_and_cleans_parts(self):
        self.assertEqual(make_id("Asset", "EXTR01"), "bsk:asset:EXTR01")
        self.assertEqual(make_id("Component", "EXTR01", "Barrel Zone 1"), "bsk:component:EXTR01/Barrel-Zone-1")
        self.assertEqual(make_id("Alarm", "EXTR01", "DIE_PLUG"), "bsk:alarm:EXTR01/DIE_PLUG")
        with self.assertRaises(ValueError):
            clean_part("  //  ")

    def test_parse_and_validate(self):
        self.assertEqual(parse_id("bsk:signal:EXTR01/die_pressure_bar"), ("signal", ["EXTR01", "die_pressure_bar"]))
        self.assertTrue(is_valid_id("bsk:operating-limit:EXTR01/x/normal"))
        for bad in ("EXTR01", "bsk:Asset:EXTR01", "bsk:asset:", "bsk:asset:a b"):
            self.assertFalse(is_valid_id(bad), bad)
        with self.assertRaises(ValueError):
            parse_id("asset:EXTR01")


class SchemaTests(SimpleTestCase):
    def setUp(self):
        self.schema = Schema.from_yaml()

    def test_default_schema(self):
        self.assertEqual(len(self.schema.entity_types), 12)
        self.assertIn("CONNECTED_TO", self.schema.relationship_types)
        self.assertEqual(self.schema.label_for_slug("operating-limit"), "OperatingLimit")

    def test_relationship_checks(self):
        self.assertEqual(self.schema.check_relationship("Asset", "HAS_COMPONENT", "Component"), "HAS_COMPONENT")
        with self.assertRaisesMessage(SchemaError, "not allowed"):
            self.schema.check_relationship("Signal", "HAS_COMPONENT", "Component")
        with self.assertRaisesMessage(SchemaError, "unknown relationship"):
            self.schema.check_relationship("Asset", "OWNS", "Component")
        with self.assertRaisesMessage(SchemaError, "unknown entity type"):
            self.schema.check_label("Robot")

    def test_validation_errors(self):
        definition = _default_definition()
        definition["entity_types"]["bad label"] = {}
        definition["entity_types"]["Entity"] = {}
        definition["relationship_types"]["lower_case"] = {"pairs": [["Asset", "Component"]]}
        definition["relationship_types"]["FEEDS"] = {"pairs": [["Asset", "Robot"]]}
        definition["relationship_types"]["EMPTY"] = {"pairs": []}
        errors = " | ".join(validate_definition(definition))
        for expected in ("'bad label' must be PascalCase", "'Entity' is reserved", "'lower_case' must be UPPER_SNAKE",
                         "'Robot' is not an entity type", "'EMPTY' needs at least one"):
            self.assertIn(expected, errors)

    def test_slug_collision_is_rejected(self):
        definition = _default_definition()
        definition["entity_types"]["OpcNode"] = {}  # same ID prefix as OPCNode
        self.assertTrue(any("same ID prefix" in e for e in validate_definition(definition)))

    def test_constraint_statements(self):
        statements = self.schema.constraint_statements()
        self.assertIn("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE", statements)
        self.assertTrue(any("FOR (n:OperatingLimit) ON (n.name)" in s for s in statements))


class SeedPlanTests(SimpleTestCase):
    def setUp(self):
        self.schema = Schema.from_yaml()

    def test_resolve_ref(self):
        self.assertEqual(resolve_ref(self.schema, "Asset:EXTR01"), ("Asset", "bsk:asset:EXTR01"))
        self.assertEqual(resolve_ref(self.schema, "bsk:alarm:EXTR01/DIE_PLUG"), ("Alarm", "bsk:alarm:EXTR01/DIE_PLUG"))
        with self.assertRaises(SchemaError):
            resolve_ref(self.schema, "Robot:R1")

    def test_extr01_seed_is_valid(self):
        nodes, rels = plan_seed(yaml.safe_load(SEED.read_text()), self.schema)
        self.assertEqual((len(nodes), len(rels)), (87, 136))
        self.assertTrue(all(props["source"] == "seed:extr01-v1" for _, _, props in nodes))

    def test_invalid_seed_reports_all_problems(self):
        document = {
            "nodes": [{"ref": "Asset:A1", "name": "A"}, {"ref": "Asset:A1", "name": "dup"},
                      {"ref": "Robot:R1"}, {"ref": "Signal:A1/s", "name": "s"}],
            "relationships": [
                {"from": "Signal:A1/s", "type": "HAS_COMPONENT", "to": "Asset:A1"},
                {"from": "Asset:A1", "type": "MONITORED_BY", "to": "Signal:A1/missing"},
            ],
        }
        with self.assertRaises(SchemaError) as ctx:
            plan_seed(document, self.schema)
        message = str(ctx.exception)
        for expected in ("duplicate node bsk:asset:A1", "unknown entity type 'Robot'", "not allowed",
                         "doesn't define"):
            self.assertIn(expected, message)


class RegistryTests(TestCase):
    def test_falls_back_to_default_without_versions(self):
        schema = registry.active_schema()
        self.assertIsNone(schema.version)
        self.assertEqual(schema.name, "Industrial")

    def test_versions_are_append_only_with_one_active(self):
        v1 = registry.save_version(_default_definition(), note="v1")
        definition = _default_definition()
        definition["entity_types"]["Sensor"] = {"description": "Physical sensor"}
        v2 = registry.save_version(definition, note="add Sensor")

        self.assertEqual((v1.version, v2.version), (1, 2))
        self.assertEqual(list(GraphSchemaVersion.objects.filter(is_active=True)), [v2])
        self.assertIn("Sensor", registry.active_schema().entity_types)
        self.assertEqual(registry.active_schema().version, 2)

    def test_invalid_definition_is_not_stored(self):
        with self.assertRaises(SchemaError):
            registry.save_version({"entity_types": {"bad label": {}}, "relationship_types": {}})
        self.assertFalse(GraphSchemaVersion.objects.exists())


class ViewTests(SimpleTestCase):
    def setUp(self):
        self.client = APIClient()

    @patch("context_graph.views.services.get_asset_context", return_value={"asset": {"id": "bsk:asset:EXTR01"}})
    def test_ids_with_colons_and_slashes_route(self, mock_context):
        r = self.client.get("/api/graph/assets/bsk:asset:EXTR01/context/")
        self.assertEqual(r.status_code, 200)
        mock_context.assert_called_once_with("bsk:asset:EXTR01")
        with patch("context_graph.views.services.node_detail", return_value={}) as mock_node:
            self.client.get("/api/graph/nodes/bsk:component:EXTR01/die-head/")
        mock_node.assert_called_once_with("bsk:component:EXTR01/die-head")

    @patch("context_graph.views.services.search_nodes", return_value={"items": []})
    def test_query_params_are_parsed_and_clamped(self, mock_search):
        self.client.get("/api/graph/nodes/", {"type": "Signal", "q": "temp", "limit": "5000", "offset": "10"})
        mock_search.assert_called_once_with("Signal", "temp", limit=1000, offset=10)
        with patch("context_graph.views.services.graph_data", return_value={}) as mock_data:
            self.client.get("/api/graph/data/", {"root": "bsk:asset:EXTR01", "depth": "9", "types": "Signal,Alarm"})
        mock_data.assert_called_once_with("bsk:asset:EXTR01", depth=4, labels=["Signal", "Alarm"], limit=300,
                                          layer="curated", doc_key=None)

    def test_errors_map_to_statuses(self):
        cases = [
            (GraphUnavailable("disabled"), 503),
            (NodeNotFound("bsk:asset:NOPE"), 404),
            (SchemaError("unknown entity type 'Robot'"), 400),
        ]
        for exc, code in cases:
            with patch("context_graph.views.services.search_nodes", side_effect=exc):
                self.assertEqual(self.client.get("/api/graph/nodes/").status_code, code, exc)
        self.assertEqual(self.client.get("/api/graph/nodes/", {"limit": "abc"}).status_code, 400)


class SchemaDiffAndParseTests(SimpleTestCase):
    def test_diff_reports_types_and_pairs(self):
        from context_graph.schema import diff, is_empty_diff

        old = Schema.from_yaml()
        definition = _default_definition()
        definition["entity_types"]["Sensor"] = {}
        definition["relationship_types"]["MONITORED_BY"]["pairs"].append(["Component", "Sensor"])
        definition["relationship_types"]["MONITORED_BY"]["pairs"].remove(["Asset", "Signal"])
        del definition["relationship_types"]["CONNECTED_TO"]
        change = diff(old, Schema.from_definition(definition))
        self.assertEqual(change["added_entity_types"], ["Sensor"])
        self.assertEqual(change["removed_relationship_types"], ["CONNECTED_TO"])
        self.assertEqual(change["changed_pairs"]["MONITORED_BY"],
                         {"added": [["Component", "Sensor"]], "removed": [["Asset", "Signal"]]})
        self.assertTrue(is_empty_diff(diff(old, Schema.from_yaml())))

    def test_parse_yaml_and_json(self):
        from context_graph.services import parse_schema_text

        self.assertEqual(parse_schema_text('{"entity_types": {}}', "json"), {"entity_types": {}})
        self.assertEqual(parse_schema_text("entity_types: {}\n", "yaml"), {"entity_types": {}})
        for text, fmt, message in [("entity_types: [", "yaml", "could not parse YAML"),
                                   ("- a\n- b", "yaml", "must be a mapping"),
                                   ("{}", "xml", "format must be")]:
            with self.assertRaisesMessage(SchemaError, message):
                parse_schema_text(text, fmt)


class SchemaRegistryEditTests(TestCase):
    def test_export_and_activate(self):
        from context_graph.services import export_schema

        registry.save_version(_default_definition(), note="v1")
        definition = _default_definition()
        definition["entity_types"]["Sensor"] = {}
        registry.save_version(definition, note="v2")
        self.assertIn("Sensor:", export_schema(None, "yaml"))
        self.assertNotIn('"Sensor"', export_schema(1, "json"))
        registry.activate(1)
        self.assertEqual(registry.active_schema().version, 1)
        self.assertEqual([v["is_active"] for v in registry.versions()], [False, True])
        with self.assertRaises(SchemaError):
            registry.activate(99)


class SchemaEditViewTests(SimpleTestCase):
    def setUp(self):
        self.client = APIClient()

    def test_validate_never_raises_on_bad_input(self):
        r = self.client.post("/api/graph/schema/validate/", {"text": "entity_types: [", "format": "yaml"}, format="json")
        self.assertEqual((r.status_code, r.data["valid"]), (200, False))
        self.assertIn("could not parse YAML", r.data["errors"][0])

    @patch("context_graph.views.services.save_schema")
    def test_conflicts_are_409_with_details(self, mock_save):
        from context_graph.services import SchemaConflict

        mock_save.side_effect = SchemaConflict([{"kind": "entity_type", "name": "Procedure", "count": 8,
                                                 "message": "Procedure is used by 8 node(s)"}])
        r = self.client.post("/api/graph/schema/", {"definition": _default_definition(), "note": "x"}, format="json")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.data["conflicts"][0]["name"], "Procedure")

    @patch("context_graph.views.services.export_schema", return_value="name: Industrial\n")
    def test_export_is_a_download(self, mock_export):
        r = self.client.get("/api/graph/schema/export/", {"fmt": "yaml", "version": "2"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r["Content-Disposition"], 'attachment; filename="graph-schema-v2.yaml"')
        mock_export.assert_called_once_with(2, "yaml")

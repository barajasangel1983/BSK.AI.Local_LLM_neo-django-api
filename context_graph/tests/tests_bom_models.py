"""P26d (2026-10-25): a BOM's Manufacturer + Model (and optional Family, Variant, keys) give each installed part its
device model and family in the same import; an existing model is matched by id, then by manufacturer and exact name;
a changed variant or a part's changed model is staged for review; the variant reaches the extraction's context."""

from pathlib import Path
from unittest.mock import patch

import yaml
from django.test import SimpleTestCase, override_settings

from context_graph import extraction, registry, structured
from context_graph.identity import EntityIndex, normalize
from context_graph.models import DataFile
from context_graph.tests.tests_structured import ImportTestCase

SCHEMAS = Path(extraction.__file__).parent / "schemas"
V7 = yaml.safe_load((SCHEMAS / "industrial_v7.yaml").read_text())
ASSET = "bsk:asset:EXTR01"
G120C_75 = "SINAMICS G120C, 7.5 kW, 400 V, PROFINET"
MODEL_75 = "bsk:device-model:siemens/sinamics-g120c-7.5-kw-400-v-profinet"
FAMILY = "bsk:product-family:siemens/sinamics-g120c"
HEAD = "Item,Parent,Description,Type,Manufacturer,Model,Part Number,Qty,Family,Variant\n"


@override_settings(GRAPH_ENABLED=False)
class BomModelTests(ImportTestCase):
    def build(self, csv: str, index: EntityIndex | None = None):
        registry.save_version(V7, note="v7")
        file_id = self.upload("EXTR01_bom.csv", csv.encode())["data_file"]["id"]
        mapping = self.suggested(file_id, "bom")
        df = DataFile.objects.get(pk=file_id)
        self.assertEqual(structured.check(df, mapping, ASSET), [])
        return mapping, structured.build(df, mapping, ASSET, index=index or EntityIndex())

    def links(self, result):
        return sorted((t.subject_id.split(":", 2)[2], t.predicate, t.object_id.split(":", 2)[2])
                      for t in result.triples if t.predicate in ("INSTANCE_OF", "VARIANT_OF"))

    def test_without_manufacturer_and_model_nothing_changes(self):
        mapping, result = self.build("Item,Parent,Description,Type\nfeeder,,Feeder,Feeder\n")
        self.assertNotIn("model", {e["key"] for e in mapping["entities"]})
        self.assertEqual(self.links(result), [])

    def test_parts_get_their_model_and_family_twins_share_one(self):
        _, result = self.build(HEAD
            + f'cutter-drive,,Cutter drive,Drive,Siemens,"{G120C_75}",6SL3210-1KE21-7UF1,1,SINAMICS G120C,frame size B\n'
            + 'zone-1-tc,,Zone 1 thermocouple,Sensor,Watlow,TM411,TM411,1,,\n'
            + 'zone-2-tc,,Zone 2 thermocouple,Sensor,Watlow,TM411,TM411,1,,\n'
            + 'hopper,,Hopper,Hopper,,,,1,,\n')
        self.assertEqual(self.links(result), [
            ("EXTR01/cutter-drive", "INSTANCE_OF", "siemens/sinamics-g120c-7.5-kw-400-v-profinet"),
            ("EXTR01/zone-1-tc", "INSTANCE_OF", "watlow/tm411"),
            ("EXTR01/zone-2-tc", "INSTANCE_OF", "watlow/tm411"),
            ("siemens/sinamics-g120c-7.5-kw-400-v-profinet", "VARIANT_OF", "siemens/sinamics-g120c"),
        ])
        drive = next(t for t in result.triples if t.predicate == "INSTANCE_OF" and t.object_id == MODEL_75)
        self.assertEqual((drive.object_name, drive.object_props),
                         (G120C_75, {"manufacturer": "Siemens", "variant": "frame size B"}))
        part = next(t for t in result.triples if t.predicate == "HAS_COMPONENT" and t.object_id.endswith("cutter-drive"))
        self.assertEqual(part.object_props["model"], G120C_75)          # the part keeps its own model text

    def test_an_existing_model_is_matched_by_manufacturer_and_name(self):
        edited = "bsk:device-model:siemens/g120c-cutter"                # an id Angel chose in P26a
        with patch.object(structured, "_products", return_value={("DeviceModel", normalize("Siemens"), normalize(G120C_75)): edited}):
            _, result = self.build(HEAD + f'cutter-drive,,Cutter drive,Drive,Siemens,"{G120C_75}",,1,,\n'
                                   + 'other-drive,,Other drive,Drive,ABB,"SINAMICS G120C, 7.5 kW, 400 V, PROFINET",,1,,\n')
        by_part = {t.subject_id.rsplit("/", 1)[1]: (t.object_id, t.object_match)
                   for t in result.triples if t.predicate == "INSTANCE_OF"}
        self.assertEqual(by_part["cutter-drive"], (edited, "id"))
        self.assertEqual(by_part["other-drive"][1], "new")              # another manufacturer: not the same model

    def test_keys_in_the_file_win(self):
        _, result = self.build("Item,Description,Manufacturer,Model,Model Key,Family,Family Key\n"
                               f'cutter-drive,Cutter drive,Siemens,"{G120C_75}",siemens/g120c-75,SINAMICS G120C,siemens/sinamics-g120c\n')
        self.assertEqual(self.links(result), [("EXTR01/cutter-drive", "INSTANCE_OF", "siemens/g120c-75"),
                                              ("siemens/g120c-75", "VARIANT_OF", "siemens/sinamics-g120c")])

    def test_a_changed_variant_is_staged_and_other_properties_are_not(self):
        index = EntityIndex()
        index.add(MODEL_75, "DeviceModel", G120C_75)
        graph = {"variant": "frame size D", "manufacturer": "SIEMENS AG", "order_code": "6SL3210"}
        with patch.object(structured._GraphProps, "get", return_value=graph):
            _, result = self.build(HEAD + f'cutter-drive,,Cutter drive,Drive,Siemens,"{G120C_75}",,1,,frame size B\n', index)
        drive = next(t for t in result.triples if t.predicate == "INSTANCE_OF")
        self.assertEqual(drive.replaces[MODEL_75], {"variant": ["frame size D", "frame size B"]})

    def test_a_part_whose_model_changed_is_staged_as_a_replaced_link(self):
        index = EntityIndex()
        index.add("bsk:component:EXTR01/cutter-drive", "Component", "Cutter drive")
        old = "bsk:device-model:siemens/sinamics-g120c-5.5-kw-400-v-profinet"
        with patch.object(structured, "_single_links", return_value={old}):
            _, result = self.build(HEAD + f'cutter-drive,,Cutter drive,Drive,Siemens,"{G120C_75}",,1,,\n', index)
        drive = next(t for t in result.triples if t.predicate == "INSTANCE_OF")
        self.assertEqual(drive.replaces["bsk:component:EXTR01/cutter-drive"]["INSTANCE_OF"], [old, MODEL_75])
        self.assertIn("approving removes that link", drive.issue)

    def test_the_same_model_again_is_no_change(self):
        index = EntityIndex()
        index.add("bsk:component:EXTR01/cutter-drive", "Component", "Cutter drive")
        with patch.object(structured, "_single_links", return_value={MODEL_75}):
            _, result = self.build(HEAD + f'cutter-drive,,Cutter drive,Drive,Siemens,"{G120C_75}",,1,,\n', index)
        drive = next(t for t in result.triples if t.predicate == "INSTANCE_OF")
        self.assertEqual((drive.issue, drive.replaces), ("", {}))


class VariantContextTests(SimpleTestCase):
    def model(self, name, variant=""):
        return {"name": name, "properties": {"variant": variant} if variant else {}}

    def test_a_model_run_names_its_variant(self):
        self.assertEqual(extraction._variant_line("model", {"variant": "frame size B"}, []), "\n- Its variant: frame size B")
        self.assertEqual(extraction._variant_line("model", {}, []), "")

    def test_a_family_run_names_the_variant_of_its_models_in_use(self):
        same = [self.model("G120C 7.5 kW", "frame size B"), self.model("G120C 5.5 kW", "frame size B")]
        self.assertEqual(extraction._variant_line("family", {}, same), "\n- The variant of its models in use: frame size B")
        mixed = [self.model("A", "frame size B"), self.model("B")]
        self.assertEqual(extraction._variant_line("family", {}, mixed),
                         "\n- The variants of its models in use: A: frame size B; B: not given")
        self.assertEqual(extraction._variant_line("family", {}, [self.model("A")]), "")


class ReplacedLinkTests(SimpleTestCase):
    def test_a_replaced_links_dates_are_kept_as_text_and_named(self):
        from neo4j.time import DateTime
        from context_graph.triples import _plain_props
        out = _plain_props({"source": "structured", "created_at": DateTime(2026, 10, 22, 3, 11, 6), "triple_ids": [7984]})
        self.assertEqual(out["dates"], ["created_at"])
        self.assertEqual(out["props"]["triple_ids"], [7984])
        self.assertTrue(out["props"]["created_at"].startswith("2026-10-22T03:11:06"))

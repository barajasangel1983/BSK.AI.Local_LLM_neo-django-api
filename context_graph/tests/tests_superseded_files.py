"""P26f (2026-10-25): staging a file makes the other files with the same filename and asset its older versions;
the Import list hides them unless ?all=1 and says how many there are (and how many still have pending triples)."""

from django.test import override_settings

from context_graph import structured
from context_graph.models import CandidateTriple, DataFile
from context_graph.tests.tests_structured import ASSET, ImportTestCase

BOM = "Item,Parent,Description,Type\nfeeder,,Feeder,Feeder\n"


@override_settings(GRAPH_ENABLED=False)
class SupersededFileTests(ImportTestCase):
    def staged(self, name: str, content: str, asset: str = ASSET) -> DataFile:
        file_id = self.upload(name, content.encode())["data_file"]["id"]
        r = self.post(file_id, "stage", self.suggested(file_id, "bom"), asset)
        self.assertEqual(r.status_code, 200, r.content)
        return DataFile.objects.get(pk=file_id)

    def listed(self, all_files: bool = False) -> dict:
        return self.client.get("/api/graph/datafiles/" + ("?all=1" if all_files else "")).json()

    def test_a_newer_version_hides_the_older_one(self):
        old = self.staged("EXTR01_bom.csv", BOM)
        new = self.staged("EXTR01_bom.csv", BOM + "hopper,feeder,Hopper,Hopper\n")
        other_name = self.staged("EXTR01_limits_bom.csv", BOM + "x,,X,X\n")
        other_asset = self.staged("EXTR01_bom.csv", BOM + "y,,Y,Y\n", asset="bsk:asset:EXTR02")
        old.refresh_from_db()
        self.assertEqual(old.superseded_by_id, new.pk)
        self.assertIsNone(DataFile.objects.get(pk=new.pk).superseded_by_id)
        self.assertIsNone(DataFile.objects.get(pk=other_name.pk).superseded_by_id)
        self.assertIsNone(DataFile.objects.get(pk=other_asset.pk).superseded_by_id)
        listed = self.listed()
        self.assertNotIn(str(old.pk), [f["id"] for f in listed["data_files"]])
        self.assertEqual(listed["older_versions"]["count"], 1)
        shown = {f["id"]: f for f in self.listed(all_files=True)["data_files"]}
        self.assertEqual(shown[str(old.pk)]["superseded_by"], str(new.pk))
        self.assertEqual(shown[str(old.pk)]["superseded_by_uploaded_at"], new.uploaded_at.isoformat())

    def test_pending_triples_of_an_older_version_are_counted(self):
        old = self.staged("EXTR01_bom.csv", BOM)
        self.assertTrue(old.triples.filter(status=CandidateTriple.Status.PENDING).exists())
        self.staged("EXTR01_bom.csv", BOM + "hopper,feeder,Hopper,Hopper\n")
        self.assertEqual(self.listed()["older_versions"], {"count": 1, "with_pending": 1})
        self.assertTrue(old.triples.exists())             # its triples stay

    def test_staging_an_older_version_again_makes_it_current(self):
        old = self.staged("EXTR01_bom.csv", BOM)
        new = self.staged("EXTR01_bom.csv", BOM + "hopper,feeder,Hopper,Hopper\n")
        self.post(str(old.pk), "stage", self.suggested(str(old.pk), "bom"))
        self.assertEqual((DataFile.objects.get(pk=old.pk).superseded_by_id, DataFile.objects.get(pk=new.pk).superseded_by_id),
                         (None, old.pk))

    def test_deleting_the_newer_version_brings_the_older_back(self):
        old = self.staged("EXTR01_bom.csv", BOM)
        new = self.staged("EXTR01_bom.csv", BOM + "hopper,feeder,Hopper,Hopper\n")
        structured.delete_data_file(new)
        self.assertIsNone(DataFile.objects.get(pk=old.pk).superseded_by_id)

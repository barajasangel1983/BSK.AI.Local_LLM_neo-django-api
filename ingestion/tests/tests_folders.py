"""Library folders and the unattended ingest (P14d-2): parts of long PDFs, waiting for BSK / the DGX,
embeddings after the parse, batched Chroma writes, library_ingest. Docling and the embedder are mocked."""

import io
import json
from datetime import timedelta
from unittest.mock import patch

import pypdfium2 as pdfium
import requests
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from ingestion import folders, library
from ingestion.chunker import PAGE_BREAK
from ingestion.docling_client import DoclingError
from ingestion.models import Document, Folder, Job
from ingestion.vector_store import VectorStore

from .tests_library import PARSED, LibraryTestCase

STANDARD = ["Manuals", "Datasheets", "Drawings", "SOPs", "Work Instructions", "Quality Alerts", "Visual Aids",
            "Training", "Reports"]


def pdf(pages: int) -> bytes:
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(612, 792)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def docling_for_parts(data: bytes, filename: str, with_layout: bool = False, **kw) -> dict:
    """A Docling answer for a part: one paragraph per page, page breaks between, a figure on page 2."""
    n = len(pdfium.PdfDocument(data))
    md = f"\n\n{PAGE_BREAK}\n\n".join(f"Text of {filename} page {i + 1}." for i in range(n))
    layout = {"pages": {"2": {"size": {"width": 612, "height": 792}}}, "texts": [],
              "pictures": [{"prov": [{"page_no": 2, "bbox": {"l": 100, "t": 500, "r": 400, "b": 200}}]}]}
    return {"status": "success", "errors": [], "document": {"md_content": md, "json_content": layout}}


class FolderTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        self.client = APIClient()

    def test_standard_folders(self):
        tree = self.client.get("/api/folders/").json()
        top = [f for f in tree["folders"] if f["parent"] is None]
        self.assertEqual([f["name"] for f in top], STANDARD)
        self.assertEqual({f["kind"] for f in top if f["name"] == "Quality Alerts"}, {"quality_alert"})
        self.assertTrue(all(f["standard"] for f in top))
        self.assertEqual(tree["unfiled_count"], 0)

    def test_create_rename_move_and_delete(self):
        manuals = Folder.objects.get(name="Manuals", parent=None)
        r = self.client.post("/api/folders/", {"name": "Drives", "parent": str(manuals.id)}, format="json")
        self.assertEqual(r.status_code, 201)
        drives = Folder.objects.get(pk=r.json()["id"])
        self.assertEqual((drives.kind, folders.path(drives)), ("manual", "Manuals/Drives"))
        self.assertEqual(self.client.post("/api/folders/", {"name": "drives", "parent": str(manuals.id)},
                                          format="json").json()["error"], "there is already a folder 'drives' here")
        self.assertEqual(self.client.post("/api/folders/", {"name": "a/b"}, format="json").status_code, 400)
        abb = folders.create_folder("ABB", drives)
        self.assertIn("inside itself", self.client.patch(f"/api/folders/{drives.id}/", {"parent": str(abb.id)},
                                                         format="json").json()["error"])
        # Moving under SOPs makes the whole subtree SOP-kind.
        sops = Folder.objects.get(name="SOPs", parent=None)
        self.client.patch(f"/api/folders/{drives.id}/", {"name": "Drive SOPs", "parent": str(sops.id)}, format="json")
        abb.refresh_from_db()
        self.assertEqual((abb.kind, folders.path(abb)), ("sop", "SOPs/Drive SOPs/ABB"))

        doc, _ = self.upload()
        self.client.post("/api/documents/move/", {"ids": [str(doc.id)], "folder": str(abb.id)}, format="json")
        r = self.client.delete(f"/api/folders/{drives.id}/")
        self.assertEqual(r.status_code, 409)
        self.assertIn("holds 1 document(s)", r.json()["error"])
        self.assertEqual(self.client.delete(f"/api/folders/{drives.id}/?move_to={abb.id}").status_code, 409)
        r = self.client.delete(f"/api/folders/{drives.id}/?move_to=unfiled").json()
        self.assertEqual((r["folders_deleted"], r["documents_moved"], r["unfiled_count"]), (2, 1, 1))
        doc.refresh_from_db()
        self.assertIsNone(doc.folder)                                   # moved, never deleted
        self.assertTrue(Document.objects.filter(pk=doc.pk).exists())

    def test_upload_into_a_folder_and_moving_updates_the_chunks(self):
        sops = Folder.objects.get(name="SOPs", parent=None)
        r = self.client.post("/api/documents/", {"files": [SimpleUploadedFile("Die change.pdf", b"%PDF sop")],
                                                 "folder": str(sops.id), "then_embed": "1"}, format="multipart")
        doc = Document.objects.get(pk=r.json()["results"][0]["document"]["id"])
        self.assertEqual(r.json()["results"][0]["document"]["folder"], str(sops.id))
        self.drain()                                                    # parse, then embed by itself
        doc.refresh_from_db()
        self.assertEqual((doc.parse_status, doc.rag_status), ("parsed", "done"))
        _, metas = self.chunks(doc.doc_key)
        self.assertEqual({(m["doc_type"], m["folder_path"]) for m in metas}, {("sop", "SOPs")})

        alerts = folders.create_folder("Extruder", Folder.objects.get(name="Quality Alerts", parent=None))
        self.client.post("/api/documents/move/", {"ids": [str(doc.id)], "folder": str(alerts.id)}, format="json")
        _, metas = self.chunks(doc.doc_key)
        self.assertEqual({(m["doc_type"], m["folder_path"]) for m in metas}, {("quality_alert", "Quality Alerts/Extruder")})
        self.assertEqual(self.mock_embedder.return_value.embed_passages.call_count, 1)   # no new embeddings
        self.client.post("/api/documents/move/", {"ids": [str(doc.id)], "folder": "unfiled"}, format="json")
        self.assertEqual({m["doc_type"] for m in self.chunks(doc.doc_key)[1]}, {"unfiled"})


@override_settings(LIBRARY_PARSE_PART_PAGES=25)
class UnattendedTests(LibraryTestCase):
    def test_a_long_pdf_is_parsed_in_parts(self):
        self.mock_convert.side_effect = docling_for_parts
        doc, _ = self.upload("Long manual.pdf", pdf(60))
        self.drain()
        doc.refresh_from_db()
        self.assertEqual((doc.parse_status, doc.page_count), ("parsed", 60))
        self.assertEqual([c.args[1] for c in self.mock_convert.call_args_list],
                         ["Long manual p1-25.pdf", "Long manual p26-50.pdf", "Long manual p51-60.pdf"])
        md = library.load_parsed(doc)["document"]["md_content"]
        self.assertEqual(md.count(PAGE_BREAK), 59)
        pages = md.split(PAGE_BREAK)
        self.assertIn("p26-50.pdf page 1.", pages[25])                  # the PDF's page 26
        figs = json.loads((library.doc_dir(doc) / "figures.json").read_text())["figures"]
        self.assertEqual([(f["index"], f["page"]) for f in figs], [(0, 2), (1, 27), (2, 52)])
        self.assertFalse((library.doc_dir(doc) / "parts").exists())

    def test_bsk_away_waits_and_resumes_at_the_missing_part(self):
        calls = []

        def flaky(data, filename, **kw):
            calls.append(filename)
            if len(calls) == 2:
                raise DoclingError("Docling unavailable") from requests.ConnectionError("BSK is off")
            return docling_for_parts(data, filename)

        self.mock_convert.side_effect = flaky
        doc, _ = self.upload("Long manual.pdf", pdf(60))
        job = library.claim_next()
        library.run_job(job)
        job.refresh_from_db()
        self.assertEqual(job.status, "queued")
        self.assertIn("waiting for BSK", job.message)
        self.assertGreater(job.run_after, timezone.now() + timedelta(minutes=4))
        self.assertIsNone(library.claim_next())                         # not before run_after
        Job.objects.filter(pk=job.pk).update(run_after=timezone.now())
        self.drain()
        doc.refresh_from_db()
        self.assertEqual((doc.parse_status, doc.page_count), ("parsed", 60))
        self.assertEqual(calls, ["Long manual p1-25.pdf", "Long manual p26-50.pdf", "Long manual p26-50.pdf",
                                 "Long manual p51-60.pdf"])            # part 1 not parsed again

    def test_a_part_that_fails_twice_fails_the_job_with_its_pages(self):
        self.mock_convert.side_effect = lambda data, filename, **kw: (
            (_ for _ in ()).throw(DoclingError("HTTP 500")) if "p26-50" in filename else docling_for_parts(data, filename))
        doc, _ = self.upload("Long manual.pdf", pdf(60))
        self.drain()
        doc.refresh_from_db()
        self.assertEqual(doc.parse_status, "failed")
        self.assertIn("pages 26–50: DoclingError: HTTP 500", doc.parse_error)
        self.assertEqual(len(list((library.doc_dir(doc) / "parts").iterdir())), 1)   # part 1 kept for a retry

    def test_dgx_away_waits_then_gives_up_after_the_retry_days(self):
        self.mock_embedder.return_value.embed_passages.side_effect = requests.ConnectionError("DGX is off")
        doc, _ = library.add_document(SimpleUploadedFile("a.pdf", b"%PDF a"), then_embed=True)
        self.drain()
        embed = doc.jobs.get(kind="embed")
        self.assertEqual(embed.status, "queued")
        self.assertIn("waiting for the DGX", embed.message)
        Job.objects.filter(pk=embed.pk).update(run_after=timezone.now(), created_at=timezone.now() - timedelta(days=8))
        self.drain()
        embed.refresh_from_db()
        doc.refresh_from_db()
        self.assertEqual((embed.status, doc.rag_status), ("failed", "failed"))
        self.assertIn("gave up after 7 days", doc.rag_error)

    def test_chunks_are_written_in_batches(self):
        with patch.object(VectorStore, "max_batch", return_value=2):
            doc, _ = library.add_document(SimpleUploadedFile("a.pdf", b"%PDF a"), then_embed=True)
            self.drain()
        doc.refresh_from_db()
        ids, _ = self.chunks(doc.doc_key)
        self.assertGreater(doc.rag_chunk_count, 2)
        self.assertEqual(len(ids), doc.rag_chunk_count)


class LibraryIngestTests(LibraryTestCase):
    def test_manifest_to_library(self):
        out = library.Path(self.tmp) / "manuals"
        out.mkdir()
        entries = []
        for name, title, maker, body in [("ACS880 Firmware.pdf", "ACS880 Firmware manual", "ABB", b"%PDF " + b"a" * 900),
                                         ("Opta.pdf", "Arduino Opta Datasheet", "Arduino", b"%PDF opta")]:
            (out / name).write_bytes(body)
            entries.append({"url": f"https://example/{name}", "title": title, "file": name, "manufacturer": maker,
                            "status": "downloaded", "size": len(body), "sha256": __import__("hashlib").sha256(body).hexdigest(),
                            "downloaded_at": "2026-10-15T03:54:37+00:00"})
        entries.append({"url": "https://phoenix/x", "title": "QUINT4", "file": "q.pdf", "status": "by hand"})
        (out / "manifest.json").write_text(json.dumps({"manuals": entries}))
        manifest = str(out / "manifest.json")

        dry = io.StringIO()
        call_command("library_ingest", manifest=manifest, dry_run=True, stdout=dry)
        self.assertEqual(Document.objects.count(), 0)
        self.assertIn("Datasheets/Arduino", dry.getvalue())

        call_command("library_ingest", manifest=manifest, asset="bsk:asset:EXTR01", embed=True, stdout=io.StringIO())
        docs = list(Document.objects.order_by("uploaded_at"))
        self.assertEqual([(d.filename, folders.path(d.folder)) for d in docs],
                         [("Opta.pdf", "Datasheets/Arduino"), ("ACS880 Firmware.pdf", "Manuals/ABB")])   # smallest first
        abb = docs[1]
        self.assertEqual((abb.title, abb.source_url, abb.plant_asset, abb.fetched_at.isoformat()),
                         ("ACS880 Firmware manual", "https://example/ACS880 Firmware.pdf", "bsk:asset:EXTR01",
                          "2026-10-15T03:54:37+00:00"))
        self.assertEqual(abb.jobs.get().params, {"then_embed": True})

        again = io.StringIO()
        call_command("library_ingest", manifest=manifest, stdout=again)
        self.assertIn("added 0, already in the library 2", again.getvalue())
        self.drain()
        status = io.StringIO()
        call_command("library_ingest", manifest=manifest, status=True, stdout=status)
        self.assertIn("embedded: 2", status.getvalue())


EN = "The motor must be lifted only by its lifting lugs. Check the bearing temperature after the first hours of operation. " * 3
ES = "El motor debe levantarse solo por sus cáncamos de elevación. Compruebe la temperatura de los rodamientos tras las primeras horas. " * 3
IT = "Il motore deve essere sollevato solo dai golfari. Controllare la temperatura dei cuscinetti dopo le prime ore di funzionamento. " * 3
MULTI = f"\n\n{PAGE_BREAK}\n\n".join([f"# Safety\n\n{EN}", "| 1 | 2 |", f"# Seguridad\n\n{ES}", f"# Sicurezza\n\n{IT}", "| 3 | 4 |"])


@override_settings(LIBRARY_PARSE_PART_PAGES=25)
class LanguagesAndFiguresPassTests(LibraryTestCase):
    def test_page_languages_inherit_on_short_pages(self):
        from ingestion import languages
        self.assertEqual(languages.page_languages(MULTI), ["en", "en", "es", "it", "it"])
        self.assertEqual(languages.kept_pages(MULTI, ["en", "es"]), {1, 2, 3})
        self.assertIsNone(languages.kept_pages(MULTI, []))

    def test_embedding_keeps_only_the_chosen_languages(self):
        layout = {"pages": {str(p): {"size": {"width": 612, "height": 792}} for p in range(1, 6)}, "texts": [],
                  "pictures": [{"prov": [{"page_no": p, "bbox": {"l": 100, "t": 500, "r": 400, "b": 200}}]} for p in (1, 4)]}
        self.mock_convert.return_value = {"status": "success", "errors": [],
                                          "document": {"md_content": MULTI, "json_content": layout}}
        doc, _ = library.add_document(SimpleUploadedFile("motors.pdf", b"%PDF motors"))
        Document.objects.filter(pk=doc.pk).update(languages=["en", "es"])
        library.enqueue(doc, Job.Kind.EMBED)
        self.drain()
        _, metas = self.chunks(doc.doc_key)
        self.assertEqual({m["language"] for m in metas}, {"en", "es"})          # the Italian page is left out
        self.assertFalse({m["page_start"] for m in metas} & {4, 5})

        with patch("ingestion.figures.describe", side_effect=lambda d, f: f.update(status="done", description="a lug")):
            library.enqueue(doc, Job.Kind.FIGURES, {"then_embed": True})
            self.drain()
        figs = {f["page"]: f for f in json.loads((library.doc_dir(doc) / "figures.json").read_text())["figures"]}
        self.assertEqual(figs[1]["status"], "done")
        self.assertEqual((figs[4]["status"], figs[4]["skip_reason"]), ("skipped", "page in another language (kept: en, es)"))
        self.assertEqual(doc.jobs.filter(kind="embed", status="done").count(), 2)  # embedded again after its figures

    def test_the_figures_pass_follows_parse_and_embedding(self):
        def wordy(data, filename, **kw):   # enough text per page to make chunks
            out = docling_for_parts(data, filename)
            out["document"]["md_content"] = out["document"]["md_content"].replace(".", f". {EN}")
            return out
        self.mock_convert.side_effect = wordy
        with patch("ingestion.figures.describe", side_effect=lambda d, f: f.update(status="done", description="x")) as describe:
            doc, _ = library.add_document(SimpleUploadedFile("a.pdf", pdf(3)), then_embed=True)
            job = doc.jobs.get(kind="parse")
            job.params = {**job.params, "then_figures": True}
            job.save()
            self.drain()
        kinds = list(doc.jobs.order_by("created_at").values_list("kind", "status"))
        self.assertEqual(kinds, [("parse", "done"), ("embed", "done"), ("figures", "done"), ("embed", "done")])
        self.assertEqual(describe.call_count, 1)

    def test_a_figures_pass_waits_for_bsk(self):
        from ingestion import figures as figs
        self.mock_convert.side_effect = docling_for_parts
        doc, _ = library.add_document(SimpleUploadedFile("a.pdf", pdf(3)))
        self.drain()
        library.enqueue(doc, Job.Kind.FIGURES, {"then_embed": True})
        with patch("ingestion.figures.describe", side_effect=figs.BskUnavailable("BSK is off")):
            self.drain()
        job = doc.jobs.get(kind="figures")
        self.assertEqual(job.status, "queued")
        self.assertIn("waiting for BSK", job.message)

    def test_library_languages_command(self):
        self.mock_convert.return_value = {"status": "success", "errors": [], "document": {"md_content": MULTI}}
        doc, _ = library.add_document(SimpleUploadedFile("motors.pdf", b"%PDF motors"), then_embed=True)
        self.drain()
        out = io.StringIO()
        call_command("library_languages", doc.doc_key, keep="en,es", dry_run=True, stdout=out)
        self.assertIn("keep en, es: 3 of 5 pages", out.getvalue())
        doc.refresh_from_db()
        self.assertEqual(doc.languages, [])
        call_command("library_languages", doc.doc_key, keep="es,en", stdout=io.StringIO())
        doc.refresh_from_db()
        self.assertEqual(doc.languages, ["en", "es"])
        self.assertTrue(doc.jobs.filter(kind="embed", status="queued").exists())   # embedded again

    def test_embeddings_go_before_figures(self):
        self.mock_convert.side_effect = docling_for_parts
        a, _ = library.add_document(SimpleUploadedFile("a.pdf", pdf(2)))
        b, _ = library.add_document(SimpleUploadedFile("b.pdf", pdf(3)))
        self.drain()
        library.enqueue(a, Job.Kind.FIGURES, {"then_embed": True})
        library.enqueue(b, Job.Kind.EMBED)
        self.assertEqual((library.claim_next().kind, library.claim_next().kind), ("embed", "figures"))

    def test_upload_full_processing_and_describe_re_embeds(self):
        def wordy(data, filename, **kw):
            out = docling_for_parts(data, filename)
            out["document"]["md_content"] = out["document"]["md_content"].replace(".", f". {EN}")
            return out
        self.mock_convert.side_effect = wordy
        client = APIClient()
        self.assertEqual(client.post("/api/documents/", {"files": [SimpleUploadedFile("x.pdf", pdf(2))], "processing": "all"},
                                     format="multipart").status_code, 400)
        r = client.post("/api/documents/", {"files": [SimpleUploadedFile("manual.pdf", pdf(3))], "processing": "full"},
                        format="multipart")
        doc = Document.objects.get(pk=r.json()["results"][0]["document"]["id"])
        with patch("ingestion.figures.describe", side_effect=lambda d, f: f.update(status="done", description="x")):
            self.drain()
            self.assertEqual(list(doc.jobs.order_by("created_at").values_list("kind", flat=True)),
                             ["parse", "embed", "figures", "embed"])
            # "Describe figures" on an embedded document embeds it again by itself.
            r = client.post(f"/api/documents/{doc.id}/figures/describe/", {"force": True}, format="json")
            self.assertEqual(r.json()["params"]["then_embed"], True)
            self.drain()
        self.assertEqual(doc.jobs.filter(kind="embed", status="done").count(), 3)
        self.mock_convert.side_effect, self.mock_convert.return_value = None, PARSED
        notes = client.post("/api/documents/", {"files": [SimpleUploadedFile("notes.md", b"# Notes\n\n" + EN.encode())],
                                                "processing": "full"}, format="multipart")
        notes_doc = Document.objects.get(pk=notes.json()["results"][0]["document"]["id"])
        self.drain()
        self.assertEqual(list(notes_doc.jobs.order_by("created_at").values_list("kind", flat=True)), ["parse", "embed"])  # no figures for text

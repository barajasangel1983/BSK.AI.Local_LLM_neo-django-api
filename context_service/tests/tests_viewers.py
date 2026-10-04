"""Read-only views for clients (P11b): the document library, figures, a page's text, the asset graph."""

import asyncio
import io
import shutil
import tempfile
from unittest.mock import patch

from django.test import TestCase, TransactionTestCase, override_settings
from PIL import Image

from chat.test_asset_chat import ASSET, node
from context_service import mcp_server, service
from context_service.views import MCP_TOOLS
from ingestion import figures, library
from ingestion.chunker import PAGE_BREAK
from ingestion.models import Document

MD = f"# Drive\n\n<!-- image -->\n\nThe motor drives the screw.\n\n<!-- image -->\n\n{PAGE_BREAK}\n\n# Page two\n\nGearbox lubrication."
FIGURES = {"source": "pdf", "document_revision": "1", "figures": [
    {"index": 0, "page": 1, "bbox": None, "page_size": None, "caption": "Fig. 1", "area": 0.2, "status": "done",
     "skip_reason": "", "kind": "schematic", "description": "A VFD feeds motor M101."},
    {"index": 1, "page": 1, "bbox": None, "page_size": None, "caption": "", "area": 0.01, "status": "skipped",
     "skip_reason": "too small"},
]}


class ViewerTestCase(TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        override = override_settings(LIBRARY_BASE=self.tmp)
        override.enable()
        self.addCleanup(override.disable)
        self.doc = Document.objects.create(filename="Drive.pdf", doc_key="DRIVE", sha256="0" * 64, parse_status="parsed",
                                           page_count=2, rag_status="done", rag_chunk_count=7)
        library.doc_dir(self.doc).mkdir(parents=True)
        library.parsed_path(self.doc).write_text('{"document": {"md_content": %s}}' % __import__("json").dumps(MD))
        figures.save(self.doc, FIGURES)


class DocumentTests(ViewerTestCase):
    def test_list_and_detail_show_only_described_figures(self):
        listed = service.list_documents()["documents"]
        self.assertEqual([(d["filename"], d["pages"], d["searchable"], d["chunks"], d["figures_found"], d["figures_described"])
                          for d in listed], [("Drive.pdf", 2, True, 7, 2, 1)])
        detail = service.get_document(str(self.doc.id))
        self.assertEqual(detail["figures"], [{"index": 0, "page": 1, "kind": "schematic", "caption": "Fig. 1",
                                              "description": "A VFD feeds motor M101."}])
        for bad in ("nope", "00000000-0000-0000-0000-000000000000"):
            with self.assertRaisesMessage(service.ContextError, "unknown document"):
                service.get_document(bad)

    def test_page_text_has_figure_descriptions_and_no_placeholders(self):
        page = service.get_document_page(str(self.doc.id), 1)
        self.assertEqual((page["page"], page["pages"]), (1, 2))
        self.assertIn("[Figure p.1 (schematic): A VFD feeds motor M101.]", page["text"])
        self.assertNotIn("<!-- image -->", page["text"])
        self.assertEqual(service.get_document_page(str(self.doc.id), 2)["text"], "# Page two\n\nGearbox lubrication.")
        with self.assertRaisesMessage(service.ContextError, "page must be between 1 and 2"):
            service.get_document_page(str(self.doc.id), 3)

    def test_figure_image_only_for_described_figures(self):
        buf = io.BytesIO()
        Image.new("RGB", (40, 30), "red").save(buf, format="JPEG")
        path = figures.image_path(self.doc, 0)
        path.parent.mkdir(parents=True)
        path.write_bytes(buf.getvalue())
        self.assertEqual(service.figure_image(str(self.doc.id), 0), buf.getvalue())
        with self.assertRaisesMessage(service.ContextError, "unknown figure 1"):
            service.figure_image(str(self.doc.id), 1)          # skipped: not served


class GraphTests(TestCase):
    @patch("context_graph.services.node_detail", return_value={**node(ASSET, "Asset", "Extruder EXTR01"), "relationships": []})
    @patch("context_graph.services.graph_data")
    def test_curated_layer_only_and_internal_properties_hidden(self, graph_data, detail):
        graph_data.return_value = {
            "nodes": [{"id": ASSET, "label": "Asset", "name": "Extruder EXTR01",
                       "properties": {"asset_type": "Extruder", "evidence_ids": [1], "created_at": "x"}}],
            "links": [{"source": ASSET, "target": "bsk:line:1", "type": "PART_OF", "properties": {"source": "seed"}}]}
        g = service.get_graph(ASSET, depth=9)
        self.assertEqual(graph_data.call_args.kwargs["layer"], "curated")
        self.assertEqual(g["depth"], 3)
        self.assertEqual(g["nodes"][0]["properties"], {"asset_type": "Extruder"})
        self.assertEqual(g["links"], [{"source": ASSET, "target": "bsk:line:1", "type": "PART_OF"}])
        with self.assertRaisesMessage(service.ContextError, "not an asset id"):
            service.get_graph("bsk:component:EXTR01/die-head")


class ToolTests(TransactionTestCase):
    def test_viewer_tools_are_listed_and_the_image_tool_returns_an_image(self):
        server = mcp_server.build_server()
        self.assertEqual([t.name for t in asyncio.run(server.list_tools())], MCP_TOOLS)
        for name in ("list_documents", "get_document", "get_document_page", "get_figure_image", "get_graph"):
            self.assertIn(name, MCP_TOOLS)
        buf = io.BytesIO()
        Image.new("RGB", (10, 10), "blue").save(buf, format="JPEG")
        with patch.object(service, "figure_image", return_value=buf.getvalue()):
            result = asyncio.run(server.call_tool("get_figure_image", {"document_id": "d", "figure_index": 0}))
        content = result[0] if isinstance(result, tuple) else result
        self.assertEqual((content[0].type, content[0].mimeType), ("image", "image/jpeg"))

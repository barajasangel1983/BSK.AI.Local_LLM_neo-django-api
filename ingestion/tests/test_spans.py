"""P25b (2026-10-25): where a chunk, a window or an excerpt sits in a document's text snapshot."""

from django.test import SimpleTestCase

from ingestion import spans
from ingestion.chunker import PAGE_BREAK

PAGES = [
    "# 1. Safety\n\nRead the instructions before you start.\n\n<!-- image -->\n\nWear gloves when you touch the heat sink.",
    "## 2.1 Terminals\n\nTerminal 5 is digital input DI 0.\n\n| Terminal | Function |\n|---|---|\n| 5 | DI 0 |\n| 6 | DI 1 |\n\n"
    "- Wire the inputs with shielded cable.\n- Ground the shield at one end.",
    "## WARNING\n\nHazardous voltage. Wait 5 minutes.\n\n## CAUTION\n\nHot surface.\n\nRead the instructions before you start.",
]
TEXT = f"\n\n{PAGE_BREAK}\n\n".join(PAGES)


class LocatorTests(SimpleTestCase):
    def setUp(self):
        self.snap = spans.Snapshot(TEXT)

    def at(self, span):
        return TEXT[span.start:span.end]

    def test_whitespace_and_markers_do_not_matter(self):
        span = self.snap.locate("Read the instructions   before you start. Wear gloves when\nyou touch the heat sink.", 1, 1)
        self.assertEqual(span.status, spans.EXACT)
        self.assertTrue(self.at(span).startswith("Read the instructions") and self.at(span).endswith("heat sink."))

    def test_the_pages_narrow_the_search_and_repeats_are_ambiguous(self):
        self.assertEqual(self.snap.locate("Read the instructions before you start.", 3, 3).status, spans.EXACT)
        self.assertEqual(self.snap.locate("Read the instructions before you start.").status, spans.AMBIGUOUS)
        self.assertIsNone(self.snap.locate("Read the instructions before you start.").start)       # never guessed

    def test_a_chunk_across_a_heading_and_a_tables_cells(self):
        self.assertEqual(self.snap.locate("Hazardous voltage. Wait 5 minutes. Hot surface.", 3, 3).status, spans.EXACT)
        self.assertEqual(self.snap.locate("5 DI 0 6 DI 1", 2, 2).status, spans.EXACT)
        self.assertEqual(self.snap.locate("Wire the inputs with shielded cable. Ground the shield at one end.", 2, 2).status,
                         spans.EXACT)

    def test_a_window_is_located_with_its_table_its_stripped_text_only_partly(self):
        long_intro = "Terminal 5 is digital input DI 0. " + "The inputs are 24 V DC. " * 8
        text = TEXT.replace("Terminal 5 is digital input DI 0.", long_intro.strip())
        snap = spans.Snapshot(text)
        table = "| Terminal | Function |\n|---|---|\n| 5 | DI 0 |\n| 6 | DI 1 |"
        bullets = "- Wire the inputs with shielded cable.\n- Ground the shield at one end."
        whole = f"{long_intro.strip()}\n\n{table}\n\n{bullets}"          # the window as chunked (extraction keeps it)
        span = snap.locate(whole, 2, 2)
        self.assertEqual(span.status, spans.EXACT)
        self.assertIn("| 5 | DI 0 |", text[span.start:span.end])
        # the same window after its table was read apart (a stored excerpt): only its start is known to be there
        self.assertEqual(snap.locate(f"{long_intro.strip()}\n\n{bullets}", 2, 2).status, spans.PARTIAL)

    def test_nowhere_and_empty(self):
        self.assertEqual(self.snap.locate("Torque 0.8 Nm", 2, 2), spans.NONE)
        self.assertEqual(self.snap.locate("   "), spans.NONE)

    def test_the_hash_and_the_saved_file(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            snap = spans.Snapshot.save(Path(tmp), TEXT)
            self.assertEqual((Path(tmp) / f"{snap.hash}.md").read_text(), TEXT)
            spans.Snapshot.save(Path(tmp), TEXT)          # the same version again: one file
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)
        self.assertEqual(snap.page_of(TEXT.index("Hot surface")), 3)

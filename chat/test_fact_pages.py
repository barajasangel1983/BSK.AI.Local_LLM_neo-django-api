"""P17a: the graph facts' own manual pages join the reranker's candidates; long excerpts are cut to the lines
that match the question; table padding is squeezed. All off by default (settings)."""

from unittest.mock import patch

from django.test import SimpleTestCase, TestCase, override_settings

from chat import asset_context, retrieval
from chat.asset_context import Line, fact_pages
from chat.retrieval import V2Chunk, search_v2
from chat.excerpts import excerpt_window, squeeze_padding
from chat.views import build_rag_context
from context_graph.models import Evidence
from ingestion.models import Document

TABLE = "\n".join([
    "| Code | Name | Cause | What to do |",
    "|------|------|-------|------------|",
    *[f"| A5{n:02d} | Other {n} | Cause {n} text | Do thing {n} |" for n in range(70, 81)],
    "| A581 | Fan | Cooling fan feedback missing. | Check 95.20 bit 14 and the auxiliary code. |",
    *[f"| A5{n:02d} | Other {n} | Cause {n} text | Do thing {n} |" for n in range(82, 99)],
])


class ExcerptWindowTests(SimpleTestCase):
    def test_the_matching_row_and_its_neighbours_under_the_header(self):
        out = excerpt_window(TABLE, "What does warning A581 Fan mean?", 400)
        self.assertLessEqual(len(out), 400)
        lines = out.split("\n")
        self.assertEqual(lines[0], "| Code | Name | Cause | What to do |")
        self.assertIn("| A581 | Fan | Cooling fan feedback missing.", out)
        self.assertIn("…", lines)
        self.assertIn("A582", out)                     # the rows after it first

    def test_no_match_keeps_the_start_and_short_text_is_untouched(self):
        self.assertEqual(excerpt_window(TABLE, "lubricant grade", 300), TABLE[:300])
        self.assertEqual(excerpt_window("short", "A581", 300), "short")

    def test_squeeze(self):
        self.assertEqual(squeeze_padding("| A581   | Fan        |\n|---------|------|"), "| A581 | Fan |\n|---|---|")

    def entry(self, text, page):
        return {"text": text, "source": "fw.pdf", "page_start": page}

    @override_settings(CHAT_EXCERPT_WINDOW=False, CHAT_EXCERPT_SQUEEZE=False)
    def test_off_is_unchanged(self):
        block, used = build_rag_context([self.entry(TABLE, 601), self.entry("x" * 500, 602)], 500,
                                        header="H", footer="F", question="A581")
        self.assertEqual(len(used), 1)
        self.assertIn(TABLE[:300], block)               # cut from the start, as before
        self.assertNotIn("A581", block)

    @override_settings(CHAT_EXCERPT_WINDOW=True, CHAT_EXCERPT_SQUEEZE=True)
    def test_on_the_row_is_kept_and_the_next_excerpt_gets_the_rest(self):
        block, used = build_rag_context([self.entry(TABLE, 601), self.entry("Fan replacement " + "y" * 900, 602)],
                                        1000, header="H", footer="F", question="A581 fan")
        self.assertIn("Cooling fan feedback missing", block)
        self.assertEqual(len(used), 2)
        self.assertLessEqual(len(block), 1000)


class FactPagesTests(TestCase):
    def setUp(self):
        doc = Document.objects.create(title="FW", filename="fw.pdf", doc_key="FW-MANUAL", sha256="x" * 64)
        self.e601 = Evidence.objects.create(source_kind="document", document=doc, page_start=601)
        self.e590 = Evidence.objects.create(source_kind="document", document=doc, page_start=590)
        self.lines = [
            Line("PART_ALARMS", "- A581 Fan on Main drive", node_id="al:a581", related=("pr:a581",)),
            Line("PART_PROCEDURES", "- A581: what to do for Main drive", node_id="pr:a581"),
            Line("PART_ALARMS", "- 64B1 Internal SSW fault", node_id="al:64b1"),
            Line("PART_ALARMS", "- 2340 Short circuit", node_id="al:2340"),
        ]
        self.props = {"al:a581": {"evidence_ids": [self.e601.pk]}, "pr:a581": {"evidence_ids": [self.e601.pk]},
                      "al:64b1": {"evidence_ids": [self.e590.pk]}, "al:2340": {"evidence_ids": [999999]}}

    def test_focus_first_then_word_matches_each_page_once(self):
        self.assertEqual(fact_pages(self.lines, "Internal SSW fault and the A581 fan", {"al:a581"}, self.props, 4),
                         [("FW-MANUAL", 601), ("FW-MANUAL", 590)])

    def test_unmatched_facts_give_no_pages_and_zero_is_off(self):
        self.assertEqual(fact_pages(self.lines, "lubricant grade", set(), self.props, 4), [])
        self.assertEqual(fact_pages(self.lines, "A581", {"al:a581"}, self.props, 0), [])


class ExtraCandidatesTests(SimpleTestCase):
    def chunk(self, cid, score=0.5):
        return V2Chunk(id=cid, text=cid, source="fw.pdf", asset_id="FW-MANUAL", vector_score=score)

    def test_page_chunks_join_the_reranker_once(self):
        seen = {}

        def rerank(query, chunks, top_n):
            seen["ids"] = [c.id for c in chunks]
            return chunks[::-1][:top_n]

        with patch.object(retrieval, "_vector_search", return_value=[self.chunk("a"), self.chunk("b")]), \
                patch.object(retrieval, "_rerank", side_effect=rerank):
            result = search_v2("A581", top_n=5, doc_keys=["FW-MANUAL"], extra=[self.chunk("b", 0), self.chunk("p601", 0)])
        self.assertEqual(seen["ids"], ["a", "b", "p601"])
        self.assertEqual([c.id for c in result.chunks], ["p601", "b", "a"])


class BuildCarriesPagesTests(SimpleTestCase):
    def test_asset_context_has_pages_default(self):
        self.assertEqual(asset_context.AssetContext(asset_id="a", name="n", text="").pages, [])


class RerankFocusTests(SimpleTestCase):
    def chunk(self):
        return V2Chunk(id="p601", text=TABLE.replace(" | ", "     |     "), source="fw.pdf", asset_id="FW",
                       section_path=["FW", "Warnings"])

    @override_settings(RAG_RERANK_FOCUS=False)
    def test_off_scores_the_chunk_as_before(self):
        self.assertEqual(retrieval._rerank_text(self.chunk(), "A581"), "fw > FW > Warnings\n" + self.chunk().text)

    @override_settings(RAG_RERANK_FOCUS=True, RAG_RERANK_FOCUS_CHARS=300)
    def test_on_scores_the_matching_rows_squeezed(self):
        text = retrieval._rerank_text(self.chunk(), "What is warning A581?")
        self.assertTrue(text.startswith("fw > FW > Warnings\n"))
        self.assertIn("| A581 | Fan | Cooling fan feedback missing.", text)
        self.assertLess(len(text), 340)

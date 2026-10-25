"""P25d/P25e (2026-10-25): the line of an evidence span that holds a fact, by string matching; never guessed."""

from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from context_graph import quotes

TABLE = "| Frame | Torque |\n|---|---|\n| 280 | 0,8 Nm |\n| 315 | 1,2 Nm |\n"
PROSE = ("The drive starts after the precharge. " * 5 + "When operating at an ambient temperature of &gt; 60 °C, a power "
         "derating of 2.5 %/K should be observed. Up to 40 °C the static boost can be used for a sustained period.")


class FindTests(SimpleTestCase):
    def at(self, text, found):
        return text[found[1]:found[2]]

    def test_a_table_row_with_a_decimal_comma(self):
        found = quotes.find(TABLE, "Tightening torque (frame 315)", {"value": "1.2", "unit": "Nm"})
        self.assertEqual((found[0], self.at(TABLE, found)), ("exact", "| 315 | 1,2 Nm |"))

    def test_a_long_paragraph_is_cut_into_sentences(self):
        found = quotes.find(PROSE, "Power derating", {"value": "2.5 %/K", "condition": "ambient temperature > 60 °C"})
        self.assertEqual(found[0], "exact")
        self.assertTrue(self.at(PROSE, found).startswith("When operating") and self.at(PROSE, found).endswith("observed."))

    def test_a_name_and_its_value_on_two_lines(self):
        text = "Hold-up time\n20 ms\n\nSwitch-on time\n< 1 s\n"
        found = quotes.find(text, "Hold-up time", {"value": "20", "unit": "ms"})
        self.assertEqual((found[0], self.at(text, found)), ("exact", "Hold-up time\n20 ms"))

    def test_the_same_line_in_two_places_is_ambiguous(self):
        text = "Fuse 10 A slow\n\nThe fan runs.\n\nFuse 10 A slow\n"
        self.assertEqual(quotes.find(text, "Fuse", {"value": "10 A"}), ("ambiguous", None, None))

    def test_a_value_not_in_the_text_and_a_short_name_need_all_their_words(self):
        self.assertEqual(quotes.find(TABLE, "Torque (frame 400)", {"value": "2.5"})[0], "not_found")
        self.assertEqual(quotes.find("DI 3 is reserved for a safety function.", "DI 0", {})[0], "not_found")
        self.assertEqual(quotes.find("Terminal 5 is DI 0.", "DI 0", {})[0], "exact")

    def test_offsets_are_in_the_snapshot_inside_the_span(self):
        snapshot = "x" * 100 + TABLE
        e = SimpleNamespace(char_start=100, char_end=len(snapshot))
        fields = quotes.fields(e, "Torque (frame 280)", {"value": "0.8"}, snapshot)
        self.assertEqual(snapshot[fields["quote_start"]:fields["quote_end"]], "| 280 | 0,8 Nm |")
        self.assertEqual(quotes.fields(SimpleNamespace(char_start=None, char_end=None), "Torque", {}, snapshot), {})


class PassageTests(SimpleTestCase):
    def evidence(self, **kw):
        base = dict(document_id="d", text_hash="h", char_start=None, char_end=None, quote_start=None, quote_end=None,
                    span_status="", quote_status="", page_start=3, page_end=3, row_start=None, row_end=None,
                    excerpt="the stored excerpt")
        return SimpleNamespace(**{**base, **kw})

    def test_the_span_with_the_quote_and_the_excerpt_without_a_span(self):
        text = "x" * 50 + TABLE
        with patch.object(quotes, "snapshot_text", return_value=text):
            p = quotes.passage(self.evidence(char_start=50, char_end=len(text), quote_start=50 + TABLE.index("| 315"),
                                             quote_end=len(text) - 1, quote_status="exact", span_status="exact"))
            self.assertEqual((p["kind"], p["text"][p["quote"]["start"]:p["quote"]["end"]]), ("span", "| 315 | 1,2 Nm |"))
            self.assertEqual(quotes.passage(self.evidence())["text"], "the stored excerpt")

    def test_a_long_span_is_cut_around_the_quote(self):
        text = "a" * 30000 + "QUOTE" + "b" * 30000
        with patch.object(quotes, "snapshot_text", return_value=text):
            p = quotes.passage(self.evidence(char_start=0, char_end=len(text), quote_start=30000, quote_end=30005))
        self.assertTrue(p["truncated"])
        self.assertEqual(len(p["text"]), quotes.PASSAGE_CHARS)
        self.assertEqual(p["text"][p["quote"]["start"]:p["quote"]["end"]], "QUOTE")

"""P26b-2 (2026-10-23): one value per property, one id per channel in a run, and the dry run on the real path. The
G120C graph had `terminal: 5; DI 0; 5; DI 0; F-DI 0` (values joined from several pages) and DI 0 as both X138-5 and
DI-0 (one channel, two ids); 69 IOPoints that should have been 25."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings

from context_graph import extraction, registry
from context_graph.extraction import Channels, Votes, differing, first_terminal, merge_properties, strip_as_channel
from context_graph.models import CandidateTriple
from ingestion import library
from ingestion.chunker import PAGE_BREAK
from ingestion.tests.tests_library import LibraryTestCase

SCHEMAS = Path(extraction.__file__).parent / "schemas"
V7 = yaml.safe_load((SCHEMAS / "industrial_v7.yaml").read_text())


def staged(sname="Converter", stype="Component", pred="HAS_IO", oname="DI 0", otype="IOPoint", oid="", props=None):
    return SimpleNamespace(subject_name=sname, subject_type=stype, subject_id="bsk:component:A/x", subject_existing=True,
                           predicate=pred, object_name=oname, object_type=otype, object_id=oid, object_existing=False,
                           subject_props={}, object_props=dict(props or {}), issue="")


class OneValueTests(SimpleTestCase):
    def test_first_terminal(self):
        self.assertEqual([first_terminal(t) for t in ("5", "18, 19, 20", "3 (AI 0+), 4 (AI 0-)", "13/14", "DI 0", "L1",
                                                        "", None, "1.5")],
                         ["5", "18", "3", "13", "", "", "", "", ""])

    def test_a_text_keeps_its_value_lists_and_free_text_are_joined(self):
        self.assertEqual(merge_properties({"terminal": "5", "aliases": ["a"], "summary": "Step 1."},
                                          {"terminal": "F-DI 0", "aliases": ["b"], "summary": "Step 2.", "unit": "V"}),
                         {"terminal": "5", "aliases": ["a", "b"], "summary": "Step 1.; Step 2.", "unit": "V"})

    @override_settings(GRAPH_SINGLE_VALUES=False)
    def test_the_old_behaviour_joins(self):
        self.assertEqual(merge_properties({"terminal": "5"}, {"terminal": "DI 0"}), {"terminal": "5; DI 0"})
        self.assertEqual(differing({"terminal": "5"}, {"terminal": "DI 0"}), [])

    def test_the_most_frequent_value_wins_and_the_others_are_the_issue(self):
        t, votes = staged(props={"terminal": "5"}), Votes()
        for value in ("5", "F-DI 0", "5 ", "5"):                 # four finds; "5 " is the same value
            votes.add(t, "object", {"terminal": value, "signal_type": "24 V DC"})
        self.assertEqual(votes.settle([t]), 1)
        self.assertEqual(t.object_props["terminal"], "5")
        self.assertEqual(t.issue, "pages disagree on terminal: 5 (3×) | F-DI 0")
        tie, votes = staged(props={"subtype": "Digital input"}), Votes()
        for value in ("Digital input", "Safety input"):
            votes.add(tie, "object", {"subtype": value})
        votes.settle([tie])
        self.assertEqual(tie.object_props["subtype"], "Digital input")          # a tie: the first found
        kept = staged(props={"unit": "V"})
        kept.issue = "a rating with no value"
        votes = Votes()
        for value in ("V", "mV"):
            votes.add(kept, "object", {"unit": value})
        votes.settle([kept])
        self.assertEqual(kept.issue, "a rating with no value")                 # an issue already there is kept

    def test_a_fold_of_two_pending_triples_notes_the_other_value(self):
        self.assertEqual(differing({"default": "1", "basis": "p.197"}, {"default": "r2090.2", "basis": "p.12"}),
                         ["pages disagree on default: 1 | r2090.2"])


class ChannelTests(SimpleTestCase):
    def test_one_channel_one_id_in_a_run(self):
        ch = Channels()
        by_name = staged(pred="CONFIGURED_BY", oname="DI 0", oid="bsk:io-point:A/DI-0")
        with_terminal = staged(oname="DI 0", oid="bsk:io-point:A/X138-5", props={"connector": "-X138", "terminal": "5"})
        other_name = staged(oname="Digital input 0", oid="bsk:io-point:A/Digital-input-0",
                            props={"connector": "-X138", "terminal": "5"})
        for t in (by_name, with_terminal, other_name):
            ch.unify(t)
        self.assertEqual({by_name.object_id, with_terminal.object_id, other_name.object_id}, {"bsk:io-point:A/DI-0"})

    def test_the_same_number_on_another_strip_is_another_channel(self):
        ch = Channels()
        relay1 = staged(oname="Relay output 1", oid="bsk:io-point:A/XRO1-1", props={"connector": "XRO1", "terminal": "1, 2, 3"})
        relay2 = staged(oname="Relay output 2", oid="bsk:io-point:A/XRO2-1", props={"connector": "XRO2", "terminal": "1, 2, 3"})
        for t in (relay1, relay2):
            ch.unify(t)
        self.assertNotEqual(relay1.object_id, relay2.object_id)

    def test_a_strip_is_not_a_channel(self):
        for name in ("Terminal strip -X136", "terminal strips", "Connector X1", "-X138", "X2", "Terminal block 3"):
            self.assertTrue(strip_as_channel(staged(oname=name)), name)
        for name in ("DI 0", "Relay output 1", "Signaling connection terminal blocks", "X138-5", "+24V OUT"):
            self.assertFalse(strip_as_channel(staged(oname=name)), name)
        self.assertFalse(strip_as_channel(staged(oname="Connector X1", otype="Interface")))   # IOPoints only

    def test_an_existing_node_keeps_its_id(self):
        ch = Channels()
        first = staged(oname="DI 0", oid="bsk:io-point:A/DI-0")
        known = staged(oname="DI 0", oid="bsk:io-point:A/X138-5")
        known.object_existing = True
        for t in (first, known):
            ch.unify(t)
        self.assertEqual(known.object_id, "bsk:io-point:A/X138-5")


PAGES = ["# 4. Terminals\n\nTerminal 5 is digital input DI 0, 24 V DC. " * 6,
         "# 4.2 Terminal functions\n\nDI 0 switches the motor on. Terminal 5 again. " * 6,
         "# 9. Safety\n\nDI 0 can be used as a failsafe input F-DI 0. " * 6]
MANUAL = {"status": "success", "errors": [], "document": {"md_content": f"\n\n{PAGE_BREAK}\n\n".join(PAGES)}}


@override_settings(GRAPH_ENABLED=False)
class DryRunTests(LibraryTestCase):
    """The same terminal on three pages: one IOPoint, one terminal value, the other value in the issue; nothing staged."""

    def setUp(self):
        super().setUp()
        self.mock_convert.return_value = MANUAL
        terminals = iter(["5", "5", "F-DI 0"])

        def llm(messages):
            terminal = next(terminals, "5")
            return json.dumps({"triples": [
                {"subject": "Converter", "subject_type": "Component", "predicate": "HAS_IO", "object": "DI 0",
                 "object_type": "IOPoint", "object_subtype": "Digital input",
                 "object_properties": {"connector": "-X138", "terminal": terminal, "signal_type": "24 V DC"},
                 "confidence": 0.9}]})
        p = patch("context_graph.extraction.call_llm", side_effect=llm)
        p.start()
        self.addCleanup(p.stop)
        registry.save_version(V7, note="v7")
        self.doc, _ = library.add_document(SimpleUploadedFile("conv.pdf", b"%PDF conv"))
        self.drain()
        self.doc.refresh_from_db()

    def dry(self, **settings):
        params = extraction.validate_params({"pages": "1-3"})
        params.update(dry_run=True, chunking={"strategy": "structure", "params": {"max_tokens": 200, "overlap_sentences": 0}})
        with override_settings(**settings):
            return extraction._extract(None, self.doc, params)

    def test_one_value_and_nothing_staged(self):
        out = self.dry()
        self.assertEqual(CandidateTriple.objects.count(), 0)
        [t] = [t for t in out["triples"] if t.object_type == "IOPoint"]
        self.assertEqual(t.object_props["terminal"], "5")
        self.assertRegex(t.issue, r"^pages disagree on terminal: 5 \(\d+×\) \| F-DI 0$")
        self.assertEqual(out["disagreed"], 1)

    def test_the_old_code_joins_them(self):
        out = self.dry(GRAPH_SINGLE_VALUES=False)
        [t] = [t for t in out["triples"] if t.object_type == "IOPoint"]
        self.assertEqual(t.object_props["terminal"], "5; F-DI 0")
        self.assertEqual(CandidateTriple.objects.count(), 0)

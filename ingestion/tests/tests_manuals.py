"""Manuals named in a BOM (P14d): the download list, the downloads and their manifest. No network."""

import io
import json
import shutil
import tempfile
from pathlib import Path

import requests
from django.core.management import call_command
from django.test import SimpleTestCase

from ingestion import manuals

BOM = """Item,Manufacturer,Manual URL,Manual Title
vfd,ABB,https://search.abb.com/a.pdf | https://search.abb.com/b.pdf,Hardware manual | Firmware manual
rtd-1,Endress+Hauser,https://bdih-download.endress.com/tm411.pdf,iTHERM TM411/TM412 Operating Instructions
rtd-2,Endress+Hauser,https://bdih-download.endress.com/tm411.pdf,iTHERM TM411/TM412 Operating Instructions
psu,Phoenix Contact,https://www.phoenixcontact.com/psu.pdf,QUINT4 Data sheet
blog,Someone,https://example.com/manual.pdf,Copied manual
nir,MoistTech,,
"""


class FakeResponse:
    def __init__(self, status=200, body=b"", content_type="application/pdf"):
        self.status_code, self.body, self.headers = status, body, {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, size):
        for i in range(0, len(self.body), size):
            yield self.body[i:i + size]


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def get(self, url, **kwargs):
        self.calls.append(url)
        r = self.responses[url]
        if isinstance(r, Exception):
            raise r
        return r


class ManualsTests(SimpleTestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.bom = self.tmp / "bom.csv"
        self.bom.write_text(BOM)
        self.out = self.tmp / "out"
        self.out.mkdir()

    def test_download_list(self):
        items = manuals.read_bom(self.bom)
        self.assertEqual([(m.url.rsplit("/", 1)[1], m.title, m.items) for m in items], [
            ("a.pdf", "Hardware manual", ["vfd"]), ("b.pdf", "Firmware manual", ["vfd"]),
            ("tm411.pdf", "iTHERM TM411/TM412 Operating Instructions", ["rtd-1", "rtd-2"]),   # one manual, two items
            ("psu.pdf", "QUINT4 Data sheet", ["psu"]), ("manual.pdf", "Copied manual", ["blog"])])
        self.assertEqual(items[2].file, "iTHERM TM411 TM412 Operating Instructions.pdf")
        self.assertEqual([manuals.allowed(m.url) for m in items], [True, True, True, True, False])
        self.assertFalse(manuals.allowed("http://search.abb.com/a.pdf"))          # https only
        self.assertFalse(manuals.allowed("https://abb.com.example.org/a.pdf"))    # the host, not a prefix
        bad = self.tmp / "bad.csv"
        bad.write_text("Part,Link\nx,y\n")
        with self.assertRaises(manuals.ManualsError):
            manuals.read_bom(bad)

    def test_fetch_records_pdfs_and_lists_the_rest(self):
        items = {m.url.rsplit("/", 1)[1]: m for m in manuals.read_bom(self.bom)}
        pdf = b"%PDF-1.7\n" + b"x" * 200_000
        session = FakeSession({
            "https://search.abb.com/a.pdf": FakeResponse(body=pdf),
            "https://search.abb.com/b.pdf": FakeResponse(body=b"<html>login</html>", content_type="text/html; charset=utf-8"),
            "https://bdih-download.endress.com/tm411.pdf": requests.ConnectionError("reset"),
            "https://www.phoenixcontact.com/psu.pdf": FakeResponse(status=403),
        })
        for m in items.values():
            manuals.fetch(m, self.out, None, 10 * 1024 * 1024, session)
        a = items["a.pdf"]
        self.assertEqual((a.status, a.size, (self.out / a.file).read_bytes()), ("downloaded", len(pdf), pdf))
        self.assertEqual(len(a.sha256), 64)
        self.assertEqual((items["b.pdf"].status, items["b.pdf"].reason), ("by hand", "not a PDF (text/html)"))
        self.assertTrue(items["tm411.pdf"].reason.startswith("ConnectionError"))
        self.assertEqual(items["psu.pdf"].reason, "HTTP 403")
        self.assertEqual(items["manual.pdf"].reason, "not a manufacturer's https server")
        self.assertNotIn("https://example.com/manual.pdf", session.calls)          # never requested
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), [a.file])    # no partial files left

        moved = manuals.read_bom(self.bom)[0]
        away = FakeResponse(body=pdf)
        away.url = "https://mirror.example.net/a.pdf"
        manuals.fetch(moved, self.tmp, None, 10 * 1024 * 1024, FakeSession({moved.url: away}))
        self.assertEqual((moved.status, moved.reason), ("by hand", "redirected to mirror.example.net"))

        too_big = manuals.read_bom(self.bom)[0]
        manuals.fetch(too_big, self.tmp, None, 100_000, FakeSession({too_big.url: FakeResponse(body=pdf)}))
        self.assertEqual((too_big.status, too_big.reason), ("by hand", "larger than the 100,000 byte limit"))

    def test_command_keeps_what_it_has(self):
        pdf = b"%PDF-1.4\nmanual"
        session = FakeSession({url: FakeResponse(body=pdf + url.encode()) for url in (
            "https://search.abb.com/a.pdf", "https://search.abb.com/b.pdf", "https://bdih-download.endress.com/tm411.pdf",
            "https://www.phoenixcontact.com/psu.pdf")})
        original = manuals.requests.Session
        manuals.requests.Session = lambda: session
        self.addCleanup(setattr, manuals.requests, "Session", original)

        dry = io.StringIO()
        call_command("fetch_manuals", bom=str(self.bom), out=str(self.out), dry_run=True, stdout=dry)
        self.assertIn("5 distinct manuals", dry.getvalue())
        self.assertEqual((session.calls, list(self.out.iterdir())), ([], []))   # a dry run fetches nothing

        call_command("fetch_manuals", bom=str(self.bom), out=str(self.out), stdout=io.StringIO())
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual([m["status"] for m in manifest["manuals"]], ["downloaded"] * 4 + ["by hand"])
        self.assertEqual(len(session.calls), 4)

        again = io.StringIO()
        call_command("fetch_manuals", bom=str(self.bom), out=str(self.out), stdout=again)
        self.assertEqual(len(session.calls), 4)                                  # nothing downloaded twice
        self.assertIn("downloaded 0, kept 4, by hand 1", again.getvalue())

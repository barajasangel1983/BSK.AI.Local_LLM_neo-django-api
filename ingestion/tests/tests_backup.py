"""Backups and their restore test (P25a). Docker, Neo4j and Chroma are mocked; the live run and its restore
test against the real containers are in the session log (2026-10-21)."""

import json
import shutil
import sqlite3
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase, override_settings

from ingestion import backup
from ingestion.models import Document, Job

NEO4J = {"nodes": 4868, "relationships": 8544}
CHROMA = {"bsk_rag_v2": 14801}


class BackupTests(TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.db = self.tmp / "studio.sqlite3"
        with sqlite3.connect(self.db) as db:
            db.execute("CREATE TABLE triples (id INTEGER)")
            db.executemany("INSERT INTO triples VALUES (?)", [(n,) for n in range(5)])
        db.close()
        self.data = self.tmp / "data"
        (self.data / "library" / "doc").mkdir(parents=True)
        (self.data / "library" / "doc" / "manual.pdf").write_bytes(b"%PDF")
        (self.data / "gpu.lock").write_text("")
        self.edge = self.tmp / "edge"
        (self.edge / "branding").mkdir(parents=True)
        (self.edge / "branding" / "logo.png").write_bytes(b"png")
        with sqlite3.connect(self.edge / "edge.sqlite3") as db:
            db.execute("CREATE TABLE users (id INTEGER)")
        db.close()
        self.chroma = self.tmp / "chroma_index"
        self.chroma.mkdir()
        (self.chroma / "chroma.sqlite3").write_bytes(b"x")
        self.root = self.tmp / "backups"
        self.calls = []

        def docker(*args, timeout=900):
            self.calls.append(args)
            if args[:1] == ("run",) and "dump" in args:
                folder = Path(next(a for a in args if a.endswith(":/backups")).rsplit(":", 1)[0])
                (folder / f"{args[args.index('dump') + 1]}.dump").write_bytes(b"dump")
            return ""
        for target, kwargs in (("ingestion.backup.docker", {"side_effect": docker}),
                               ("ingestion.backup.studio_db", {"return_value": self.db}),
                               ("ingestion.backup.studio_data", {"return_value": self.data}),
                               ("ingestion.backup.neo4j_counts", {"return_value": NEO4J}),
                               ("ingestion.backup.chroma_counts", {"return_value": CHROMA})):
            p = patch(target, **kwargs)
            p.start()
            self.addCleanup(p.stop)
        self.settings_override = override_settings(CHROMA_DIR=str(self.chroma), BACKUP_EDGE_DATA=str(self.edge))
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)

    def test_a_run_writes_every_part_with_its_manifest(self):
        folder = backup.run(self.root, log=lambda m: None)
        self.assertTrue(backup.RUN_RE.match(folder.name))
        self.assertEqual(stat.S_IMODE(folder.stat().st_mode), 0o700)
        names = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
        self.assertEqual(names, {"db.sqlite3", "edge.sqlite3", "data.tar", "edge-data.tar", "chroma.tar",
                                 "neo4j/neo4j.dump", "neo4j/system.dump", "manifest.json"})
        self.assertTrue(all(stat.S_IMODE((folder / n).stat().st_mode) == 0o600 for n in names))
        manifest = json.loads((folder / "manifest.json").read_text())
        self.assertEqual(manifest["counts"]["db.sqlite3"], {"triples": 5})
        self.assertEqual((manifest["counts"]["data.tar"], manifest["counts"]["edge-data.tar"]), (1, 1))  # no gpu.lock, no edge db
        self.assertEqual((manifest["counts"]["neo4j"], manifest["counts"]["chroma"]), (NEO4J, CHROMA))
        self.assertEqual(set(manifest["files"]), names - {"manifest.json"})
        # Each container is stopped, then started again; the graph is dumped in between.
        order = [a[:2] for a in self.calls if a[0] in ("stop", "start")]
        self.assertEqual(order, [("stop", "-t"), ("start", "bsklab-chroma"), ("stop", "-t"), ("start", "bsklab-neo4j")])
        self.assertEqual(sum("dump" in a for a in self.calls), 2)

    def test_the_graph_is_started_again_when_the_dump_fails(self):
        def failing(*args, timeout=900):
            self.calls.append(args)
            if "dump" in args:
                raise backup.BackupError("docker run failed: disk full")
            return ""
        with patch("ingestion.backup.docker", side_effect=failing), self.assertRaises(backup.BackupError):
            backup.run(self.root, log=lambda m: None)
        self.assertIn(("start", "bsklab-neo4j"), [a[:2] for a in self.calls])

    def test_no_backup_while_a_library_job_runs(self):
        doc = Document.objects.create(title="manual", filename="manual.pdf")
        Job.objects.create(document=doc, kind=Job.Kind.EXTRACT, status=Job.Status.RUNNING)
        with self.assertRaisesRegex(backup.BackupError, "library job"):
            backup.run(self.root, log=lambda m: None)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.root.exists())

    def test_old_runs_are_pruned_and_hand_copies_kept(self):
        self.root.mkdir()
        for name in ("2026-10-18T0300Z", "2026-10-19T0300Z", "2026-10-20T0300Z", "2026-10-21-before-batch"):
            (self.root / name).mkdir()
        backup.prune(self.root, keep=1, log=lambda m: None)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["2026-10-20T0300Z", "2026-10-21-before-batch"])

    def test_the_restore_test_compares_every_count(self):
        folder = backup.run(self.root, log=lambda m: None)
        with patch("ingestion.backup.restored_neo4j_counts", return_value=NEO4J), \
                patch("ingestion.backup.restored_chroma_counts", return_value=CHROMA):
            self.assertEqual(backup.restore_test(folder, log=lambda m: None), [])
        with patch("ingestion.backup.restored_neo4j_counts", return_value={**NEO4J, "nodes": 4000}), \
                patch("ingestion.backup.restored_chroma_counts", return_value=CHROMA):
            problems = backup.restore_test(folder, log=lambda m: None)
        self.assertEqual(problems, ["Neo4j: expected {'nodes': 4868, 'relationships': 8544}, "
                                    "found {'nodes': 4000, 'relationships': 8544}"])
        (folder / "data.tar").write_bytes(b"changed")
        with patch("ingestion.backup.restored_neo4j_counts", return_value=NEO4J), \
                patch("ingestion.backup.restored_chroma_counts", return_value=CHROMA):
            problems = backup.restore_test(folder, log=lambda m: None)
        self.assertIn("data.tar: SHA-256 differs from the manifest", problems)

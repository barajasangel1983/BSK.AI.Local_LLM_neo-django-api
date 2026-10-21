"""Backups of the Context Studio's data and their restore test (P25a, 2026-10-21).

One run writes ~/backups/<UTC time>/ (folder 700, files 600):
  db.sqlite3, edge.sqlite3   SQLite's online backup of the Studio's and EDGE's databases
  data.tar                   the backend's data/ (library, graph data files, chat files, manuals)
  edge-data.tar              EDGE's data/ without its database (chat pictures, branding)
  chroma.tar                 the Chroma folder, copied with bsklab-chroma stopped (seconds)
  neo4j/<db>.dump            neo4j-admin database dump, bsklab-neo4j stopped (about a minute)
  manifest.json              counts taken just before (SQLite rows, Neo4j nodes and links, Chroma items),
                             and each file's size and SHA-256

It refuses to start while a library job is queued or running (a job reading the graph while Neo4j is stopped
matches nothing; an embed job writes to Chroma), and always starts the containers again. The .env files are
not backed up (secrets are restored by hand). The restore test loads a backup into throwaway containers
(127.0.0.1, other ports, own volume) and compares every count with the manifest.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from django.conf import settings

NEO4J_CONTAINER = "bsklab-neo4j"
NEO4J_VOLUME = "bsklab_neo4j_data"
NEO4J_IMAGE = "neo4j:5.26-community"
NEO4J_DATABASES = ("neo4j", "system")
CHROMA_CONTAINER = "bsklab-chroma"
CHROMA_IMAGE = "chromadb/chroma:1.5.5"
TEST_NEO4J = "bsklab-neo4j-restoretest"
TEST_CHROMA = "bsklab-chroma-restoretest"
TEST_CHROMA_PORT = 8199
RUN_FORMAT = "%Y-%m-%dT%H%MZ"
RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{4}Z$")     # only folders made by this module are pruned
DATA_EXCLUDE = {"gpu.lock"}


class BackupError(Exception):
    pass


def docker(*args: str, timeout: int = 900) -> str:
    r = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise BackupError(f"docker {args[0]} failed: {(r.stderr or r.stdout).strip()[-500:]}")
    return r.stdout


def backups_root() -> Path:
    return Path(getattr(settings, "BACKUP_DIR", "") or Path.home() / "backups")


def edge_data() -> Path:
    return Path(getattr(settings, "BACKUP_EDGE_DATA", "") or Path.home() / "repos" / "BSK.AI_bsklab-edge" / "data")


def studio_db() -> Path:
    return Path(settings.DATABASES["default"]["NAME"])


def studio_data() -> Path:
    return Path(settings.BASE_DIR) / "data"


def active_jobs() -> int:
    from ingestion.models import Job
    return Job.objects.filter(status__in=[Job.Status.QUEUED, Job.Status.RUNNING]).count()


# --- the parts ---------------------------------------------------------------------------------------

def sqlite_copy(src: Path, dst: Path) -> None:
    """SQLite's online backup: a consistent copy while the API and the timers keep writing."""
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30) as source, sqlite3.connect(dst) as target:
        source.backup(target)
    source.close()
    target.close()


def sqlite_counts(path: Path) -> dict[str, int]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        counts = {t: db.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in sorted(tables)}
    db.close()
    return counts


def sqlite_check(path: Path) -> str:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        result = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.close()
    return result


def tar_folder(src: Path, dst: Path, exclude: set[str] = frozenset()) -> int:
    """An uncompressed tar of a folder (manuals and pictures do not compress); returns the number of files."""
    files = 0

    def keep(info: tarfile.TarInfo):
        nonlocal files
        if Path(info.name).name in exclude:
            return None
        files += info.isfile()
        return info
    with tarfile.open(dst, "w") as tar:
        tar.add(src, arcname=src.name, filter=keep)
    return files


def tar_files(path: Path) -> int:
    with tarfile.open(path) as tar:
        return sum(m.isfile() for m in tar.getmembers())


def neo4j_counts() -> dict[str, int]:
    from context_graph.driver import session
    with session() as s:
        return {"nodes": s.run("MATCH (n) RETURN count(n) AS n").single()["n"],
                "relationships": s.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"]}


def chroma_counts(client=None) -> dict[str, int]:
    if client is None:
        from ingestion.chroma_client import get_client
        client = get_client()
    return {c.name: c.count() for c in client.list_collections()}


def wait_for(check, what: str, seconds: int = 180):
    deadline = time.monotonic() + seconds
    while True:
        try:
            return check()
        except Exception as exc:        # noqa: BLE001 — any failure means "not up yet"
            if time.monotonic() > deadline:
                raise BackupError(f"{what} did not come back within {seconds} s: {exc}") from exc
            time.sleep(3)


def neo4j_dump(folder: Path) -> None:
    """Stop the graph, dump each database, start it again (Community edition dumps only a stopped database)."""
    folder.mkdir(mode=0o700)
    docker("stop", "-t", "60", NEO4J_CONTAINER)
    try:
        for db in NEO4J_DATABASES:
            docker("run", "--rm", "--user", "0:0", "--entrypoint", "neo4j-admin", "-v", f"{NEO4J_VOLUME}:/data",
                   "-v", f"{folder}:/backups", NEO4J_IMAGE, "database", "dump", db, "--to-path=/backups")
    finally:
        docker("start", NEO4J_CONTAINER)
        docker("run", "--rm", "--entrypoint", "chown", "-v", f"{folder}:/backups", NEO4J_IMAGE,
               "-R", f"{os.getuid()}:{os.getgid()}", "/backups")
        wait_for(neo4j_counts, "Neo4j")      # the driver reconnects by itself


def chroma_copy(src: Path, dst: Path) -> int:
    docker("stop", "-t", "30", CHROMA_CONTAINER)
    try:
        return tar_folder(src, dst)
    finally:
        docker("start", CHROMA_CONTAINER)
        wait_for(chroma_counts, "Chroma")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- a run -------------------------------------------------------------------------------------------

def run(root: Path | None = None, keep: int = 7, log=print) -> Path:
    if active_jobs():
        raise BackupError("a library job is queued or running: no backup now (Neo4j and Chroma are stopped during it)")
    root = root or backups_root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc)
    folder = root / started.strftime(RUN_FORMAT)
    folder.mkdir(mode=0o700)
    manifest: dict = {"started": started.isoformat(timespec="seconds"), "host": os.uname().nodename,
                      "sources": {}, "counts": {}, "files": {}}
    db, data = studio_db(), studio_data()

    log("SQLite (Studio)")
    sqlite_copy(db, folder / "db.sqlite3")
    manifest["sources"]["db.sqlite3"] = str(db)
    manifest["counts"]["db.sqlite3"] = sqlite_counts(folder / "db.sqlite3")
    edge = edge_data()
    if (edge / "edge.sqlite3").exists():
        log("SQLite (EDGE)")
        sqlite_copy(edge / "edge.sqlite3", folder / "edge.sqlite3")
        manifest["sources"]["edge.sqlite3"] = str(edge / "edge.sqlite3")
        manifest["counts"]["edge.sqlite3"] = sqlite_counts(folder / "edge.sqlite3")
        log("EDGE data/")
        manifest["counts"]["edge-data.tar"] = tar_folder(edge, folder / "edge-data.tar", exclude={"edge.sqlite3"})
        manifest["sources"]["edge-data.tar"] = str(edge)
    log("backend data/")
    manifest["counts"]["data.tar"] = tar_folder(data, folder / "data.tar", exclude=DATA_EXCLUDE)
    manifest["sources"]["data.tar"] = str(data)

    log("Chroma (stopped for the copy)")
    manifest["counts"]["chroma"] = chroma_counts()
    chroma_dir = Path(settings.CHROMA_DIR)
    manifest["counts"]["chroma.tar"] = chroma_copy(chroma_dir, folder / "chroma.tar")
    manifest["sources"]["chroma.tar"] = str(chroma_dir)

    log("Neo4j (stopped for the dump)")
    manifest["counts"]["neo4j"] = neo4j_counts()
    neo4j_dump(folder / "neo4j")
    manifest["sources"]["neo4j"] = f"volume {NEO4J_VOLUME}"

    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        path.chmod(0o600)
        manifest["files"][str(path.relative_to(folder))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (folder / "manifest.json").chmod(0o600)
    prune(root, keep, log)
    return folder


def prune(root: Path, keep: int, log=print) -> list[Path]:
    """Keep the newest `keep` runs; folders not named by this module (hand copies) are never touched."""
    runs = sorted((p for p in root.iterdir() if p.is_dir() and RUN_RE.match(p.name)), key=lambda p: p.name)
    removed = runs[:-keep] if keep > 0 else []
    for p in removed:
        shutil.rmtree(p)
        log(f"removed old backup {p.name}")
    return removed


# --- the restore test --------------------------------------------------------------------------------

def restore_test(folder: Path, log=print) -> list[str]:
    """Every count of the manifest, found again in the restored copy; returns the differences (empty: passed)."""
    manifest = json.loads((folder / "manifest.json").read_text())
    problems: list[str] = []

    def compare(what: str, expected, found):
        log(f"{what}: {'ok' if expected == found else 'DIFFERENT'}")
        if expected != found:
            problems.append(f"{what}: expected {expected}, found {found}")

    for name, info in manifest["files"].items():
        if sha256(folder / name) != info["sha256"]:
            problems.append(f"{name}: SHA-256 differs from the manifest")
    for name in ("db.sqlite3", "edge.sqlite3"):
        if name in manifest["counts"]:
            try:
                check = sqlite_check(folder / name)
                if check != "ok":
                    problems.append(f"{name}: integrity_check says {check}")
                compare(f"{name} rows", manifest["counts"][name], sqlite_counts(folder / name))
            except sqlite3.DatabaseError as exc:
                problems.append(f"{name}: cannot be read ({exc})")
    for name in ("data.tar", "edge-data.tar"):
        if name in manifest["counts"]:
            try:
                compare(f"{name} files", manifest["counts"][name], tar_files(folder / name))
            except tarfile.TarError as exc:
                problems.append(f"{name}: cannot be read ({exc})")
    compare("Neo4j", manifest["counts"]["neo4j"], restored_neo4j_counts(folder / "neo4j"))
    compare("Chroma", manifest["counts"]["chroma"], restored_chroma_counts(folder / "chroma.tar"))
    return problems


def restored_neo4j_counts(dumps: Path) -> dict[str, int]:
    volume = "bsklab_neo4j_restoretest"
    _remove(TEST_NEO4J)
    docker("volume", "create", volume)
    try:
        docker("run", "--rm", "--user", "0:0", "--entrypoint", "neo4j-admin", "-v", f"{volume}:/data",
               "-v", f"{dumps}:/backups:ro", NEO4J_IMAGE, "database", "load", "neo4j", "--from-path=/backups",
               "--overwrite-destination")
        # No published port: the counts are read inside the container (auth off on this throwaway copy only).
        docker("run", "-d", "--name", TEST_NEO4J, "-e", "NEO4J_AUTH=none",
               "-e", "NEO4J_server_memory_heap_max__size=512m", "-e", "NEO4J_server_memory_pagecache_size=256m",
               "-v", f"{volume}:/data", NEO4J_IMAGE)

        def count(query: str) -> int:
            out = docker("exec", TEST_NEO4J, "cypher-shell", "--format", "plain", query, timeout=60)
            return int(out.strip().splitlines()[-1])
        wait_for(lambda: count("RETURN 1"), "the restored Neo4j")
        return {"nodes": count("MATCH (n) RETURN count(n)"), "relationships": count("MATCH ()-[r]->() RETURN count(r)")}
    finally:
        _remove(TEST_NEO4J)
        docker("volume", "rm", volume, timeout=60)


def restored_chroma_counts(archive: Path) -> dict[str, int]:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    work = Path(tempfile.mkdtemp(prefix="chroma-restoretest-", dir=archive.parent))
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(work, filter="data")
        data = next(p for p in work.iterdir() if p.is_dir())
        _remove(TEST_CHROMA)
        docker("run", "-d", "--name", TEST_CHROMA, "--user", f"{os.getuid()}:{os.getgid()}",
               "-p", f"127.0.0.1:{TEST_CHROMA_PORT}:8000", "-v", f"{data}:/data", CHROMA_IMAGE)
        client = wait_for(lambda: _heartbeat(chromadb.HttpClient(
            host="127.0.0.1", port=TEST_CHROMA_PORT, settings=ChromaSettings(anonymized_telemetry=False))),
            "the restored Chroma", 120)
        return chroma_counts(client)
    finally:
        _remove(TEST_CHROMA)
        shutil.rmtree(work, ignore_errors=True)


def _heartbeat(client):
    client.heartbeat()
    return client


def _remove(container: str) -> None:
    """Remove a throwaway container if it exists (left over from an interrupted test)."""
    if subprocess.run(["docker", "inspect", container], capture_output=True, text=True).returncode == 0:
        docker("rm", "-f", container, timeout=60)

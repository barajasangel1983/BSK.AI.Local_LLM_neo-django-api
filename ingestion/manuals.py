"""Manuals named in a bill of materials (P14d): the download list and the downloads.

The BOM's `Manual URL` column holds one URL or several separated by " | ", with the
titles in `Manual Title` in the same order. Only PDFs from the manufacturers' own
servers (MANUFACTURER_HOSTS) are fetched; anything else (an HTML page, a login, a
refusal) is listed for fetching by hand, never retried. Each download is recorded in
`manifest.json` beside the files: source URL, date, size and SHA-256. Nothing is
added to the library here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

SEPARATOR = " | "
MANUFACTURER_HOSTS = (
    "abb.com", "arduino.cc", "endress.com", "gefran.com", "grundfos.com", "hardysolutions.com",
    "optris.com", "phoenixcontact.com", "sew-eurodrive.com", "siemens.com", "spiraxsarco.com", "watlow.com",
)
USER_AGENT = "BSKLAB-manual-fetch/1.0 (+industrial documentation for an internal library)"
TIMEOUT = (15, 120)
CHUNK = 1 << 16


class ManualsError(ValueError):
    """An unreadable BOM."""


@dataclass
class Manual:
    url: str
    title: str
    items: list[str] = field(default_factory=list)
    manufacturer: str = ""
    file: str = ""
    status: str = "pending"      # pending | downloaded | kept | by hand
    reason: str = ""
    size: int = 0
    sha256: str = ""
    downloaded_at: str = ""


def allowed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return urlparse(url).scheme == "https" and any(host == h or host.endswith("." + h) for h in MANUFACTURER_HOSTS)


def filename_for(title: str, url: str, taken: set[str]) -> str:
    """A readable file name from the title (the library derives the document key from it)."""
    base = re.sub(r"[^A-Za-z0-9.+()-]+", " ", title or Path(urlparse(url).path).stem).strip()
    base = re.sub(r"\s+", " ", base)[:120].rstrip(" .") or "manual"
    name, n = f"{base}.pdf", 2
    while name.lower() in taken:
        name, n = f"{base} ({n}).pdf", n + 1
    taken.add(name.lower())
    return name


def read_bom(path: Path) -> list[Manual]:
    """One Manual per distinct URL, with every BOM item that uses it."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ManualsError(f"{path}: {exc}") from exc
    if rows and not {"Item", "Manual URL"} <= set(rows[0]):
        raise ManualsError(f"{path}: needs the columns Item and Manual URL")
    manuals: dict[str, Manual] = {}
    for row in rows:
        urls = [u.strip() for u in (row.get("Manual URL") or "").split(SEPARATOR) if u.strip()]
        titles = [t.strip() for t in (row.get("Manual Title") or "").split(SEPARATOR)]
        for i, url in enumerate(urls):
            m = manuals.setdefault(url, Manual(url, titles[i] if i < len(titles) else "", manufacturer=row.get("Manufacturer", "")))
            if row["Item"] not in m.items:
                m.items.append(row["Item"])
    taken: set[str] = set()
    for m in manuals.values():
        m.file = filename_for(m.title, m.url, taken)
    return list(manuals.values())


def load_manifest(out: Path) -> dict[str, dict]:
    try:
        return {e["url"]: e for e in json.loads((out / "manifest.json").read_text())["manuals"]}
    except (OSError, ValueError, KeyError):
        return {}


def save_manifest(out: Path, bom: Path, manuals: list[Manual]) -> None:
    data = {"bom": str(bom), "written_at": _now(), "manuals": [asdict(m) for m in manuals]}
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(out / "manifest.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(m: Manual, out: Path, known: dict | None, max_bytes: int, session: requests.Session | None = None) -> Manual:
    """Download one manual (or keep the copy already recorded). Sets status and reason."""
    if not allowed(m.url):
        m.status, m.reason = "by hand", "not a manufacturer's https server"
        return m
    target = out / m.file
    if known and known.get("status") in ("downloaded", "kept") and known.get("file") == m.file and target.exists() \
            and _sha256(target) == known.get("sha256"):
        m.status, m.size, m.sha256, m.downloaded_at = "kept", known["size"], known["sha256"], known["downloaded_at"]
        return m
    http = session or requests.Session()
    part = out / (m.file + ".part")
    try:
        with http.get(m.url, stream=True, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as r:
            if r.status_code != 200:
                m.status, m.reason = "by hand", f"HTTP {r.status_code}"
                return m
            final = getattr(r, "url", m.url) or m.url
            if not allowed(final):   # a redirect may leave the manufacturer's servers
                m.status, m.reason = "by hand", f"redirected to {urlparse(final).hostname}"
                return m
            digest, size, head = hashlib.sha256(), 0, b""
            with part.open("wb") as f:
                for chunk in r.iter_content(CHUNK):
                    if not head:
                        head = chunk[:5]
                        if not head.startswith(b"%PDF"):
                            m.status = "by hand"
                            m.reason = f"not a PDF ({r.headers.get('Content-Type', 'no content type').split(';')[0]})"
                            break
                    size += len(chunk)
                    if size > max_bytes:
                        m.status, m.reason = "by hand", f"larger than the {max_bytes:,} byte limit"
                        break
                    digest.update(chunk)
                    f.write(chunk)
        if m.status == "by hand":
            part.unlink(missing_ok=True)
            return m
        if not size:
            part.unlink(missing_ok=True)
            m.status, m.reason = "by hand", "empty response"
            return m
        part.replace(target)
        m.status, m.size, m.sha256, m.downloaded_at, m.reason = "downloaded", size, digest.hexdigest(), _now(), ""
    except requests.RequestException as exc:
        part.unlink(missing_ok=True)
        m.status, m.reason = "by hand", f"{type(exc).__name__}: {str(exc)[:200]}"
    return m

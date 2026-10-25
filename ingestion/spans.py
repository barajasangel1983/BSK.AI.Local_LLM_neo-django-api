"""Source spans (P25b, 2026-10-25): where a chunk, an extraction window or a quote sits in a document's text.

The canonical text of a document is what both pipelines chunk: `library.load_parsed(doc)` Markdown with the figure
descriptions in place. Its SHA-256 is the text version; each version is kept once as an immutable snapshot
(`LIBRARY_BASE/<doc id>/text/<hash>.md`), so old evidence always resolves against its own text.

A span is [char_start, char_end) in a snapshot, found *after* chunking: the chunkers are not changed, so chunk lists,
Chroma ids and extraction windows stay as they are. The search is whitespace-normalized (chunkers join sentences and
paragraphs with their own spacing) and narrowed to the chunk's pages.

    exact      the text is that place (markers, headings, table marks and bullets aside: the chunkers drop them)
    partial    the start and end are found but the text between differs, or only the start (an excerpt) is found
    ambiguous  several places, even within the pages: marked, never guessed (no offsets)
    not_found  nowhere on those pages
"""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path

from .chunker import PAGE_BREAK

EXACT, PARTIAL, AMBIGUOUS, NOT_FOUND = "exact", "partial", "ambiguous", "not_found"
ANCHOR = 120            # normalized characters of a text's start and end searched when the whole is not found
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Span:
    start: int | None
    end: int | None
    status: str

    def fields(self, text_hash: str) -> dict:
        """The Evidence / Chroma fields of this span."""
        return {"text_hash": text_hash, "char_start": self.start, "char_end": self.end, "span_status": self.status}


NONE = Span(None, None, NOT_FOUND)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return _SPACE.sub(" ", text or "").strip()


_MARKUP = re.compile(r"<!--\s*(?:page-break|image)\s*-->")
_HEADING = re.compile(r"(?m)^[ \t]*#{1,6}[ \t].*$")
RANK = {EXACT: 0, PARTIAL: 1, AMBIGUOUS: 2, NOT_FOUND: 3}


def _blank(pattern: re.Pattern, text: str) -> str:
    return pattern.sub(lambda m: " " * len(m.group()), text)


def _markup(text: str) -> str:
    """The parser's page and image markers blanked (same length): the chunkers drop or move them."""
    return _blank(_MARKUP, text)


_TABLE_MARKS = re.compile(r"\||^[ \t]*:?-{3,}:?(?=[ \t|]|$)|(?<=\s):?-{3,}:?(?=\s)", re.M)
_BULLET = re.compile(r"(?m)^[ \t]*(?:[-*+\u2022]|\d+[.)])[ \t]+")


def _markup_and_headings(text: str) -> str:
    """Also the Markdown heading lines: the structure chunker keeps a heading as the section path, not as text."""
    return _blank(_HEADING, _blank(_MARKUP, text))


def _plain(text: str) -> str:
    """Also the table cell separators and rules (a chunk can hold a table's cells as text) and list bullets."""
    return _blank(_BULLET, _blank(_TABLE_MARKS, _markup_and_headings(text)))


def _without_tables(text: str) -> str:
    """A text with its Markdown table lines removed (extraction reads tables apart) and the parser's page and image
    markers (the chunkers drop or move them)."""
    return _MARKUP.sub(" ", "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("|")))


class _View:
    """The text with some markup blanked, whitespace-normalized, each character mapped back to its real offset."""

    def __init__(self, text: str, blank):
        self.blank = blank
        norm, offsets = [], []
        last_space = True
        for i, ch in enumerate(blank(text)):
            if ch.isspace():
                if not last_space:
                    norm.append(" ")
                    offsets.append(i)
                last_space = True
            else:
                norm.append(ch)
                offsets.append(i)
                last_space = False
        self.norm = "".join(norm)
        self.offsets = offsets

    def find_all(self, needle: str, lo: int, hi: int, limit: int = 2) -> list[int]:
        found, at = [], self.norm.find(needle, lo, hi)
        while at != -1 and len(found) < limit:
            found.append(at)
            at = self.norm.find(needle, at + 1, hi)
        return found

    def real(self, n_start: int, n_end: int) -> tuple[int, int]:
        return self.offsets[n_start], self.offsets[n_end - 1] + 1


class Snapshot:
    """One version of a document's text, searched in up to three views, each tried only when the ones before give
    no exact place: the parser's markers blanked; also the headings; also the table marks and list bullets."""

    def __init__(self, text: str):
        self.text = text
        self.hash = text_hash(text)
        self.views: list[_View | None] = [_View(text, _markup), None, None]      # the others built when first needed
        starts, pos = [0], 0
        for page in text.split(PAGE_BREAK)[:-1]:
            pos += len(page) + len(PAGE_BREAK)
            starts.append(pos)
        self.page_starts = starts       # real offset of each page (1-based page n starts at page_starts[n - 1])

    # --- files ---------------------------------------------------------------------------------------------------

    @classmethod
    def save(cls, folder: Path, text: str) -> "Snapshot":
        """The snapshot of `text`, written under `folder` the first time this version is seen (never overwritten)."""
        snap = cls(text)
        path = folder / f"{snap.hash}.md"
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
        return snap

    # --- search --------------------------------------------------------------------------------------------------

    def _region(self, view: _View, page_start: int | None, page_end: int | None) -> tuple[int, int]:
        """Normalized bounds of the pages (one page more on each side: a page marker may sit inside a paragraph)."""
        if not page_start:
            return 0, len(view.norm)
        first = max(1, page_start - 1)
        last = min(len(self.page_starts), (page_end or page_start) + 1)
        real_start = self.page_starts[first - 1]
        real_end = self.page_starts[last] if last < len(self.page_starts) else len(self.text)
        return bisect_left(view.offsets, real_start), bisect_right(view.offsets, real_end)

    def locate(self, text: str, page_start: int | None = None, page_end: int | None = None) -> Span:
        """The span of `text` on its pages (see the module docstring for the statuses)."""
        best = NONE
        for n, blank in enumerate((_markup, _markup_and_headings, _plain)):
            if self.views[n] is None:
                self.views[n] = _View(self.text, blank)
            found = self._locate(self.views[n], text, page_start, page_end)
            if RANK[found.status] < RANK[best.status]:
                best = found
            if best.status == EXACT:
                break
        return best

    def _locate(self, view: _View, text: str, page_start, page_end) -> Span:
        needle = normalize(view.blank(text or ""))
        if not needle:
            return NONE
        lo, hi = self._region(view, page_start, page_end)
        hits = view.find_all(needle, lo, hi)
        if len(hits) == 1:
            return Span(*view.real(hits[0], hits[0] + len(needle)), EXACT)
        if hits:
            return Span(None, None, AMBIGUOUS)
        if len(needle) <= 2 * ANCHOR:
            return NONE
        heads = view.find_all(needle[:ANCHOR], lo, hi)
        if len(heads) > 1:
            return Span(None, None, AMBIGUOUS)
        if not heads:
            return NONE
        head = heads[0]
        # the end, within a window somewhat longer than the text (a table read apart was in it)
        tail_at = view.norm.find(needle[-ANCHOR:], head + len(needle) - 2 * ANCHOR - 1,
                                 min(hi, head + 3 * len(needle) + 4000))
        if tail_at == -1:
            return Span(*view.real(head, min(head + len(needle), len(view.norm))), PARTIAL)
        start, end = view.real(head, tail_at + ANCHOR)
        same = normalize(_without_tables(view.blank(self.text[start:end]))) == normalize(_without_tables(view.blank(text)))
        return Span(start, end, EXACT if same else PARTIAL)

    def page_of(self, offset: int) -> int:
        return bisect_right(self.page_starts, offset)

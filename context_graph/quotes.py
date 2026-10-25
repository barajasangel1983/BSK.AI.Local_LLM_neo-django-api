"""Supporting quotes (P25d, 2026-10-25): the line of an evidence span that holds a fact, by string matching only (D3).

The span's text is cut into lines (a table row is one line; a long paragraph is cut into sentences). A quote is one
line or two neighbouring text lines (never a table row and its header) that:

- contain the fact's values: a rating's value / min / max, a parameter's code (a number may be written with a decimal
  comma), each part of a value given as "a; b";
- without values, hold the words of the object's name (at least half of them when it has more than two).

Lines are scored by the name's words, then by the words of the name's parenthesis, the condition and the default. The
best line wins; a tie between different places is `ambiguous`, nothing found is `not_found`. Neither has offsets: a
quote is never guessed. The offsets are real offsets in the document's text snapshot, inside the evidence span.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from pathlib import Path

from django.conf import settings

from ingestion import spans

EXACT, AMBIGUOUS, NOT_FOUND = "exact", "ambiguous", "not_found"
VALUE_KEYS = ("value", "min", "max", "code", "nominal")
LONG_VALUE = 60         # a value this long is a description, not a value: the name alone is matched
LONG_LINE = 220         # a longer line that isn't a table row is cut into sentences
QUOTE_CHARS = 300       # the quote's text in the API
PASSAGE_CHARS = 12000   # the passage sent to the screens
_STOP = frozenset("a an the of and or for in on at to with by from is are be as per".split())
_SENTENCE = re.compile(r".+?(?:[.!?](?=\s+[A-Z(\"])|$)", re.S)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_PAREN = re.compile(r"\(([^)]*)\)")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip().lower()


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", _norm(text)) if w not in _STOP]


def _lines(text: str) -> list[tuple[int, int]]:
    """(start, end) of each line or sentence, without its surrounding whitespace; table rules left out."""
    out, pos = [], 0
    for line in text.splitlines(keepends=True):
        s = line.strip()
        if s and not set(s) <= set("|-: "):
            if len(s) > LONG_LINE and not s.startswith("|"):
                pieces = [(pos + m.start(), pos + m.end()) for m in _SENTENCE.finditer(line)]
            else:
                pieces = [(pos, pos + len(line))]
            for a, b in pieces:
                piece = text[a:b]
                if piece.strip():
                    a += len(piece) - len(piece.lstrip())
                    b -= len(piece) - len(piece.rstrip())
                    out.append((a, b))
        pos += len(line)
    return out


def _needles(name: str, props: dict) -> tuple[list[str], list[str], list[str]]:
    """The name's words, the values each quote must hold, and the words that only add to a line's score."""
    props = props or {}
    values = [str(props[k]) for k in VALUE_KEYS if props.get(k) not in (None, "")]
    values = [_norm(part) for v in values if len(v) < LONG_VALUE for part in v.split(";") if part.strip()]
    extra = _words(" ".join(_PAREN.findall(name)) + " " + str(props.get("condition") or "") + " "
                   + str(props.get("default") or ""))
    return _words(_PAREN.sub(" ", name)), values, extra


def _has_value(line: str, value: str) -> bool:
    line = _norm(line)
    if value in line:
        return True
    numbers = _NUMBER.findall(value)
    return bool(numbers) and all(
        re.search(rf"(?<![\d.,]){re.escape(n)}(?!\d)", line) or
        re.search(rf"(?<![\d.,]){re.escape(n.replace('.', ','))}(?!\d)", line) for n in numbers)


def find(text: str, name: str, props: dict | None) -> tuple[str, int | None, int | None]:
    """(status, start, end) of the quote of a fact (its object's name and properties) in `text`."""
    words, values, extra = _needles(name, props)
    lines = _lines(text)
    if not lines or not (words or values):
        return NOT_FOUND, None, None
    needed = len(words) if len(words) <= 2 else (len(words) + 1) // 2     # a short name: all its words
    candidates = [(a, b, False) for a, b in lines] + [         # two neighbouring text lines (a table row stands alone)
        (a[0], b[1], True) for a, b in zip(lines, lines[1:]) if not (text[a[0]] == "|" or text[b[0]] == "|")]
    scored = []
    for a, b, pair in candidates:
        line = text[a:b]
        if values and not all(_has_value(line, v) for v in values):
            continue
        present = set(_words(line))
        hits = sum(w in present for w in words)
        if not values and hits < needed:
            continue
        scored.append((hits * 10 + sum(w in present for w in extra) - (5 if pair else 0), a, b))
    if not scored:
        return NOT_FOUND, None, None
    best = max(s for s, _, _ in scored)
    top = [(a, b) for s, a, b in scored if s == best]
    if len({a for a, _ in top}) > 1:
        return AMBIGUOUS, None, None
    return EXACT, top[0][0], top[0][1]


def fields(evidence, name: str, props: dict | None, snapshot_text: str) -> dict:
    """The quote fields of one evidence row: searched within its span; none without a span."""
    if evidence.char_start is None or evidence.char_end is None or not snapshot_text:
        return {}
    status, start, end = find(snapshot_text[evidence.char_start:evidence.char_end], name, props)
    if start is not None:
        start, end = evidence.char_start + start, evidence.char_start + end
    return {"quote_status": status, "quote_start": start, "quote_end": end}


def apply(evidence, name: str, props: dict | None, snapshot_text: str) -> None:
    for k, v in fields(evidence, name, props, snapshot_text).items():
        setattr(evidence, k, v)


def snapshot_text(doc_or_id, text_hash: str) -> str:
    """The text of one snapshot of a document ("" when it isn't there)."""
    if not text_hash:
        return ""
    return _snapshot(str(getattr(doc_or_id, "pk", doc_or_id)), text_hash, settings.LIBRARY_BASE)


@lru_cache(maxsize=16)
def _snapshot(doc_id: str, text_hash: str, base: str) -> str:
    """Snapshots never change (a new text is a new hash), so they can be kept in memory."""
    path = Path(base) / doc_id / "text" / f"{text_hash}.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def quote_text(evidence) -> str:
    """The quote's text (cut at QUOTE_CHARS), from the snapshot."""
    if evidence.quote_start is None or evidence.quote_end is None or not evidence.document_id:
        return ""
    text = snapshot_text(evidence.document_id, evidence.text_hash)
    return spans.normalize(text[evidence.quote_start:evidence.quote_end])[:QUOTE_CHARS]


def passage(evidence) -> dict:
    """The evidence's text for the screens (P25e): its span in the snapshot with the quote's place in it, or the stored
    excerpt when there is no span."""
    text = snapshot_text(evidence.document_id, evidence.text_hash) if evidence.document_id else ""
    out = {"span_status": evidence.span_status, "quote_status": evidence.quote_status,
           "page_start": evidence.page_start, "page_end": evidence.page_end,
           "row_start": evidence.row_start, "row_end": evidence.row_end, "quote": None, "truncated": False}
    if not text or evidence.char_start is None or evidence.char_end is None:
        return {**out, "kind": "excerpt", "text": evidence.excerpt}
    start, end = evidence.char_start, evidence.char_end
    if end - start > PASSAGE_CHARS:      # a long span: the part around the quote
        centre = evidence.quote_start if evidence.quote_start is not None else start
        start = max(evidence.char_start, min(centre - PASSAGE_CHARS // 2, end - PASSAGE_CHARS))
        end = start + PASSAGE_CHARS
        out["truncated"] = True
    if evidence.quote_start is not None and start <= evidence.quote_start and evidence.quote_end <= end:
        out["quote"] = {"start": evidence.quote_start - start, "end": evidence.quote_end - start}
    return {**out, "kind": "span", "text": text[start:end]}

"""Excerpt text for the reranker and the prompt (P17a, 2026-10-17): table padding squeezed, and a long excerpt
cut to its lines that best match the question instead of its start."""

from __future__ import annotations

import re

from .asset_context import _tokens


def squeeze_padding(text: str) -> str:
    """Parsed tables pad their columns with spaces and their separator rows with dashes (about 46 % of the
    library's excerpt text, P17a); one space and three dashes say the same."""
    return re.sub(r"-{4,}", "---", re.sub(r"[ \t]{2,}", " ", text))


def _row_weight(line: str, words: set[str]) -> int:
    """How well a line of an excerpt matches the question's words: a word with a digit (a code, a parameter)
    counts 3, others 1."""
    return sum(3 if any(ch.isdigit() for ch in w) else 1 for w in words & _tokens(line))


def excerpt_window(text: str, question: str, room: int) -> str:
    """The lines of a long excerpt that best match the question, within `room` characters (P17a), instead of
    its start: a fault table's matching row with its neighbours, under the table's header row. No match: the start."""
    if len(text) <= room:
        return text
    lines = text.split("\n")
    words = _tokens(question)
    weights = [_row_weight(line, words) for line in lines]
    best = max(range(len(lines)), key=lambda i: (weights[i], -i))
    if not weights[best]:
        return text[:room]
    header: list[str] = []
    if lines[0].lstrip().startswith("|") and lines[best].lstrip().startswith("|") and best > 1:
        header = lines[:2] if set(lines[1].strip()) <= set("|-: ") else lines[:1]
    first = len(header)
    size = sum(len(h) + 1 for h in header) + len(lines[best]) + 4      # + the "…" lines
    if size > room:
        return lines[best][:room]
    lo = hi = best
    while True:
        grown = False
        for j in (hi + 1, lo - 1):                                       # the rows after first: the row's details
            if first <= j < len(lines) and not lo <= j <= hi and size + len(lines[j]) + 1 <= room:
                size += len(lines[j]) + 1
                lo, hi = min(lo, j), max(hi, j)
                grown = True
        if not grown:
            break
    body = lines[lo:hi + 1]
    return "\n".join([*header, *(["…"] if lo > first else []), *body, *(["…"] if hi < len(lines) - 1 else [])])

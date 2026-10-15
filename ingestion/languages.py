"""Languages of a document's pages, for multilingual manuals (P14d-2).

A manual printed in many languages (ABB's low-voltage motors manual has some 20) would fill
the vector database with text nobody searches in. A document can keep some languages
(`Document.languages`, e.g. ["en", "es"]): only chunks from pages in those languages are
embedded, and figures on the other pages are not described. The parsed text stays whole.

The language is decided per page (whole pages are in one language in such manuals, and a page
has enough words for a reliable guess). A page with little text (a table, a drawing, a cover)
takes the language of the page before it; before any page with text, it counts as neutral
and is kept.
"""

from __future__ import annotations

import re

from .chunker import PAGE_BREAK

MIN_LETTERS = 200          # fewer letters than this: too little to tell, inherit
NEUTRAL = ""


def _plain(md: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", md, flags=re.S)          # image placeholders
    text = re.sub(r"[#*|`>_\-\[\]()]+", " ", text)
    return re.sub(r"\s+", " ", text)


def page_languages(md: str) -> list[str]:
    """One ISO 639-1 code per page ("" = neutral), in page order."""
    import py3langid

    out, last = [], NEUTRAL
    for page in md.split(PAGE_BREAK):
        text = _plain(page)
        if sum(c.isalpha() for c in text) >= MIN_LETTERS:
            last = py3langid.classify(text)[0]
        out.append(last)
    return out


def kept_pages(md: str, languages: list[str]) -> set[int] | None:
    """1-based pages to keep, or None when the document keeps every language."""
    if not languages:
        return None
    keep = set(languages)
    return {i + 1 for i, lang in enumerate(page_languages(md)) if lang == NEUTRAL or lang in keep}


def summary(md: str) -> dict[str, int]:
    """Pages per language (for the command and the report)."""
    counts: dict[str, int] = {}
    for lang in page_languages(md):
        counts[lang or "neutral"] = counts.get(lang or "neutral", 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

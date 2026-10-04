"""Chunking strategies for "Generate embeddings" (document library).

    structure  paragraphs within heading sections (ingestion.chunker.chunk_document)
    sentence   whole sentences grouped up to max_tokens, with sentence overlap
    fixed      token windows of `size` with `overlap` tokens

All strategies keep section paths and page ranges (from Docling page-break
markers), and never split tables or lists into sentences.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any, Dict, List

import pysbd

from .chunker import (
    ENC,
    Chunk,
    _content_type,
    _count_tokens,
    _extract_sections_from_markdown,
    _hard_split_by_tokens,
    _paragraphs,
    _strip_embedded_images,
    chunk_document,
)

# Defaults and bounds per strategy (also served to the UI via /api/rag/config/).
STRATEGIES: Dict[str, Dict[str, Any]] = {
    "structure": {
        "label": "Structure-aware",
        "description": "Paragraphs within sections; best for manuals and long explanations.",
        "params": {
            "max_tokens": {"default": 800, "min": 200, "max": 1200},
            "overlap_sentences": {"default": 1, "min": 0, "max": 3},
        },
    },
    "sentence": {
        "label": "Sentence",
        "description": "Whole sentences up to a token budget; sharper retrieval of specs, limits and steps.",
        "params": {
            "max_tokens": {"default": 300, "min": 64, "max": 800},
            "overlap_sentences": {"default": 1, "min": 0, "max": 3},
        },
    },
    "fixed": {
        "label": "Fixed window",
        "description": "Token windows with overlap; for unstructured text.",
        "params": {
            "size": {"default": 512, "min": 64, "max": 1500},
            "overlap": {"default": 64, "min": 0, "max": 512},
        },
    },
}
DEFAULT_STRATEGY = "structure"
MIN_CHUNK_TOKENS = 20


class ChunkingError(ValueError):
    pass


def resolve_params(strategy: str, params: Dict[str, Any] | None) -> Dict[str, int]:
    """Validate a strategy and fill in defaults; raises ChunkingError."""
    spec = STRATEGIES.get(strategy)
    if spec is None:
        raise ChunkingError(f"unknown chunking strategy {strategy!r}")
    params = params or {}
    unknown = set(params) - set(spec["params"])
    if unknown:
        raise ChunkingError(f"unknown parameter(s) for {strategy}: {', '.join(sorted(unknown))}")
    resolved = {}
    for name, bounds in spec["params"].items():
        value = params.get(name, bounds["default"])
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ChunkingError(f"{name} must be an integer")
        if not bounds["min"] <= value <= bounds["max"]:
            raise ChunkingError(f"{name} must be between {bounds['min']} and {bounds['max']}")
        resolved[name] = value
    if strategy == "fixed" and resolved["overlap"] >= resolved["size"]:
        raise ChunkingError("overlap must be smaller than size")
    return resolved


def variant_key(strategy: str, params: Dict[str, int]) -> str:
    """Short, stable identifier of strategy + params (part of chunk ids / ingest key)."""
    digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    return f"{strategy}-{digest}"


def chunk_parsed(doc: Dict[str, Any], strategy: str = DEFAULT_STRATEGY, params: Dict[str, Any] | None = None) -> List[Chunk]:
    resolved = resolve_params(strategy, params)
    if strategy == "structure":
        return chunk_document(doc, max_tokens=resolved["max_tokens"], overlap_sentences=resolved["overlap_sentences"])
    sections = _sections(doc)
    if strategy == "sentence":
        chunks = _sentence_chunks(sections, resolved["max_tokens"], resolved["overlap_sentences"])
    else:
        chunks = _fixed_chunks(sections, resolved["size"], resolved["overlap"])
    # Drop tiny paragraph fragments, but never short tables/lists (e.g. a 3-step procedure).
    return [c for c in chunks if c.token_count >= MIN_CHUNK_TOKENS or c.content_type in ("table", "list", "figure")]


# --- helpers -------------------------------------------------------------------

def _sections(doc: Dict[str, Any]):
    document = doc.get("document", doc)
    if not isinstance(document, dict):
        return []
    md = document.get("md_content", "") or document.get("text_content", "")
    if not md:
        return []
    return _extract_sections_from_markdown(_strip_embedded_images(md))


@lru_cache(maxsize=1)
def _segmenter():
    return pysbd.Segmenter(language="en", clean=False)


def _chunk(text: str, path: List[str], pages: List[int], content_type: str) -> Chunk:
    return Chunk(
        text=text.strip(),
        section_path=path or ["Document"],
        page_start=min(pages),
        page_end=max(pages),
        content_type=content_type,
        token_count=_count_tokens(text),
    )


def _sentence_chunks(sections, max_tokens: int, overlap: int) -> List[Chunk]:
    """Group whole sentences per section up to `max_tokens`, carrying `overlap` sentences over.

    A chunk is emitted only when it holds at least one sentence that isn't
    overlap. Tables/lists (and any single over-long sentence) are emitted on
    their own, hard-split by tokens only if they exceed the budget; overlap
    never crosses them or a section boundary.
    """
    chunks: List[Chunk] = []

    for section in sections:
        path = section["path"]
        buf: List[tuple[str, int, int]] = []   # (sentence, page_start, page_end)
        carried = 0                             # leading sentences in buf that are overlap

        def emit():
            chunks.append(_chunk(" ".join(b[0] for b in buf), path,
                                 [b[1] for b in buf] + [b[2] for b in buf], "paragraph"))

        def emit_whole(text: str, pages: List[int], ctype: str):
            nonlocal buf, carried
            if len(buf) > carried:
                emit()
            buf, carried = [], 0
            pieces = [text] if _count_tokens(text) <= max_tokens else _hard_split_by_tokens(text, max_tokens)
            chunks.extend(_chunk(piece, path, pages, ctype) for piece in pieces)

        for para, p_start, p_end in _paragraphs(section["lines"]):
            ctype = _content_type(para)
            if ctype in ("table", "list", "figure"):
                emit_whole(para, [p_start, p_end], ctype)
                continue
            for sentence in (s.strip() for s in _segmenter().segment(para)):
                if not sentence:
                    continue
                if _count_tokens(sentence) > max_tokens:
                    emit_whole(sentence, [p_start, p_end], "paragraph")
                    continue
                if buf and len(buf) > carried and \
                        _count_tokens(" ".join([b[0] for b in buf] + [sentence])) > max_tokens:
                    emit()
                    buf = buf[-overlap:] if overlap else []
                    carried = len(buf)
                    if _count_tokens(" ".join([b[0] for b in buf] + [sentence])) > max_tokens:
                        buf, carried = [], 0  # overlap + sentence wouldn't fit: drop the overlap
                buf.append((sentence, p_start, p_end))

        if len(buf) > carried:
            emit()
    return chunks


def _fixed_chunks(sections, size: int, overlap: int) -> List[Chunk]:
    """Token windows over each section's text (page range from the paragraphs a window touches)."""
    chunks: List[Chunk] = []
    for section in sections:
        path = section["path"]
        tokens: List[int] = []
        token_pages: List[int] = []
        for para, p_start, p_end in _paragraphs(section["lines"]):
            ids = ENC.encode(para + "\n\n")
            tokens.extend(ids)
            # spread the paragraph's page range over its tokens
            token_pages.extend(p_start if i < len(ids) / 2 or p_start == p_end else p_end for i in range(len(ids)))
        step = size - overlap
        for start in range(0, len(tokens), step):
            window = tokens[start:start + size]
            if not window:
                break
            text = ENC.decode(window)
            chunks.append(_chunk(text, path, token_pages[start:start + len(window)], _content_type(text)))
            if start + size >= len(tokens):
                break
    return chunks

"""Structure-aware chunker for Docling JSON output.

Input: Docling conversion result (the JSON returned by /v1/result/{task_id}).
Output: list of Chunk objects suitable for embedding + metadata storage.

Chunking rules (per Phase 4 plan):
- Target 600–1000 tokens, hard max 1200–1500 (tiktoken cl100k_base)
- Prefer natural boundaries: heading > section > paragraph > table > procedure
- Tables stay with their containing section; oversized tables split parent/child
- Minimal overlap: last sentence of previous chunk is prepended to the next
- Every chunk carries: text, section_path, page_start, page_end, content_type,
  tables_or_figure_refs
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
TARGET_TOKENS = 800
MAX_TOKENS = 1400
OVERLAP_SENTENCES = 1


@dataclass
class Chunk:
    text: str
    section_path: List[str]
    page_start: int
    page_end: int
    content_type: str  # heading | paragraph | table | list | procedure | mixed
    tables_or_figure_refs: List[str] = field(default_factory=list)
    token_count: int = 0


def _count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def _split_sentences(text: str) -> List[str]:
    # Simple sentence splitter that keeps the delimiter with the sentence
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _extract_text_and_meta(node: Dict[str, Any]) -> tuple[str, str, List[str], int, int]:
    """Return (text, content_type, section_path, page_start, page_end) for a node."""
    text = node.get("text", "") or node.get("content", "")
    ctype = node.get("label", node.get("type", "paragraph"))
    if ctype in ("heading", "title"):
        ctype = "heading"
    elif ctype in ("table", "tabular"):
        ctype = "table"
    elif ctype in ("list", "ordered_list", "unordered_list"):
        ctype = "list"
    else:
        ctype = "paragraph"

    path = node.get("section_path", node.get("path", []))
    if isinstance(path, str):
        path = [path]

    pages = node.get("page_numbers", node.get("pages", [1]))
    if isinstance(pages, (int, float)):
        pages = [int(pages)]
    page_start = min(pages) if pages else 1
    page_end = max(pages) if pages else 1

    refs = node.get("table_refs", []) + node.get("figure_refs", [])
    return text.strip(), ctype, path, page_start, page_end


def chunk_document(doc: Dict[str, Any]) -> List[Chunk]:
    """Main entry point. Accepts the full Docling result JSON."""
    chunks: List[Chunk] = []
    current_section: List[str] = []
    overlap_buffer: str = ""

    # Docling top-level keys we care about (from the OpenAPI + Phase-2 outputs)
    body = doc.get("document", doc.get("body", doc))

    # Walk the hierarchical structure
    def walk(node: Any):
        nonlocal current_section, overlap_buffer

        if isinstance(node, dict):
            label = node.get("label", node.get("type", ""))
            if label in ("heading", "title", "section"):
                current_section = node.get("section_path", [node.get("text", "")])

            if "children" in node or "items" in node:
                for child in node.get("children", node.get("items", [])):
                    walk(child)
                return

            text, ctype, path, pstart, pend = _extract_text_and_meta(node)
            if not text:
                return

            # Merge with overlap from previous chunk
            candidate = (overlap_buffer + " " + text).strip() if overlap_buffer else text
            tok_count = _count_tokens(candidate)

            if tok_count <= MAX_TOKENS:
                chunk = Chunk(
                    text=candidate,
                    section_path=path or current_section,
                    page_start=pstart,
                    page_end=pend,
                    content_type=ctype,
                    token_count=tok_count,
                )
                chunks.append(chunk)
                sents = _split_sentences(candidate)
                overlap_buffer = " ".join(sents[-OVERLAP_SENTENCES:]) if len(sents) > 1 else ""
            else:
                sents = _split_sentences(text)
                buf = overlap_buffer
                for sent in sents:
                    trial = (buf + " " + sent).strip() if buf else sent
                    if _count_tokens(trial) > MAX_TOKENS and buf:
                        chunks.append(
                            Chunk(
                                text=buf,
                                section_path=path or current_section,
                                page_start=pstart,
                                page_end=pend,
                                content_type=ctype,
                                token_count=_count_tokens(buf),
                            )
                        )
                        buf = sent
                    else:
                        buf = trial
                if buf:
                    overlap_buffer = " ".join(_split_sentences(buf)[-OVERLAP_SENTENCES:])
                    if _count_tokens(buf) <= MAX_TOKENS:
                        chunks.append(
                            Chunk(
                                text=buf,
                                section_path=path or current_section,
                                page_start=pstart,
                                page_end=pend,
                                content_type=ctype,
                                token_count=_count_tokens(buf),
                            )
                        )
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(body)

    # Final cleanup: drop any empty or tiny chunks
    chunks = [c for c in chunks if c.token_count >= 20]
    return chunks


def chunks_to_dicts(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    """Convenience for JSON serialization / embedding pipeline."""
    return [
        {
            "text": c.text,
            "section_path": c.section_path,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "content_type": c.content_type,
            "tables_or_figure_refs": c.tables_or_figure_refs,
            "token_count": c.token_count,
        }
        for c in chunks
    ]

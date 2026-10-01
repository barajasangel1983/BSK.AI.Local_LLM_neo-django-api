"""Structure-aware chunker for Docling JSON output.

Input: Docling conversion result (the JSON returned by /v1/convert/source).
Output: list of Chunk objects suitable for embedding + metadata storage.

The Docling sync endpoint returns:
{
  "document": {
    "filename": "...",
    "md_content": "# Heading\\n\\nParagraph text...",
    "json_content": null,
    ...
  },
  "status": "success",
  "errors": []
}

Chunking rules (per Phase 4 plan):
- Target 600–1000 tokens, hard max 1200–1500 (tiktoken cl100k_base)
- Prefer natural boundaries: heading > section > paragraph
- Minimal overlap: last sentence of previous chunk is prepended to the next
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
TARGET_TOKENS = 600
MAX_TOKENS = 800  # Must be well under 4096 to account for "passage: " prefix + overhead
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
    """Simple sentence splitter that keeps the delimiter with the sentence."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _strip_embedded_images(md_content: str) -> str:
    """Remove base64-encoded images from markdown to prevent huge chunks.
    
    Docling sometimes embeds figures as data:image/... URIs in md_content.
    We strip them since they're not useful for text RAG embedding.
    """
    # Remove markdown image references with data URIs: ![alt](data:image/...)
    md_content = re.sub(r'!\[[^\]]*\]\(data:image/[^)]+\)', '[figure]', md_content)
    # Also remove any raw data:image blocks that might not be in markdown syntax
    md_content = re.sub(r'data:image/[^)\s]+', '[figure]', md_content)
    return md_content


def _extract_sections_from_markdown(md_content: str) -> List[Dict[str, Any]]:
    """Parse markdown into sections based on headings."""
    lines = md_content.split("\n")
    sections = []
    current_section = []
    current_heading = "Document"
    current_path = []

    for line in lines:
        # Check for markdown headings (# ## ### etc.)
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            # Save previous section
            if current_section:
                sections.append({
                    "heading": current_heading,
                    "path": current_path,
                    "content": "\n".join(current_section).strip()
                })
            # Start new section
            level = len(heading_match.group(1))
            current_heading = heading_match.group(2).strip()
            current_path = current_path[:level-1] + [current_heading]
            current_section = []
        else:
            current_section.append(line)

    # Don't forget the last section
    if current_section:
        sections.append({
            "heading": current_heading,
            "path": current_path,
            "content": "\n".join(current_section).strip()
        })

    return sections


def chunk_document(doc: Dict[str, Any]) -> List[Chunk]:
    """Main entry point. Accepts the full Docling result JSON."""
    chunks: List[Chunk] = []
    overlap_buffer: str = ""

    # Extract the document from the result
    document = doc.get("document", doc)

    # Get the markdown content
    md_content = document.get("md_content", "") if isinstance(document, dict) else ""
    if not md_content:
        # Fallback: try to get text from other fields
        md_content = document.get("text_content", "") if isinstance(document, dict) else ""

    if not md_content:
        return chunks

    # Strip embedded base64 images (95%+ of content for figure-heavy PDFs)
    md_content = _strip_embedded_images(md_content)

    # Parse into sections
    sections = _extract_sections_from_markdown(md_content)

    for section in sections:
        text = section["content"]
        if not text or len(text) < 50:
            continue

        path = section["path"]
        heading = section["heading"]

        # Split the section into paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)

        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 20:
                continue

            # Merge with overlap from previous chunk
            candidate = (overlap_buffer + " " + para).strip() if overlap_buffer else para
            tok_count = _count_tokens(candidate)

            if tok_count <= MAX_TOKENS:
                # Fits comfortably
                chunk = Chunk(
                    text=candidate,
                    section_path=path,
                    page_start=1,
                    page_end=1,
                    content_type="paragraph" if len(path) == 0 else "heading",
                    token_count=tok_count,
                )
                chunks.append(chunk)

                # Prepare overlap for next chunk
                sents = _split_sentences(candidate)
                overlap_buffer = " ".join(sents[-OVERLAP_SENTENCES:]) if len(sents) > 1 else ""
            else:
                # Paragraph is too large — split on sentence boundaries
                sents = _split_sentences(para)
                buf = overlap_buffer
                for sent in sents:
                    trial = (buf + " " + sent).strip() if buf else sent
                    if _count_tokens(trial) > MAX_TOKENS and buf:
                        # Flush current buf as a chunk
                        chunks.append(
                            Chunk(
                                text=buf,
                                section_path=path,
                                page_start=1,
                                page_end=1,
                                content_type="paragraph",
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
                                section_path=path,
                                page_start=1,
                                page_end=1,
                                content_type="paragraph",
                                token_count=_count_tokens(buf),
                            )
                        )
                    else:
                        # Last resort: hard split by token count (no sentence boundary found)
                        hard_chunks = _hard_split_by_tokens(buf, MAX_TOKENS)
                        for hc in hard_chunks:
                            chunks.append(
                                Chunk(
                                    text=hc,
                                    section_path=path,
                                    page_start=1,
                                    page_end=1,
                                    content_type="paragraph",
                                    token_count=_count_tokens(hc),
                                )
                            )

    # Final cleanup: drop any empty or tiny chunks
    chunks = [c for c in chunks if c.token_count >= 20]

    # Ensure no chunk has an empty section_path (Chroma rejects empty list metadata)
    for c in chunks:
        if not c.section_path:
            c.section_path = ["Document"]

    return chunks


def _hard_split_by_tokens(text: str, max_tokens: int) -> List[str]:
    """Hard-split text into chunks of at most max_tokens (last resort)."""
    tokens = ENC.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(ENC.decode(chunk_tokens))
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

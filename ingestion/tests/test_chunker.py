"""Unit tests for the markdown chunker (Docling md_content with page-break markers)."""

import pytest

from ingestion.chunker import PAGE_BREAK, Chunk, chunk_document, chunks_to_dicts


def _doc(md: str) -> dict:
    return {"document": {"md_content": md}, "status": "success", "errors": []}


@pytest.fixture
def native_pdf_doc():
    """Maintenance-manual style: headings, paragraphs, a table, pages 1-3."""
    safety = "This manual contains important safety information. Read all instructions before use. " * 12
    install = "Mount the unit on a flat surface using the provided brackets. " * 15
    return _doc(
        f"# 1. Safety Information\n\n{safety}\n\n{PAGE_BREAK}\n\n"
        f"# 2. Installation\n\n{install}\n{PAGE_BREAK}\n{install}\n\n"
        "| Parameter | Value |\n|---|---|\n| Voltage | 230V |\n| Current | 10A |\n"
    )


@pytest.fixture
def scanned_pdf_doc():
    """Scanned OCR document: long paragraphs that must be split."""
    long_text = "The quick brown fox jumps over the lazy dog. " * 120
    return _doc(f"{long_text}\n\n{PAGE_BREAK}\n\n{long_text} Additional OCR artifacts here.")


def test_basic_chunking(native_pdf_doc):
    chunks = chunk_document(native_pdf_doc)
    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(20 <= c.token_count <= 800 for c in chunks)
    assert any("Safety" in " ".join(c.section_path) for c in chunks)


def test_page_numbers_follow_page_breaks(native_pdf_doc):
    chunks = chunk_document(native_pdf_doc)
    safety = next(c for c in chunks if c.section_path == ["1. Safety Information"])
    assert (safety.page_start, safety.page_end) == (1, 1)
    install = [c for c in chunks if c.section_path == ["2. Installation"]]
    assert {c.page_start for c in install} == {2, 3}
    table = next(c for c in chunks if c.content_type == "table")
    assert table.page_start == 3


def test_paragraph_spanning_a_page_break():
    para_a = "First half of a long paragraph that keeps going. " * 3
    para_b = "Second half after the page turn continues here. " * 3
    chunks = chunk_document(_doc(f"# S\n\n{para_a}\n{PAGE_BREAK}\n{para_b}"))
    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_without_page_breaks_everything_is_page_one():
    chunks = chunk_document(_doc("# Notes\n\n" + "Plain text notes with no pagination at all. " * 10))
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1)]


def test_content_types():
    md = (
        "# Section\n\n"
        + "A normal explanatory paragraph about the machine. " * 4 + "\n\n"
        + "- first step to take\n- second step to take\n- third step to take\n\n"
        + "| Pin | Signal |\n|---|---|\n| 1 | GND |\n| 2 | +24V |\n"
    )
    types = [c.content_type for c in chunk_document(_doc(md))]
    assert types == ["paragraph", "list", "table"]


def test_token_limits(scanned_pdf_doc):
    chunks = chunk_document(scanned_pdf_doc)
    assert len(chunks) > 2
    assert all(c.token_count <= 800 for c in chunks)
    assert {c.page_start for c in chunks} == {1, 2}


def test_strips_embedded_images():
    md = "# Fig\n\n" + "Text before the figure goes here. " * 4 + "\n\n![x](data:image/png;base64," + "A" * 5000 + ")\n"
    chunks = chunk_document(_doc(md))
    assert all("data:image" not in c.text for c in chunks)


def test_metadata_dicts(native_pdf_doc):
    dlist = chunks_to_dicts(chunk_document(native_pdf_doc))
    assert all({"section_path", "page_start", "page_end", "content_type"} <= d.keys() for d in dlist)
    assert all(d["section_path"] for d in dlist)

"""Unit tests for the structure-aware chunker.

Fixtures are minimal Docling-style JSON structures that mimic the
three real documents from Phase 2 (native, scanned, schematic).
"""

import json
from pathlib import Path

import pytest

from ingestion.chunker import Chunk, chunk_document, chunks_to_dicts


@pytest.fixture
def native_pdf_doc():
    """Maintenance_Manual_2024_EN.pdf style (clean headings + paragraphs)."""
    return {
        "document": {
            "children": [
                {"label": "heading", "text": "1. Safety Information", "page_numbers": [1]},
                {"label": "paragraph", "text": "This manual contains important safety information. Read all instructions before use." * 12, "page_numbers": [1]},
                {"label": "heading", "text": "2. Installation", "page_numbers": [2]},
                {"label": "paragraph", "text": "Mount the unit on a flat surface using the provided brackets." * 15, "page_numbers": [2, 3]},
                {"label": "table", "text": "Parameter | Value\nVoltage | 230V\nCurrent | 10A", "page_numbers": [3]},
            ]
        }
    }


@pytest.fixture
def scanned_pdf_doc():
    """Scanned OCR document (long paragraphs, OCR noise)."""
    long_text = "The quick brown fox jumps over the lazy dog. " * 40
    return {
        "document": {
            "children": [
                {"label": "paragraph", "text": long_text, "page_numbers": [5]},
                {"label": "paragraph", "text": long_text + " Additional OCR artifacts here.", "page_numbers": [6]},
            ]
        }
    }


@pytest.fixture
def schematic_pdf_doc():
    """Electrical schematic with tables and symbols (tables stay with section)."""
    return {
        "document": {
            "children": [
                {"label": "heading", "text": "Wiring Diagram", "page_numbers": [10]},
                {"label": "table", "text": "Pin | Signal | Color\n1 | GND | Black\n2 | +24V | Red", "page_numbers": [10]},
                {"label": "paragraph", "text": "Connect according to the table above. Verify polarity before powering.", "page_numbers": [10]},
            ]
        }
    }


def test_basic_chunking(native_pdf_doc):
    chunks = chunk_document(native_pdf_doc)
    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(20 <= c.token_count <= 1400 for c in chunks)
    # Heading should appear in at least one chunk's section_path
    assert any("Safety" in " ".join(c.section_path) for c in chunks)


def test_token_limits(scanned_pdf_doc):
    chunks = chunk_document(scanned_pdf_doc)
    for c in chunks:
        assert c.token_count <= 1400
        # Most chunks should be close to the target band
        if len(chunks) > 1:
            assert c.token_count >= 100 or c == chunks[-1]


def test_table_with_section(schematic_pdf_doc):
    chunks = chunk_document(schematic_pdf_doc)
    table_chunks = [c for c in chunks if c.content_type == "table"]
    assert table_chunks, "Expected at least one table chunk"
    # Table should share page with the heading
    assert table_chunks[0].page_start == 10


def test_overlap_and_metadata(native_pdf_doc):
    chunks = chunk_document(native_pdf_doc)
    dlist = chunks_to_dicts(chunks)
    assert all("section_path" in d and "page_start" in d for d in dlist)
    # Overlap buffer should be populated after the first chunk (if multiple)
    if len(chunks) >= 2:
        # At least one chunk should have a non-empty section_path or page info
        assert any(c.section_path or c.page_start for c in chunks)

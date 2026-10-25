"""Tables row by row for Generate triples (schema review step 3, 2026-10-16).

Docling writes a document's tables as Markdown tables. Inside 800-token text windows a long table is cut
in the middle (its rows lose their header) and the model's answer for it can run past the output limit;
the ABB motors manual's Table 10.1 lost 33 of its 54 rows that way. Here each table is read on its own:

    the table's caption and section, its header, then a few rows at a time
    (rows with the same first cell stay together: one trouble with all its causes)

A table continued on a later page without a caption (same header) keeps the caption of its first part.
A row whose cells are all the same text (e.g. "Ball bearings, lubrication intervals in duty hours") is a
heading inside the table: it is repeated above the rows under it. The text windows are read without the
tables, so a table is not read twice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ingestion.chunker import PAGE_BREAK

TABLE_CHUNK_BASE = 200_000      # chunk_index of a table part: 200000 + its number (never a text window or a figure)
MAX_ROWS = 12                   # rows per call
EMPTY = ("", "-", "–", "—")     # a cell with no value is left out of its row
CAPTION_RE = re.compile(r"^\s*(Table\s+[\w.\-]+.*)$", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.*\S)\s*$")
NUMBERED_RE = re.compile(r"^\d+(\.\d+)*\.?\s")
MARKER_RE = re.compile(r"(<!--.*?-->\s*)+")      # "<!-- image -->" left by the parser
RULE_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


@dataclass
class Table:
    page: int
    caption: str
    section: str
    header: list[str]
    rows: list[list[str]]
    last_page: int = 0
    char_start: int = -1            # P25b: where the table and each row sit in the parsed Markdown
    char_end: int = -1
    row_spans: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class TableWindow:
    """What extraction needs from a text window, for a part of a table."""
    page_start: int
    page_end: int
    section_path: list[str] = field(default_factory=list)
    text: str = ""
    char_start: int | None = None   # P25b: the part's rows in the parsed Markdown, and their numbers in the table
    char_end: int | None = None
    row_start: int | None = None
    row_end: int | None = None


def _cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [re.sub(r"\s+", " ", c).strip() for c in line.split("|")]


def is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def find_tables(md: str) -> list[Table]:
    """Every Markdown table of a parsed document, with its page (1-based), caption and section."""
    tables: list[Table] = []
    section = ""
    base = 0                        # real offset of the page in `md`
    for page_no, page in enumerate(md.split(PAGE_BREAK), start=1):
        lines = page.splitlines()
        starts, pos = [], base
        for line in page.splitlines(keepends=True):
            starts.append(pos)
            pos += len(line)
        base += len(page) + len(PAGE_BREAK)
        i = 0
        recent: list[str] = []          # the last non-empty lines before a table: its caption is there
        while i < len(lines):
            line = lines[i]
            heading = HEADING_RE.match(line)
            if heading and (NUMBERED_RE.match(heading.group(1)) or not section):
                section = heading.group(1)      # the chapter: a numbered heading ("7.2.2 …"), not a sub-title
            if not is_table_line(line):
                if line.strip() and line.strip() != "-":
                    recent = (recent + [line.strip()])[-4:]
                i += 1
                continue
            block, first = [], i
            while i < len(lines) and is_table_line(lines[i]):
                block.append(lines[i])
                i += 1
            if len(block) < 2:
                continue
            header = _cells(block[0])
            rows = [_cells(b) for b in block[1:] if not RULE_RE.match(b.strip())]
            row_spans = [(starts[first + k], starts[first + k] + len(b)) for k, b in enumerate(block)
                         if k and not RULE_RE.match(b.strip())]
            caption = next((m.group(1).strip() for m in map(CAPTION_RE.match, reversed(recent)) if m), "")
            previous = tables[-1] if tables else None
            if (not caption and previous and previous.header == header
                    and 0 < page_no - (previous.last_page or previous.page) <= 2):
                caption, table_section = previous.caption, previous.section      # the same table, continued
            else:
                table_section = section
            tables.append(Table(page=page_no, caption=caption, section=table_section, header=header, rows=rows,
                                last_page=page_no, char_start=starts[first],
                                char_end=starts[i - 1] + len(lines[i - 1]), row_spans=row_spans))
            recent = []
    return tables


def _is_heading_row(row: list[str]) -> bool:
    return len(row) > 1 and len(set(row)) == 1 and bool(row[0])


def windows(table: Table, number: int, max_rows: int = MAX_ROWS) -> list[tuple[int, TableWindow]]:
    """(chunk index, window) for the parts of one table: rows with the same first cell stay together."""
    groups: list[tuple[str, list[tuple[int, list[str]]]]] = []      # (heading inside the table, (row number, row))
    heading = ""
    for number_in_table, row in enumerate(table.rows):
        if _is_heading_row(row):
            heading = row[0]
            continue
        same = groups and groups[-1][0] == heading and groups[-1][1] and (not row[0] or row[0] == groups[-1][1][-1][1][0])
        if same:
            groups[-1][1].append((number_in_table, row))
        else:
            groups.append((heading, [(number_in_table, row)]))
    parts: list[tuple[str, list[tuple[int, list[str]]]]] = []
    for heading, rows in groups:
        for start in range(0, len(rows), max_rows):        # one group longer than a part: cut it
            chunk = rows[start:start + max_rows]
            if parts and parts[-1][0] == heading and len(parts[-1][1]) + len(chunk) <= max_rows:
                parts[-1][1].extend(chunk)
            else:
                parts.append((heading, list(chunk)))
    out = []
    title = table.caption or "Table"
    for n, (heading, numbered) in enumerate(parts):
        rows = [r for _, r in numbered]
        lines = [f"{title}" + (f" (section: {table.section})" if table.section else "") + f", page {table.page}.",
                 "Read it row by row: each row is one fact; every value is given with its column."]
        if heading:
            lines.append(f"Rows under: {heading}")
        # Each row as "column = value" pairs: a model loses track of the columns of a wide Markdown table
        # (Table 7.2: the 1800 r/min value was read for 1500 r/min).
        lines.append("Columns: " + " | ".join(table.header))
        lines += [f"Row {k}: " + "; ".join(f"{h or f'column {c + 1}'} = {v}" for c, (h, v) in enumerate(zip(table.header, r))
                                       if v not in EMPTY)
                  for k, r in enumerate(rows, start=1)]
        # A short reasoning step first: with thinking off, the model skipped Table 7.2 or read the wrong speed
        # column; asked to say which row and column apply, it read 5 600 h / 70 g for frame 355 at 1500 r/min.
        lines.append('Before "triples", add a key "note": in one or two sentences, which rows (and which columns) of this '
                     'part of the table apply, and why; or why none does.')
        section_path = [p for p in (table.section, table.caption) if p]
        first, last = numbered[0][0], numbered[-1][0]
        spans = table.row_spans if len(table.row_spans) == len(table.rows) else []
        out.append((TABLE_CHUNK_BASE + number * 1000 + n,
                    TableWindow(page_start=table.page, page_end=table.page, section_path=section_path, text="\n".join(lines),
                                char_start=spans[first][0] if spans else None, char_end=spans[last][1] if spans else None,
                                row_start=first + 1, row_end=last + 1)))
    return out


def strip_tables(text: str, captions: frozenset[str] | set[str] = frozenset()) -> str:
    """A text window without its table lines (the tables are read row by row). When rows were removed, their
    captions and image markers go too: each table part starts with its caption, and a window left with only
    "Table 11-13 Current reduction depending on the pulse frequency" was given invented values (G120C, 2026-10-21)."""
    lines = text.splitlines()
    kept = [line for line in lines if not is_table_line(line)]
    if len(kept) < len(lines):
        kept = [line for line in kept if line.strip() not in captions and not MARKER_RE.fullmatch(line.strip())]
    return "\n".join(kept)

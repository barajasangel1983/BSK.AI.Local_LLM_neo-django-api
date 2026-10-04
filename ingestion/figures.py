"""Figures of library documents (P9a.2): found by Docling, described by the VLM on BSK.

    parse (Docling md + json) ──► figures.json: page + box of every picture
    "Describe figures" job     ──► crop each figure from the stored file → VLM → description
    Generate embeddings / triples ──► `[Figure p.N (kind): description]` replaces the n-th
                                       `<!-- image -->` placeholder of the parsed Markdown

The crop follows contract v1.2 (frontend repo, claude/VLM_service_brief.md): Docling's box
is in PDF points (BOTTOMLEFT origin), pages are 1-based, 5 % padding, JPEG <= 1280 px.
An image file in the library is one figure: the whole picture.

figures.json (next to parsed.json):
    {"source": "pdf" | "image", "document_revision": "1",
     "figures": [{"index", "page", "bbox", "page_size", "caption", "area",
                  "status": "pending" | "done" | "skipped" | "failed", "skip_reason",
                  "kind", "description", "error", "model", "described_at", "width", "height"}]}
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from gpu import images, vlm
from gpu import orchestrator as gpu

from .chunker import FIGURE_PREFIX, IMAGE_PLACEHOLDER
from .models import Document

logger = logging.getLogger("chat")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
DECORATIVE = "decorative"
KINDS = ("drawing", "schematic", "diagram", "chart", "table", "form", "photo", "screenshot", DECORATIVE)
# Figure kinds worth turning into graph relationships (P9a.3).
DRAWING_KINDS = ("drawing", "schematic", "diagram")
PROMPT_VERSION = "figure-v1"
PROMPT = (
    "You are describing a figure from a technical document for engineers who cannot see it.\n"
    "Answer in exactly this format:\n"
    f"Kind: <one of: {', '.join(KINDS)}>\n"
    "Description: <2 to 4 sentences: what the figure shows and how its parts relate>\n"
    "Text: <the labels, tag numbers and values that are legible, separated by '; ', or 'none'>\n"
    f"Use the kind '{DECORATIVE}' for logos, icons, background pictures and stock photos that carry no "
    "technical information; then keep the description to one short sentence.\n"
    "Copy labels and numbers exactly as written. Do not guess text that is not legible."
)
MAX_DESCRIPTION_CHARS = 1500


class BskUnavailable(Exception):
    """BSK (or its VLM) can't be reached: the remaining figures stay pending."""


# --- storage -------------------------------------------------------------------

def _dir(doc: Document) -> Path:
    return Path(settings.LIBRARY_BASE) / str(doc.id)


def figures_path(doc: Document) -> Path:
    return _dir(doc) / "figures.json"


def image_path(doc: Document, index: int) -> Path:
    return _dir(doc) / "figures" / f"{index}.jpg"


def is_image(doc: Document) -> bool:
    return Path(doc.filename).suffix.lower() in IMAGE_SUFFIXES


def is_pdf(doc: Document) -> bool:
    return Path(doc.filename).suffix.lower() == ".pdf"


def load(doc: Document) -> dict | None:
    """The document's figure list, or None when it was parsed before figures existed."""
    path = figures_path(doc)
    return json.loads(path.read_text()) if path.exists() else None


def save(doc: Document, data: dict) -> None:
    figures_path(doc).write_text(json.dumps(data, indent=1))


# --- from the Docling result -------------------------------------------------------

def from_docling(doc: Document, layout: dict | None) -> dict:
    """Build the figure list from Docling's JSON, keeping descriptions of unchanged figures."""
    source = "image" if is_image(doc) else "pdf" if is_pdf(doc) else "other"
    figures: list[dict] = []
    if source == "image":
        # The whole picture is the figure (Docling's boxes inside it are parts of it).
        figures.append({"index": 0, "page": 1, "bbox": None, "page_size": None, "caption": "", "area": 1.0,
                        "status": "pending", "skip_reason": ""})
    elif source == "pdf" and layout:
        pages = layout.get("pages") or {}
        texts = layout.get("texts") or []
        for index, picture in enumerate(layout.get("pictures") or []):
            prov = (picture.get("prov") or [{}])[0]
            page = int(prov.get("page_no") or 0)
            size = (pages.get(str(page)) or {}).get("size") or {}
            bbox = prov.get("bbox")
            figure = {"index": index, "page": page, "bbox": bbox, "page_size": size or None,
                      "caption": _caption(picture, texts), "area": None, "status": "pending", "skip_reason": ""}
            if not bbox or not size.get("width") or not size.get("height"):
                figure.update(status="skipped", skip_reason="no position in the parsed document")
            else:
                area = abs((bbox["r"] - bbox["l"]) * (bbox["t"] - bbox["b"])) / (size["width"] * size["height"])
                figure["area"] = round(area, 4)
                if area < settings.FIGURE_MIN_AREA:
                    figure.update(status="skipped", skip_reason="too small (under 2 % of the page)")
                elif area > settings.FIGURE_MAX_AREA:
                    figure.update(status="skipped", skip_reason="covers the whole page")
            figures.append(figure)

    previous = {f["index"]: f for f in (load(doc) or {}).get("figures", [])}
    for figure in figures:
        old = previous.get(figure["index"])
        if old and old.get("status") in ("done", "failed") and (old.get("page"), old.get("bbox")) == (
                figure["page"], figure["bbox"]) and figure["status"] == "pending":
            figure.update({k: old[k] for k in ("status", "kind", "description", "error", "model", "described_at",
                                               "width", "height", "skip_reason") if k in old})
    return {"source": source, "document_revision": doc.document_revision, "figures": figures}


def _caption(picture: dict, texts: list) -> str:
    parts = []
    for ref in picture.get("captions") or []:
        match = re.fullmatch(r"#/texts/(\d+)", ref.get("$ref", ""))
        if match and int(match.group(1)) < len(texts):
            parts.append((texts[int(match.group(1))].get("text") or "").strip())
    return " ".join(p for p in parts if p)[:500]


def summary(data: dict | None) -> dict:
    """Counts for the document list."""
    figures = (data or {}).get("figures", [])
    by = lambda status: sum(1 for f in figures if f["status"] == status)  # noqa: E731
    return {"found": len(figures), "eligible": len(figures) - by("skipped"), "described": by("done"),
            "pending": by("pending"), "failed": by("failed"), "skipped": by("skipped")}


# --- crop + describe ---------------------------------------------------------------

def crop(doc: Document, figure: dict) -> tuple[bytes, int, int]:
    """The figure as the JPEG the VLM gets (cached). Raises images.ImageError when it is too small."""
    path = image_path(doc, figure["index"])
    source = _dir(doc) / doc.filename
    if figure.get("bbox") is None:      # an image document: the whole picture
        jpeg, width, height = images.to_jpeg(source.read_bytes())
    else:
        jpeg, width, height, raw_w, raw_h = images.crop_pdf_figure(
            source, figure["page"], figure["bbox"], figure["page_size"],
            padding=settings.FIGURE_PADDING, max_scale=settings.FIGURE_RENDER_MAX_SCALE)
        if min(raw_w, raw_h) < settings.FIGURE_MIN_SIDE_PX:
            raise images.ImageError(f"too small ({raw_w} x {raw_h} px)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jpeg)
    return jpeg, width, height


def parse_reply(reply: str) -> tuple[str, str]:
    """(kind, description text) from the VLM's 'Kind: / Description: / Text:' answer."""
    reply = reply.replace("*", "")       # the model sometimes answers with Markdown bold
    kind = ""
    match = re.search(r"^\s*\**kind\**\s*:\s*\**\s*([A-Za-z]+)", reply, re.IGNORECASE | re.MULTILINE)
    if match and match.group(1).lower() in KINDS:
        kind = match.group(1).lower()
    description = re.search(r"description\**\s*:\s*(.*?)(?=^\s*\**text\**\s*:|\Z)", reply,
                            re.IGNORECASE | re.DOTALL | re.MULTILINE)
    text = re.search(r"^\s*\**text\**\s*:\s*(.*)\Z", reply, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    body = " ".join(description.group(1).split()) if description else ""
    labels = " ".join(text.group(1).split()) if text else ""
    if labels and labels.lower().strip(" .") != "none":
        body = f"{body} Text: {labels}".strip()
    if not body:        # the model ignored the format: keep what it said
        body = " ".join(re.sub(r"^\s*\**kind\**\s*:.*$", "", reply, flags=re.IGNORECASE | re.MULTILINE).split())
    return kind, body[:MAX_DESCRIPTION_CHARS]


def describe(doc: Document, figure: dict) -> None:
    """Describe one figure in place. Raises BskUnavailable when BSK can't be reached."""
    try:
        jpeg, width, height = crop(doc, figure)
    except images.ImageError as exc:
        figure.update(status="skipped", skip_reason=str(exc))
        return
    prompt = PROMPT + (f"\nCaption in the document: {figure['caption']}" if figure.get("caption") else "")
    try:
        reply = vlm.complete(jpeg, prompt, max_tokens=settings.FIGURE_MAX_TOKENS, purpose="vlm-figure")
    except gpu.GpuError as exc:
        raise BskUnavailable(str(exc)) from exc
    except Exception as exc:        # one bad figure doesn't stop the document
        logger.warning("figure failed doc=%s figure=%s: %s", doc.doc_key, figure["index"], exc)
        figure.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:300])
        return
    kind, description = parse_reply(reply)
    figure.update(kind=kind, description=description, error="", width=width, height=height,
                  model=settings.VLM_MODEL, prompt_version=PROMPT_VERSION, described_at=timezone.now().isoformat())
    if kind == DECORATIVE:
        figure.update(status="skipped", skip_reason="decorative (logo, icon or background)")
    else:
        figure.update(status="done", skip_reason="")


def targets(data: dict, only: list[int] | None = None, force: bool = False) -> list[dict]:
    """Figures a run should send to the VLM."""
    chosen = []
    for figure in data["figures"]:
        # Skipped for its size or position stays skipped; "decorative" was the VLM's call and can be redone.
        redoable = figure["status"] != "skipped" or figure.get("kind") == DECORATIVE
        if only is not None:
            # Asked for by number: also a figure skipped for its size ("describe anyway").
            if figure["index"] in only and (redoable or figure.get("bbox")):
                chosen.append(figure)
        elif figure["status"] in ("pending", "failed") or (force and redoable):
            chosen.append(figure)
    return chosen


# --- into the text -------------------------------------------------------------------

def figure_line(figure: dict) -> str:
    kind = f" ({figure['kind']})" if figure.get("kind") else ""
    return f"{FIGURE_PREFIX}{figure['page']}{kind}: {figure['description']}]"


def apply_to_markdown(md: str, data: dict | None) -> str:
    """Put each described figure where Docling left its placeholder.

    Docling writes one `<!-- image -->` per picture, in the order of its picture list,
    so the n-th placeholder is figure n. Each description is its own paragraph.
    """
    if not data:
        return md
    described = {f["index"]: f for f in data["figures"] if f["status"] == "done" and f.get("description")}
    if not described:
        return md
    if data.get("source") == "image":
        return f"{figure_line(described[0])}\n\n{md}" if 0 in described else md
    if md.count(IMAGE_PLACEHOLDER) != len(data["figures"]):
        # Shouldn't happen (same Docling result); don't guess which placeholder is which.
        logger.warning("figure placeholders (%d) != figures (%d); descriptions appended per page",
                       md.count(IMAGE_PLACEHOLDER), len(data["figures"]))
        from .chunker import PAGE_BREAK
        pages = md.split(PAGE_BREAK)
        for figure in described.values():
            if 1 <= figure["page"] <= len(pages):
                pages[figure["page"] - 1] = pages[figure["page"] - 1].rstrip() + f"\n\n{figure_line(figure)}\n\n"
        return PAGE_BREAK.join(pages)
    counter = iter(range(len(data["figures"])))

    def replace(_match):
        figure = described.get(next(counter))
        return f"\n\n{figure_line(figure)}\n\n" if figure else IMAGE_PLACEHOLDER
    return re.sub(re.escape(IMAGE_PLACEHOLDER), replace, md)

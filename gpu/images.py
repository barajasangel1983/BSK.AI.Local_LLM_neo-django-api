"""Images for the BSK VLM (contract v1.2): JPEG, long side <= 1280 px, never upscaled."""

from __future__ import annotations

import io
from pathlib import Path

from django.conf import settings


class ImageError(ValueError):
    """The file is not a usable image / PDF page."""


def to_jpeg(data: bytes, long_side: int | None = None) -> tuple[bytes, int, int]:
    """Decode an image and return (jpeg bytes, width, height), downscaled to `long_side`."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageError("not a readable image") from exc
    return _encode(ImageOps.exif_transpose(img), long_side)


def _encode(img, long_side: int | None = None) -> tuple[bytes, int, int]:
    from PIL import Image

    limit = long_side or settings.VLM_IMAGE_LONG_SIDE
    if img.mode in ("RGBA", "LA", "P"):
        # Transparent areas become white (a black background hides dark line drawings).
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > limit:
        img.thumbnail((limit, limit), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=settings.VLM_IMAGE_JPEG_QUALITY)
    return out.getvalue(), img.width, img.height


def pdf_page_count(path: str | Path) -> int:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as exc:
        raise ImageError("not a readable PDF") from exc
    try:
        return len(pdf)
    finally:
        pdf.close()


def render_pdf_page(path: str | Path, page: int, long_side: int | None = None) -> tuple[bytes, int, int]:
    """Render page `page` (1-based) of a PDF as (jpeg bytes, width, height)."""
    import pypdfium2 as pdfium

    limit = long_side or settings.VLM_IMAGE_LONG_SIDE
    try:
        pdf = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as exc:
        raise ImageError("not a readable PDF") from exc
    try:
        if not 1 <= page <= len(pdf):
            raise ImageError(f"page {page} is out of range (1–{len(pdf)})")
        pdf_page = pdf[page - 1]
        width, height = pdf_page.get_size()
        img = pdf_page.render(scale=limit / max(width, height)).to_pil()
        return _encode(img, limit)
    finally:
        pdf.close()


def crop_pdf_figure(path: str | Path, page: int, bbox: dict, page_size: dict, padding: float = 0.05,
                    max_scale: float = 4.0, long_side: int | None = None) -> tuple[bytes, int, int, int, int]:
    """Crop a figure from a PDF page (contract v1.2 crop spec).

    `bbox` is Docling's box in PDF points: {l, t, r, b, coord_origin} with origin
    BOTTOMLEFT (y grows upwards) or TOPLEFT; `page_size` is Docling's {width, height}.
    The page is rendered so the figure gets up to `long_side` pixels (at most
    `max_scale` x 72 DPI, at least the page at `long_side`), cropped with `padding`
    on each side (clamped to the page) and only ever downscaled.

    Returns (jpeg, width, height, crop width before downscaling, crop height before downscaling).
    """
    import pypdfium2 as pdfium

    limit = long_side or settings.VLM_IMAGE_LONG_SIDE
    page_w, page_h = float(page_size["width"]), float(page_size["height"])
    left, right = sorted((float(bbox["l"]), float(bbox["r"])))
    if str(bbox.get("coord_origin", "BOTTOMLEFT")).upper() == "TOPLEFT":
        top, bottom = sorted((float(bbox["t"]), float(bbox["b"])))
    else:       # y flips: the top of the box has the larger y
        top, bottom = sorted((page_h - float(bbox["t"]), page_h - float(bbox["b"])))
    box_w, box_h = right - left, bottom - top
    if box_w <= 0 or box_h <= 0:
        raise ImageError("the figure box is empty")
    pad_x, pad_y = box_w * padding, box_h * padding
    left, right = max(0.0, left - pad_x), min(page_w, right + pad_x)
    top, bottom = max(0.0, top - pad_y), min(page_h, bottom + pad_y)

    scale = max(limit / max(page_w, page_h), min(max_scale, limit / max(right - left, bottom - top)))
    try:
        pdf = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as exc:
        raise ImageError("not a readable PDF") from exc
    try:
        if not 1 <= page <= len(pdf):
            raise ImageError(f"page {page} is out of range (1–{len(pdf)})")
        img = pdf[page - 1].render(scale=scale).to_pil()
    finally:
        pdf.close()
    # Normalised by Docling's page size, so a different render size still lines up.
    fx, fy = img.width / page_w, img.height / page_h
    crop = img.crop((round(left * fx), round(top * fy), round(right * fx), round(bottom * fy)))
    jpeg, width, height = _encode(crop, limit)
    return jpeg, width, height, crop.width, crop.height

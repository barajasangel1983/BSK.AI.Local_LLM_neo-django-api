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

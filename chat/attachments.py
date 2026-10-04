"""Chat attachments (P9a.1): images and PDFs that stay with their conversation.

Nothing here writes to the document library, Chroma or the Context Graph.

- An image is stored once as the JPEG the VLM gets (<= 1280 px) and is answered
  by the vision model.
- A PDF is either read as text (parsed once with Docling, cached, added to the
  prompt of the text models within a budget) or one of its pages is rendered and
  shown to the vision model ("look at page 3").

Endpoints:
- POST   /api/chat/attachments/                    upload (multipart: file, conversation_id?)
- DELETE /api/chat/attachments/<uuid>/             remove an upload that was never sent
- GET    /api/chat/attachments/<uuid>/file/        the stored image / PDF
- GET    /api/chat/attachments/<uuid>/pages/<n>/   a PDF page as JPEG (preview and what the VLM sees)
"""

from __future__ import annotations

import logging
import shutil
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.http import FileResponse
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from gpu import images
from gpu import orchestrator as gpu
from ingestion.chunker import PAGE_BREAK
from ingestion.docling_client import DoclingError, convert_file

from .models import ChatAttachment, Conversation, Message

logger = logging.getLogger("chat")

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
ATTACHED_HEADER = "ATTACHED DOCUMENT"
MIN_PAGE_CHARS = 500     # smallest useful slice when the first page alone exceeds the budget


class AttachmentError(ValueError):
    """The upload can't be used (wrong type, too large, unreadable)."""


# ---------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------

def attachment_dir(att: ChatAttachment) -> Path:
    return Path(settings.CHAT_FILES_DIR) / str(att.id)


def file_path(att: ChatAttachment) -> Path:
    return attachment_dir(att) / ("image.jpg" if att.kind == ChatAttachment.Kind.IMAGE else "document.pdf")


def text_path(att: ChatAttachment) -> Path:
    return attachment_dir(att) / "text.md"


def page_path(att: ChatAttachment, page: int) -> Path:
    return attachment_dir(att) / f"page-{page}.jpg"


@receiver(post_delete, sender=ChatAttachment)
def _remove_files(sender, instance, **kwargs):
    shutil.rmtree(attachment_dir(instance), ignore_errors=True)


def remove_orphans() -> int:
    """Delete uploads that were never sent with a message."""
    cutoff = timezone.now() - timedelta(hours=settings.CHAT_ATTACHMENT_ORPHAN_HOURS)
    orphans = ChatAttachment.objects.filter(conversation__isnull=True, created_at__lt=cutoff)
    count = orphans.count()
    for att in orphans:
        att.delete()
    return count


def create(uploaded, owner, conversation: Conversation | None = None) -> ChatAttachment:
    """Validate and store an uploaded image or PDF."""
    name = Path(uploaded.name or "file").name
    suffix = Path(name).suffix.lower()
    if uploaded.size > settings.CHAT_ATTACHMENT_MAX_BYTES:
        raise AttachmentError(f"The file is larger than {settings.CHAT_ATTACHMENT_MAX_BYTES // (1024 * 1024)} MB.")
    if suffix != ".pdf" and suffix not in IMAGE_TYPES:
        raise AttachmentError("Only images (PNG, JPEG, WebP, GIF, BMP) and PDF files can be attached in chat.")
    data = uploaded.read()

    kind = ChatAttachment.Kind.PDF if suffix == ".pdf" else ChatAttachment.Kind.IMAGE
    att = ChatAttachment(owner=owner, conversation=conversation, kind=kind, filename=name[:512], size=len(data))
    folder = attachment_dir(att)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        if kind == ChatAttachment.Kind.IMAGE:
            jpeg, att.width, att.height = images.to_jpeg(data)
            file_path(att).write_bytes(jpeg)
        else:
            file_path(att).write_bytes(data)
            att.page_count = images.pdf_page_count(file_path(att))
            if att.page_count > settings.CHAT_ATTACHMENT_MAX_PAGES:
                raise AttachmentError(f"The PDF has {att.page_count} pages; the limit in chat is "
                                      f"{settings.CHAT_ATTACHMENT_MAX_PAGES}. Add it to the library in RAG Lab instead.")
    except (images.ImageError, AttachmentError) as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise AttachmentError(str(exc) if isinstance(exc, AttachmentError) else f"The file could not be read: {exc}.")
    att.save()
    return att


def to_json(att: ChatAttachment | None, page: int | None = None) -> dict | None:
    if att is None:
        return None
    return {
        "id": str(att.id), "kind": att.kind, "filename": att.filename, "size": att.size,
        "page_count": att.page_count, "width": att.width, "height": att.height, "page": page,
    }


# ---------------------------------------------------------------------
# What a chat turn uses
# ---------------------------------------------------------------------

def page_jpeg(att: ChatAttachment, page: int) -> bytes:
    """A PDF page as the JPEG the VLM gets (rendered once, then cached)."""
    path = page_path(att, page)
    if not path.exists():
        jpeg, _, _ = images.render_pdf_page(file_path(att), page)
        path.write_bytes(jpeg)
    return path.read_bytes()


def image_bytes(att: ChatAttachment, page: int | None) -> bytes | None:
    """The image a message shows to the vision model, if any."""
    if att.kind == ChatAttachment.Kind.IMAGE:
        return file_path(att).read_bytes()
    if page:
        return page_jpeg(att, page)
    return None


def latest_image(conversation: Conversation) -> tuple[ChatAttachment, int | None] | None:
    """The most recent image (or PDF page) shown in the conversation: follow-up questions re-send only this one."""
    msgs = (conversation.messages.filter(role="user", attachment__isnull=False)
            .select_related("attachment").order_by("-created_at"))
    for msg in msgs:
        if msg.attachment.kind == ChatAttachment.Kind.IMAGE or msg.attachment_page:
            return msg.attachment, msg.attachment_page
    return None


def document_text(att: ChatAttachment, wait: float | None = None) -> str:
    """Markdown of a PDF attachment: parsed with Docling on first use, then cached with the file."""
    path = text_path(att)
    if path.exists():
        return path.read_text(encoding="utf-8")
    result = convert_file(file_path(att).read_bytes(), att.filename, wait=wait)
    text = ((result.get("document") or {}).get("md_content") or "").strip()
    path.write_text(text, encoding="utf-8")
    return text


def text_documents(conversation: Conversation) -> list[ChatAttachment]:
    """PDFs of the conversation that were attached to be read as text, newest first."""
    ids: list = []
    msgs = (conversation.messages.filter(role="user", attachment__kind=ChatAttachment.Kind.PDF,
                                         attachment_page__isnull=True).order_by("-created_at"))
    for attachment_id in msgs.values_list("attachment_id", flat=True):
        if attachment_id not in ids:
            ids.append(attachment_id)
    by_id = ChatAttachment.objects.in_bulk(ids)
    return [by_id[i] for i in ids if i in by_id]


def build_document_context(docs: list[ChatAttachment], budget_chars: int,
                           wait: float | None = None) -> tuple[str, list[dict]]:
    """Prompt block with the attached PDFs' text, whole pages first-to-last within `budget_chars`.

    Returns (block or "", citations). A citation says how many pages were read, so the
    UI can tell the user when a long document was cut.
    """
    blocks: list[str] = []
    citations: list[dict] = []
    remaining = budget_chars
    for att in docs:
        text = document_text(att, wait=wait)
        pages = [p.strip() for p in text.split(PAGE_BREAK)] if text else []
        header = f"{ATTACHED_HEADER}: {att.filename}\n"
        remaining -= len(header)
        taken: list[str] = []
        for i, page in enumerate(pages):
            piece = f"[page {i + 1}]\n{page}\n"
            if len(piece) > remaining:
                if not taken and not blocks and remaining >= MIN_PAGE_CHARS:
                    taken.append(piece[:remaining])
                    remaining = 0
                break
            taken.append(piece)
            remaining -= len(piece)
        if not taken:
            if not pages:
                citations.append(_citation(att, 0, 0))     # nothing readable (e.g. an empty scan)
            break
        blocks.append(header + "\n".join(taken))
        citations.append(_citation(att, len(taken), len(pages)))
        if len(taken) < len(pages):
            break
    if not blocks:
        return "", citations
    footer = "---\nThe user attached the document(s) above to this conversation. Use them when answering."
    return "\n\n".join(blocks) + "\n" + footer, citations


def _citation(att: ChatAttachment, pages_read: int, pages_total: int) -> dict:
    return {
        "kind": "attachment", "source": att.filename, "attachment_id": str(att.id),
        "pages_read": pages_read, "page_count": pages_total or att.page_count,
        "truncated": pages_read < (pages_total or 0),
    }


# ---------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------

def _owned(pk):
    from .views import get_current_user
    return ChatAttachment.objects.filter(pk=pk, owner=get_current_user()).first()


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload(request):
    """POST /api/chat/attachments/ (multipart: file, conversation_id?) → the attachment."""
    from .views import get_current_user

    uploaded = request.FILES.get("file")
    if uploaded is None:
        return Response({"error": "file is required"}, status=status.HTTP_400_BAD_REQUEST)
    owner = get_current_user()
    conversation = None
    if request.data.get("conversation_id"):
        conversation = Conversation.objects.filter(pk=request.data["conversation_id"], owner=owner).first()
        if conversation is None:
            return Response({"error": "conversation not found"}, status=status.HTTP_404_NOT_FOUND)
    try:
        remove_orphans()
    except Exception:
        logger.exception("could not remove unsent chat attachments")
    try:
        att = create(uploaded, owner, conversation)
    except AttachmentError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    logger.info("chat attachment id=%s kind=%s size=%d pages=%s", att.id, att.kind, att.size, att.page_count)
    return Response(to_json(att), status=status.HTTP_201_CREATED)


@api_view(["DELETE"])
def detail(request, pk):
    """DELETE /api/chat/attachments/<uuid>/ — only before it was sent with a message."""
    att = _owned(pk)
    if att is None:
        return Response({"error": "attachment not found"}, status=status.HTTP_404_NOT_FOUND)
    if Message.objects.filter(attachment=att).exists():
        return Response({"error": "The attachment is part of a conversation; delete the conversation to remove it."},
                        status=status.HTTP_409_CONFLICT)
    att.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(["GET"])
def file(request, pk):
    """GET /api/chat/attachments/<uuid>/file/"""
    att = _owned(pk)
    if att is None or not file_path(att).exists():
        return Response({"error": "attachment not found"}, status=status.HTTP_404_NOT_FOUND)
    content_type = "image/jpeg" if att.kind == ChatAttachment.Kind.IMAGE else "application/pdf"
    return FileResponse(file_path(att).open("rb"), content_type=content_type, filename=att.filename)


@api_view(["GET"])
def page(request, pk, number: int):
    """GET /api/chat/attachments/<uuid>/pages/<n>/ — a PDF page as JPEG."""
    att = _owned(pk)
    if att is None or att.kind != ChatAttachment.Kind.PDF:
        return Response({"error": "attachment not found"}, status=status.HTTP_404_NOT_FOUND)
    try:
        page_jpeg(att, number)
    except images.ImageError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    return FileResponse(page_path(att, number).open("rb"), content_type="image/jpeg")

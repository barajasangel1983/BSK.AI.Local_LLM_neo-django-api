"""P29d: an approved report becomes knowledge.

On approval the report (the template in problems.report, as Markdown) is added to the document library, in
`<publish folder>/<machine>`, with its embeddings queued; when the ticket is completed (every action done and
validated) that document is replaced with the final report: the chat can find it the next time the problem happens. Its graph facts are not written here:
the document goes through the normal extraction and review in GraphLab, like any other document.

A ticket closed without analysis is never published. With PROBLEMS_PUBLISH=0 (a preview) nothing is written.
"""

from __future__ import annotations

import logging
import os

from django.core.files.uploadedfile import SimpleUploadedFile

from . import context
from .models import ProblemSettings, Ticket

logger = logging.getLogger("chat")

PICTURE_PROMPT = ("This picture was taken on the plant floor for a problem-solving report. Describe what it shows "
                  "that matters for the problem (parts, damage, wear, leaks, readings), in two or three sentences.")


def describe_pictures(ticket: Ticket) -> None:
    """What the vision model sees in each picture not yet described (best effort: the BSK GPU may be busy)."""
    from gpu import vlm
    from . import service
    for pic in ticket.pictures.filter(description=""):
        try:
            data = service.picture_bytes(ticket.pk, pic.pk)
            pic.description = vlm.complete(data, PICTURE_PROMPT + (f" The engineer's caption: {pic.caption}" if pic.caption else ""),
                                           max_tokens=200, purpose="problem-picture", wait=20).strip()[:2000]
            pic.save(update_fields=["description"])
        except Exception as exc:
            logger.warning("problems: picture %s of %s not described (%s)", pic.pk, ticket.number, exc)


def markdown(ticket: Ticket) -> str:
    from .report import markdown as template
    return template(ticket)


def publish(ticket: Ticket) -> Ticket:
    """Approved: the report goes into the library. Completed: the document is replaced with the final report."""
    completed = ticket.status == Ticket.Status.COMPLETED
    if ticket.status not in (Ticket.Status.APPROVED, Ticket.Status.COMPLETED) or (ticket.document_id and not completed):
        return ticket
    if os.getenv("PROBLEMS_PUBLISH", "1") != "1":
        ticket.publish_error = "Publishing is switched off here (PROBLEMS_PUBLISH=0)."
        ticket.save(update_fields=["publish_error", "updated_at"])
        return ticket
    from ingestion import folders, library
    from ingestion.models import Document
    old = ticket.document_id
    try:
        describe_pictures(ticket)
        s = ProblemSettings.get()
        folder = folders.get_by_path(f"{s.publish_folder}/{ticket.asset_name or ticket.asset_id.split(':')[-1]}", create=True)
        stamp = ticket.completed_at if completed else ticket.approved_at
        name = f"{ticket.number} {ticket.asset_name} {stamp:%Y-%m-%d}{' completed' if completed else ''}.md".replace("/", "-")
        upload = SimpleUploadedFile(name, markdown(ticket).encode("utf-8"), content_type="text/markdown")
        doc, _ = library.add_document(upload, folder=folder, then_embed=True, title=f"{ticket.number}: {ticket.title}",
                                      plant_asset=ticket.asset_id)
        ticket.document_id, ticket.publish_error = str(doc.pk), ""
        if old and old != str(doc.pk):                      # the approved version gives way to the completed one
            previous = Document.objects.filter(pk=old).first()
            if previous is not None:
                library.delete_document(previous)
        logger.info("problems: %s published as document %s%s", ticket.number, doc.pk, f" (replaces {old})" if old else "")
    except Exception as exc:
        logger.exception("problems: %s could not be published", ticket.number)
        ticket.publish_error = f"{type(exc).__name__}: {exc}"[:500]
    ticket.save(update_fields=["document_id", "publish_error", "updated_at"])
    return ticket

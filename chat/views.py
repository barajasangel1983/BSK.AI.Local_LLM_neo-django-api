# chat/views.py
# Views for the chat API and RAG lab.
# Endpoints:
# - GET  /api/ping/
# - POST /api/chat/
# - GET  /api/conversations/
# - GET  /api/conversations/<uuid>/
# - PATCH /api/conversations/<uuid>/   (rename: {"title": ...})
# - DELETE /api/conversations/<uuid>/
# - GET  /api/models/
# - /api/chat/attachments/...   (chat/attachments.py)
# - POST /api/rag/query/
# - POST /api/rag/upload/   (legacy; RAG Lab now uploads via /api/rag/ingest/)
# - GET  /api/usage/summary/
# - GET  /api/health/...
# RAG Lab document/chunk/config endpoints live in chat/rag_lab_views.py.

import logging
import os
import time
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db.models import Max, Count
from django.http import JsonResponse
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from pathlib import Path
from os import getenv

from usage import recorder as usage

from .models import Conversation, Message
from .retrieval import search_v2
from .serializers import ConversationSummarySerializer, ConversationDetailSerializer
from .titles import DEFAULT_TITLE, RENAME_MAX_CHARS, generate_title, normalize_title

logger = logging.getLogger("chat")

FACTORY_KEYWORDS = ["extruder", "extr01", "extr1", "shift", "oee", "throughput", "downtime", "alarm"]


def is_factory_question(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in FACTORY_KEYWORDS)

# Legacy retrieval over the old `bsk_rag` collection, only used for
# plc_historian shift summaries (factory questions); documents are retrieved
# from bsk_rag_v2 via chat.retrieval.search_v2.
from .legacy_retrieval import query_chunks
from context_graph.driver import GraphUnavailable

from . import asset_context
from . import attachments
from gpu import orchestrator as gpu
from gpu import vlm
from ingestion.docling_client import DoclingError


# Base directory for RAG uploads (raw docs). For now, point directly at the
# GraphRAG repo's data/raw directory; later we can parameterize this further.
RAG_UPLOAD_BASE = Path(
    os.getenv(
        "RAG_UPLOAD_BASE",
        "/home/barajas_angel/repos/BSK.AI.Local_LLM_neo4j-graphrag/data/raw",
    )
)


def get_current_user():
    """TEMP: single-user fallback until real auth is wired.

    For now we always use (or create) an 'admin' user.
    """

    user, _ = User.objects.get_or_create(username="admin", defaults={"is_staff": True})
    return user


# ---------------------------------------------------------------------
# Simple health check view
# ---------------------------------------------------------------------


def ping(request):
    """Basic non-DRF view for quick health checks.

    Called by: GET /api/ping/
    """

    return JsonResponse({"status": "ok", "message": "Neo LLM API is alive"})


# ---------------------------------------------------------------------
# Dummy model backend for v0
# ---------------------------------------------------------------------


def generate_dummy_reply(message: str, model: str, use_rag: bool) -> str:
    """Temporary fake LLM backend used for non-Grok models.

    For now, we just echo the message and note model + RAG mode.
    """

    rag_text = " with RAG" if use_rag else ""
    return f"[{model}{rag_text}] Echo: {message}"


def call_grok_chat(messages: list[dict]) -> str:
    """Call xAI Grok chat completions and return the reply text.

    Uses GROK_* settings from neo_llm_api.settings.
    """

    from django.conf import settings

    api_key = settings.GROK_API_KEY
    if not api_key:
        raise RuntimeError("GROK_API_KEY is not set")

    base_url = settings.GROK_API_BASE.rstrip("/")
    model = settings.GROK_CHAT_MODEL
    url = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    with usage.track(None, "external-gpt") as call:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Grok follows the OpenAI-style response format
        reply = data["choices"][0]["message"]["content"]
        call.from_response(data)
        call.estimate(usage.messages_text(messages), reply)
    return reply


def call_dgx_gpt_oss_20b(messages: list[dict]) -> str:
    """Call DGX vLLM server hosting openai/gpt-oss-20b and return reply text."""

    from django.conf import settings

    base_url = settings.DGX_API_BASE.rstrip("/")
    model = settings.DGX_CHAT_MODEL
    url = f"{base_url}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 2048,
    }

    # Asset-scoped prompts are longer and the DGX is shared: 60 s was too tight.
    with usage.track(None, f"dgx-{model}") as call:
        resp = requests.post(url, headers=headers, json=payload, timeout=settings.DGX_CHAT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        call.from_response(data)
        call.estimate(usage.messages_text(messages), reply)
    return reply


def call_ollama_qwen3_8b(messages: list[dict]) -> str:
    """Call local Ollama running qwen3:8b on the bsk-ai machine.

    Uses Ollama chat API over Tailscale. Override with OLLAMA_BASE_URL env var.
    """

    base_url = os.getenv("OLLAMA_BASE_URL", "http://100.76.107.3:11434")
    url = f"{base_url}/api/chat"

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "qwen3:8b",
        "messages": messages,
        "stream": False,
    }

    with usage.track(None, "ollama-qwen3-8b") as call:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Ollama chat API returns the final message under data["message"]["content"]
        reply = data["message"]["content"]
        call.from_response(data)
        call.estimate(usage.messages_text(messages), reply)
    return reply


# ---------------------------------------------------------------------
# Chat context: history window + RAG sources footer
# ---------------------------------------------------------------------

DGX_MODEL_IDS = ("dgx-gpt-oss-20b", "dgx-qwen38-27b-fp8")

DEFAULT_SYSTEM_PROMPTS = {
    "dgx": "You are Neo, an assistant running on the DGX Spark box.",
    "external-gpt": "You are Neo, an AI assistant helping build and debug a local LLM stack.",
    "ollama-qwen3-8b": "You are Neo running on Ollama (Qwen3 8B) on Angel's local PC.",
}

# Plain-text sources footer that assistant replies carried before citations
# moved to Message.sources. No longer written, but older stored replies still
# contain it, so it is stripped from history.
RAG_FOOTER_SEPARATOR = "\n\n---\nSources (RAG): "


def strip_rag_footer(content: str) -> str:
    """Remove the legacy RAG sources footer so it is not fed back to the model."""

    idx = content.rfind(RAG_FOOTER_SEPARATOR)
    return content[:idx] if idx != -1 else content


def context_budget_chars(model_id: str) -> int:
    """Total prompt budget (characters) for a model's context window."""

    from django.conf import settings

    if model_id == "ollama-qwen3-8b":
        return settings.CHAT_CONTEXT_MAX_CHARS_OLLAMA
    return settings.CHAT_CONTEXT_MAX_CHARS


RAG_CONTEXT_HEADER = (
    "You are Neo, an assistant helping with Django / RAG / DGX and factory analytics questions.\n"
    "\nHere is relevant context from the knowledge base:"
)
RAG_CONTEXT_FOOTER = "---\nUse this context when answering the user."
# Smallest useful slice when the top chunk alone exceeds the RAG budget.
RAG_MIN_CHUNK_CHARS = 200


def citation_label(entry: dict) -> str:
    """Human-readable source label, e.g. 'paper.pdf — 3 Model Architecture, p.3'."""

    label = entry.get("source") or "unknown"
    section_path = entry.get("section_path") or []
    # The first element is usually the document title; show the section below it.
    section = section_path[-1] if len(section_path) > 1 else ""
    if section:
        label = f"{label} — {section}"
    if entry.get("page_start"):
        label = f"{label}, p.{entry['page_start']}"
    return label


def to_citation(entry: dict) -> dict:
    """Citation stored on the assistant Message and returned to the UI."""

    return {
        "kind": "document",
        "source": entry.get("source", ""),
        "asset_id": entry.get("asset_id", ""),
        "section_path": entry.get("section_path") or [],
        "page_start": entry.get("page_start"),
        "page_end": entry.get("page_end"),
        "content_type": entry.get("content_type", ""),      # "figure" = a VLM figure description
        "snippet": (entry.get("text") or "").strip()[:200],
        "score": entry.get("score"),
        "vector_score": entry.get("vector_score"),
        "rerank_score": entry.get("rerank_score"),
    }


def build_rag_context(entries: list[dict], budget_chars: int, header: str = RAG_CONTEXT_HEADER,
                      footer: str = RAG_CONTEXT_FOOTER) -> tuple[str | None, list[dict]]:
    """Build the RAG system prompt from ranked entries within `budget_chars`.

    Entries are added in order; ones that don't fit are skipped. If even the
    top entry doesn't fit, it is truncated so RAG still contributes context.
    Returns (system_prompt or None, entries actually used).
    """

    remaining = budget_chars - len(header) - len(footer) - 2
    blocks: list[str] = []
    used: list[dict] = []
    for entry in entries:
        label = f"[{len(used) + 1}] (source: {citation_label(entry)})\n"
        text = (entry.get("text") or "").strip()
        cost = len(label) + len(text) + 2
        if cost > remaining:
            if used:
                continue
            room = remaining - len(label) - 2
            if room < RAG_MIN_CHUNK_CHARS:
                break
            text = text[:room]
            cost = len(label) + len(text) + 2
        blocks.append(f"{label}{text}\n")
        used.append(entry)
        remaining -= cost

    if not used:
        return None, []
    return "\n".join([header, *blocks, footer]), used


def build_chat_messages(
    history: list[tuple[str, str]],
    user_message: str,
    system_prompt: str,
    max_messages: int,
    max_chars: int,
) -> list[dict]:
    """Build an OpenAI-style messages list: system + recent history + user.

    `history` is (role, content) pairs, oldest first, NOT including the
    current user message. History fills whatever budget is left after the
    system prompt and current message; the oldest messages are dropped first,
    whole messages only. User messages that never got a reply (e.g. the model
    call failed) are skipped so turns stay user/assistant alternating, and the
    window never starts with an assistant message.
    """

    turns = [(r, c) for r, c in history if r in ("user", "assistant")]
    # Drop user messages not followed by an assistant reply.
    turns = [
        (r, c)
        for i, (r, c) in enumerate(turns)
        if r != "user" or (i + 1 < len(turns) and turns[i + 1][0] == "assistant")
    ]

    budget = max_chars - len(system_prompt) - len(user_message)
    selected: list[dict] = []
    for role, content in reversed(turns):
        if len(selected) >= max_messages:
            break
        if role == "assistant":
            content = strip_rag_footer(content)
        if len(content) > budget:
            break
        selected.append({"role": role, "content": content})
        budget -= len(content)

    selected.reverse()
    while selected and selected[0]["role"] == "assistant":
        selected.pop(0)

    return [
        {"role": "system", "content": system_prompt},
        *selected,
        {"role": "user", "content": user_message},
    ]


def generate_reply_backend(
    message: str,
    model_id: str,
    use_rag: bool,
    history: list[tuple[str, str]],
    system_prompt: str | None = None,
    extra_context: str = "",
) -> tuple[str, int]:
    """Central routing for model calls.

    - 'external-gpt' -> Grok / xAI backend.
    - 'dgx-qwen38-27b-fp8' (or legacy 'dgx-gpt-oss-20b') -> DGX Spark vLLM backend.
    - 'ollama-qwen3-8b' -> Ollama on the bsk-ai machine.
    - anything else -> dummy echo backend for now.

    `history` is the conversation's prior (role, content) messages, oldest
    first. When `use_rag` is True, callers can pass a `system_prompt` that
    already includes RAG context; otherwise each backend's default is used.
    `extra_context` (attached-document text) is appended to whichever prompt is used.

    Returns (reply_text, history_messages_sent).
    """

    from django.conf import settings

    if model_id in DGX_MODEL_IDS:
        backend, default_prompt = call_dgx_gpt_oss_20b, DEFAULT_SYSTEM_PROMPTS["dgx"]
    elif model_id in ("external-gpt", "ollama-qwen3-8b"):
        backend = call_grok_chat if model_id == "external-gpt" else call_ollama_qwen3_8b
        default_prompt = DEFAULT_SYSTEM_PROMPTS[model_id]
    else:
        # Default: dummy echo
        return generate_dummy_reply(message, model_id, use_rag), 0

    messages = build_chat_messages(
        history=history,
        user_message=message,
        system_prompt=(system_prompt or default_prompt) + (f"\n\n{extra_context}" if extra_context else ""),
        max_messages=settings.CHAT_HISTORY_MAX_MESSAGES,
        max_chars=context_budget_chars(model_id),
    )
    return backend(messages), len(messages) - 2


VLM_SYSTEM_PROMPT = ("You are Neo, an assistant for industrial engineers. When an image is attached, answer from what "
                     "is visible in it: read labels, tag numbers and values exactly, and say so when something is "
                     "not legible.")


def generate_vlm_reply(message: str, history: list[tuple[str, str]], jpeg: bytes | None) -> tuple[str, int]:
    """Reply from the vision model on BSK: a short text history plus at most one image
    (on the current message). Raises gpu.GpuBusy / gpu.GpuUnavailable."""

    from django.conf import settings

    messages = build_chat_messages(
        history=history,
        user_message=message,
        system_prompt=VLM_SYSTEM_PROMPT,
        max_messages=settings.CHAT_HISTORY_MAX_MESSAGES,
        max_chars=settings.VLM_CONTEXT_MAX_CHARS,
    )
    if jpeg:
        messages[-1] = {"role": "user", "content": [{"type": "text", "text": message}, vlm.image_part(jpeg)]}
    reply = vlm.chat(messages, max_tokens=settings.VLM_CHAT_MAX_TOKENS, wait=settings.VLM_CHAT_LOCK_WAIT)
    return reply, len(messages) - 2


def _gpu_response(exc: Exception) -> Response:
    """503 for the two ways the BSK GPU can be unavailable to a chat request."""
    busy = isinstance(exc, gpu.GpuBusy) or isinstance(exc.__cause__, gpu.GpuBusy)
    if busy:
        return Response({"error": "The GPU on BSK is busy with another job. Try again in a few minutes.",
                         "code": "gpu_busy"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    return Response({"error": f"The BSK PC isn't reachable or its service didn't start: {exc}",
                     "code": "gpu_unavailable"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def rag_upload(request):
    """Upload one or more files for RAG ingestion.

    v0 behavior:
    - Save uploaded files under the GraphRAG repo's data/raw directory.
    - Return basic metadata about saved files.
    - Ingestion into Chroma is still triggered separately (e.g. via a script).
    """

    files = request.FILES.getlist("files")
    visibility = request.data.get("visibility", "private")  # "private" or "public"

    if not files:
        return Response(
            {"error": "No files uploaded (expected 'files' form field)"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    saved = []

    RAG_UPLOAD_BASE.mkdir(parents=True, exist_ok=True)

    for f in files:
        target_path = RAG_UPLOAD_BASE / f.name

        with target_path.open("wb+") as dest:
            for chunk in f.chunks():
                dest.write(chunk)

        saved.append(
            {
                "name": f.name,
                "size": f.size,
                "content_type": f.content_type,
                "path": str(target_path),
                "visibility": visibility,
            }
        )

    # Optional: auto-ingest if configured
    AUTO_INGEST = getenv("RAG_AUTO_INGEST", "false").lower() == "true"

    chunks_added = 0
    if AUTO_INGEST and settings.CHROMA_HOST:
        # The GraphRAG repo's ingest_files opens the Chroma folder directly, which
        # isn't safe while the Chroma server owns it. Use the document library.
        logger.warning("RAG_AUTO_INGEST ignored in Chroma server mode; use /api/documents/ instead")
    elif AUTO_INGEST:
        try:
            from rag.ingestion import ingest_files

            owner = get_current_user()
            paths = [f["path"] for f in saved]
            # For now we record owner_user_id + visibility in metadata;
            # retrieval is still global, but metadata prepares us for
            # per-user / visibility-aware filtering.
            chunks_added = ingest_files(
                paths,
                owner_user_id=str(owner.id),
                visibility=str(visibility),
            )
        except Exception as e:
            # Log the error but don't fail the upload
            logger.exception("Auto-ingestion failed: %s", e)

    return Response(
        {
            "message": "Files uploaded successfully (ingestion is a separate step)",
            "files": saved,
            "chunks_added": chunks_added,
        },
        status=status.HTTP_201_CREATED,
    )


# ---------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------


CHAT_PURPOSES = ("chat", "compare", "regenerate")


@api_view(["POST"])
def chat_view(request):
    """POST /api/chat/ — see _chat. `purpose` (chat | compare | regenerate) labels the model calls for Analytics."""
    purpose = request.data.get("purpose")
    with usage.scope(purpose=purpose if purpose in CHAT_PURPOSES else "chat"):
        return _chat(request)


def _chat(request):
    """POST /api/chat/

    Body:
    {
      "conversation_id": "uuid or null",
      "message": "user text",
      "model": "local-small",
      "use_rag": true/false   (default true)
      "asset_id": "bsk:asset:EXTR01" | "" | absent   (scope; absent = keep the conversation's)
      "attachment_id": "uuid"   (optional; an image or PDF uploaded to /api/chat/attachments/)
      "page": 3                 (optional; show this page of the PDF to the vision model)
    }

    Attachments stay with the conversation (never added to the library / RAG):
    - an image, or a PDF page, is answered by the vision model (bsk-qwen3-vl-4b); follow-up
      questions to that model re-send the most recent image only
    - a PDF without `page` is read as text (Docling, cached) and added to the prompt of the
      text models for this message and the following ones

    Behavior:
    - If conversation_id is null -> create a new Conversation.
    - Else -> load existing Conversation (404 if not found).
    - Save a user Message.
    - With RAG: retrieve from bsk_rag_v2 (plus plc_historian summaries from the
      legacy bsk_rag for factory questions) within the model's RAG budget.
    - Generate the assistant reply and save it with its RAG citations.
    - Return the full Conversation (with messages[] incl. sources).
    """

    conversation_id = request.data.get("conversation_id")
    user_message = request.data.get("message")
    model_id = request.data.get("model", "local-small")
    use_rag = bool(request.data.get("use_rag", True))
    requested_asset = request.data.get("asset_id")

    if not user_message:
        return Response(
            {"error": "message is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Asset scope: validated before anything is saved.
    if requested_asset:
        try:
            asset_context.check_asset(str(requested_asset))
        except asset_context.AssetScopeError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except GraphUnavailable as exc:
            return Response({"error": f"Context Graph unavailable: {exc}"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    owner = get_current_user()
    is_vlm = model_id == settings.VLM_CHAT_MODEL_ID

    # --- Attachment: validated before anything is saved ---

    attachment, attachment_page = None, None
    if request.data.get("attachment_id"):
        try:
            attachment = attachments.ChatAttachment.objects.filter(pk=request.data["attachment_id"], owner=owner).first()
        except (ValueError, ValidationError):
            attachment = None
        if attachment is None:
            return Response({"error": "attachment not found"}, status=status.HTTP_404_NOT_FOUND)
        if attachment.conversation_id and str(attachment.conversation_id) != str(conversation_id or ""):
            return Response({"error": "the attachment belongs to another conversation"},
                            status=status.HTTP_400_BAD_REQUEST)
        page_raw = request.data.get("page")
        if page_raw not in (None, ""):
            try:
                attachment_page = int(page_raw)
            except (TypeError, ValueError):
                attachment_page = 0
            if attachment.kind != attachments.ChatAttachment.Kind.PDF or not (
                    1 <= attachment_page <= (attachment.page_count or 0)):
                return Response({"error": f"page must be between 1 and {attachment.page_count or 1} of a PDF attachment"},
                                status=status.HTTP_400_BAD_REQUEST)
        shows_image = attachment.kind == attachments.ChatAttachment.Kind.IMAGE or bool(attachment_page)
        if shows_image and not is_vlm:
            return Response({"error": "This model can't read images. Choose Qwen3-VL 4B (BSK).",
                             "code": "vision_model_required"}, status=status.HTTP_400_BAD_REQUEST)
        if not shows_image and is_vlm:
            return Response({"error": "The vision model reads one page at a time: choose a page of the PDF, "
                                      "or pick a text model to read the whole document.",
                             "code": "text_model_required"}, status=status.HTTP_400_BAD_REQUEST)

    # --- Get or create conversation ---

    if conversation_id:
        try:
            conversation = Conversation.objects.get(id=conversation_id, owner=owner)
        except Conversation.DoesNotExist:
            return Response(
                {"error": "conversation not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
    else:
        conversation = Conversation.objects.create(owner=owner, title=DEFAULT_TITLE)

    usage.set_conversation(conversation.id)

    if attachment is not None and attachment.conversation_id is None:
        attachment.conversation = conversation
        attachment.save(update_fields=["conversation"])

    # --- Prior messages (before saving this one), oldest first ---

    # Fetch a little more than the window so orphaned user messages can be skipped.
    recent = conversation.messages.order_by("-created_at").values_list("role", "content")[
        : settings.CHAT_HISTORY_MAX_MESSAGES * 2
    ]
    history = list(reversed(recent))

    # --- Save user message ---

    Message.objects.create(
        conversation=conversation,
        role="user",
        content=user_message,
        attachment=attachment,
        attachment_page=attachment_page,
    )

    if requested_asset is not None and conversation.asset_id != (requested_asset or ""):
        conversation.asset_id = requested_asset or ""
        conversation.save(update_fields=["asset_id"])
    scope = conversation.asset_id

    total_budget = context_budget_chars(model_id)

    # --- Attached PDFs read as text (this message's and earlier ones in the conversation) ---

    document_block, attachment_citations = "", []
    reads_documents = model_id in DGX_MODEL_IDS or model_id in ("external-gpt", "ollama-qwen3-8b")
    if reads_documents:
        text_docs = attachments.text_documents(conversation)
        if text_docs:
            try:
                document_block, attachment_citations = attachments.build_document_context(
                    text_docs, int(total_budget * settings.CHAT_ATTACHMENT_SHARE), wait=settings.VLM_CHAT_LOCK_WAIT)
            except gpu.GpuError as exc:
                return _gpu_response(exc)
            except DoclingError as exc:
                logger.warning("attachment parse failed conversation=%s: %s", conversation.id, exc)
                if isinstance(exc.__cause__, gpu.GpuError):
                    return _gpu_response(exc.__cause__)
                return Response({"error": f"The attached PDF could not be read: {exc}"},
                                status=status.HTTP_502_BAD_GATEWAY)
            # Asset facts and RAG share what is left.
            total_budget -= len(document_block)

    # The vision model's context is small: no asset facts or RAG for it.
    if is_vlm:
        scope, use_rag = "", False

    # --- Asset facts (independent of the RAG toggle) ---

    asset_ctx = None
    asset_note = ""
    if scope:
        facts_budget, _ = asset_context.budgets(total_budget)
        try:
            asset_ctx = asset_context.build(scope, user_message, facts_budget)
        except (GraphUnavailable, asset_context.AssetScopeError) as exc:
            logger.warning("asset context failed conversation=%s asset=%s: %s", conversation.id, scope, exc)
            asset_note = (f"You are Neo, the assistant for asset {scope}. Its facts are unavailable right now "
                          f"(Context Graph unreachable); say so if the question needs them.")

    # --- Optional RAG context ---

    system_prompt: str | None = None
    citations: list[dict] = []
    rag_block: str | None = None
    if use_rag:
        rag_start = time.time()
        entries: list[dict] = []
        v2_top_n = settings.RAG_CHAT_TOP_N

        # Asset-scoped chats already carry the asset's historian values in its fact sheet;
        # the legacy shift summaries would only use up document budget there.
        if is_factory_question(user_message) and query_chunks is not None and not asset_ctx:
            # Historian-first: plc_historian shift summaries live in the legacy
            # bsk_rag collection; documents come second from v2.
            try:
                for c in query_chunks(query=user_message, top_k=5, where={"source": "plc_historian"}):
                    entries.append({"text": c.text, "source": c.source})
            except Exception:
                logger.exception("historian retrieval failed conversation=%s", conversation.id)
            v2_top_n = 3

        reranker = "error"
        doc_scope = "library"
        try:
            relevant = lambda result: [  # noqa: E731
                c.to_dict() for c in result.chunks
                if c.rerank_score is None or c.rerank_score >= settings.RAG_MIN_RERANK_SCORE
            ]
            found: list[dict] = []
            if asset_ctx and asset_ctx.doc_keys:
                # The asset's own documents first; the whole library when none of them match.
                result = search_v2(user_message, top_n=v2_top_n, doc_keys=asset_ctx.doc_keys)
                found, reranker, doc_scope = relevant(result), result.reranker, "asset"
            if not found:
                result = search_v2(user_message, top_n=v2_top_n)
                found, reranker = relevant(result), result.reranker
                doc_scope = "library"
            entries.extend(found)
        except Exception:
            logger.exception("v2 retrieval failed conversation=%s", conversation.id)

        if asset_ctx:
            _, rag_budget = asset_context.budgets(total_budget)
            rag_block, used = build_rag_context(entries, rag_budget, header="DOCUMENT EXCERPTS:", footer="---")
        else:
            rag_budget = int(total_budget * settings.RAG_CONTEXT_SHARE)
            system_prompt, used = build_rag_context(entries, rag_budget)
            rag_block = system_prompt
        citations = [to_citation(e) for e in used]
        logger.info(
            "rag conversation=%s model=%s retrieved=%d used=%d reranker=%s "
            "rag_chars=%d/%d doc_scope=%s latency_ms=%d",
            conversation.id, model_id, len(entries), len(used), reranker,
            len(rag_block or ""), rag_budget, doc_scope, round((time.time() - rag_start) * 1000),
        )

    if asset_ctx:
        system_prompt = asset_context.system_prompt(asset_ctx) + (f"\n\n{rag_block}" if rag_block else "")
        citations = asset_context.citations(asset_ctx) + citations
        logger.info("asset context conversation=%s asset=%s facts=%d/%d chars=%d historian=%s docs_linked=%d",
                    conversation.id, scope, len(asset_ctx.facts), asset_ctx.total_facts, len(asset_ctx.text),
                    bool(asset_ctx.historian), len(asset_ctx.doc_keys))
    elif asset_note:
        system_prompt = asset_note + (f"\n\n{rag_block}" if rag_block else "")

    # --- Generate assistant reply ---

    citations = attachment_citations + citations

    start = time.time()
    try:
        if is_vlm:
            shown = (attachment, attachment_page) if attachment is not None else attachments.latest_image(conversation)
            jpeg = attachments.image_bytes(*shown) if shown else None
            assistant_reply, history_sent = generate_vlm_reply(user_message, history, jpeg)
        else:
            assistant_reply, history_sent = generate_reply_backend(
                message=user_message,
                model_id=model_id,
                use_rag=use_rag or bool(scope),
                history=history,
                system_prompt=system_prompt,
                extra_context=document_block,
            )
    except gpu.GpuError as exc:
        logger.warning("chat gpu unavailable conversation=%s model=%s: %s", conversation.id, model_id, exc)
        return _gpu_response(exc)
    except Exception as exc:
        logger.exception(
            "chat failed conversation=%s model=%s latency_ms=%d",
            conversation.id, model_id, round((time.time() - start) * 1000),
        )
        return Response(
            {"error": f"Model backend failed: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    logger.info(
        "chat ok conversation=%s model=%s use_rag=%s history_sent=%d/%d "
        "rag_context_chars=%d rag_sources=%s latency_ms=%d",
        conversation.id, model_id, use_rag, history_sent, len(history),
        len(system_prompt or ""), [c["source"] for c in citations], round((time.time() - start) * 1000),
    )

    # --- Save assistant message ---

    Message.objects.create(
        conversation=conversation,
        role="assistant",
        content=assistant_reply,
        sources=citations,
    )

    # Track the model used on the conversation for basic analytics
    conversation.model_id = model_id
    update_fields = ["model_id", "updated_at"]

    # Title the conversation from its first exchange. Only untitled conversations
    # without an earlier reply qualify, so a user's rename is never overwritten.
    first_exchange = not any(role == "assistant" for role, _ in history)
    if first_exchange and conversation.title in ("", DEFAULT_TITLE):
        conversation.title, title_source = generate_title(user_message, assistant_reply)
        update_fields.append("title")
        logger.info("title conversation=%s source=%s title=%r", conversation.id, title_source, conversation.title)

    conversation.save(update_fields=update_fields)

    serializer = ConversationDetailSerializer(conversation)
    return Response(serializer.data, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------
# Conversation list + detail
# ---------------------------------------------------------------------


@api_view(["GET"])

def list_conversations(request):
    """GET /api/conversations/

    Returns a list of conversation summaries for the sidebar.
    """

    owner = get_current_user()

    qs = (
        Conversation.objects.filter(owner=owner)
        .annotate(last_message_at=Max("messages__created_at"))
        .order_by("-last_message_at", "-created_at")
    )

    serializer = ConversationSummarySerializer(qs, many=True)
    return Response(serializer.data)


@api_view(["GET", "PATCH", "DELETE"])

def conversation_detail(request, pk):
    """GET/PATCH/DELETE /api/conversations/<uuid:pk>/

    PATCH body: {"title": "new name"} — renames the conversation.
    """

    owner = get_current_user()
    try:
        conversation = Conversation.objects.get(pk=pk, owner=owner)
    except Conversation.DoesNotExist:
        return Response(
            {"error": "conversation not found"},
            status=status.HTTP_404_NOT_FOUND,
        )

    if request.method == "DELETE":
        conversation.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    if request.method == "PATCH":
        title = normalize_title(str(request.data.get("title", "")))
        if not title:
            return Response({"error": "title must not be empty"}, status=status.HTTP_400_BAD_REQUEST)
        conversation.title = title[:RENAME_MAX_CHARS]
        conversation.save(update_fields=["title"])
        return Response(ConversationSummarySerializer(conversation).data)

    serializer = ConversationDetailSerializer(conversation)
    return Response(serializer.data)


# ---------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------


@api_view(["GET"])

def list_models(request):
    """GET /api/models/

    Returns a static list of available model/backend options.
    """

    data = [
        {
            "id": "local-small",
            "label": "Local Small (Dummy)",
            "description": "Placeholder local model for development.",
        },
        {
            "id": "ollama-qwen3-8b",
            "label": "Ollama Qwen3 8B (Local)",
            "description": "Qwen3:8B served via Ollama on the bsk-ai machine.",
        },
        {
            "id": "dgx-qwen38-27b-fp8",
            "label": "DGX Qwen3.8-27B-FP8",
            "description": "DGX Spark vLLM backend for Qwen3.8-27B-FP8.",
        },
        {
            "id": "external-gpt",
            "label": "External GPT (Grok)",
            "description": "xAI Grok backend (external GPT-style API).",
        },
        {
            "id": settings.VLM_CHAT_MODEL_ID,
            "label": "Qwen3-VL 4B (BSK)",
            "description": "Vision model on the BSK PC: answers questions about an attached image or PDF page. "
                           "Starts on demand (about 20 s).",
            "vision": True,
        },
    ]
    return Response(data)


# ---------------------------------------------------------------------
# RAG docs listing + delete + query endpoint (Chroma-backed)
# ---------------------------------------------------------------------


@api_view(["POST"])

def rag_query(request):
    """POST /api/rag/query/

    Retrieval debugger for the RAG Lab: same path as chat (bsk_rag_v2 vector
    search + DGX rerank, falling back to vector order if the reranker is down).

    Request body (JSON):
    {
      "query": "user question",
      "top_k": 5   # optional (default RAG_QUERY_DEFAULT_TOP_K), clamped to 1..RAG_QUERY_MAX_TOP_K
    }

    Response body:
    {
      "query": "...", "top_k": 5,
      "reranker": "ok" | "fallback", "candidates": 20, "latency_ms": 123,
      "results": [
        {
          "id": "chunk-id", "text": "chunk text",
          "source": "file.pdf", "document_path": "file.pdf",
          "asset_id": "...", "section_path": ["Title", "Section"],
          "page_start": 3, "page_end": 3, "content_type": "paragraph",
          "score": 0.98,          # rerank_score if available, else vector_score
          "vector_score": 0.71,   # cosine similarity (higher is better)
          "rerank_score": 0.98    # null when the reranker fell back
        },
        ...
      ]
    }
    """

    from django.conf import settings

    query = str(request.data.get("query", ""))
    try:
        top_k = int(request.data.get("top_k", settings.RAG_QUERY_DEFAULT_TOP_K))
    except (TypeError, ValueError):
        return Response({"error": "top_k must be an integer"}, status=status.HTTP_400_BAD_REQUEST)
    top_k = max(1, min(top_k, settings.RAG_QUERY_MAX_TOP_K))

    if not query.strip():
        return Response(
            {"error": "query is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        result = search_v2(query, top_n=top_k)
    except Exception as exc:
        logger.exception("rag_query failed")
        return Response(
            {"error": f"RAG query failed: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    logger.info(
        "rag_query top_k=%d candidates=%d returned=%d reranker=%s latency_ms=%d",
        top_k, result.candidates, len(result.chunks), result.reranker, result.latency_ms,
    )
    return Response(
        {
            "query": query,
            "top_k": top_k,
            "reranker": result.reranker,
            "candidates": result.candidates,
            "latency_ms": result.latency_ms,
            "results": [{**c.to_dict(), "document_path": c.source} for c in result.chunks],
        }
    )


# ---------------------------------------------------------------------
# Usage analytics (simple summary)
# ---------------------------------------------------------------------


@api_view(["GET"])
def usage_summary(request):
    """GET /api/usage/summary/?days=7|14|30|90&purpose=<purpose>

    Measured model / AI-service calls (usage.ModelCall): totals, per model, per
    purpose and a daily series with prompt and completion tokens, requests,
    errors and latency. Plus conversations per model (from the chat history).
    """

    from usage.summary import RANGES, summary

    try:
        days = int(request.query_params.get("days", 14))
    except ValueError:
        days = 0
    if days not in RANGES:
        return Response({"error": f"days must be one of {', '.join(map(str, RANGES))}"},
                        status=status.HTTP_400_BAD_REQUEST)

    owner = get_current_user()
    per_model = (
        Conversation.objects.filter(owner=owner)
        .values("model_id")
        .annotate(count=Count("id"))
        .order_by("model_id")
    )
    data = summary(days, request.query_params.get("purpose") or None)
    data["total_conversations"] = sum(row["count"] for row in per_model)
    data["conversations_per_model"] = [
        {"model_id": row["model_id"] or "unknown", "conversations": row["count"]} for row in per_model
    ]
    return Response(data)


# ---------------------------------------------------------------------
# Health monitoring
# ---------------------------------------------------------------------

# In-memory uptime tracker: {endpoint_id: {"total": int, "ok": int}}
_uptime_tracker: dict = defaultdict(lambda: {"total": 0, "ok": 0})

TRACKED_ENDPOINTS = [
    {
        "id": "neo-django-api",
        "name": "Neo Django API",
        "url": "http://127.0.0.1:8000/api/ping/",
        "model": "backend",
        "can_restart": False,
    },
    {
        "id": "dgx-vllm",
        "name": "DGX vLLM (Qwen3.8-27B-FP8)",
        "url": "http://100.74.225.3:8004/v1/models",
        "model": "dgx-qwen38-27b-fp8",
        "can_restart": False,
    },
    # RAG v2: every document chunk and every RAG question is embedded, and retrieved chunks
    # are reranked, on the DGX.
    {
        "id": "dgx-embed",
        "name": f"DGX Embeddings ({settings.DGX_EMBED_MODEL})",
        "url": settings.DGX_EMBED_URL.split("/v1/")[0].rstrip("/") + "/health",
        "model": "embeddings",
        "can_restart": False,
    },
    {
        "id": "dgx-rerank",
        "name": f"DGX Reranker ({settings.DGX_RERANK_MODEL})",
        "url": settings.DGX_RERANK_URL.rsplit("/", 1)[0] + "/health",
        "model": "reranker",
        "can_restart": False,
    },
    {
        "id": "grok-xai",
        "name": "Grok / xAI",
        "url": "https://api.x.ai/v1/models",
        "model": "external-gpt",
        "can_restart": False,
    },
    {
        "id": "ollama-qwen3-8b",
        "name": "Ollama Qwen3 8B (Local)",
        "url": "http://100.76.107.3:11434/api/tags",
        "model": "ollama-qwen3-8b",
        "can_restart": False,
    },
    {
        "id": "rag-chroma",
        "name": "RAG / Chroma",
        "url": "",
        "model": "rag",
        "can_restart": True,
    },
    {
        "id": "historian-db",
        "name": "Historian Postgres (plc_1_historian)",
        "url": "",
        "model": "historian",
        "can_restart": False,
    },
    {
        "id": "context-graph",
        "name": "Context Graph (Neo4j)",
        "url": "",
        "model": "graph",
        "can_restart": False,
    },
    # BSK desktop: one GPU shared by Docling and the VLM, started on demand by the orchestrator.
    {
        "id": "gpu-orchestrator",
        "name": "GPU orchestrator (BSK)",
        "url": "",
        "model": "gpu",
        "can_restart": False,
    },
    {
        "id": "docling-bsk",
        "name": "Docling (BSK)",
        "url": "",
        "model": "docling",
        "can_restart": False,
    },
    {
        "id": "vlm-bsk",
        "name": "VLM Qwen3-VL-4B (BSK)",
        "url": "",
        "model": "vlm",
        "can_restart": False,
    },
]


_BSK_URLS = {
    "gpu-orchestrator": lambda: f"{settings.GPU_ORCHESTRATOR_URL.rstrip('/')}/gpu/status",
    "docling-bsk": lambda: settings.DOCLING_URL,
    "vlm-bsk": lambda: settings.VLM_URL,
}


def _bsk_status(ep_id: str) -> tuple[str, int]:
    """BSK services: never probe the VLM / Docling directly while the orchestrator owns them.

    Returns (status, latency_ms). "idle" = not running by design (started on demand)."""
    from gpu import orchestrator as gpu

    summary = gpu.health_summary()
    if not summary["enabled"]:
        if ep_id == "docling-bsk":  # direct mode (as before the orchestrator)
            start = time.time()
            try:
                ok = requests.get(f"{settings.DOCLING_URL.rstrip('/')}/health", timeout=5).status_code == 200
            except requests.RequestException:
                ok = False
            return ("online" if ok else "offline"), round((time.time() - start) * 1000) if ok else 0
        return "idle", 0     # orchestrator / VLM not in use yet
    if not summary["reachable"]:
        return "offline", 0
    if ep_id == "gpu-orchestrator":
        return ("degraded" if summary.get("health") == "error" else "online"), summary.get("latency_ms", 0)
    service = "docling" if ep_id == "docling-bsk" else "vl"
    if summary.get("active") != service:
        return "idle", 0
    return {"ok": "online", "starting": "degraded"}.get(summary.get("health"), "offline"), summary.get("latency_ms", 0)


def _check_single_endpoint(ep: dict) -> dict:
    """Ping one endpoint and return its status dict."""
    from django.conf import settings
    from django.db import connections

    ep_id = ep["id"]

    if ep_id == "rag-chroma":
        start = time.time()
        try:
            from ingestion.chroma_client import get_client

            get_client().heartbeat()
            latency = round((time.time() - start) * 1000)
            status_val = "online"
        except Exception:
            latency = 0
            status_val = "offline"
    elif ep_id == "context-graph":
        from context_graph.driver import health as graph_health

        result = graph_health()
        # disabled / not_configured / offline all show as offline on the page.
        status_val = "online" if result["status"] == "online" else "offline"
        latency = result.get("latency_ms", 0)
    elif ep_id in ("gpu-orchestrator", "docling-bsk", "vlm-bsk"):
        status_val, latency = _bsk_status(ep_id)
    elif ep_id == "historian-db":
        start = time.time()
        try:
            with connections["historian"].cursor() as cursor:
                cursor.execute("SELECT 1;")
                cursor.fetchone()
            latency = round((time.time() - start) * 1000)
            status_val = "online"
        except Exception:
            latency = 0
            status_val = "offline"
    else:
        url = ep["url"]
        headers = {}
        if ep_id == "grok-xai":
            api_key = getattr(settings, "GROK_API_KEY", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        start = time.time()
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            latency = round((time.time() - start) * 1000)
            status_val = "online" if resp.status_code < 500 and latency < 2000 else "degraded"
        except requests.exceptions.ConnectionError:
            status_val = "offline"
            latency = 0
        except requests.exceptions.Timeout:
            status_val = "degraded"
            latency = 5000
        except Exception:
            status_val = "offline"
            latency = 0

    tracker = _uptime_tracker[ep_id]
    tracker["total"] += 1
    if status_val in ("online", "idle"):   # idle = stopped by design, started on demand
        tracker["ok"] += 1
    uptime = round(tracker["ok"] / tracker["total"] * 100, 1) if tracker["total"] > 0 else 100.0

    return {
        "id": ep_id,
        "name": ep["name"],
        "url": settings.NEO4J_URI if ep_id == "context-graph" else _BSK_URLS.get(ep_id, lambda: ep.get("url", ""))(),
        "model": ep["model"],
        "status": status_val,
        "latency": latency,
        "uptime": uptime,
        "last_checked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "can_restart": ep["can_restart"],
    }


@api_view(["GET"])
def health_status(request):
    """GET /api/health/status/ — check all tracked endpoints in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=len(TRACKED_ENDPOINTS)) as executor:
        futures = {executor.submit(_check_single_endpoint, ep): ep for ep in TRACKED_ENDPOINTS}
        for future in as_completed(futures):
            ep = futures[future]
            try:
                results.append(future.result())
            except Exception:
                results.append({
                    "id": ep["id"],
                    "name": ep["name"],
                    "url": ep.get("url", ""),
                    "model": ep["model"],
                    "status": "offline",
                    "latency": 0,
                    "uptime": 0.0,
                    "last_checked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "can_restart": ep["can_restart"],
                })

    order = {ep["id"]: i for i, ep in enumerate(TRACKED_ENDPOINTS)}
    results.sort(key=lambda x: order.get(x["id"], 99))
    return Response(results)


@api_view(["POST"])
def health_check_one(request, endpoint_id: str):
    """POST /api/health/check/<endpoint_id>/ — re-check a single endpoint."""
    ep = next((e for e in TRACKED_ENDPOINTS if e["id"] == endpoint_id), None)
    if ep is None:
        return Response({"error": f"Unknown endpoint: {endpoint_id}"}, status=status.HTTP_404_NOT_FOUND)
    return Response(_check_single_endpoint(ep))


@api_view(["POST"])
def health_restart(request, endpoint_id: str):
    """POST /api/health/restart/<endpoint_id>/ — attempt restart of a service."""
    ep = next((e for e in TRACKED_ENDPOINTS if e["id"] == endpoint_id), None)
    if ep is None:
        return Response({"error": f"Unknown endpoint: {endpoint_id}"}, status=status.HTTP_404_NOT_FOUND)

    restart_message = "Restart not supported — re-checking connectivity"

    if endpoint_id == "rag-chroma":
        try:
            from ingestion.chroma_client import get_client, legacy_collection

            v2 = get_client().get_collection(name=settings.RAG_COLLECTION_V2).count()
            restart_message = f"Chroma reconnected — {v2} document chunks, {legacy_collection().count()} historian summaries"
        except Exception as exc:
            restart_message = f"Chroma reconnect failed: {exc}"

    result = _check_single_endpoint(ep)
    result["restart_message"] = restart_message
    return Response(result)

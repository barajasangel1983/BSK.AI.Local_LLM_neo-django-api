# chat/views.py
# Views for the chat API and RAG lab.
# Endpoints:
# - GET  /api/ping/
# - POST /api/chat/
# - GET  /api/conversations/
# - GET  /api/conversations/<uuid>/
# - DELETE /api/conversations/<uuid>/
# - GET  /api/models/
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

from django.db.models import Max, Count
from django.http import JsonResponse
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from pathlib import Path
from os import getenv

from .models import Conversation, Message
from .retrieval import search_v2
from .serializers import ConversationSummarySerializer, ConversationDetailSerializer

logger = logging.getLogger("chat")

FACTORY_KEYWORDS = ["extruder", "extr01", "extr1", "shift", "oee", "throughput", "downtime", "alarm"]


def is_factory_question(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in FACTORY_KEYWORDS)

# Legacy retrieval over the old `bsk_rag` collection – imported from GraphRAG repo.
# Only used for plc_historian shift summaries (factory questions); documents
# are retrieved from bsk_rag_v2 via chat.retrieval.search_v2.
# NOTE: this assumes the BSK.AI.Local_LLM_neo4j-graphrag repo is on PYTHONPATH
# when running Django (we can adjust PYTHONPATH in manage.py or venv later).
try:  # pragma: no cover - defensive import
    from rag.retrieval import query_chunks
except Exception:  # pragma: no cover
    query_chunks = None  # type: ignore


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

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Grok follows the OpenAI-style response format
    return data["choices"][0]["message"]["content"]


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

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    return data["choices"][0]["message"]["content"]


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

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Ollama chat API returns the final message under data["message"]["content"]
    return data["message"]["content"]


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
        "source": entry.get("source", ""),
        "asset_id": entry.get("asset_id", ""),
        "section_path": entry.get("section_path") or [],
        "page_start": entry.get("page_start"),
        "page_end": entry.get("page_end"),
        "snippet": (entry.get("text") or "").strip()[:200],
        "score": entry.get("score"),
        "vector_score": entry.get("vector_score"),
        "rerank_score": entry.get("rerank_score"),
    }


def build_rag_context(entries: list[dict], budget_chars: int) -> tuple[str | None, list[dict]]:
    """Build the RAG system prompt from ranked entries within `budget_chars`.

    Entries are added in order; ones that don't fit are skipped. If even the
    top entry doesn't fit, it is truncated so RAG still contributes context.
    Returns (system_prompt or None, entries actually used).
    """

    remaining = budget_chars - len(RAG_CONTEXT_HEADER) - len(RAG_CONTEXT_FOOTER) - 2
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
    return "\n".join([RAG_CONTEXT_HEADER, *blocks, RAG_CONTEXT_FOOTER]), used


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
) -> tuple[str, int]:
    """Central routing for model calls.

    - 'external-gpt' -> Grok / xAI backend.
    - 'dgx-qwen38-27b-fp8' (or legacy 'dgx-gpt-oss-20b') -> DGX Spark vLLM backend.
    - 'ollama-qwen3-8b' -> Ollama on the bsk-ai machine.
    - anything else -> dummy echo backend for now.

    `history` is the conversation's prior (role, content) messages, oldest
    first. When `use_rag` is True, callers can pass a `system_prompt` that
    already includes RAG context; otherwise each backend's default is used.

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
        system_prompt=system_prompt or default_prompt,
        max_messages=settings.CHAT_HISTORY_MAX_MESSAGES,
        max_chars=context_budget_chars(model_id),
    )
    return backend(messages), len(messages) - 2


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
    if AUTO_INGEST:
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


@api_view(["POST"])

def chat_view(request):
    """POST /api/chat/

    Body:
    {
      "conversation_id": "uuid or null",
      "message": "user text",
      "model": "local-small",
      "use_rag": true/false   (default true)
    }

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

    if not user_message:
        return Response(
            {"error": "message is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # --- Get or create conversation ---

    owner = get_current_user()

    if conversation_id:
        try:
            conversation = Conversation.objects.get(id=conversation_id, owner=owner)
        except Conversation.DoesNotExist:
            return Response(
                {"error": "conversation not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
    else:
        conversation = Conversation.objects.create(owner=owner, title="New Conversation")

    # --- Prior messages (before saving this one), oldest first ---
    from django.conf import settings

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
    )

    # --- Optional RAG context ---

    system_prompt: str | None = None
    citations: list[dict] = []
    if use_rag:
        rag_start = time.time()
        entries: list[dict] = []
        v2_top_n = settings.RAG_CHAT_TOP_N

        if is_factory_question(user_message) and query_chunks is not None:
            # Historian-first: plc_historian shift summaries live in the legacy
            # bsk_rag collection; documents come second from v2.
            try:
                for c in query_chunks(query=user_message, top_k=5, where={"source": "plc_historian"}):
                    entries.append({"text": c.text, "source": c.source})
            except Exception:
                logger.exception("historian retrieval failed conversation=%s", conversation.id)
            v2_top_n = 3

        reranker = "error"
        try:
            result = search_v2(user_message, top_n=v2_top_n)
            entries.extend(
                c.to_dict()
                for c in result.chunks
                if c.rerank_score is None or c.rerank_score >= settings.RAG_MIN_RERANK_SCORE
            )
            reranker = result.reranker
        except Exception:
            logger.exception("v2 retrieval failed conversation=%s", conversation.id)

        rag_budget = int(context_budget_chars(model_id) * settings.RAG_CONTEXT_SHARE)
        system_prompt, used = build_rag_context(entries, rag_budget)
        citations = [to_citation(e) for e in used]
        logger.info(
            "rag conversation=%s model=%s retrieved=%d used=%d reranker=%s "
            "rag_chars=%d/%d latency_ms=%d",
            conversation.id, model_id, len(entries), len(used), reranker,
            len(system_prompt or ""), rag_budget, round((time.time() - rag_start) * 1000),
        )

    # --- Generate assistant reply ---

    start = time.time()
    try:
        assistant_reply, history_sent = generate_reply_backend(
            message=user_message,
            model_id=model_id,
            use_rag=use_rag,
            history=history,
            system_prompt=system_prompt,
        )
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
    conversation.save(update_fields=["model_id", "updated_at"])

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


@api_view(["GET", "DELETE"])

def conversation_detail(request, pk):
    """GET/DELETE /api/conversations/<uuid:pk>/"""

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
    """GET /api/usage/summary/

    Return a very simple usage summary for the Analytics page.

    For now we only report per-model conversation counts and total
    conversations, scoped to the current owner. Later we can extend
    this with token counts and latency when we start logging them.
    """

    owner = get_current_user()

    # Per-model conversation counts
    per_model = (
        Conversation.objects.filter(owner=owner)
        .values("model_id")
        .annotate(count=Count("id"))
        .order_by("model_id")
    )

    total_conversations = sum(row["count"] for row in per_model)

    data = {
        "total_conversations": total_conversations,
        "per_model": [
            {
                "model_id": row["model_id"] or "unknown",
                "conversations": row["count"],
            }
            for row in per_model
        ],
    }

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
]


def _check_single_endpoint(ep: dict) -> dict:
    """Ping one endpoint and return its status dict."""
    from django.conf import settings
    from django.db import connections

    ep_id = ep["id"]

    if ep_id == "rag-chroma":
        start = time.time()
        try:
            from rag.retrieval import get_collection
            col = get_collection()
            col.count()
            latency = round((time.time() - start) * 1000)
            status_val = "online"
        except Exception:
            latency = 0
            status_val = "offline"
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
    if status_val == "online":
        tracker["ok"] += 1
    uptime = round(tracker["ok"] / tracker["total"] * 100, 1) if tracker["total"] > 0 else 100.0

    return {
        "id": ep_id,
        "name": ep["name"],
        "url": ep.get("url", ""),
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
            import chromadb
            from rag.config import CHROMA_DIR
            from chromadb.config import Settings as ChromaSettings
            client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            col = client.get_or_create_collection(name="bsk_rag")
            count = col.count()
            restart_message = f"Chroma reconnected — {count} chunks indexed"
        except Exception as exc:
            restart_message = f"Chroma reconnect failed: {exc}"

    result = _check_single_endpoint(ep)
    result["restart_message"] = restart_message
    return Response(result)

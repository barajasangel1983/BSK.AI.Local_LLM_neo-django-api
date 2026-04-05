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

import os
import requests

from django.db.models import Max
from django.http import JsonResponse
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from pathlib import Path
from os import getenv

from .models import Conversation, Message
from .serializers import ConversationSummarySerializer, ConversationDetailSerializer

# RAG lab (vector-only, Chroma) – imported from GraphRAG repo.
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


def call_grok_chat(message: str, system_prompt: str | None = None) -> str:
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
        "messages": [
            {
                "role": "system",
                "content": system_prompt
                or "You are Neo, an AI assistant helping build and debug a local LLM stack.",
            },
            {"role": "user", "content": message},
        ],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Grok follows the OpenAI-style response format
    return data["choices"][0]["message"]["content"]


def call_dgx_gpt_oss_20b(message: str, system_prompt: str | None = None) -> str:
    """Call DGX vLLM server hosting openai/gpt-oss-20b and return reply text."""

    from django.conf import settings

    base_url = settings.DGX_API_BASE.rstrip("/")
    model = settings.DGX_CHAT_MODEL
    url = f"{base_url}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
                or "You are Neo, an assistant running on the DGX Spark box.",
            },
            {"role": "user", "content": message},
        ],
        "max_tokens": 2048,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    return data["choices"][0]["message"]["content"]


def generate_reply_backend(
    message: str,
    model_id: str,
    use_rag: bool,
    conversation: Conversation,
    system_prompt: str | None = None,
) -> str:
    """Central routing for model calls.

    - 'external-gpt' -> Grok / xAI backend.
    - 'dgx-gpt-oss-20b' -> DGX Spark vLLM backend.
    - anything else -> dummy echo backend for now.

    When `use_rag` is True, callers can pass a `system_prompt` that already
    includes RAG context; backends that support system prompts will use it.
    """

    if model_id == "dgx-gpt-oss-20b":
        # DGX is the primary RAG target; apply system_prompt when provided.
        return call_dgx_gpt_oss_20b(message, system_prompt=system_prompt)

    if model_id == "external-gpt":
        # Optional: also allow Grok to use RAG context when available.
        return call_grok_chat(message, system_prompt=system_prompt)

    # Default: dummy echo
    return generate_dummy_reply(message, model_id, use_rag)


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
    visibility = request.data.get("visibility", "private")  # reserved for future use

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

            paths = [f["path"] for f in saved]
            chunks_added = ingest_files(paths)
        except Exception as e:
            # Log the error but don't fail the upload
            print(f"Auto-ingestion failed: {e}")

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
      "use_rag": true/false
    }

    Behavior:
    - If conversation_id is null -> create a new Conversation.
    - Else -> load existing Conversation (404 if not found).
    - Save a user Message.
    - Generate a dummy assistant reply and save it.
    - Return the full Conversation (with messages[]).
    """

    conversation_id = request.data.get("conversation_id")
    user_message = request.data.get("message")
    model_id = request.data.get("model", "local-small")
    use_rag = bool(request.data.get("use_rag", False))

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

    # --- Save user message ---

    Message.objects.create(
        conversation=conversation,
        role="user",
        content=user_message,
    )

    # --- Optional RAG context ---

    system_prompt: str | None = None
    rag_sources: list[str] = []
    if use_rag and query_chunks is not None:
        try:
            rag_chunks = query_chunks(query=user_message, top_k=3)
            if rag_chunks:
                context_lines = [
                    "You are Neo, an assistant helping with Django / RAG / DGX questions.",
                    "\nHere is relevant context from the knowledge base:",
                ]
                for idx, c in enumerate(rag_chunks, start=1):
                    context_lines.append(
                        f"[{idx}] (source: {c.source})\n{c.text.strip()}\n"
                    )
                    if c.source and c.source not in rag_sources:
                        rag_sources.append(c.source)
                context_lines.append("---\nUse this context when answering the user.")
                system_prompt = "\n".join(context_lines)
        except Exception as exc:
            # If RAG fails for any reason, fall back to normal behavior.
            print(f"RAG context generation failed: {exc}")
            system_prompt = None
            rag_sources = []

    # --- Generate assistant reply ---

    assistant_reply = generate_reply_backend(
        message=user_message,
        model_id=model_id,
        use_rag=use_rag,
        conversation=conversation,
        system_prompt=system_prompt,
    )

    # Append simple sources footer when RAG was used and we have sources
    if use_rag and rag_sources:
        sources_str = ", ".join(rag_sources)
        assistant_reply = f"{assistant_reply}\n\n---\nSources: {sources_str}"

    # --- Save assistant message ---

    Message.objects.create(
        conversation=conversation,
        role="assistant",
        content=assistant_reply,
    )

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
            "label": "Local Small",
            "description": "Placeholder local model for development.",
        },
        {
            "id": "dgx-gpt-oss-20b",
            "label": "DGX gpt-oss-20b",
            "description": "DGX Spark vLLM backend for openai/gpt-oss-20b.",
        },
        {
            "id": "external-gpt",
            "label": "External GPT (Grok)",
            "description": "xAI Grok backend (external GPT-style API).",
        },
    ]
    return Response(data)


# ---------------------------------------------------------------------
# RAG query endpoint (Chroma-backed)
# ---------------------------------------------------------------------


@api_view(["POST"])

def rag_query(request):
    """POST /api/rag/query/

    Simple RAG query endpoint backed by the Chroma collection.

    Request body (JSON):
    {
      "query": "user question",
      "top_k": 5   # optional
    }

    Response body:
    {
      "query": "...",
      "results": [
        {
          "id": "chunk-id",
          "text": "chunk text",
          "document_path": "path/to/file.txt",
          "source": "file.txt",
          "score": 0.123
        },
        ...
      ]
    }
    """

    if query_chunks is None:
        return Response(
            {"error": "RAG backend not available (query_chunks import failed)"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    query = request.data.get("query", "")
    top_k = int(request.data.get("top_k", 5))

    if not query.strip():
        return Response(
            {"error": "query is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        chunks = query_chunks(query=query, top_k=top_k)
    except Exception as exc:  # pragma: no cover - simple safety net
        return Response(
            {"error": f"RAG query failed: {exc}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    results = [
        {
            "id": c.id,
            "text": c.text,
            "document_path": c.document_path,
            "source": c.source,
            "score": c.score,
        }
        for c in chunks
    ]

    return Response({"query": query, "results": results})

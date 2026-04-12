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
# - GET  /api/rag/docs/
# - DELETE /api/rag/docs/<name>/
# - GET  /api/usage/summary/

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

    # Append a compact, deduplicated sources footer when RAG was used.
    if use_rag and rag_sources:
        unique_sources = list(dict.fromkeys(rag_sources))  # preserve order, remove dups
        max_sources = 3
        shown = unique_sources[:max_sources]
        remaining = len(unique_sources) - len(shown)
        sources_str = ", ".join(shown)
        if remaining > 0:
            sources_str = f"{sources_str}, +{remaining} more"
        assistant_reply = f"{assistant_reply}\n\n---\nSources (RAG): {sources_str}"

    # --- Save assistant message ---

    Message.objects.create(
        conversation=conversation,
        role="assistant",
        content=assistant_reply,
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
# RAG docs listing + delete + query endpoint (Chroma-backed)
# ---------------------------------------------------------------------


@api_view(["GET"])

def rag_docs(request):
    """GET /api/rag/docs/

    Return a simple listing of raw RAG documents based on files in
    RAG_UPLOAD_BASE. This is used by the RAG Lab UI to show the corpus.
    """

    RAG_UPLOAD_BASE.mkdir(parents=True, exist_ok=True)

    # Optional: enrich with chunk/token stats from Chroma when available.
    stats_by_source: dict[str, dict[str, int]] = {}
    try:  # pragma: no cover - best-effort enrichment
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from rag.config import CHROMA_DIR

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False)
        )
        collection = client.get_or_create_collection(name="bsk_rag")
        data = collection.get(include=["documents", "metadatas"], limit=None)
        documents = data.get("documents", []) or []
        metadatas = data.get("metadatas", []) or []

        for doc_text, meta in zip(documents, metadatas):
            meta = meta or {}
            source = str(meta.get("source") or "")
            if not source:
                continue
            bucket = stats_by_source.setdefault(source, {"chunks": 0, "tokens": 0})
            bucket["chunks"] += 1
            text = doc_text or ""
            # Rough token estimate: 4 characters ≈ 1 token (good enough for UI stats).
            bucket["tokens"] += max(1, len(text) // 4) if text else 0
    except Exception as exc:
        # If anything goes wrong, we still return the basic docs list.
        print(f"rag_docs: failed to compute Chroma stats: {exc}")

    docs = []
    for p in sorted(RAG_UPLOAD_BASE.iterdir()):
        if not p.is_file():
            continue
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        stats = stats_by_source.get(p.name, {"chunks": 0, "tokens": 0})
        docs.append(
            {
                "name": p.name,
                "size": size,
                "chunks": stats.get("chunks", 0),
                "tokens": stats.get("tokens", 0),
            }
        )

    return Response({"documents": docs})


@api_view(["DELETE"])

def rag_delete_doc(request, name: str):
    """DELETE /api/rag/docs/<name>/

    Delete a raw RAG document file and its associated chunks from Chroma.
    """

    # Remove the file from disk (if it exists)
    path = RAG_UPLOAD_BASE / name
    if path.exists() and path.is_file():
      try:
        path.unlink()
      except OSError as exc:
        return Response({"error": f"Failed to delete file: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Best-effort delete from Chroma based on "source" metadata
    try:
      import chromadb
      from rag.config import CHROMA_DIR
      from chromadb.config import Settings

      client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
      collection = client.get_or_create_collection(name="bsk_rag")
      collection.delete(where={"source": name})
    except Exception as exc:
      # Log but do not fail hard; at worst, stale chunks remain.
      print(f"Chroma delete failed for {name}: {exc}")

    return Response(status=status.HTTP_204_NO_CONTENT)


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
        "name": "DGX vLLM (gpt-oss-20b)",
        "url": "http://100.74.225.3:8000/v1/models",
        "model": "dgx-gpt-oss-20b",
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
        "id": "lm-studio",
        "name": "LM Studio (Local)",
        "url": "http://100.111.50.52:1234/v1/models",
        "model": "lm-studio",
        "can_restart": False,
    },
    {
        "id": "rag-chroma",
        "name": "RAG / Chroma",
        "url": "",
        "model": "rag",
        "can_restart": True,
    },
]


def _check_single_endpoint(ep: dict) -> dict:
    """Ping one endpoint and return its status dict."""
    from django.conf import settings

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

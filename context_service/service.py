"""Context Service (P10a): one governed way to ask "what do we know that is relevant to this
question?", for the Studio chat, REST clients and the MCP server (BSKLAB EDGE, other agents).

It does not hold knowledge of its own. It asks the existing parts and returns one versioned
**Context Packet**:

    curated Context Graph  → graph facts (with where each came from) and relationships
    document library (RAG) → reranked excerpts, the asset's own documents first, figures included
    historian              → current state and values against their normal range

Rules: curated, approved facts only (never the lab layer or pending triples); every item
carries its source; a source that is down yields a warning, not an error; an ambiguous
machine or component is returned as candidates, never guessed.

The packet contract is described in the frontend repo: claude/P10_context_service_mcp_plan.md.
Changes within version "1" are additive only.
"""

from __future__ import annotations

import logging
import re
import time

from django.conf import settings

from chat import asset_context
from chat.retrieval import search_v2
from context_graph import provenance, services
from context_graph.driver import GraphUnavailable
from context_graph.repository import NodeNotFound

logger = logging.getLogger("chat")

PACKET_VERSION = "1"
MIN_BUDGET, MAX_BUDGET = 2000, 60000
FOCUS_LABELS = ("Component", "Signal", "Alarm", "Procedure")
ANSWER_PROMPT = (
    "You are the assistant of the BSKLAB plant. Answer the question from the CONTEXT below.\n"
    "- Graph facts are authoritative; cite them as [G]. Cite document excerpts as [1], [2], … by their number.\n"
    "- Current values are the last recorded ones; state their time and the machine state when you use them.\n"
    "- If the context does not contain the answer, say so plainly. Do not invent values, limits or procedures.\n\n"
    "CONTEXT\n{context}"
)


class ContextError(ValueError):
    """A request the service can't act on (unknown asset, empty question)."""


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _mentions(question_norm: str, key: str) -> bool:
    key = _norm(key)
    return len(key) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", question_norm) is not None


# --- assets and resolution --------------------------------------------------------

def list_assets() -> list[dict]:
    """Assets a client can ask about: [{id, name, asset_type, description}]."""
    return [{"id": a["id"], "name": a["name"], "asset_type": a["properties"].get("asset_type", ""),
             "description": a["properties"].get("description", "")} for a in services.list_assets()]


def _asset_keys(asset: dict) -> list[str]:
    return [asset["name"], asset["id"].split(":", 2)[2]]


def resolve(text: str, asset_id: str | None = None) -> dict:
    """What a question is about: {asset, focus, ambiguous}.

    - `asset`: the given one, or the only asset named in the text (by name or key, e.g. "EXTR01").
    - `focus`: that asset's components / signals / alarms / procedures named in the text.
    - `ambiguous`: several assets named, or one phrase matching several entities — the client asks the user.
    """
    question = _norm(text)
    out = {"asset": None, "focus": [], "ambiguous": []}
    if asset_id:
        try:
            name = asset_context.check_asset(asset_id)
        except asset_context.AssetScopeError as exc:
            raise ContextError(str(exc))
        out["asset"] = {"id": asset_id, "name": name, "match": "given"}
    else:
        named = [a for a in list_assets() if any(_mentions(question, k) for k in _asset_keys(a))]
        if len(named) == 1:
            out["asset"] = {"id": named[0]["id"], "name": named[0]["name"], "match": "named"}
        elif len(named) > 1:
            out["ambiguous"].append({"phrase": "asset", "candidates": [{"id": a["id"], "name": a["name"]} for a in named]})
    if not out["asset"]:
        return out

    hits: dict[str, list[dict]] = {}      # matched phrase -> entities
    for entity in asset_context.entities(out["asset"]["id"]):
        if entity["label"] not in FOCUS_LABELS:
            continue
        matched = max((k for k in entity["keys"] if _mentions(question, k)), key=len, default=None)
        if matched:
            hits.setdefault(_norm(matched), []).append(entity)
    # A longer phrase wins over a shorter one inside it ("die head temperature" over "die head").
    phrases = sorted(hits, key=len, reverse=True)
    kept = [p for i, p in enumerate(phrases) if not any(p in longer and p != longer for longer in phrases[:i])]
    for phrase in kept:
        entities = hits[phrase]
        if len(entities) == 1:
            e = entities[0]
            out["focus"].append({"id": e["id"], "label": e["label"], "name": e["name"], "match": phrase})
        else:
            out["ambiguous"].append({"phrase": phrase, "candidates": [{"id": e["id"], "label": e["label"], "name": e["name"]}
                                                                    for e in entities]})
    return out


# --- single lookups ---------------------------------------------------------------

def entity_context(entity_id: str) -> dict:
    """One curated entity with its direct relationships (lab-layer nodes are not served)."""
    try:
        node = services.node_detail(entity_id)
    except NodeNotFound:
        raise ContextError(f"unknown entity {entity_id!r}")
    return {"id": node["id"], "label": node.get("label"), "name": node.get("name"),
            "properties": {k: v for k, v in (node.get("properties") or {}).items()
                           if k not in ("evidence_ids", "created_at", "updated_at")},
            "relationships": node.get("relationships") or []}


def entity_sources(entity_id: str) -> dict:
    """Where an entity came from: its origin and every evidence record (P8c)."""
    try:
        return provenance.node_sources(entity_id)
    except NodeNotFound:
        raise ContextError(f"unknown entity {entity_id!r}")


def operational_state(asset_id: str) -> dict:
    """Current machine state and signal values against their normal range (historian; OPC UA later)."""
    try:
        ctx = asset_context.build(asset_id, "", MAX_BUDGET)
    except asset_context.AssetScopeError as exc:
        raise ContextError(str(exc))
    return _operational(ctx)


def _operational(ctx) -> dict:
    h = ctx.historian
    return {
        "source": "plc_1_historian" if h else None,
        "current_state": {"ts": h["latest_ts"], "machine_state": h["state"], "running_ts": h["running_ts"]} if h else None,
        "values": [{**v, "out_of_range": v["status"].endswith("normal range")} for v in ctx.values],
        "recent_events": [],        # filled by monitoring (P11)
        "trends": [],
    }


def search_documents(query: str, asset_id: str | None = None, top_k: int | None = None,
                     doc_keys: list[str] | None = None) -> dict:
    """Reranked document excerpts: the asset's own documents first, then the whole library."""
    top_k = max(1, min(int(top_k or settings.RAG_CHAT_TOP_N), settings.RAG_QUERY_MAX_TOP_K))
    if doc_keys is None and asset_id:
        try:
            doc_keys = asset_context.build(asset_id, query, MIN_BUDGET).doc_keys
        except asset_context.AssetScopeError as exc:
            raise ContextError(str(exc))
    relevant = lambda result: [c for c in result.chunks  # noqa: E731
                               if c.rerank_score is None or c.rerank_score >= settings.RAG_MIN_RERANK_SCORE]
    found, scope, reranker = [], "library", "skipped"
    if doc_keys:
        result = search_v2(query, top_n=top_k, doc_keys=doc_keys)
        found, scope, reranker = relevant(result), "asset", result.reranker
    if not found:
        result = search_v2(query, top_n=top_k)
        found, scope, reranker = relevant(result), "library", result.reranker
    return {"scope": scope, "reranker": reranker, "results": [_excerpt(c, scope) for c in found]}


def _excerpt(chunk, scope: str) -> dict:
    return {"document": chunk.source, "doc_key": chunk.asset_id, "page": chunk.page_start, "page_end": chunk.page_end,
            "section_path": chunk.section_path, "kind": "figure" if chunk.content_type == "figure" else "text",
            "text": chunk.text, "score": chunk.score, "scope": scope}


# --- read-only views for clients (BSKLAB EDGE's RAG and Graph tabs) ---------------------

GRAPH_MAX_NODES = 600
_HIDDEN_PROPS = ("evidence_ids", "created_at", "updated_at")


def _document(doc_id):
    from ingestion.models import Document
    try:
        doc = Document.objects.filter(pk=doc_id).first()
    except (ValueError, Exception):     # not a UUID
        doc = None
    if doc is None:
        raise ContextError(f"unknown document {doc_id!r}")
    return doc


def _document_json(doc, data=...) -> dict:
    from ingestion import figures
    if data is ...:
        data = figures.load(doc) if doc.parse_status == "parsed" else None
    counts = figures.summary(data)
    return {
        "id": str(doc.id), "filename": doc.filename, "doc_key": doc.doc_key, "pages": doc.page_count,
        "parsed": doc.parse_status == "parsed", "searchable": doc.rag_status == "done", "chunks": doc.rag_chunk_count,
        "figures_found": counts["found"], "figures_described": counts["described"],
        "uploaded_at": doc.uploaded_at.isoformat(),
    }


def list_documents() -> dict:
    """The document library: what exists, whether it is searchable, how many figures are described."""
    from ingestion.models import Document
    return {"documents": [_document_json(d) for d in Document.objects.all()]}


def get_document(document_id: str) -> dict:
    """One document with its described figures (page, kind, caption, description)."""
    from ingestion import figures
    doc = _document(document_id)
    data = figures.load(doc) if doc.parse_status == "parsed" else None
    shown = [{"index": f["index"], "page": f["page"], "kind": f.get("kind", ""), "caption": f.get("caption", ""),
              "description": f.get("description", "")}
             for f in (data or {}).get("figures", []) if f.get("status") == "done"]
    return {**_document_json(doc, data), "figures": shown}


def get_document_page(document_id: str, page: int) -> dict:
    """The text of one page (Markdown as parsed, with figure descriptions in place)."""
    from ingestion import library
    from ingestion.chunker import IMAGE_PLACEHOLDER, PAGE_BREAK
    doc = _document(document_id)
    if doc.parse_status != "parsed" or not library.parsed_path(doc).exists():
        raise ContextError("the document is not parsed yet")
    md = (library.load_parsed(doc).get("document") or {}).get("md_content") or ""
    pages = md.split(PAGE_BREAK)
    if not 1 <= int(page) <= len(pages):
        raise ContextError(f"page must be between 1 and {len(pages)}")
    text = pages[int(page) - 1].replace(IMAGE_PLACEHOLDER, "").strip()
    return {"document_id": str(doc.id), "filename": doc.filename, "page": int(page), "pages": len(pages), "text": text}


def figure_image(document_id: str, figure_index: int) -> bytes:
    """A described figure as JPEG (the same crop the vision model saw)."""
    from gpu import images
    from ingestion import figures
    doc = _document(document_id)
    data = figures.load(doc) or {}
    figure = next((f for f in data.get("figures", []) if f["index"] == int(figure_index) and f.get("status") == "done"), None)
    if figure is None:
        raise ContextError(f"unknown figure {figure_index} of document {document_id}")
    path = figures.image_path(doc, figure["index"])
    if not path.exists():
        try:
            figures.crop(doc, figure)
        except images.ImageError as exc:
            raise ContextError(f"the figure can't be shown: {exc}")
    return path.read_bytes()


def get_graph(asset_id: str, depth: int = 2) -> dict:
    """The curated graph around an asset, for a viewer: {nodes: [{id, label, name, properties}],
    links: [{source, target, type}]}. The lab layer is never included."""
    try:
        asset_context.check_asset(asset_id)
    except asset_context.AssetScopeError as exc:
        raise ContextError(str(exc))
    depth = max(1, min(int(depth or 2), 3))
    data = services.graph_data(asset_id, depth=depth, limit=GRAPH_MAX_NODES, layer="curated")
    return {
        "asset_id": asset_id, "depth": depth,
        "nodes": [{"id": n["id"], "label": n.get("label"), "name": n.get("name"),
                   "properties": {k: v for k, v in (n.get("properties") or {}).items() if k not in _HIDDEN_PROPS}}
                  for n in data["nodes"]],
        "links": [{"source": l["source"], "target": l["target"], "type": l["type"]} for l in data["links"]],
        "truncated": len(data["nodes"]) >= GRAPH_MAX_NODES,
    }


# --- the packet -----------------------------------------------------------------------

def assemble(query: str, asset_id: str | None = None, include_documents: bool = True,
             budget_chars: int | None = None) -> dict:
    """The Context Packet for a question (see the module docstring)."""
    query = (query or "").strip()
    if not query:
        raise ContextError("query is required")
    budget = max(MIN_BUDGET, min(int(budget_chars or settings.CHAT_CONTEXT_MAX_CHARS), MAX_BUDGET))
    started = time.monotonic()
    timings: dict[str, int] = {}
    warnings: list[str] = []

    def timed(name, fn):
        t = time.monotonic()
        try:
            return fn()
        finally:
            timings[name] = round((time.monotonic() - t) * 1000)

    scope = {"asset": None, "focus": [], "ambiguous": []}
    try:
        scope = timed("resolve", lambda: resolve(query, asset_id))
    except GraphUnavailable as exc:
        warnings.append("graph unavailable")
        logger.warning("context: graph unavailable during resolve: %s", exc)

    asset = scope["asset"]
    ctx = None
    if asset:
        facts_budget = int(budget * (0.6 if include_documents else 1.0))
        try:
            ctx = timed("graph", lambda: asset_context.build(asset["id"], query, facts_budget,
                                                               {f["id"] for f in scope["focus"]}))
        except GraphUnavailable as exc:
            warnings.append("graph unavailable")
            logger.warning("context: graph unavailable asset=%s: %s", asset["id"], exc)
    elif not scope["ambiguous"] and "graph unavailable" not in warnings:
        warnings.append("no asset in scope: documents only")

    facts = [{"id": f"f{i + 1}", **item, "source": {"label": item["source"]}} for i, item in enumerate(ctx.items)] if ctx else []
    used = len(ctx.text) if ctx else 0

    documents: list[dict] = []
    if include_documents:
        try:
            found = timed("documents", lambda: search_documents(query, doc_keys=ctx.doc_keys if ctx else None))
            if found["reranker"] == "fallback":
                warnings.append("reranker unavailable: document order is by vector similarity")
            remaining = budget - used
            for excerpt in found["results"]:
                if len(excerpt["text"]) > remaining:
                    if documents or remaining < 400:
                        continue
                    excerpt = {**excerpt, "text": excerpt["text"][:remaining], "truncated": True}
                documents.append({"id": f"d{len(documents) + 1}", **excerpt})
                remaining -= len(excerpt["text"])
            used = budget - remaining
        except Exception as exc:        # embeddings / Chroma down: the packet still has the graph part
            warnings.append("documents unavailable")
            logger.warning("context: document search failed: %s", exc)

    packet = {
        "packet_version": PACKET_VERSION,
        "query": {"text": query, "intent": None},
        "resolved_scope": scope,
        "graph_facts": facts,
        "relationships": [{"from": f["entity_id"][5:].split(">")[0], "to": f["entity_id"][5:].split(">")[1], "text": f["text"],
                           "source": f["source"]} for f in facts if f["entity_id"].startswith("edge:")],
        "document_evidence": documents,
        "operational_context": _operational(ctx) if ctx else None,
        "provenance": {
            "graph_entities": sorted({f["entity_id"] for f in facts if f["entity_id"] and not f["entity_id"].startswith("edge:")}),
            "documents": sorted({d["document"] for d in documents}),
        },
        "limits": {"budget_chars": budget, "used_chars": used, "facts_total": ctx.total_facts if ctx else 0,
                   "facts_included": len([f for f in facts if f["section"] != "asset"]), "chunks": len(documents)},
        "warnings": warnings,
    }
    packet["context_text"] = render(packet, ctx)
    timings["total"] = round((time.monotonic() - started) * 1000)
    packet["timings_ms"] = timings
    logger.info("context packet asset=%s focus=%d facts=%d docs=%d used=%d/%d warnings=%s ms=%d",
                asset["id"] if asset else "-", len(scope["focus"]), len(facts), len(documents), used, budget,
                warnings, timings["total"])
    return packet


def render(packet: dict, ctx=None) -> str:
    """The packet as text for a model's prompt (what a small client can pass on as is)."""
    parts: list[str] = []
    if ctx is not None:
        parts.append("GRAPH FACTS [G] (curated, authoritative):\n" + ctx.text)
        out = [f"{v['name']}: {v['value']} {v['unit']} ({v['status']})".strip() for v in ctx.values
               if v["status"].endswith("normal range")]
        if out:
            parts.append("Out of normal range in the last running sample: " + "; ".join(out))
    if packet["document_evidence"]:
        lines = []
        for n, d in enumerate(packet["document_evidence"], 1):
            where = f"{d['document']}" + (f", p.{d['page']}" if d.get("page") else "") + (" (figure)" if d["kind"] == "figure" else "")
            lines.append(f"[{n}] ({where})\n{d['text']}")
        parts.append("DOCUMENT EXCERPTS:\n" + "\n\n".join(lines))
    if not parts:
        parts.append("(nothing relevant was found)")
    return "\n\n".join(parts)


# --- answering (for clients without a model of their own) -----------------------------------

def ask(query: str, asset_id: str | None = None, model: str | None = None, include_documents: bool = True) -> dict:
    """Answer a question with one of the Studio's models, from the packet. Returns {answer, model, packet}."""
    from chat.views import DGX_MODEL_IDS, call_dgx_gpt_oss_20b, call_grok_chat, call_ollama_qwen3_8b, context_budget_chars
    from usage import recorder as usage

    model = model or DGX_MODEL_IDS[-1]
    backends = {**{m: call_dgx_gpt_oss_20b for m in DGX_MODEL_IDS}, "external-gpt": call_grok_chat,
                "ollama-qwen3-8b": call_ollama_qwen3_8b}
    if model not in backends:
        raise ContextError(f"unknown model {model!r}; choose one of {', '.join(backends)}")
    # Leave room for the instructions, the question and the answer.
    packet = assemble(query, asset_id, include_documents, int(context_budget_chars(model) * 0.8))
    if packet["resolved_scope"]["ambiguous"] and not packet["resolved_scope"]["asset"]:
        return {"answer": None, "model": model, "packet": packet,
                "needs_choice": packet["resolved_scope"]["ambiguous"]}
    messages = [{"role": "system", "content": ANSWER_PROMPT.format(context=packet["context_text"])},
                {"role": "user", "content": packet["query"]["text"]}]
    with usage.scope(purpose="context-ask"):
        answer = backends[model](messages)
    return {"answer": answer, "model": model, "packet": packet}

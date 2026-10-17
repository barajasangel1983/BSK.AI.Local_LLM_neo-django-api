"""Run a chat reference set and score the answers (P16a, 2026-10-16).

    .venv/bin/python manage.py chat_eval --model dgx-qwen38-27b-fp8 [--set chat/evals/extr01_reference.json]
        [--only Q1,Q6] [--total-chars 24000] [--out run.json]

Each question goes through the chat's own answer path (POST /api/chat/, asset scope, RAG on) in a new
conversation, which is deleted afterwards: an eval run leaves nothing in the chat list. An answer is scored by
the facts it states (`expect`: each fact a list of accepted spellings): ✓ all, ◐ some, ✗ none. The score is a
first pass; a reviewer reads the saved answers (`--out`) to confirm it.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.test import override_settings
from rest_framework.test import APIRequestFactory

from chat import views
from chat.models import Conversation

DEFAULT_SET = Path(__file__).resolve().parents[2] / "evals" / "extr01_reference.json"


def normalize(text: str) -> str:
    """Lower case, one space, and no thousands separators ("5,600", "5 600" -> "5600")."""
    text = re.sub(r"(?<=\d)[ ,.  ](?=\d{3}(?!\d))", "", str(text).lower())
    return re.sub(r"\s+", " ", text.replace("–", "-").replace(" ", " "))


def score(answer: str, expect: list[list[str]]) -> tuple[str, list[bool]]:
    """(✓ | ◐ | ✗, which expected facts the answer states)."""
    text = normalize(answer)
    found = [any(normalize(alt) in text for alt in alternatives) for alternatives in expect]
    mark = "✓" if all(found) else "◐" if any(found) else "✗"
    return mark, found


class Command(BaseCommand):
    help = "Run a chat reference set (questions with expected facts) against a model and score the answers."

    def add_arguments(self, parser):
        parser.add_argument("--model", required=True, help="chat model id, e.g. dgx-qwen38-27b-fp8 or ollama-qwen3-8b")
        parser.add_argument("--set", default=str(DEFAULT_SET), help="reference set (JSON)")
        parser.add_argument("--only", default="", help="question ids, comma-separated (default: all)")
        parser.add_argument("--total-chars", type=int, default=0,
                            help="the model's prompt budget for this run (characters); default: the setting")
        parser.add_argument("--out", default="", help="save the answers and scores here (JSON)")

    def handle(self, *args, **opts):
        try:
            ref = json.loads(Path(opts["set"]).read_text())
        except (OSError, ValueError) as exc:
            raise CommandError(f"reference set: {exc}")
        only = {q.strip() for q in opts["only"].split(",") if q.strip()}
        questions = [q for q in ref["questions"] if not only or q["id"] in only]
        overrides = {"ALLOWED_HOSTS": [*settings.ALLOWED_HOSTS, "testserver"]}
        if opts["total_chars"]:
            key = "CHAT_CONTEXT_MAX_CHARS_OLLAMA" if opts["model"] == "ollama-qwen3-8b" else "CHAT_CONTEXT_MAX_CHARS"
            overrides[key] = opts["total_chars"]
        factory = APIRequestFactory()
        results = []
        with override_settings(**overrides):
            for q in questions:
                started = time.time()
                request = factory.post("/api/chat/", {"conversation_id": None, "message": q["question"],
                                                      "model": opts["model"], "use_rag": True,
                                                      "asset_id": ref["asset_id"]}, format="json")
                response = views.chat_view(request)
                data = response.data if hasattr(response, "data") else {}
                messages = data.get("messages") or []
                reply = messages[-1] if messages and messages[-1].get("role") == "assistant" else {}
                answer = reply.get("content") or data.get("error") or f"HTTP {response.status_code}"
                if data.get("id"):
                    Conversation.objects.filter(pk=data["id"]).delete()     # leave nothing in the chat list
                mark, found = score(answer, q["expect"]) if reply else ("✗", [False] * len(q["expect"]))
                sources = reply.get("sources") or []
                graph = next((s for s in sources if s.get("kind") == "graph"), {})
                results.append({
                    "id": q["id"], "question": q["question"], "expected": q["answer"], "source": q["source"],
                    "mark": mark, "found": found, "seconds": round(time.time() - started, 1), "status": response.status_code,
                    "answer": answer, "graph_facts": graph.get("snippet", ""),
                    "documents": [s.get("source") for s in sources if s.get("kind") not in ("graph", "historian")],
                })
                self.stdout.write(f"{q['id']:>4} {mark} {sum(found)}/{len(found)}  {results[-1]['seconds']:>5}s  "
                                  f"{graph.get('snippet', '')}")
        totals = {m: sum(r["mark"] == m for r in results) for m in ("✓", "◐", "✗")}
        summary = (f"{opts['model']}: {totals['✓']} ✓, {totals['◐']} ◐, {totals['✗']} ✗ of {len(results)}"
                   f" (facts {sum(sum(r['found']) for r in results)}/{sum(len(r['found']) for r in results)})")
        self.stdout.write(summary)
        if opts["out"]:
            Path(opts["out"]).write_text(json.dumps({"set": ref["name"], "model": opts["model"],
                                                     "total_chars": opts["total_chars"] or None, "summary": summary,
                                                     "results": results}, indent=1, ensure_ascii=False))

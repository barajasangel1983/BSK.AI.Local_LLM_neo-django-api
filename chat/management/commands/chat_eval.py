"""Run a chat reference set and score the answers (P16a, 2026-10-16).

    .venv/bin/python manage.py chat_eval --model dgx-qwen38-27b-fp8 [--set chat/evals/extr01_reference.json]
        [--only Q1,Q6] [--total-chars 24000 | --context-tokens 16384] [--out run.json]

Each question goes through the chat's own answer path (POST /api/chat/, asset scope, RAG on) in a new
conversation, which is deleted afterwards: an eval run leaves nothing in the chat list. An answer is scored by
the facts it states (`expect`: each fact a list of accepted spellings): ✓ all, ◐ some, ✗ none. A question with
`abstain` expects the answer to say the fact is not documented; one with `forbid` (regexes) is ✗ when the answer
states one of them (a wrong value). A data question (P18b, `measures`) has no fixed answer: its numbers are
computed from the historian when it is asked (`historian_window.stats`, the numbers the chat's block states),
just before and just after, and each measure is found when the answer states it within its tolerance of
either. The score is a first pass; a reviewer reads the saved answers (`--out`) to
confirm it. Each result also records the chat call's prompt and completion tokens (Analytics' ModelCall).
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

from chat import historian_window, views
from chat.models import Conversation
from usage.models import ModelCall

DEFAULT_SET = Path(__file__).resolve().parents[2] / "evals" / "extr01_reference.json"


def normalize(text: str) -> str:
    """Lower case, one space, and no thousands separators ("5,600", "5 600" -> "5600")."""
    text = re.sub(r"(?<=\d)[ ,.  ](?=\d{3}(?!\d))", "", str(text).lower())
    return re.sub(r"\s+", " ", text.replace("–", "-").replace(" ", " "))


# The answer says the fact is not in what it was given.
ABSTAIN = re.compile(
    r"\bnot (explicitly |specifically |directly |clearly )?"
    r"(in|found|available|documented|specified|listed|provided|included|mentioned|given|stated|covered|contain)"
    r"|\bno (information|data|documentation|details|mention|specific|manual|record)"
    r"|\b(don't|do not|doesn't|does not|didn't|did not) (have|contain|include|specify|mention|list|provide|say|state|cover)"
    r"|\b(cannot|can't|unable to) (find|answer|determine|confirm|provide|give|say)"
    r"|\b(isn't|is not|aren't|are not) (in|listed|documented|specified|available|provided|given|stated|mentioned)")


def wrong(answer: str, forbid: list[str]) -> list[str]:
    """The `forbid` patterns the answer states."""
    text = normalize(answer)
    return [pattern for pattern in forbid if re.search(pattern, text)]


def score(answer: str, expect: list[list[str]], forbid: list[str] = (), abstain: bool = False) -> tuple[str, list[bool]]:
    """(✓ | ◐ | ✗, which expected facts the answer states). A stated `forbid` value makes it ✗: always when the
    answer should say "not documented"; otherwise only when it misses an expected fact (a right answer may name
    the other values as context, e.g. "315 °C; the K series takes 538 °C")."""
    text = normalize(answer)
    if abstain:
        found = [bool(ABSTAIN.search(text))]
    else:
        found = [any(normalize(alt) in text for alt in alternatives) for alternatives in expect]
    mark = "✓" if all(found) else "◐" if any(found) else "✗"
    return ("✗" if is_wrong(answer, found, forbid, abstain) else mark), found


def is_wrong(answer: str, found: list[bool], forbid: list[str], abstain: bool) -> bool:
    return bool(wrong(answer, forbid)) and (abstain or not all(found))


# A data question's measures (P18b): the number in hour_stats and how near the answer must be.
PERCENT = ("oee", "availability", "performance", "quality")
MINUTES = ("running_min", "downtime_min", "not_running_min")
NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?:\s*(%|percent|minutes?|mins?|min|hours?|hrs?|hr|h|kg)\b)?")
NONE_OUT = re.compile(r"\bno (signals?|values?|readings?)\b[^.]*\b(out|outside|left|exceed)|\b(none|nothing) (of the signals |was |were )?"
                      r"(out|outside)|\b(stayed|remained|were|was|all) (with)?in (its|their|the|normal)|\bnot out of")


def expected(stats: dict, measure: str):
    """The value a measure expects, from hour_stats: a number, or for alarms/out_of_range a list of names."""
    if measure.startswith("state:"):
        return stats["states"].get(measure[6:], 0)
    if measure == "alarm_top":                                   # the three alarms that started most often
        return [code for code, _ in sorted(stats["alarms"].items(), key=lambda x: -x[1])[:3]]
    if measure == "out_of_range":
        return [o["name"] for o in stats["out_of_range"]]
    return stats.get(measure)


def tolerance(measure: str, value: float) -> float:
    if measure in PERCENT:
        return 0.5
    if measure == "throughput":
        return max(5.0, abs(value) * 0.01)
    return max(2.0, abs(value) * 0.02)                           # minutes (running, downtime, a state)


def numbers(answer: str) -> list[tuple[float, str]]:
    """The numbers an answer states with their unit word ("17.3 h" -> (17.3, "h"))."""
    return [(float(m.group(1)), (m.group(2) or "").lower()) for m in NUMBER.finditer(normalize(answer))]


def _words(text: str) -> str:
    """"OUT_OF_SPEC_DENSITY", "out-of-spec density" -> "out of spec density"."""
    return re.sub(r"[\s_-]+", " ", text.lower())


def states_measure(answer: str, measure: str, values: list) -> bool:
    """Does the answer state `measure` near any of `values` (the numbers computed before and after it ran)?"""
    text = normalize(answer)
    if measure == "alarm_top":
        return any(all(_words(c) in _words(text) for c in v) for v in values)
    if measure == "out_of_range":
        return any((all(n.lower() in text for n in v) if v else bool(NONE_OUT.search(text))) for v in values)
    said = numbers(answer)
    for v in values:
        if v is None:
            continue
        tol = tolerance(measure, v)
        for n, unit in said:
            if unit in ("h", "hr", "hrs", "hour", "hours") and (measure in MINUTES or measure.startswith("state:")):
                if abs(n * 60 - v) <= max(tol, 6.0):             # "17.3 hours" for 1,035 min
                    return True
            elif abs(n - v) <= tol:
                return True
    return False


def score_data(answer: str, measures: list[str], values: dict[str, list]) -> tuple[str, list[bool]]:
    found = [states_measure(answer, m, values[m]) for m in measures]
    return ("✓" if all(found) else "◐" if any(found) else "✗"), found


class Command(BaseCommand):
    help = "Run a chat reference set (questions with expected facts) against a model and score the answers."

    def add_arguments(self, parser):
        parser.add_argument("--model", required=True, help="chat model id, e.g. dgx-qwen38-27b-fp8 or ollama-qwen3-8b")
        parser.add_argument("--set", default=str(DEFAULT_SET), help="reference set (JSON)")
        parser.add_argument("--only", default="", help="question ids, comma-separated (default: all)")
        parser.add_argument("--total-chars", type=int, default=0,
                            help="the model's prompt budget for this run (characters); default: the setting")
        parser.add_argument("--context-tokens", type=int, default=0,
                            help="the model's context in tokens for this run (P16b: CHAT_CONTEXT_MAX_TOKENS_*)")
        parser.add_argument("--setting", action="append", default=[], metavar="NAME=VALUE",
                            help="override a setting for this run (int, float or text), e.g. CHAT_FACT_PAGES=4")
        parser.add_argument("--rescore", default="",
                            help="score a saved run (--out file) again with the current key, without asking the model")
        parser.add_argument("--out", default="", help="save the answers and scores here (JSON)")

    def handle(self, *args, **opts):
        try:
            ref = json.loads(Path(opts["set"]).read_text())
        except (OSError, ValueError) as exc:
            raise CommandError(f"reference set: {exc}")
        if opts["rescore"]:
            return self._rescore(ref, Path(opts["rescore"]))
        only = {q.strip() for q in opts["only"].split(",") if q.strip()}
        questions = [q for q in ref["questions"] if not only or q["id"] in only]
        overrides = {"ALLOWED_HOSTS": [*settings.ALLOWED_HOSTS, "testserver"]}
        if opts["total_chars"]:
            key = "CHAT_CONTEXT_MAX_CHARS_OLLAMA" if opts["model"] == "ollama-qwen3-8b" else "CHAT_CONTEXT_MAX_CHARS"
            overrides[key] = opts["total_chars"]
        if opts["context_tokens"]:
            key = "CHAT_CONTEXT_MAX_TOKENS_OLLAMA" if opts["model"] == "ollama-qwen3-8b" else "CHAT_CONTEXT_MAX_TOKENS_DGX"
            overrides[key] = opts["context_tokens"]
        for item in opts["setting"]:
            name, _, value = item.partition("=")
            if not name or not hasattr(settings, name):
                raise CommandError(f"--setting {item}: unknown setting")
            overrides[name] = _typed(value)
        factory = APIRequestFactory()
        results = []
        with override_settings(**overrides):
            for q in questions:
                started = time.time()
                measures = q.get("measures") or []
                before = historian_window.stats(ref["asset_id"], q["question"]) if measures else None
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
                forbid, abstain = q.get("forbid") or [], bool(q.get("abstain"))
                data_check = {}
                if measures:
                    after = historian_window.stats(ref["asset_id"], q["question"])
                    found_stats = [s for s in (before, after) if s]
                    values = {m: [expected(st, m) for _, st in found_stats] for m in measures}
                    mark, found = score_data(answer, measures, values) if reply else ("✗", [False] * len(measures))
                    window = next((s.get("source", "") for s in reply.get("sources") or [] if s.get("kind") == "historian"
                                   and s.get("source", "").startswith("Historian — ")), "")
                    labels = {f"Historian — {w.label}" for w, _ in found_stats}
                    data_check = {"measures": measures, "values": values, "window": window,
                                  "window_ok": window in labels, "window_expected": sorted(labels)}
                else:
                    mark, found = (score(answer, q["expect"], forbid, abstain) if reply
                                   else ("✗", [False] * len(q["expect"])))
                call = ModelCall.objects.filter(conversation_id=data.get("id"), purpose="chat").first() \
                    if data.get("id") else None
                sources = reply.get("sources") or []
                graph = next((s for s in sources if s.get("kind") == "graph"), {})
                results.append({
                    "id": q["id"], "kind": q.get("kind", ""), "question": q["question"], "expected": q["answer"], "source": q["source"],
                    "mark": mark, "found": found, "seconds": round(time.time() - started, 1), "status": response.status_code,
                    "answer": answer, "wrong": wrong(answer, forbid) if reply and is_wrong(answer, found, forbid, abstain) else [],
                    "prompt_tokens": call.prompt_tokens if call else None,
                    "completion_tokens": call.completion_tokens if call else None,
                    "graph_facts": graph.get("snippet", ""),
                    "documents": [s.get("source") for s in sources if s.get("kind") not in ("graph", "historian")],
                    **data_check,
                })
                r = results[-1]
                self.stdout.write(f"{q['id']:>4} {mark} {sum(found)}/{len(found)}  {r['seconds']:>5}s  "
                                  f"{r['prompt_tokens'] or '-':>6} tok  {graph.get('snippet', '')}"
                                  + (f"  WRONG {r['wrong']}" if r["wrong"] else "")
                                  + (f"  window {'ok' if r['window_ok'] else 'WRONG ' + repr(r['window'])}" if measures else ""))
        summary = f"{opts['model']}: {_totals(results)}"
        tokens = [r["prompt_tokens"] for r in results if r["prompt_tokens"]]
        if tokens:
            summary += f"; prompt tokens avg {sum(tokens) // len(tokens)}, max {max(tokens)}"
        if results:
            summary += f"; {sum(r['seconds'] for r in results) / len(results):.1f} s per answer"
        self.stdout.write(summary)
        kinds = sorted({r["kind"] for r in results if r["kind"]})
        for kind in kinds if len(kinds) > 1 else ():
            self.stdout.write(f"  {kind}: {_totals([r for r in results if r['kind'] == kind])}")
        if opts["out"]:
            Path(opts["out"]).write_text(json.dumps({"set": ref["name"], "model": opts["model"],
                                                     "total_chars": opts["total_chars"] or None,
                                                     "context_tokens": opts["context_tokens"] or None,
                                                     "settings": opts["setting"], "summary": summary,
                                                     "results": results}, indent=1, ensure_ascii=False))

    def _rescore(self, ref: dict, path: Path):
        run = json.loads(path.read_text())
        key = {q["id"]: q for q in ref["questions"]}
        for r in run["results"]:
            q = key[r["id"]]
            forbid, abstain = q.get("forbid") or [], bool(q.get("abstain"))
            ok = r.get("status") == 200
            if q.get("measures"):                       # the numbers computed when it was asked
                r["mark"], r["found"] = score_data(r["answer"], q["measures"], r["values"]) if ok \
                    else ("✗", [False] * len(q["measures"]))
                r["expected"], r["kind"] = q["answer"], q.get("kind", "")
                self.stdout.write(f"{r['id']:>4} {r['mark']} {sum(r['found'])}/{len(r['found'])}")
                continue
            r["mark"], r["found"] = score(r["answer"], q["expect"], forbid, abstain) if ok else ("✗", [False] * len(q["expect"]))
            r["wrong"] = wrong(r["answer"], forbid) if ok and is_wrong(r["answer"], r["found"], forbid, abstain) else []
            r["expected"], r["kind"] = q["answer"], q.get("kind", "")
            self.stdout.write(f"{r['id']:>4} {r['mark']} {sum(r['found'])}/{len(r['found'])}"
                              + (f"  WRONG {r['wrong']}" if r["wrong"] else ""))
        run["summary"] = f"{run['model']}: {_totals(run['results'])}"
        self.stdout.write(run["summary"])
        kinds = sorted({r["kind"] for r in run["results"] if r["kind"]})
        for kind in kinds if len(kinds) > 1 else ():
            self.stdout.write(f"  {kind}: {_totals([r for r in run['results'] if r['kind'] == kind])}")
        path.write_text(json.dumps(run, indent=1, ensure_ascii=False))


def _typed(value: str):
    for kind in (int, float):
        try:
            return kind(value)
        except ValueError:
            pass
    return value


def _totals(results: list[dict]) -> str:
    marks = {m: sum(r["mark"] == m for r in results) for m in ("✓", "◐", "✗")}
    text = (f"{marks['✓']} ✓, {marks['◐']} ◐, {marks['✗']} ✗ of {len(results)}"
            f" (facts {sum(sum(r['found']) for r in results)}/{sum(len(r['found']) for r in results)})")
    wrong_answers = sum(bool(r["wrong"]) for r in results)
    return text + (f", {wrong_answers} wrong" if wrong_answers else "")

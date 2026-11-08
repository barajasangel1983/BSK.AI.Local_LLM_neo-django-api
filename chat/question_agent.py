"""The question agent (P24-2, ported from BSKLAB EDGE's P11e): reads the user's words before anything is searched.

A spoken or hurried question often holds several topics, filler words, or a name that came out wrong ("BEX80").
Searched as it is, no document excerpt is relevant enough. The agent returns one of:

    search   one to three short document searches, one per topic
    confirm  a name looks like a known one: ask "Did you mean BX80?"
    clarify  the question has no clear subject: ask one question back

It asks at most once: when the last reply in the conversation was its own question, it searches with what it has.
Its work is shown under the answer ("Searched for: …"). It never blocks a question: if the model fails or returns
something unusable, the question is searched as written. The answer model still receives the question exactly as
the user asked it. One engine for the Studio chat, EDGE and the Context Service (P24): the same agent for all.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
from dataclasses import dataclass, field

from usage import recorder as usage

logger = logging.getLogger("chat")

MAX_QUERIES = 3
NAMES_TTL = 300             # seconds the machine and document names are kept
NAME_SIMILARITY = 0.75      # how close a word must be to a known name to ask "did you mean"
_names_cache: dict = {"at": 0.0, "names": []}

INSTRUCTIONS = """You prepare a plant-floor question for a document search. You do not answer it.
Reply with one JSON object and nothing else:
{{"action": "search" | "clarify", "queries": ["...", "..."], "message": "..."}}

- "search" (the usual case): "queries" is a list of 1 to 3 plain strings, one per topic the user asked about,
  in the user's language. Each string names the subject the user named and one topic.
  Drop greetings and filler. Keep names, codes and numbers exactly as written. Do not add topics.
  Never add a machine or document the user did not name.
- "clarify": only when the question has no subject you can search for (for example "how do I install it?"
  with nothing earlier in the conversation). "message" is one short question to the user, in their language.
  When in doubt, search.

Example. User: "hola, dime el rango de temperatura del BX80 y cómo se limpia"
{{"action": "search", "queries": ["rango de temperatura del BX80", "limpieza del BX80"], "message": ""}}
{selected}{known}{asked}"""

CONFIRM = {"en": "Did you mean {right}? I heard “{wrong}”, and the closest name I know is {right}.",
           "es": "¿Quiso decir {right}? Entendí «{wrong}», y el nombre más parecido que conozco es {right}."}
NOTHING = {
    "en": "I searched the library for {queries} and found nothing relevant enough to answer from.{closest} "
          "Can you ask about one topic at a time, or name the document or section?",
    "es": "Busqué en la biblioteca {queries} y no encontré nada lo bastante relevante para responder.{closest} "
          "¿Puede preguntar por un tema a la vez, o indicar el documento o la sección?",
}
CLOSEST = {"en": " The closest documents were: {names}.", "es": " Los documentos más cercanos fueron: {names}."}


@dataclass
class Plan:
    action: str = "search"                  # search | confirm | clarify
    queries: list[str] = field(default_factory=list)
    message: str = ""                       # the question back to the user (confirm, clarify)
    language: str = "en"
    used: bool = False                      # False = the agent could not be used; searched as written

    def shown(self) -> dict:
        """What is kept with the reply and shown under it (a citation of kind "agent")."""
        return {"kind": "agent", "action": self.action, "queries": self.queries, "used": self.used,
                "language": self.language, "source": "Question agent"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]*", text)


def _codes(text: str) -> set[str]:
    """Words that look like a name or code: they hold a digit (EXTR01, BX80, CX0)."""
    return {t.upper() for t in _tokens(text) if len(t) >= 3 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)}


def known_names() -> list[str]:
    """Machine names and document titles (kept for a few minutes)."""
    if time.monotonic() - _names_cache["at"] < NAMES_TTL and _names_cache["names"]:
        return _names_cache["names"]
    from context_service import service
    names: list[str] = []
    try:
        names += [a["name"] for a in service.list_assets()]
        names += [re.sub(r"\.[A-Za-z0-9]{2,5}$", "", d["filename"]) for d in service.list_documents().get("documents", [])]
    except Exception as exc:          # the graph or the library is down: no names, the agent still runs
        logger.warning("question agent: names could not be read: %s", exc)
        return names
    _names_cache.update(at=time.monotonic(), names=names)
    return names


def near_miss(question: str, names: list[str]) -> tuple[str, str] | None:
    """(what was written, the known name) when a code in the question is close to a known one but not it."""
    known = sorted({code for name in names for code in _codes(name)})
    for written in sorted(_codes(question)):
        if written in known:
            continue
        match = difflib.get_close_matches(written, known, n=1, cutoff=NAME_SIMILARITY)
        if match:
            return written, match[0]
    return None


def already_asked(last_sources) -> bool:
    """The last reply was the agent's own question back (a Studio citation list or EDGE's sources dict)."""
    if isinstance(last_sources, dict):
        last_sources = [last_sources.get("agent") or {}]
    return any(isinstance(s, dict) and s.get("kind", "agent") == "agent" and s.get("action") in ("confirm", "clarify")
               for s in (last_sources or []))


def _language(text: str) -> str:
    return "es" if re.search(r"[áéíóúñ¿¡]|\b(el|la|los|las|del|que|cómo|como|para|una|qué|por)\b", text.lower()) else "en"


def _parse(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    match = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(match.group(0) if match else raw)
    if not isinstance(data, dict):
        raise ValueError("not an object")
    return data


def _complete(model_id: str, messages: list[dict]) -> str:
    from .views import _backend_stream
    with usage.scope(purpose="question-agent"):
        return "".join(_backend_stream(model_id, messages))


def prepare(model_id: str, question: str, history: list[dict], asset_name: str = "",
            asked_before: bool = False, has_picture: bool = False) -> Plan:
    """Decide what to search for, or what to ask back. `history`: [{role, content}], oldest first."""
    language = _language(question)
    names = known_names()

    # A name that is almost a known one is asked about, never silently corrected.
    if not asked_before:
        miss = near_miss(question, names)
        if miss:
            return Plan("confirm", [], CONFIRM[language].format(wrong=miss[0], right=miss[1]), language, used=True)

    system = INSTRUCTIONS.format(
        selected=f"\nThe user has selected the machine {asset_name}: it is the subject when none is named.\n" if asset_name else "",
        known=("\nNames in the plant, only to check spelling: " + "; ".join(names[:60]) + "\n") if names else "",
        asked=("\nThe user was already asked a question back, or a picture is attached. Use \"search\"; build the queries "
               "from the whole conversation.\n") if asked_before or has_picture else "")
    try:
        raw = _complete(model_id, [{"role": "system", "content": system}, *history, {"role": "user", "content": question}])
        data = _parse(raw)
    except Exception as exc:
        logger.warning("question agent: not used (%s: %s)", type(exc).__name__, str(exc)[:120])
        return Plan("search", [question], language=language)

    message = str(data.get("message") or "").strip()
    if data.get("action") == "clarify" and message and not (asked_before or has_picture or asset_name):
        return Plan("clarify", [], message[:400], language, used=True)
    queries = []
    for q in data.get("queries") or []:
        if isinstance(q, dict):         # a small model sometimes returns {"subject": …, "topic": …}
            q = " ".join(str(v) for v in q.values() if isinstance(v, str))
        if isinstance(q, str) and q.strip():
            queries.append(q.strip()[:200])
    # A search may only name a machine or document that the user named: the model is not trusted to add one.
    said = _codes(" ".join([question, asset_name, *(m["content"] for m in history)]))
    known = {code for name in names for code in _codes(name)}
    queries = [q for q in dict.fromkeys(queries) if not (_codes(q) & known) - said][:MAX_QUERIES]
    if not queries:
        return Plan("search", [question], language=language)
    return Plan("search", queries, language=language, used=True)


def nothing_found(plan: Plan, closest: list[dict]) -> str:
    """The reply when no excerpt was relevant enough: where it looked, never a guess."""
    queries = ", ".join(f"“{q}”" for q in plan.queries)
    names = ", ".join(c["document"] for c in closest)
    lang = plan.language if plan.language in NOTHING else "en"
    return NOTHING[lang].format(queries=queries, closest=CLOSEST[lang].format(names=names) if names else "")

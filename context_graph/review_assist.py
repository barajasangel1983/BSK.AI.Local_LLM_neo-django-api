"""P27 (2026-10-27): assisted review, a feasibility test. Nothing here changes a triple's status or the graph.

Each staged triple gets a tier from three layers, all on the Lab Ring (never Claude):

- `code_checks`: rules that need no model. A link the schema doesn't allow and a Safety fact without an exact quote
  are rejects; a Safety measure whose hazard can't be approved waits; several hazards from one quote are one notice
  split (sent to the reviewer as a group); flags from staging (pages disagree, likely duplicate) go to the reviewer.
- `duplicate_groups`: facts of the same kind repeated on another page (same anchor, near in embedding space, DGX
  embeddings). The first page is kept; the repeats are rejects, and the facts hung on a repeated hazard are compared
  with the kept hazard's (a new one would be approved under the kept hazard's name).
- `judge`: the DGX model asked about one fact, its quote and its passage, three times in three wordings. A pass is
  three votes of "stated, specific, applies".

`tier` joins them: reject / leave / reviewer / auto. Judge rules v2 (2026-11-06): a measure stated inside its hazard's
notice is "stated" without the word "protects" (v1 sent ~40 % of the Safety facts to a person, most of them right). The thresholds are not set here: `manage.py review_replay`
reports how the tiers compare with the review (Angel's decisions) or with a reference sheet.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests
from django.conf import settings

from . import quotes, triples as triple_ops
from .models import CandidateTriple

PASSAGE_CHARS = 1500    # the judge's view of the source around the quote
SIMILAR_NAME = 0.90     # cosine of two names at which two facts of the same kind on one anchor are the same fact
SIMILAR_QUOTE = 0.90    # the same for their quote sentences (0.95 on 10-27 missed p.43/p.90 at 0.91, 2026-11-06)
MIN_QUOTE = 30          # a shorter quote says too little to compare
NAME_FLOOR = 0.60       # a quote match also needs names this alike (stock sentences end different notices)
BOTH_CLOSE = 0.80       # or name and quote both this alike (p.86/p.84: 0.83 and 0.82): only a *possible* repeat.
# P27c C9 (2026-11-07): in that band right and wrong repeats score the same (0.82: SIMOTICS p.86/p.84 is one notice,
# gear units "improper startup" / "improper foundation" are two; "failsafe digital output" / "input" 0.87), so it
# sends the pair to the reviewer instead of rejecting, and the later fact keeps its own measures (no alias).
# Not used: a measure repeated under a differently named hazard. Stock steps ("Switch off and disconnect the machine",
# "Wait for complete standstill") repeat word for word under different hazards; merging on them joined rotating parts
# with electric shock (2026-11-06). Hazards merge on their own names and quotes, and their measures follow.
ANCHORS = ("ProductFamily", "DeviceModel", "Component")
JUDGE_TOKENS = 300

# --- code checks -----------------------------------------------------------------------------------------------------


def _exact(t: CandidateTriple) -> bool:
    return any(e.quote_status == "exact" for e in t.evidence.all())


def _quote_key(t: CandidateTriple):
    for e in t.evidence.all():
        if e.quote_status == "exact" and e.quote_start is not None:
            return (t.document_id, e.text_hash, e.quote_start, e.quote_end)
    return None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def code_checks(rows: list[CandidateTriple]) -> dict[int, tuple[str, str]]:
    """{triple id: (verdict, reason)} for the triples a rule decides; the others are absent."""
    out: dict[int, tuple[str, str]] = {}
    safety = [t for t in rows if triple_ops.is_safety(t)]
    hazards_blocked = set()   # (document, anchor, hazard name) whose HAS_HAZARD row can't be approved
    by_quote = defaultdict(list)
    for t in safety:
        if t.predicate == "HAS_HAZARD":
            if t.document_id and not _exact(t):
                hazards_blocked.add((t.document_id, t.anchor_id, _norm(t.object_name)))
            elif (k := _quote_key(t)):
                by_quote[k].append(t)
    for t in rows:
        issue = t.issue or ""
        if "not allowed by the schema" in issue:
            out[t.id] = ("reject", "link not allowed by the schema")
        elif triple_ops.is_safety(t) and t.document_id and not _exact(t):
            out[t.id] = ("reject", "Safety fact without an exact quote")
        elif (triple_ops.is_safety(t) and t.subject_type == "Hazard"
              and (t.document_id, t.anchor_id, _norm(t.subject_name)) in hazards_blocked):
            out[t.id] = ("leave", "its hazard has no exact quote")
        elif triple_ops.is_safety(t) and t.object_type == "SafetyMeasure" and len(t.object_name.split()) < 2:
            out[t.id] = ("reviewer", "one-word measure")
        elif issue:
            out[t.id] = ("reviewer", f"staging flag: {issue[:80]}")
    for group in by_quote.values():
        if len(group) > 1:
            names = ", ".join(sorted({g.object_name for g in group}))[:120]
            for g in group:
                out.setdefault(g.id, ("reviewer", f"one notice split into {len(group)} hazards: {names}"))
    return out


# --- duplicate groups ------------------------------------------------------------------------------------------------


def duplicate_groups(rows: list[CandidateTriple], quote_of: dict[int, str], embed=None,
                     kept_first: set[int] | None = None) -> tuple[dict, dict]:
    """Later repeats of a fact in one document: same anchor, predicate and types, and a near-identical name
    (SIMILAR_NAME) or quote sentence (SIMILAR_QUOTE). A repeated notice usually repeats its sentence; a fact named
    again in another notice usually repeats its name.

    Facts hung on the anchor (a family's hazards) go first; a repeated one is merged into the first ("alias"), so the
    facts hung on it (its measures) are compared with the kept one's. A measure is a repeat only under the same
    (possibly merged) owner.
    `kept_first`: ids that are always kept and come first in their group (facts already approved: a pending repeat of
    one is rejected). Returns ({triple id: ("reject", reason)}, {normalized repeat name: kept name}).
    `embed(texts) -> vectors`."""
    import numpy as np

    if embed is None:
        from ingestion.embedder import Embedder
        embed = Embedder().embed_passages

    def cosines(texts):
        v = np.array(embed(texts), dtype=float)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v @ v.T

    buckets = defaultdict(list)
    for t in rows:
        buckets[(t.document_id, t.anchor_id, t.predicate, t.subject_type, t.object_type)].append(t)
    kept_first = kept_first or set()
    out, alias = {}, {}
    owner = lambda t: t.subject_type in ANCHORS or alias.get(_norm(t.subject_name), _norm(t.subject_name))
    for key in sorted(buckets, key=lambda k: k[3] not in ANCHORS):
        group = sorted(buckets[key], key=lambda t: (t.id not in kept_first, t.page_start or 0, t.id))
        if len(group) < 2:
            continue
        names = cosines([t.object_name for t in group])
        has_quote = [len(quote_of.get(t.id, "")) >= MIN_QUOTE for t in group]
        spans = [_quote_key(t) for t in group]
        sents = cosines([quote_of.get(t.id, "") or t.object_name for t in group])
        kept: list[int] = []
        strong = lambda i, j: names[i, j] >= SIMILAR_NAME or (sents[i, j] >= SIMILAR_QUOTE and names[i, j] >= NAME_FLOOR)
        for i, t in enumerate(group):
            quoted = lambda j: has_quote[i] and has_quote[j] and spans[i] != spans[j]   # one sentence, several facts
            def same(j):
                if owner(group[j]) != owner(t):
                    return 0.0
                if names[i, j] >= SIMILAR_NAME:
                    return names[i, j]
                if not quoted(j):
                    return 0.0
                if sents[i, j] >= SIMILAR_QUOTE and names[i, j] >= NAME_FLOOR:
                    return sents[i, j]
                return min(sents[i, j], names[i, j]) if min(sents[i, j], names[i, j]) >= BOTH_CLOSE else 0.0
            match = max(kept, key=same, default=None)
            if match is None or not same(match):
                kept.append(i)
                continue
            k = group[match]
            if not strong(i, match):
                if t.id not in kept_first:
                    out[t.id] = ("reviewer", f"possible repeat of {k.id} (p.{k.page_start}, similarity {same(match):.2f})")
                kept.append(i)
                continue
            if t.id not in kept_first:
                out[t.id] = ("reject", f"repeat of {k.id} (p.{k.page_start}, similarity {same(match):.2f})")
            if key[3] in ANCHORS and _norm(t.object_name) != _norm(k.object_name):
                alias[_norm(t.object_name)] = _norm(k.object_name)
    return out, alias


# --- notices ---------------------------------------------------------------------------------------------------------


def _boundaries(text: str) -> list[int]:
    """Offsets of the lines that open a notice or a section: headings and signal-word lines. A heading right after a
    signal-word line is the notice's title ("## WARNING" / "## Hazardous voltage"), not a new section."""
    out, pos, after_signal = [], 0, False
    for line in text.splitlines(keepends=True):
        bare = line.strip()
        if bare:
            signal = bool(quotes.SIGNAL_ALONE_RE.match(bare))
            if signal or (bare.startswith("#") and not after_signal):
                out.append(pos)
            after_signal = signal
        pos += len(line)
    return out


def _block(text: str, start: int, end: int) -> tuple[int, int]:
    """The notice or section around [start, end): from the last opening line at or before it to the next one after."""
    import bisect
    marks = _boundaries(text)
    i = bisect.bisect_right(marks, start) - 1
    j = bisect.bisect_right(marks, max(start, end - 1))
    return (marks[i] if i >= 0 else 0), (marks[j] if j < len(marks) else len(text))


def notice_checks(rows: list[CandidateTriple], hazards: list[CandidateTriple] | None = None) -> dict[int, tuple[str, str]]:
    """{measure id: ("reviewer", reason)} for Safety measures whose quote is outside the notice of their hazard's quote
    (SIMOTICS p.87, 10-27: three measures hung on a hazard named after another notice). `hazards`: HAS_HAZARD facts to
    look the hazard up in (default: those in `rows`)."""
    def span(t):
        for e in t.evidence.all():
            if e.quote_status == "exact" and e.quote_start is not None:
                return e.document_id, e.text_hash, e.quote_start, e.quote_end
        return None
    by_name = {}
    for h in (hazards if hazards is not None else rows):
        if h.predicate == "HAS_HAZARD" and (sp := span(h)):
            by_name.setdefault((h.document_id, h.anchor_id, _norm(h.object_name)), sp)
    out, texts = {}, {}
    for t in rows:
        if t.predicate != "PROTECTED_BY" or t.subject_type != "Hazard" or not (sp := span(t)):
            continue
        hz = by_name.get((t.document_id, t.anchor_id, _norm(t.subject_name)))
        if not hz or hz[:2] != sp[:2]:
            continue
        text = texts.setdefault(sp[:2], quotes.snapshot_text(sp[0], sp[1]))
        if not text:
            continue
        lo, hi = _block(text, hz[2], hz[3])
        if not (lo <= sp[2] < hi):
            out[t.id] = ("reviewer", "its quote is outside its hazard's notice")
    return out


# --- the judge -------------------------------------------------------------------------------------------------------

JUDGE_RULES = """You check one fact taken from a manufacturer's manual against the manual's own text.

Answer three questions:
1. stated: is the fact said by the text itself? "no" when it is only inferred (a hazard guessed from an instruction,
   a function guessed from a test step), or when the text says something else.
2. specific: is it concrete? A clear instruction or a named danger is specific even when many products share it
   ("check the temperature of parts before touching them" is specific). "no" only for a referral ("observe the
   safety instructions", "see section 5", "observe the manufacturer's information"), an empty phrase ("take
   appropriate measures"), or a name that describes the consequence instead of the thing (for a hazard: what it is,
   e.g. "hot surface", not "minor injury").
3. applies: does it apply to the product named as the subject, rather than to another series, an accessory or
   another device?

Type rules: a Hazard is a danger the text names. A SafetyMeasure is something a person or the installer must do or
fit, as the text gives it. For a SafetyMeasure, "stated" is yes when the text gives that step in the same notice as
the hazard (its instructions or its list of steps): the text does not have to say that the step "protects" against the
hazard. It is "no" when the step belongs to another notice or another danger. A SafetyFunction is a safety function the device itself provides (Safe Torque Off, Safe
Stop 1, Safely-Limited Speed), not a protection or monitoring feature and not an instruction. Check that the fact's
name matches what its quote says.

Reply with JSON only: {"stated": "yes|no|unclear", "specific": "yes|no|unclear", "applies": "yes|no|unclear",
"reason": "one short sentence"}"""

WORDINGS = (
    "Check this fact against the text.",
    "Would an expert reviewer approve this fact, given only this text? Check it question by question.",
    "Look for a reason to reject this fact. If there is none, say yes to each question.",
)


def _fact_block(t: CandidateTriple) -> str:
    props = {k: v for k, v in (t.object_props or {}).items() if v not in ("", None)}
    return (f"Subject: {t.subject_name} ({t.subject_type})\nPredicate: {t.predicate}\n"
            f"Object: {t.object_name} ({t.object_type})" + (f"\nObject properties: {json.dumps(props)}" if props else ""))


def source_of(t: CandidateTriple) -> tuple[str, str]:
    """(quote, passage) for a triple: the exact quote when there is one, and the text around it."""
    # The evidence with an exact quote first: a fact found in several windows keeps its quote on any of them (PM PLUS
    # p.22, 11-06: the first window had none, so the judge saw no quote).
    for e in sorted(t.evidence.all(), key=lambda e: (e.quote_status != "exact", e.quote_start is None, e.pk)):
        text = quotes.snapshot_text(e.document_id, e.text_hash) if e.document_id else ""
        if text and e.quote_start is not None and e.quote_end is not None:
            half = (PASSAGE_CHARS - (e.quote_end - e.quote_start)) // 2
            return text[e.quote_start:e.quote_end], text[max(0, e.quote_start - max(half, 200)):e.quote_end + max(half, 200)]
        if text and e.char_start is not None and e.char_end is not None:
            return "", text[e.char_start:e.char_end][:PASSAGE_CHARS]
        if e.excerpt:
            return "", e.excerpt[:PASSAGE_CHARS]
    return "", (t.evidence_text or "")[:PASSAGE_CHARS]


def _post(messages: list[dict]) -> str:
    resp = requests.post(
        f"{settings.DGX_API_BASE.rstrip('/')}/v1/chat/completions",
        json={"model": settings.DGX_CHAT_MODEL, "messages": messages, "temperature": 0, "max_tokens": JUDGE_TOKENS,
              "response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": False}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


def parse_vote(content: str) -> dict:
    match = re.search(r"\{.*\}", content or "", re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except ValueError:
        data = {}
    vote = {k: str(data.get(k, "unclear")).strip().lower() for k in ("stated", "specific", "applies")}
    vote["reason"] = str(data.get("reason", ""))[:200]
    return vote


def judge(t: CandidateTriple, quote: str, passage: str, post=_post) -> list[dict]:
    """Three votes, one per wording. A failed call is a vote of "error"."""
    votes = []
    for wording in WORDINGS:
        user = (f"{wording}\n\nFACT\n{_fact_block(t)}\n\nQUOTE (the sentence the fact was taken from)\n"
                f"{quote or '(none found)'}\n\nTEXT AROUND IT\n{passage}")
        try:
            votes.append(parse_vote(post([{"role": "system", "content": JUDGE_RULES}, {"role": "user", "content": user}])))
        except requests.RequestException as exc:
            votes.append({"stated": "error", "specific": "error", "applies": "error", "reason": str(exc)[:200]})
    return votes


def passes(votes: list[dict]) -> bool:
    return bool(votes) and all(v[k] == "yes" for v in votes for k in ("stated", "specific", "applies"))


def judge_all(rows: list[CandidateTriple], sources: dict[int, tuple[str, str]], workers: int = 4, post=_post) -> dict:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda t: (t.id, judge(t, *sources[t.id], post=post)), rows)
        return dict(results)


# --- tiers -----------------------------------------------------------------------------------------------------------


def follow_hazards(rows: list[CandidateTriple], tiers: dict[int, tuple[str, str]], alias: dict[str, str],
                   outside: dict[tuple, str] | None = None) -> None:
    """A measure is approved automatically only if its hazard is: a measure of a hazard that is doubtful goes to the
    reviewer with it (SIMOTICS p.87: three measures under a hazard named after another notice). A repeated hazard's
    measures follow the kept hazard. A measure whose hazard is rejected, left or missing is left. `outside` gives the
    status of hazards staged on the same anchors but not in this run, by (document, anchor, name)."""
    hazard_tier = {}
    for t in rows:
        if t.predicate == "HAS_HAZARD" and t.subject_type in ANCHORS:
            hazard_tier[(t.document_id, t.anchor_id, _norm(t.object_name))] = tiers[t.id][0]
    for t in rows:
        if t.predicate != "PROTECTED_BY" or t.subject_type != "Hazard" or tiers[t.id][0] != "auto":
            continue
        name = alias.get(_norm(t.subject_name), _norm(t.subject_name))
        h = hazard_tier.get((t.document_id, t.anchor_id, name))
        if h is None and outside:
            h = {"approved": "approved", "pending": "pending", "rejected": "rejected"}.get(
                outside.get((t.document_id, t.anchor_id, name)))
        if h == "pending":
            continue  # staged but not part of this run (a sample): nothing to follow
        if h is None:
            # Its hazard was rejected or never staged (SIMOTICS p.35): approving the measure would hang it on a
            # hazard that isn't in the graph.
            tiers[t.id] = ("leave", "its hazard is not pending or approved")
        elif h in ("reject", "leave", "rejected"):
            tiers[t.id] = ("leave", f"its hazard is {h}")
        elif h not in ("auto", "approved"):
            tiers[t.id] = ("reviewer", f"its hazard is {h}")


def tier(code: tuple[str, str] | None, dup: tuple[str, str] | None, votes: list[dict] | None) -> tuple[str, str]:
    """(tier, reason): reject / leave (the code decides) → reject (repeat) → reviewer (a flag, or the judge doubts) →
    auto (the judge passes unanimously)."""
    if code and code[0] in ("reject", "leave"):
        return code
    if dup:
        return dup
    if code:
        return code
    if votes is None:
        return "reviewer", "not judged"
    if passes(votes):
        return "auto", "judge: 3 of 3"
    doubts = [v["reason"] for v in votes if not all(v[k] == "yes" for k in ("stated", "specific", "applies"))]
    return "reviewer", f"judge doubts ({len(doubts)} of 3): {doubts[0] if doubts else ''}"[:200]

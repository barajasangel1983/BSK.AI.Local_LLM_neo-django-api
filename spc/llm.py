"""The LLM analysis of an SPC tool (P12f): the packet, the prompt, one call to a model, and two checks.

A validation tool: what the SPC figures of one tool give a language model, and what it makes of them.
One call, no memory, nothing stored but the usage record. No document store, no graph, and never the
injected-anomaly record (FR-26). Specification: claude/P12_SPCimplementation.md (Studio web repository).

    packet(tool)            everything the page shows of the tool's window, as JSON
    analyze(packet, ...)    the prompt and the packet to a model; the answer, time and tokens
    check_numbers(...)      every number of the answer looked up in the packet
    check_verdict(...)      the model's closing verdict against what the packet says
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import requests
from django.conf import settings

from historian import source as historian_source
from usage import recorder as usage

from . import engine, service
from .models import SpcTool

FIRED_MAX = 60              # fired points listed one by one; the rest are counted
EVENTS_MAX = 50
MODELS = ("ollama-qwen3-8b", "dgx-qwen38-27b-fp8", "external-gpt")    # of the chat's models: those that take a long text
CONTROL_RULES = {rule for rule, kind in engine.RULES.items() if kind == engine.CONTROL}

DEFAULT_PROMPT = """You are a process engineer experienced in statistical process control (SPC). You are given the SPC data of one signal of an industrial machine as JSON. That JSON is everything you know about it.

Write a short analysis for a process engineer, in English, under these headings:
1. State of control - is the process in statistical control in this window? Name the points beyond the control limits and the rules that fired.
2. Capability - Cp, Cpk, Pp, Ppk against the specification, or why they are not available.
3. Events and patterns - the SPC events, and what the trend, the fired points and the gaps suggest: a shift of the level, a drift, a stuck value, a spike, more noise. Give times (UTC).
4. What to check on the machine - possible causes to look at. They are possibilities, not facts.
5. What the data cannot tell.

Rules:
- Quote figures only as they are in the JSON, with their units. Do not make up values, times or causes.
- When the JSON does not support a conclusion, say so.
- Take the room the analysis needs, within the answer's token limit: be complete, but do not repeat yourself.

How to read the JSON:
- "status": stable, warning, trending, out_of_control, spec_violation, paused or configuration_required; "status_reason" says why.
- "control_limits": per group (a recipe when "group_by" is set): CL, LCL, UCL of the values and r_cl, r_ucl of the moving range, from the baseline window named there.
- "sigma_shift": the mean of the last 30 minutes minus CL, in sigmas.
- "chart.fired_points": the points that fired a rule; "chart.gaps": stretches with no point (the machine not in the tool's states, or the first seconds after a start).
- "trend": the window's values as at most 60 means, each over the points from "from" to "to" (n points).
- Cpk of 1.33 or more is usually called capable.

End with this verdict, a JSON block in ```json fences, with exactly these keys:
{"in_control": true or false, "capable": true, false or null, "events_seen": <the number of SPC events in the data>, "patterns": [{"type": "shift" | "drift" | "stuck" | "spike" | "more_noise" | "none", "start": "<UTC time or null>", "end": "<UTC time or null>"}]}"""


class ModelError(Exception):
    """The model did not answer: unreachable, timed out, or refused the request."""


def _num(value, digits: int = 5):
    """A number short enough for a prompt: 5 significant digits."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return int(value) if isinstance(value, int) else float(f"{value:.{digits}g}")


def _iso(ts) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, str):
        return ts
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(ts) -> datetime:
    return ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


# --- the packet ----------------------------------------------------------------------------------

def packet(tool: SpcTool, end: datetime | None = None, window_hours: int | None = None) -> dict:
    """Everything the page shows of the tool's window (I-MR, distribution, capability, events) as one JSON object."""
    d = service.detail(tool, window_hours, end)
    samples = d.get("samples") or []
    start, stop = d.get("window_start"), d.get("window_end")
    period = historian_source.period_seconds()
    per_sample = 1 if tool.method == engine.IMR else tool.subgroup_size
    gap_seconds = max(3 * period * per_sample, 60)

    fired = [s for s in samples if s["rules"]]
    rule_counts = Counter(rule for s in fired for rule in s["rules"])
    gaps = []
    for a, b in zip(samples, samples[1:]):
        seconds = (_parse(b["ts"]) - _parse(a["ts"])).total_seconds()
        if seconds > gap_seconds:
            gaps.append({"from": _iso(a["ts"]), "to": _iso(b["ts"]), "minutes": round(seconds / 60, 1)})
    events = []
    total_events = 0
    if start is not None and stop is not None:
        found = tool.events.filter(started_at__lte=stop, last_at__gte=start).order_by("started_at")
        total_events = found.count()
        events = [{"type": e.type, "rule": e.rule, "group": e.group, "started_at": _iso(e.started_at), "last_at": _iso(e.last_at),
                   "points": e.count, "first_value": _num(e.value), "extreme_value": _num(e.extreme),
                   "machine_state": e.machine_state, "acknowledged": e.acknowledged_at is not None}
                  for e in found[:EVENTS_MAX]]
    values = [s["value"] for s in samples]
    spans = engine.downsample_spans([s["ts"] for s in samples])
    trend = [{"from": _iso(sp["from"]), "to": _iso(sp["to"]), "n": sp["n"], "mean": _num(v)}
             for sp, v in zip(spans, engine.downsample(values))]
    cap = d.get("capability") or {}
    box = d.get("box") or {}
    dist = d.get("distribution") or None
    limits = {lim.group: lim for lim in tool.limits.all()}

    return {
        "machine": tool.asset_id,
        "tool": {
            "name": tool.name, "signal_id": tool.signal_id, "historian_column": tool.column, "unit": tool.unit,
            "chart": "I-MR" if tool.method == engine.IMR else f"X-bar/R, subgroups of {tool.subgroup_size}",
            "applies_in_states": tool.states, "seconds_ignored_after_a_start": tool.ignore_seconds,
            "group_by": tool.group_by or None, "rules_on": tool.rules, "no_variation_count": tool.no_variation_count,
            "sigma_multiplier": _num(tool.sigma_multiplier),
            "specification": {"lsl": _num(tool.lsl), "target": _num(tool.target), "usl": _num(tool.usl)},
        },
        "window": {"start": _iso(start), "end": _iso(stop), "hours": d.get("window_hours"),
                   "seconds_between_rows": period, "source": historian_source.label()},
        "now": {"status": d.get("status"), "status_reason": d.get("status_reason") or None,
                "value": _num(d.get("current_value")), "group": d.get("current_group") or None,
                "machine_state": d.get("machine_state"), "sigma_shift": _num(d.get("sigma_shift"))},
        "control_limits": [
            {"group": group or None, "source": lim.source, "baseline_start": _iso(lim.window_start), "baseline_end": _iso(lim.window_end),
             "baseline_points": lim.n_samples, "mean": _num(lim.mean), "sigma_within": _num(lim.sigma_within),
             "sigma_overall": _num(lim.sigma_overall), "cl": _num(lim.cl), "lcl": _num(lim.lcl), "ucl": _num(lim.ucl),
             "r_cl": _num(lim.r_cl), "r_ucl": _num(lim.r_ucl)}
            for group, lim in limits.items()],
        "groups_without_limits": d.get("groups_without_limits") or [],
        "chart": {
            "points": len(samples), "points_that_fired": len(fired), "fired_by_rule": dict(rule_counts),
            "fired_points": [{"ts": _iso(s["ts"]), "value": _num(s["value"]), "group": s["group"] or None, "rules": s["rules"]}
                             for s in fired[:FIRED_MAX]],
            "fired_points_not_listed": max(0, len(fired) - FIRED_MAX),
            "gaps": gaps,
            "group_runs": [{"group": lim["group"] or None, "from": _iso(sp["start"]), "to": _iso(sp["end"])}
                           for lim in d.get("limits") or [] for sp in lim.get("spans") or []] if tool.group_by else [],
        },
        "trend": trend,
        "distribution": None if not values else {
            "n": cap.get("sample_count"), "mean": _num(cap.get("mean")), "median": _num(cap.get("median")),
            "stdev_overall": _num(cap.get("sigma_overall")), "stdev_within": _num(cap.get("sigma_within")),
            "min": _num(min(values)), "max": _num(max(values)),
            "box_plot": {k: _num(box.get(k)) for k in ("q1", "median", "q3", "whisker_low", "whisker_high")}
            | {"values_beyond_whiskers": box.get("outlier_count")} if box else None,
            "quantiles": {f"{q['percent']:g}%": _num(q["value"]) for q in box.get("quantiles", [])} if box else None,
            "histogram": {"from": _num(dist["min"]), "to": _num(dist["max"]), "bins": len(dist["counts"]), "counts": dist["counts"]} if dist else None,
        },
        "capability": {"cp": _num(cap.get("cp")), "cpk": _num(cap.get("cpk")), "pp": _num(cap.get("pp")), "ppk": _num(cap.get("ppk")),
                       "not_available_because": cap.get("reason") or None},
        "events": {"in_window": total_events, "listed": events},
    }


def packet_text(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


USER_HEAD = "SPC data (JSON):\n"
CHARS_PER_TOKEN = 1.8       # a little high: Qwen3 8B read 5,098 tokens for 9,700 characters of prompt and packet (2026-10-14)


def messages_for(prompt: str, data: dict) -> list[dict]:
    return [{"role": "system", "content": prompt}, {"role": "user", "content": USER_HEAD + packet_text(data)}]


def models() -> list[dict]:
    """Of the chat's models, those offered here, each with its capacity in tokens and where that figure comes from."""
    from chat.views import model_options
    out = []
    for m in model_options():
        if m["id"] in MODELS:
            cap, source = capacity(m["id"])
            out.append({**m, "capacity": cap, "capacity_from": source})
    return out


# --- one call to a model -------------------------------------------------------------------------

THINK = re.compile(r"<think>.*?</think>", re.S)


def analyze(data: dict, prompt: str, model_id: str, input_limit: int | None = None, output_limit: int | None = None) -> dict:
    """The prompt (system) and the packet (user) to the model, once, within the run's input and output limits.
    Returns the answer with time and tokens."""
    if model_id not in MODELS:
        raise service.SpcError(f"model must be one of {', '.join(MODELS)}")
    if not prompt.strip():
        raise service.SpcError("the prompt is empty")
    messages = messages_for(prompt, data)
    input_limit, output_limit, cap = check_limits(model_id, settings.SPC_LLM_INPUT_LIMIT if input_limit is None else input_limit,
                                                  settings.SPC_LLM_OUTPUT_LIMIT if output_limit is None else output_limit, messages)
    context = ollama_context(input_limit, output_limit, cap) if model_id == "ollama-qwen3-8b" else None
    started = time.monotonic()
    timing: dict = {}
    try:
        with usage.track("spc", model_id) as call:
            if model_id == "ollama-qwen3-8b":
                raw = _ollama(messages, context, output_limit)
                ns = 1e9
                timing = {"load_seconds": round(raw.get("load_duration", 0) / ns, 2),
                          "input_seconds": round(raw.get("prompt_eval_duration", 0) / ns, 2),
                          "output_seconds": round(raw.get("eval_duration", 0) / ns, 2)}
                answer = (raw.get("message") or {}).get("content") or ""
                finished = raw.get("done_reason")
            else:
                raw = _openai(model_id, messages, output_limit)
                answer = raw["choices"][0]["message"]["content"] or ""
                finished = raw["choices"][0].get("finish_reason")
            call.from_response(raw)
            call.estimate(usage.messages_text(messages), answer)
    except requests.Timeout as exc:
        raise ModelError(f"{model_id} did not answer within {settings.SPC_LLM_TIMEOUT:g} s") from exc
    except requests.RequestException as exc:
        raise ModelError(f"{model_id} could not be reached: {exc}") from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise ModelError(f"{model_id} gave an answer that could not be read") from exc
    seconds = round(time.monotonic() - started, 2)
    tokens_in, tokens_out = call.prompt_tokens, call.completion_tokens
    for key, count in (("input", tokens_in), ("output", tokens_out)):
        if timing.get(f"{key}_seconds") and count:
            timing[f"{key}_tokens_per_second"] = round(count / timing[f"{key}_seconds"], 1)
    if tokens_out and seconds and "output_tokens_per_second" not in timing:
        timing["output_tokens_per_second_overall"] = round(tokens_out / seconds, 1)
    answer = THINK.sub("", answer).strip()
    return {"model": model_id, "answer": answer, "seconds": seconds, "input_tokens": tokens_in, "output_tokens": tokens_out,
            "tokens_estimated": call.estimated, "timing": timing,
            # "length": the answer reached the output limit and was cut
            "finished": finished or None, "cut": finished == "length",
            "settings": {"temperature": settings.SPC_LLM_TEMPERATURE, "input_limit": input_limit, "output_limit": output_limit,
                         "capacity": cap, **({"context_tokens": context} if context else {}), "thinking": False}}


MIN_LIMIT = 256
_capacity_cache: dict[str, tuple[float, int, str]] = {}


def capacity(model_id: str) -> tuple[int, str]:
    """The model's context in tokens (input and output together) and where it comes from: asked of the model
    (Ollama's model info, vLLM's max_model_len; kept 10 minutes), or SPC_LLM_CONTEXT when it cannot be asked."""
    cached = _capacity_cache.get(model_id)
    if cached and time.monotonic() - cached[0] < 600:
        return cached[1], cached[2]
    found = None
    try:
        if model_id == "ollama-qwen3-8b":
            info = requests.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/show", json={"model": "qwen3:8b"}, timeout=5).json()
            found = next((v for k, v in (info.get("model_info") or {}).items() if k.endswith(".context_length")), None)
        elif model_id == "dgx-qwen38-27b-fp8":
            listed = requests.get(f"{settings.DGX_API_BASE.rstrip('/')}/v1/models", timeout=5).json().get("data") or []
            found = next((m.get("max_model_len") for m in listed if m.get("id") == settings.DGX_CHAT_MODEL), None)
    except (requests.RequestException, ValueError):
        found = None
    result = (int(found), "model") if found else (settings.SPC_LLM_CONTEXT[model_id], "setting")
    _capacity_cache[model_id] = (time.monotonic(), *result)
    return result


def estimate_tokens(messages: list[dict]) -> int:
    """The input in tokens, estimated a little high (JSON full of figures costs about a token a digit). The page
    makes the same estimate as the prompt is edited. The model's own count comes with its answer."""
    return math.ceil(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN)


def check_limits(model_id: str, input_limit, output_limit, messages: list[dict]) -> tuple[int, int, int]:
    """The run's limits, refused when they are not whole numbers of at least MIN_LIMIT, when together they pass the
    model's capacity, or when the input is estimated above its limit. Returns (input limit, output limit, capacity)."""
    cap, _ = capacity(model_id)
    limits = []
    for name, value in (("input", input_limit), ("output", output_limit)):
        if isinstance(value, bool) or not isinstance(value, int) or value < MIN_LIMIT:
            raise service.SpcError(f"the {name} limit must be a whole number of tokens, at least {MIN_LIMIT}")
        limits.append(value)
    if limits[0] + limits[1] > cap:
        raise service.SpcError(f"the input and output limits together ({limits[0] + limits[1]:,}) pass the model's "
                               f"capacity of {cap:,} tokens")
    estimate = estimate_tokens(messages)
    if estimate > limits[0]:
        raise service.SpcError(f"the prompt and the JSON are about {estimate:,} tokens, above the input limit of {limits[0]:,}")
    return limits[0], limits[1], cap


def ollama_context(input_limit: int, output_limit: int, cap: int) -> int:
    """num_ctx for Ollama: room for both limits, in steps of 4,096, at least SPC_LLM_NUM_CTX (the Ollama app's setting:
    no reload of the model while it is enough), at most the model's capacity."""
    wanted = math.ceil((input_limit + output_limit) / 4096) * 4096
    return min(max(wanted, settings.SPC_LLM_NUM_CTX), cap)


def _ollama(messages: list[dict], context: int, output_limit: int) -> dict:
    """Ollama's chat API. num_ctx is sent (context_tokens), so that the call does not depend on the Ollama app's
    setting: a prompt longer than the context is cut without an error. The app's setting on bsk-ai is 16,384 (a call
    without num_ctx read all 5,098 tokens on 2026-10-14); a larger num_ctx makes Ollama load the model again.
    num_predict -1: no limit on the answer but the context."""
    resp = requests.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", timeout=settings.SPC_LLM_TIMEOUT, json={
        "model": "qwen3:8b", "messages": messages, "stream": False, "think": False,
        "options": {"num_ctx": context, "temperature": settings.SPC_LLM_TEMPERATURE,
                    "num_predict": output_limit}})
    resp.raise_for_status()
    return resp.json()


def _openai(model_id: str, messages: list[dict], output_limit: int) -> dict:
    """vLLM on the DGX, or Grok: the OpenAI chat format."""
    body = {"messages": messages, "temperature": settings.SPC_LLM_TEMPERATURE, "max_tokens": output_limit}
    headers = {"Content-Type": "application/json"}
    if model_id == "external-gpt":
        if not settings.GROK_API_KEY:
            raise ModelError("the Grok key is not set (GROK_API_KEY)")
        url = f"{settings.GROK_API_BASE.rstrip('/')}/chat/completions"
        body["model"] = settings.GROK_CHAT_MODEL
        headers["Authorization"] = f"Bearer {settings.GROK_API_KEY}"
    else:
        url = f"{settings.DGX_API_BASE.rstrip('/')}/v1/chat/completions"
        body.update(model=settings.DGX_CHAT_MODEL, chat_template_kwargs={"enable_thinking": False})
    resp = requests.post(url, json=body, headers=headers, timeout=settings.SPC_LLM_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# --- check 1: the numbers of the answer ----------------------------------------------------------

VERDICT = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?Z?)?\b")
TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\b")
NUMBER = re.compile(r"(?<![\w.])[-−]?\d+(?:[.,]\d+)*(?:\.\d+)?(?![\w])")
# Common SPC figures a text may name without them being in the data: rule thresholds, the capability target.
KNOWN = {1.33, 1.67, 1.5, 2.0, 3.0, 99.73, 0.27, 68.27, 95.45}
SMALL = 10                  # whole numbers up to this are counts, headings or rule sizes: not checked


def _packet_numbers(data) -> list[float]:
    out: list[float] = []
    if isinstance(data, dict):
        for value in data.values():
            out += _packet_numbers(value)
    elif isinstance(data, list):
        for value in data:
            out += _packet_numbers(value)
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        out.append(float(data))
    return out


def prose(answer: str) -> str:
    """The answer without its closing verdict block."""
    return VERDICT.sub("", answer)


def check_numbers(answer: str, data: dict) -> dict:
    """Every number of the answer's text, looked up in the packet at the precision the answer gives it.
    Dates and times are left out (check_verdict looks at times); small whole numbers and common SPC figures are not checked."""
    text = TIME.sub(" ", DATE.sub(" ", prose(answer)))
    known = _packet_numbers(data)
    checked, ignored, missing = 0, 0, []
    for m in NUMBER.finditer(text):
        raw = m.group(0).replace("−", "-").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if (decimals == 0 and abs(value) <= SMALL) or abs(value) in KNOWN:
            ignored += 1
            continue
        checked += 1
        tolerance = 0.5 * 10 ** -decimals + 1e-9
        if any(abs(abs(p) - abs(value)) <= tolerance or abs(abs(p) * 100 - abs(value)) <= tolerance for p in known):
            continue
        a, b = max(0, m.start() - 40), min(len(text), m.end() + 40)
        missing.append({"value": raw, "context": " ".join(text[a:b].split())})
    return {"checked": checked, "found": checked - len(missing), "not_checked": ignored, "not_found": missing}


# --- check 2: the verdict against the packet -----------------------------------------------------

def verdict(answer: str) -> dict | None:
    """The answer's closing JSON verdict, or None when there is none that can be read."""
    blocks = VERDICT.findall(answer)
    for raw in reversed(blocks):
        try:
            found = json.loads(raw)
        except ValueError:
            continue
        if isinstance(found, dict) and "in_control" in found:
            return found
    return None


def _times_in_window(text: str, data: dict) -> list[str]:
    """Times (HH:MM) of the answer that fall outside the window (5 minutes either side), on any of its days."""
    start, end = data["window"].get("start"), data["window"].get("end")
    if not start or not end:
        return []
    start, end = _parse(start) - timedelta(minutes=5), _parse(end) + timedelta(minutes=5)
    days = {start.date(), end.date()}
    outside = []
    for m in TIME.finditer(DATE.sub(lambda d: d.group(0)[11:] if len(d.group(0)) > 10 else " ", text)):
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 23 or minute > 59:
            continue
        candidates = [datetime(d.year, d.month, d.day, hour, minute, tzinfo=timezone.utc) for d in days]
        if not any(start <= c <= end for c in candidates):
            outside.append(m.group(0))
    return outside


def check_verdict(answer: str, data: dict) -> dict:
    """The model's verdict item by item against what the packet says; `agree` is None when the data says nothing."""
    found = verdict(answer)
    if found is None:
        return {"found": False, "items": [], "times_outside_window": _times_in_window(prose(answer), data)}
    chart = data.get("chart") or {}
    control_points = sum(n for rule, n in (chart.get("fired_by_rule") or {}).items() if rule in CONTROL_RULES)
    in_control = control_points == 0
    cpk = (data.get("capability") or {}).get("cpk")
    capable = None if cpk is None else cpk >= 1.33
    events = (data.get("events") or {}).get("in_window")
    items = [
        {"item": "In control", "model": found.get("in_control"), "data": in_control,
         "agree": found.get("in_control") is in_control,
         "basis": f"{control_points} point(s) beyond the control limits or the range limit in the window"},
        {"item": "Capable (Cpk ≥ 1.33)", "model": found.get("capable"), "data": capable,
         "agree": None if capable is None and found.get("capable") is None else found.get("capable") is capable,
         "basis": f"Cpk {cpk}" if cpk is not None else "no Cpk in the data"},
        {"item": "SPC events", "model": found.get("events_seen"), "data": events,
         "agree": found.get("events_seen") == events, "basis": "events in the window"},
    ]
    patterns = [p for p in found.get("patterns") or [] if isinstance(p, dict)]
    times = " ".join(str(p.get(k) or "") for p in patterns for k in ("start", "end"))
    return {"found": True, "items": items, "patterns": patterns,
            "times_outside_window": _times_in_window(prose(answer) + " " + times, data)}

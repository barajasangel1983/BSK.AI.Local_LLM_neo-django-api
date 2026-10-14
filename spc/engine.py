"""The SPC engine: from historian rows to samples, limits, rule violations, capability and status.

Pure functions. No database, no graph, no HTTP and no clock: the same rows and the same
configuration always give the same answer. Specification: claude/P12_SPCimplementation.md
(Studio web repository), section 5. No model computes anything here.

Two kinds of limits are kept apart everywhere:
    specification  LSL / target / USL   what the product must meet (typed by the user)
    control        LCL / CL / UCL       what the process does when it is in control (from its own variation)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal

import numpy as np

IMR, XBAR_R = "imr", "xbar_r"

# Control chart constants by subgroup size n (I-MR uses n = 2: a moving range of two).
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
D3 = {2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848, 7: 0.833, 8: 0.820, 9: 0.808, 10: 0.797}

CONTROL, SPEC, WARNING = "control_violation", "spec_violation", "statistical_warning"
# rule id → the type of event it raises
RULES = {
    "beyond_limits": CONTROL, "range_beyond_limits": CONTROL, "spec": SPEC,
    "two_of_three": WARNING, "four_of_five": WARNING, "eight_one_side": WARNING, "six_trending": WARNING,
    "no_variation": WARNING,
}
# Ticked when a new tool's block opens. The range chart and the run rules are switches the user turns on per tool:
# on single 20 s rows they raise about 2.7 events per tool and running hour on a healthy process (P12b, live trial).
DEFAULT_RULES = ["beyond_limits", "spec"]

STATUSES = ("paused", "configuration_required", "spec_violation", "out_of_control", "trending", "warning", "stable")


@dataclass
class Sample:
    """One point of the chart: a row (I-MR) or the mean of a subgroup (X-bar/R)."""
    ts: object
    value: float
    range: float | None         # moving range (None for the first sample of a segment) or subgroup range
    group: str                  # the value of the tool's `group_by` column, "" without one
    state: str
    segment: int                # samples of one segment are consecutive rows in one state and one group
    rows: list[float] = field(default_factory=list)     # the row values behind the sample


@dataclass
class Limits:
    group: str
    n_samples: int
    mean: float
    sigma_within: float
    sigma_overall: float | None
    range_bar: float
    cl: float
    lcl: float
    ucl: float
    r_cl: float
    r_lcl: float
    r_ucl: float

    def as_dict(self) -> dict:
        return asdict(self)


def chart_n(method: str, subgroup_size: int) -> int:
    """The n of the constants: 2 for a moving range, else the subgroup size."""
    return 2 if method == IMR else int(subgroup_size)


def sigma_point(limits, method: str, subgroup_size: int) -> float:
    """The standard deviation of a plotted point: of a row, or of a subgroup's mean."""
    return limits.sigma_within if method == IMR else limits.sigma_within / math.sqrt(subgroup_size)


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


# --- 5.1 from rows to samples -------------------------------------------------------------------

def samples(rows: list[dict], *, column: str, states: list[str], ignore_seconds: int, method: str,
            subgroup_size: int, group_by: str, period: int) -> list[Sample]:
    """The chart's points from rows in time order ({ts, machine_state, <column>, <group_by>}).

    A segment is a run of rows in the tool's states, at most two periods apart, with one state and
    one group. The first `ignore_seconds` of every segment are dropped, and nothing (a moving
    range, a subgroup) is ever computed across two segments.

    A subgroup is the rows of one stretch of `subgroup_size` periods on the clock (three 20 s rows:
    each whole minute), so the same rows give the same subgroups wherever a read begins. A stretch
    with a row missing, or cut by the start or end of a segment, gives no subgroup."""
    segments: list[list[tuple]] = []            # [(ts, value, group, state), …]
    current: list[tuple] = []
    for row in rows:
        state = row.get("machine_state") or ""
        if state not in states:
            current = []                        # another state ends the segment
            continue
        value = _number(row.get(column))
        if value is None:
            continue                            # a missing value is skipped; a long gap ends the segment below
        group = str(row.get(group_by) or "") if group_by else ""
        if current:
            _, _, last_group, last_state = current[-1]
            gap = (row["ts"] - current[-1][0]).total_seconds()
            if gap > 2 * period or group != last_group or state != last_state:
                current = []
        if not current:
            segments.append(current)
        current.append((row["ts"], value, group, state))

    out: list[Sample] = []
    for number, segment in enumerate(segments):
        first = segment[0][0]
        kept = [r for r in segment if (r[0] - first).total_seconds() >= ignore_seconds]
        if method == IMR:
            previous = None
            for ts, value, group, state in kept:
                out.append(Sample(ts, value, None if previous is None else abs(value - previous), group, state, number, [value]))
                previous = value
        else:
            width = subgroup_size * period
            buckets: dict[int, list[tuple]] = {}
            for r in kept:
                buckets.setdefault(int(r[0].timestamp()) // width, []).append(r)
            for part in buckets.values():
                if len(part) != subgroup_size:
                    continue
                values = [r[1] for r in part]
                out.append(Sample(part[-1][0], sum(values) / len(values), max(values) - min(values), part[-1][2], part[-1][3],
                                  number, values))
    return out


# --- 5.3 and 5.4 limits -------------------------------------------------------------------------

def _limits(group: str, n_samples: int, mean: float, sigma_within: float, sigma_overall, range_bar: float, *,
            method: str, subgroup_size: int, k: float, cl=None, lcl=None, ucl=None, r_ucl=None) -> Limits:
    n = chart_n(method, subgroup_size)
    point = sigma_within if method == IMR else sigma_within / math.sqrt(subgroup_size)
    spread = k * D3[n] * sigma_within
    return Limits(group=group, n_samples=n_samples, mean=mean, sigma_within=sigma_within, sigma_overall=sigma_overall,
                  range_bar=range_bar, cl=mean if cl is None else cl, lcl=mean - k * point if lcl is None else lcl,
                  ucl=mean + k * point if ucl is None else ucl, r_cl=range_bar, r_lcl=max(0.0, range_bar - spread),
                  r_ucl=range_bar + spread if r_ucl is None else r_ucl)


def baseline_limits(found: list[Sample], *, method: str, subgroup_size: int, k: float,
                    min_samples: int) -> tuple[dict[str, Limits], list[dict]]:
    """Control limits per group from the samples of a baseline window: ({group: Limits}, skipped groups)."""
    n = chart_n(method, subgroup_size)
    limits: dict[str, Limits] = {}
    skipped: list[dict] = []
    for group in sorted({s.group for s in found}):
        part = [s for s in found if s.group == group]
        ranges = [s.range for s in part if s.range is not None]
        if len(part) < min_samples or not ranges:
            skipped.append({"group": group, "n_samples": len(part), "reason": f"fewer than {min_samples} samples"})
            continue
        range_bar = float(np.mean(ranges))
        sigma_within = range_bar / D2[n]
        if sigma_within <= 0:
            skipped.append({"group": group, "n_samples": len(part), "reason": "no variation in the baseline"})
            continue
        row_values = [v for s in part for v in s.rows]
        limits[group] = _limits(group, len(part), float(np.mean([s.value for s in part])), sigma_within,
                                float(np.std(row_values, ddof=1)), range_bar, method=method, subgroup_size=subgroup_size, k=k)
    return limits, skipped


def fixed_limits(group: str, cl: float, lcl: float, ucl: float, r_ucl: float | None, *, method: str,
                 subgroup_size: int, k: float) -> Limits:
    """Control limits typed by the user. The sigma they imply is (UCL − CL) / k."""
    if not lcl < cl < ucl:
        raise ValueError("fixed limits need LCL < CL < UCL")
    point = (ucl - cl) / k
    sigma_within = point if method == IMR else point * math.sqrt(subgroup_size)
    range_bar = D2[chart_n(method, subgroup_size)] * sigma_within
    return _limits(group, 0, cl, sigma_within, None, range_bar, method=method, subgroup_size=subgroup_size, k=k,
                   cl=cl, lcl=lcl, ucl=ucl, r_ucl=r_ucl)


# --- 5.5 rules ----------------------------------------------------------------------------------

def judge(found: list[Sample], limits: dict[str, object], *, rules: list[str], method: str, subgroup_size: int,
          lsl: float | None = None, usl: float | None = None, no_variation_count: int = 9) -> list[list[str]]:
    """For every sample, the ids of the rules that fire on it (in the order of RULES).

    `limits` maps a group to an object with cl, lcl, ucl, r_lcl, r_ucl and sigma_within. A sample
    whose group has no limits is not judged. The run rules look back only inside the sample's segment."""
    on = [r for r in RULES if r in rules]
    out: list[list[str]] = []
    history: list[float] = []
    segment = None
    for s in found:
        if s.segment != segment:
            segment, history = s.segment, []
        lim = limits.get(s.group)
        if lim is None:
            out.append([])
            history = []
            continue
        history.append(s.value)
        sigma = sigma_point(lim, method, subgroup_size)
        fired = []
        for rule in on:
            if rule == "beyond_limits":
                hit = s.value > lim.ucl or s.value < lim.lcl
            elif rule == "range_beyond_limits":
                hit = s.range is not None and (s.range > lim.r_ucl or (lim.r_lcl > 0 and s.range < lim.r_lcl))
            elif rule == "spec":
                hit = any((usl is not None and v > usl) or (lsl is not None and v < lsl) for v in s.rows)
            elif rule == "two_of_three":
                hit = _zone(history, lim.cl, 2 * sigma, 3, 2)
            elif rule == "four_of_five":
                hit = _zone(history, lim.cl, sigma, 5, 4)
            elif rule == "eight_one_side":
                last = history[-8:]
                hit = len(last) == 8 and (all(v > lim.cl for v in last) or all(v < lim.cl for v in last))
            elif rule == "six_trending":
                last = history[-6:]
                steps = [b - a for a, b in zip(last, last[1:])]
                hit = len(last) == 6 and (all(d > 0 for d in steps) or all(d < 0 for d in steps))
            else:   # no_variation
                last = history[-no_variation_count:]
                hit = len(last) == no_variation_count and all(v == last[0] for v in last)
            if hit:
                fired.append(rule)
        out.append(fired)
    return out


def _zone(history: list[float], cl: float, distance: float, window: int, needed: int) -> bool:
    """The newest point and at least `needed` − 1 of the `window` − 1 before it are beyond cl ± distance, same side."""
    current = history[-1]
    side = 1 if current > cl + distance else -1 if current < cl - distance else 0
    if side == 0:
        return False
    return sum(1 for v in history[-window:] if (v - cl) * side > distance) >= needed


# --- 5.6 capability -----------------------------------------------------------------------------

def capability(found: list[Sample], *, method: str, subgroup_size: int, lsl: float | None, usl: float | None,
               min_samples: int) -> dict:
    """Cp and Cpk from the variation within (ranges), Pp and Ppk from the overall variation. Never mixed.

    One specification limit gives the one-sided Cpk and Ppk only. When the figures would not be
    valid, all four are None and `reason` says why."""
    values = [v for s in found for v in s.rows]
    ranges = [s.range for s in found if s.range is not None]
    out = {"lsl": lsl, "usl": usl, "sample_count": len(values), "mean": None, "median": None, "sigma_within": None,
           "sigma_overall": None, "cp": None, "cpk": None, "pp": None, "ppk": None, "reason": ""}
    if values:
        out["mean"], out["median"] = float(np.mean(values)), float(np.median(values))
    if ranges:
        out["sigma_within"] = float(np.mean(ranges)) / D2[chart_n(method, subgroup_size)]
    if len(values) > 1:
        out["sigma_overall"] = float(np.std(values, ddof=1))
    if lsl is None and usl is None:
        out["reason"] = "no specification limit"
    elif len(values) < min_samples:
        out["reason"] = f"fewer than {min_samples} values in the window"
    elif not out["sigma_within"]:
        out["reason"] = "no variation in the window"
    if out["reason"]:
        return out

    def figures(sigma):
        if not sigma:
            return None, None
        sides = [(usl - out["mean"]) / (3 * sigma)] if usl is not None else []
        sides += [(out["mean"] - lsl) / (3 * sigma)] if lsl is not None else []
        both = (usl - lsl) / (6 * sigma) if lsl is not None and usl is not None else None
        return both, min(sides)
    out["cp"], out["cpk"] = figures(out["sigma_within"])
    out["pp"], out["ppk"] = figures(out["sigma_overall"])
    return out


# --- 5.7 sigma shift ----------------------------------------------------------------------------

def sigma_shift(found: list[Sample], limits: dict[str, object], *, recent_seconds: int, min_samples: int = 10) -> float | None:
    """How far the recent mean of the newest sample's group is from its baseline mean, in baseline sigmas."""
    if not found:
        return None
    newest = found[-1]
    lim = limits.get(newest.group)
    if lim is None or not lim.sigma_within:
        return None
    recent = [s.value for s in found if s.group == newest.group and (newest.ts - s.ts).total_seconds() <= recent_seconds]
    if len(recent) < min_samples:
        return None
    return (float(np.mean(recent)) - lim.mean) / lim.sigma_within


# --- 5.8 status ---------------------------------------------------------------------------------

def status(*, enabled: bool, has_column: bool, has_limits: bool, group_has_limits: bool, group: str = "",
           active_events: list[dict] | None = None) -> tuple[str, str]:
    """(status, reason): the first line of the table in 5.8 that applies. `active_events` are {type, rule}."""
    events = active_events or []
    if not enabled:
        return "paused", ""
    if not has_column:
        return "configuration_required", "the signal has no historian column"
    if not has_limits:
        return "configuration_required", "no control limits yet: compute a baseline or enter fixed limits"
    if not group_has_limits:
        return "configuration_required", f"no control limits for {group}" if group else "no control limits"
    for wanted, name in ((SPEC, "spec_violation"), (CONTROL, "out_of_control")):
        hit = next((e for e in events if e["type"] == wanted), None)
        if hit:
            return name, hit["rule"]
    if any(e["rule"] == "six_trending" for e in events):
        return "trending", "six_trending"
    hit = next((e for e in events if e["type"] == WARNING), None)
    return ("warning", hit["rule"]) if hit else ("stable", "")


# --- 5.9 what a chart gets ----------------------------------------------------------------------

def downsample(values: list[float], points: int = 60) -> list[float]:
    """At most `points` values: the means of consecutive blocks of equal size."""
    if len(values) <= points:
        return [round(v, 4) for v in values]
    size = math.ceil(len(values) / points)
    return [round(float(np.mean(values[i:i + size])), 4) for i in range(0, len(values), size)]


QUANTILES = (100, 99.5, 97.5, 90, 75, 50, 25, 10, 2.5, 0.5, 0)
BOX_OUTLIERS_MAX = 200


def box_plot(values: list[float]) -> dict | None:
    """The outlier box plot and the quantiles of the values, as JMP's Distribution gives them.

    The box is the quartiles; a whisker reaches the farthest value within 1.5 × the interquartile range of the box;
    values beyond the whiskers are outliers (at most BOX_OUTLIERS_MAX are listed, the farthest ones; the count is exact).
    """
    if not values:
        return None
    x = np.sort(np.asarray(values, dtype=float))
    q1, median, q3 = (float(v) for v in np.percentile(x, [25, 50, 75]))
    reach = 1.5 * (q3 - q1)
    inside = x[(x >= q1 - reach) & (x <= q3 + reach)]
    outside = x[(x < q1 - reach) | (x > q3 + reach)]
    listed = outside if len(outside) <= BOX_OUTLIERS_MAX else outside[np.argsort(np.abs(outside - median))[-BOX_OUTLIERS_MAX:]]
    return {
        "q1": round(q1, 4), "median": round(median, 4), "q3": round(q3, 4),
        "whisker_low": round(float(inside[0]), 4), "whisker_high": round(float(inside[-1]), 4),
        "mean": round(float(x.mean()), 4), "outlier_count": int(len(outside)),
        "outliers": [round(float(v), 4) for v in np.sort(listed)],
        "quantiles": [{"percent": p, "value": round(float(np.percentile(x, p)), 4)} for p in QUANTILES],
    }


def histogram(values: list[float], bins: int = 20) -> dict | None:
    """Counts in `bins` equal bins between the smallest and the largest value.

    The bins cover the values only, as in Minitab and JMP: specification limits far from the data must not make the
    bars narrow. The page widens its axis to show LSL and USL.
    """
    if not values:
        return None
    low, high = min(values), max(values)
    if high <= low:
        return {"min": low, "max": high, "counts": [len(values)] + [0] * (bins - 1)}
    counts, _ = np.histogram(values, bins=bins, range=(low, high))
    return {"min": round(low, 4), "max": round(high, 4), "counts": [int(c) for c in counts]}

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Tuple

from django.db.models import Avg, Count, Min, Max, Sum

from historian.models import ExtruderSample


@dataclass
class ShiftStats:
    date: date
    shift_code: str
    extruder_id: str
    recipe_id: str

    start_ts: datetime
    end_ts: datetime

    samples: int

    avg_throughput_actual: float
    avg_throughput_target: float

    avg_availability: float
    avg_performance: float
    avg_quality: float
    avg_oee: float

    total_downtime_min: float
    running_count: int
    stopped_count: int

    moisture_avg: float
    moisture_min: float
    moisture_max: float

    diameter_avg: float
    diameter_min: float
    diameter_max: float

    total_alarms: int
    top_alarms: List[Tuple[str, int]]


def compute_shift_stats(
    start: datetime,
    end: datetime,
    extruder_id: str,
) -> List[ShiftStats]:
    """Compute per-shift aggregates for the given time window.

    Returns one ShiftStats per (date, shift_code, recipe_id).
    """

    qs = (
        ExtruderSample.objects.using("historian")
        .filter(ts__gte=start, ts__lte=end, extruder_id=extruder_id)
    )

    # group by date, shift, recipe
    grouped = (
        qs.values("ts__date", "shift_code", "recipe_id")
        .annotate(
            samples=Count("id"),
            start_ts=Min("ts"),
            end_ts=Max("ts"),
            avg_throughput_actual=Avg("throughput_actual_kg_hr"),
            avg_throughput_target=Avg("throughput_target_kg_hr"),
            avg_availability=Avg("availability_pct"),
            avg_performance=Avg("performance_pct"),
            avg_quality=Avg("quality_pct"),
            avg_oee=Avg("oee_pct"),
            total_downtime_min=Sum("downtime_min"),
        )
    )

    # running/stopped counts could be added later if needed

    stats: List[ShiftStats] = []

    for row in grouped:
        day = row["ts__date"]
        shift_code = row["shift_code"]
        recipe_id = row["recipe_id"]

        # moisture/diameter aggregates
        agg = (
            qs.filter(ts__date=day, shift_code=shift_code, recipe_id=recipe_id)
            .aggregate(
                moisture_avg=Avg("product_moisture_pct"),
                moisture_min=Min("product_moisture_pct"),
                moisture_max=Max("product_moisture_pct"),
                diameter_avg=Avg("kibble_diameter_mm"),
                diameter_min=Min("kibble_diameter_mm"),
                diameter_max=Max("kibble_diameter_mm"),
                total_alarms=Count("id", filter=None),
            )
        )

        alarm_qs = qs.filter(
            ts__date=day,
            shift_code=shift_code,
            recipe_id=recipe_id,
            alarm_active=True,
        )
        # top 3 alarm codes
        alarm_counts = (
            alarm_qs.values_list("alarm_code")
            .annotate(c=Count("id"))
            .order_by("-c")[:3]
        )
        top_alarms: List[Tuple[str, int]] = [
            (code or "", c) for code, c in alarm_counts if code
        ]

        stats.append(
            ShiftStats(
                date=day,
                shift_code=shift_code,
                extruder_id=extruder_id,
                recipe_id=recipe_id,
                start_ts=row["start_ts"],
                end_ts=row["end_ts"],
                samples=row["samples"],
                avg_throughput_actual=float(row["avg_throughput_actual"] or 0.0),
                avg_throughput_target=float(row["avg_throughput_target"] or 0.0),
                avg_availability=float(row["avg_availability"] or 0.0),
                avg_performance=float(row["avg_performance"] or 0.0),
                avg_quality=float(row["avg_quality"] or 0.0),
                avg_oee=float(row["avg_oee"] or 0.0),
                total_downtime_min=float(row["total_downtime_min"] or 0.0),
                running_count=0,
                stopped_count=0,
                moisture_avg=float(agg["moisture_avg"] or 0.0),
                moisture_min=float(agg["moisture_min"] or 0.0),
                moisture_max=float(agg["moisture_max"] or 0.0),
                diameter_avg=float(agg["diameter_avg"] or 0.0),
                diameter_min=float(agg["diameter_min"] or 0.0),
                diameter_max=float(agg["diameter_max"] or 0.0),
                total_alarms=int(alarm_qs.count()),
                top_alarms=top_alarms,
            )
        )

    return stats


def format_shift_summary(stats: ShiftStats) -> str:
    """Render a human-readable summary for a single shift."""

    # derive utilization
    if stats.avg_throughput_target > 0:
        utilization = (stats.avg_throughput_actual / stats.avg_throughput_target) * 100.0
    else:
        utilization = 0.0

    # day length in minutes for one shift is ~8 hours, but we can approximate
    downtime_pct = min(100.0, (stats.total_downtime_min / (stats.samples or 1)) * 100.0)

    lines = []
    lines.append(
        f"On {stats.date} during Shift {stats.shift_code} for extruder {stats.extruder_id} "
        f"running recipe {stats.recipe_id}:"
    )

    lines.append(
        f"- Average throughput: {stats.avg_throughput_actual/1000:.2f} t/h "
        f"(target {stats.avg_throughput_target/1000:.2f} t/h, utilization {utilization:.1f}%)."
    )
    lines.append(
        f"- OEE: {stats.avg_oee:.1f}% (Availability {stats.avg_availability:.1f}%, "
        f"Performance {stats.avg_performance:.1f}%, Quality {stats.avg_quality:.1f}%)."
    )
    lines.append(
        f"- Product moisture: avg {stats.moisture_avg:.2f}%, "
        f"range {stats.moisture_min:.2f}–{stats.moisture_max:.2f}%."
    )
    lines.append(
        f"- Kibble diameter: avg {stats.diameter_avg:.2f} mm (spec 7.5–9.5 mm), "
        f"range {stats.diameter_min:.2f}–{stats.diameter_max:.2f} mm."
    )
    lines.append(
        f"- Downtime: {stats.total_downtime_min:.0f} minutes in this shift; "
        f"approx. {downtime_pct:.1f}% of the sampled minutes had downtime."
    )

    if stats.total_alarms > 0:
        top_alarm_str = ", ".join(
            f"{code} ({count})" for code, count in stats.top_alarms
        )
        lines.append(
            f"- Alarms: {stats.total_alarms} alarmed minutes; most frequent codes: {top_alarm_str}."
        )
    else:
        lines.append("- Alarms: none recorded during this shift.")

    return "\n".join(lines)

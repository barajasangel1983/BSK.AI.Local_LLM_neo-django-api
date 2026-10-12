"""manage.py historian_check --hours N: the transfer check of the live historian, as a report.

Reads the live database (read-only role) and writes the report to the terminal. Ends with exit code 1
when a transfer check fails, so it can be run by a timer.
"""

from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db import Error as DatabaseError
from django.utils import timezone

from historian import check
from historian.models import LiveRow, LiveRowMeta, LiveSimEvent


class Command(BaseCommand):
    help = "Check the rows the PLC sent to the live historian: transfer, likeness to the historian, anomaly record."

    def add_arguments(self, parser):
        parser.add_argument("--hours", type=float, default=24, help="How far back to look (default: 24).")
        parser.add_argument("--extruder-id", type=str, default="EXTR01", help="Extruder to check (default: EXTR01).")

    def handle(self, *args, **options):
        now = timezone.now()
        since = now - timedelta(hours=options["hours"])
        window = {"extruder_id": options["extruder_id"], "ts__gte": since}
        try:
            rows = list(LiveRow.objects.filter(**window).order_by("ts").values())
            meta = {m["ts"]: m for m in LiveRowMeta.objects.filter(**window).values()}
            events = {e["ts"]: e for e in LiveSimEvent.objects.filter(**window).values()}
        except DatabaseError as exc:
            raise CommandError(f"The live historian did not answer: {exc}") from exc
        if not rows:
            raise CommandError(f"No rows for {options['extruder_id']} in the last {options['hours']:g} hours.")

        out = self.stdout.write
        first, last = rows[0]["ts"], rows[-1]["ts"]
        out(f"Live historian check: {options['extruder_id']}, last {options['hours']:g} hours")
        out(f"Run at {check._utc(now)} UTC. Rows: {len(rows):,}, from {check._utc(first)} to {check._utc(last)} UTC "
            f"({len(rows) / check.PERIODS_PER_HOUR:.1f} hours of rows; a full day is 4,320).")

        transfer = check.transfer(rows, meta, events)
        out("\n1. Transfer")
        self._checks(transfer["checks"])
        for line in transfer["info"]:
            out(f"  - {line}")

        like = check.likeness(rows, events)
        out("\n2. Likeness (figures, not pass or fail; the targets need a day or more)")
        out(f"  {'state':<11}{'rows':>7}{'share':>8}{'target':>8}{'blocks':>8}{'min each':>10}"
            f"{'kg/h':>9}{'avail':>7}{'perf':>7}{'qual':>7}{'OEE':>7}{'alarm':>7}")
        for s in like["states"]:
            if not s["rows"]:
                out(f"  {s['state']:<11}{0:>7}{0:>7.1f}%{s['target_share_pct']:>7.1f}%   (not seen)")
                continue
            out(f"  {s['state']:<11}{s['rows']:>7,}{s['share_pct']:>7.1f}%{s['target_share_pct']:>7.1f}%{s['blocks']:>8}"
                f"{s['block_minutes']:>10.0f}{s['throughput_actual_kg_hr']:>9,.0f}{s['availability_pct']:>7.1f}"
                f"{s['performance_pct']:>7.1f}{s['quality_pct']:>7.1f}{s['oee_pct']:>7.1f}{s['alarm_rows_pct']:>6.1f}%")
        running = next(s for s in like["states"] if s["state"] == "RUNNING")
        if running["rows"]:
            target = check.RUNNING_TARGET
            out(f"  - RUNNING against its targets: quality {running['quality_pct']:.1f} % ({target['quality_pct']:g}), "
                f"OEE {running['oee_pct']:.1f} % ({target['oee_pct']:g}), "
                f"rows with an alarm {running['alarm_rows_pct']:.1f} % ({target['alarm_rows_pct']:g})")
        out(f"  - an alarm starts {like['alarm_starts_per_hour']:.1f} times an hour "
            f"(target about {check.ALARM_STARTS_PER_HOUR:g}; {like['alarm_starts']} starts)")
        if like["running_rows"]:
            worst = ", ".join(f"{column} {share:.1f} %" for column, share in list(like["outside_by_signal"].items())[:4])
            out(f"  - RUNNING rows with a signal outside March's normal range: {like['outside_normal_pct']:.1f} % "
                f"of {like['running_rows']:,} (rows of an injected anomaly left out){'; most: ' + worst if worst else ''}")

        found = check.anomalies(rows, events)
        out("\n3. Injected anomalies")
        if not found["events"]:
            out(f"  none in the record (expected about {check.ANOMALIES_PER_DAY:g} a day)")
        else:
            out(f"  {len(found['events'])} anomalies on {found['rows']:,} rows, {found['per_day']:.1f} a day "
                f"(target about {check.ANOMALIES_PER_DAY:g}); by type: "
                + ", ".join(f"{name} {n}" for name, n in sorted(found["by_type"].items())))
            for e in found["events"]:
                out(f"  #{e['event_id']:<4}{e['type']:<16}{e['signal']:<28}{e['size']:>5.1f} σ  "
                    f"{check._utc(e['start'])}  {e['rows'] / 3:>5.1f} min")
            self._checks(found["checks"])

        failed = [c for c in transfer["checks"] if not c["ok"]]
        out(f"\nResult: {'the transfer checks pass' if not failed else f'{len(failed)} transfer check(s) FAILED'}.")
        if failed:
            raise SystemExit(1)

    def _checks(self, checks):
        for c in checks:
            mark = self.style.SUCCESS("PASS") if c["ok"] else self.style.ERROR("FAIL")
            self.stdout.write(f"  {mark}  {c['name']}{'' if c['ok'] or not c['detail'] else ': ' + c['detail']}")

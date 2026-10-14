"""manage.py spc_score --hours N: the score of SPC against the injected anomalies, as a report (P12d).

    --hours N         the window, ending at the historian's newest row (default 24)
    --asset-id ID     the machine (default bsk:asset:EXTR01)
    --tools saved     the machine's enabled tools as they are configured (default)
    --tools proposed  a tool for every signal that has proposed values, made in memory with its baseline
                      over the proposed days before the newest row
    --rules a,b,...   judge with these rules instead of each tool's own (in memory)

The only code that reads the injected-anomaly record (`sim_events`, FR-26). The tools' events are made
again in memory for the window (evaluate.judge_window): no tool, limit, event or acknowledgement is
written or changed, so it is safe to run while the evaluator's timer runs.
Report form: claude/P12_SPCimplementation.md, section 12.2.
"""

from __future__ import annotations

from datetime import timedelta, timezone as dt_timezone

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import Error as DatabaseError
from django.utils import timezone

from historian import source as historian_source
from historian.models import LiveSimEvent
from spc import engine, evaluate, score, service
from spc.models import SpcTool

utc = lambda ts: ts.astimezone(dt_timezone.utc).strftime("%Y-%m-%d %H:%M")    # noqa: E731


class Command(BaseCommand):
    help = "Score SPC against the injected anomalies: found, missed, delay, false alarms. Writes nothing."

    def add_arguments(self, parser):
        parser.add_argument("--hours", type=float, default=24, help="How far back from the newest row (default: 24).")
        parser.add_argument("--asset-id", type=str, default="bsk:asset:EXTR01", help="The machine (default: bsk:asset:EXTR01).")
        parser.add_argument("--tools", choices=("saved", "proposed"), default="saved",
                            help="The machine's enabled tools (default), or a tool for every proposed signal, in memory.")
        parser.add_argument("--rules", type=str, default=None, help="Comma-separated rule ids that replace each tool's rules.")

    def handle(self, *args, **options):
        if historian_source.current() != "live":
            raise CommandError("The Studio serves the recorded month: there is no injected-anomaly record to score against.")
        asset_id = options["asset_id"]
        rules = None
        if options["rules"]:
            rules = [r.strip() for r in options["rules"].split(",") if r.strip()]
            unknown = [r for r in rules if r not in engine.RULES]
            if unknown:
                raise CommandError(f"Unknown rule(s): {', '.join(unknown)}. Rules: {', '.join(engine.RULES)}")
        newest = service.newest_row(asset_id)
        if not newest:
            raise CommandError(f"No rows for {asset_id}.")
        end = newest["ts"]
        start = end - timedelta(hours=options["hours"])

        tools = self._saved(asset_id) if options["tools"] == "saved" else self._proposed(asset_id, end)
        if rules is not None:
            for tool, _ in tools:
                tool.rules = rules                       # in memory only: nothing is saved
        events: list[score.Event] = []
        for tool, limits in tools:
            for e in evaluate.judge_window(tool, limits, start, end):
                events.append(score.Event(tool=tool.name, column=tool.column, type=e.type, rule=e.rule,
                                          started_at=e.started_at, last_at=e.last_at))

        try:
            record = list(LiveSimEvent.objects.filter(extruder_id=service.asset_key(asset_id), ts__gt=start, ts__lte=end)
                          .order_by("ts").values("ts", "event_id", "event_type", "event_signal", "event_size"))
        except DatabaseError as exc:
            raise CommandError(f"The live historian did not answer: {exc}") from exc
        found = [a for a in score.anomalies(record)
                 if not LiveSimEvent.objects.filter(event_id=a.event_id, ts__lte=start).exists()]   # whole anomalies only
        result = score.score(found, events, {tool.name: tool.column for tool, _ in tools})
        # "Per day" is over the hours that have rows: the window may reach back before the record begins.
        first = (historian_source.samples().objects.filter(extruder_id=service.asset_key(asset_id), ts__gt=start)
                 .order_by("ts").values_list("ts", flat=True).first())
        self._report(asset_id, options, first or start, end, tools, result)

    # --- the tools scored --------------------------------------------------------------------------

    def _saved(self, asset_id):
        out = []
        for tool in SpcTool.objects.filter(asset_id=asset_id, enabled=True).prefetch_related("limits"):
            limits = {lim.group: lim for lim in tool.limits.all()}
            if limits:
                out.append((tool, limits))
            else:
                self.stderr.write(f"{tool.name}: no control limits, not scored")
        return out

    def _proposed(self, asset_id, end):
        """A tool, not saved, for every signal with proposed values; its limits from the proposed baseline days."""
        out = []
        for signal in service.signals(asset_id)["signals"]:
            p = signal["proposed"]
            if p is None:
                continue
            tool = SpcTool(asset_id=asset_id, signal_id=signal["signal_id"], column=signal["column"], name=signal["name"],
                           unit=signal["unit"] or "", states=p["states"], ignore_seconds=p["ignore_seconds"], method=p["method"],
                           group_by=p["group_by"], lsl=p["lsl"], target=p["target"], usl=p["usl"],
                           sigma_multiplier=p["sigma_multiplier"])
            tool.baseline_start, tool.baseline_end = end - timedelta(days=p["baseline_days"]), end
            rows = service.read_rows(asset_id, [tool.column] + ([tool.group_by] if tool.group_by else []),
                                     tool.baseline_start, tool.baseline_end)
            limits, _ = engine.baseline_limits(service._samples(tool, rows), method=tool.method, subgroup_size=tool.subgroup_size,
                                               k=tool.sigma_multiplier, min_samples=settings.SPC_MIN_BASELINE_SAMPLES)
            if limits:
                out.append((tool, limits))
            else:
                self.stderr.write(f"{tool.name}: no group has enough samples for a baseline, not scored")
        return out

    # --- the report (section 12.2) -----------------------------------------------------------------

    def _report(self, asset_id, options, first, end, tools, result: score.Score):
        out = self.stdout.write
        hours = options["hours"]
        days = max((end - first).total_seconds() / 86400, 1 / 24)
        which = "the machine's enabled tools" if options["tools"] == "saved" else "the proposed tools, made in memory"
        out(f"SPC score: {asset_id}, last {hours:g} hours to the newest row {utc(end)} UTC. Run at {utc(timezone.now())} UTC. "
            f"Tools enabled: {len(tools)} ({which}{'; rules: ' + options['rules'] if options['rules'] else ''}).")
        out(f"Rows from {utc(first)} UTC: {days * 24:.1f} hours, the base of \"per day\".")

        out("1. Injected anomalies")
        out(f"   {'#id':<5}{'type':<17}{'signal':<27}{'size':>6}  {'start':<17}{'minutes':>8}  {'found by':<31}{'rule':<16}{'delay':>6}")
        for f in result.findings:
            a = f.anomaly
            minutes = (a.end - a.start).total_seconds() / 60 + 20 / 60
            if f.event is not None:
                by, rule, delay = f.event.tool, f.event.rule, f"{f.delay_minutes:.1f}"
            else:
                by, rule, delay = ("missed" if f.monitored else "not monitored"), "", ""
            out(f"   {a.event_id:<5}{a.type:<17}{a.signal:<27}{a.size:>6.2f}  {utc(a.start):<17}{minutes:>8.1f}  {by:<31}{rule:<16}{delay:>6}")
        monitored, found = result.monitored, result.found
        not_monitored = len(result.findings) - len(monitored)
        delay = result.median_delay
        out(f"   Found {len(found)} of {len(monitored)} monitored ({not_monitored} not monitored). "
            f"Median delay {'-' if delay is None else f'{delay:.1f}'} min.")
        by_type = []
        for kind in score.TYPES.values():
            own = [f for f in monitored if f.anomaly.type == kind]
            if own or any(f.anomaly.type == kind for f in result.findings):
                by_type.append(f"{kind} {sum(f.event is not None for f in own)}/{len(own)}")
        out(f"   By type: {', '.join(by_type) if by_type else 'no anomaly in the window'}")

        out("2. False alarms")
        out(f"   {'tool':<31}{'control violations':>19}{'per day':>9}{'spec violations':>17}{'per day':>9}{'warnings':>10}{'of them false':>15}")
        for t in result.false_alarms:
            out(f"   {t.tool:<31}{t.control:>19}{t.control / days:>9.1f}{t.spec:>17}{t.spec / days:>9.1f}{t.warnings:>10}{t.false_warnings:>15}")
        total_c = sum(t.control for t in result.false_alarms)
        total_s = sum(t.spec for t in result.false_alarms)
        out(f"   {'all tools':<31}{total_c:>19}{total_c / days:>9.1f}{total_s:>17}{total_s / days:>9.1f}"
            f"{sum(t.warnings for t in result.false_alarms):>10}{sum(t.false_warnings for t in result.false_alarms):>15}")

        out("3. Configuration scored")
        out(f"   {'tool':<31}{'version':>8}  {'method':<8}{'group_by':<11}{'rules':<40}baseline window")
        for tool, limits in tools:
            window = (f"{utc(tool.baseline_start)} to {utc(tool.baseline_end)}" if tool.baseline_start and tool.baseline_end
                      else tool.limit_strategy)
            version = "-" if tool._state.adding else str(tool.version)
            out(f"   {tool.name:<31}{version:>8}  {tool.method:<8}{tool.group_by or '-':<11}{','.join(tool.rules):<40}{window}"
                f"  ({len(limits)} group{'s' if len(limits) != 1 else ''})")

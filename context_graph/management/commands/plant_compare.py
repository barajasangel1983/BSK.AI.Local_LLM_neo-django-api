"""manage.py plant_compare [--asset bsk:asset:EXTR01]: the graph against the PLC, as a report (P14a).

Signal by signal (unit, normal range, OPC node) and alarm by alarm; the PLC's spare slots. Reads the PLC over
OPC UA and the graph; writes nothing.
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from context_graph import plc_source

LABELS = {"same": "same", "differs": "DIFFERS", "only_plc": "only on the PLC", "only_graph": "only in the graph",
          "historian_only": "historian column, not a PLC slot"}


def _range(side) -> str:
    if not side or (side.get("low") is None and side.get("high") is None):
        return ""
    return f"{side.get('low')} to {side.get('high')}"


class Command(BaseCommand):
    help = "Compare the Context Graph with the PLC: signals, units, normal ranges, OPC nodes, alarms. Writes nothing."

    def add_arguments(self, parser):
        parser.add_argument("--asset", default="bsk:asset:EXTR01", help="The asset (default: bsk:asset:EXTR01).")

    def handle(self, *args, **options):
        try:
            result = plc_source.compare(options["asset"])
        except plc_source.PlcSourceError as exc:
            raise CommandError(str(exc)) from exc
        out = self.stdout.write
        out(f"The graph against the PLC: {result['asset_id']}. Read at {timezone.now():%Y-%m-%d %H:%M} UTC from {result['endpoint']}.")
        for kind in ("signal", "alarm"):
            s = result["summary"][kind]
            out(f"{kind.capitalize()}s: " + ", ".join(f"{n} {LABELS[k]}" for k, n in s.items() if n))
        out("")
        out(f"  {'signal':<28}{'state':<34}{'PLC':<38}{'graph':<50}why")
        for r in (r for r in result["rows"] if r["kind"] == "signal"):
            plc = r["plc"] or {}
            graph = r["graph"] or {}
            out(f"  {r['key']:<28}{LABELS[r['status']]:<34}{' '.join(x for x in (plc.get('slot', ''), plc.get('unit', ''), _range(plc)) if x):<38}"
                f"{' '.join(x for x in (graph.get('component', ''), graph.get('unit', ''), _range(graph)) if x):<50}{'; '.join(r['differences'])}")
        out("")
        out(f"  {'alarm':<28}{'state':<34}PLC index")
        for r in (r for r in result["rows"] if r["kind"] == "alarm"):
            out(f"  {r['key']:<28}{LABELS[r['status']]:<34}{(r['plc'] or {}).get('index', '')}")
        out("")
        spare = result["spare_slots"]
        out(f"Spare slots on the PLC, not imported ({len(spare)}): " + ", ".join(f"{s['slot']} {s['label']}" for s in spare))

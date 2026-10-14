"""The OPC UA client of the PLC page: one short session per read or command (P13f).

OPTA's server on the Mini PC (docs/VPS_HANDOFF.md in the collector repository): security None, anonymous. Only
the nodes below are read, and only the methods ExtruderSim.Start and ExtruderSim.Pause are ever called.
"""

from __future__ import annotations

import asyncio
import time

from asyncua import Client, ua
from django.conf import settings

NAMESPACE_URI = "urn:bsk:machine:v1"
SIM_NAMESPACE_URI = "urn:bsk:opta:extrudersim"
COMMANDS = ("Start", "Pause")

# Read under BSK.<asset>. in the BSK namespace
BSK_NODES = (
    "Heartbeat.State", "Heartbeat.LastCommUtc", "Sample.PeriodCounter", "Sample.TimestampUtc", "Sample.PeriodSeconds",
    "Step.StepName", "DigitalOut[0].Value", "DigitalOut[1].Value", "DigitalOut[2].Value", "DigitalOut[3].Value",
    "DigitalIn[8].Value", "Faults.ActiveLabel", "Faults.ActiveCode",
)
# Read by their own ids in the simulation namespace
SIM_NODES = (
    "ExtruderSim.Running", "ExtruderSim.ClockSet", "ExtruderSim.EventType", "ExtruderSim.EventId", "ExtruderSim.EventSignal",
    "Plc.UptimeSeconds", "Plc.BootUtc", "Plc.MapVersion", "Plc.ExpectedMapVersion", "Plc.ClockUtc", "Plc.ClockOffsetSeconds",
    "Plc.RestartCount", "Plc.LinkLossCount", "Plc.LinkUpUtc", "Plc.LinkDownUtc",
)


class OpcError(Exception):
    """The OPC UA server could not be reached, or refused the call."""


async def _read(client: Client) -> dict:
    idx = await client.get_namespace_index(NAMESPACE_URI)
    sim = await client.get_namespace_index(SIM_NAMESPACE_URI)
    nodes = [client.get_node(ua.NodeId(f"BSK.{settings.PLC_ASSET}.{path}", idx)) for path in BSK_NODES]
    nodes += [client.get_node(ua.NodeId(path, sim)) for path in SIM_NODES]
    return dict(zip(BSK_NODES + SIM_NODES, await client.read_values(nodes)))


async def _session(work, limit: float | None = None):
    try:
        async with Client(settings.PLC_OPCUA_URL, timeout=settings.PLC_TIMEOUT) as client:
            return await asyncio.wait_for(work(client), limit or settings.PLC_TIMEOUT * 2)
    except (OSError, asyncio.TimeoutError, ua.UaError) as exc:
        raise OpcError(str(exc) or type(exc).__name__) from exc


BROWSE_KEEP_SECONDS = 3600   # the server's variables are browsed again after this, or after a failed read (a browse takes about 9 s)
_browsed: dict = {}


async def _variables(client: Client, node) -> list:
    """The variables under an object, depth first. One request per object: the browse answer carries each child's
    node class (asking each child for it took over 10 s for the 774 variables)."""
    found = []
    for ref in await node.get_children_descriptions():
        child = client.get_node(ref.NodeId)
        if ref.NodeClass == ua.NodeClass.Variable:
            found.append(child)
        elif ref.NodeClass == ua.NodeClass.Object:
            found += await _variables(client, child)
    return found


async def _browse(client: Client) -> list[tuple[str, ua.NodeId]]:
    """Every variable of the asset in the BSK namespace, then ExtruderSim and Plc, in the server's order, as
    (path, NodeId). The path is the NodeId's string without the "BSK.<asset>." prefix, as OPTA's dashboard shows it."""
    idx = await client.get_namespace_index(NAMESPACE_URI)
    sim = await client.get_namespace_index(SIM_NAMESPACE_URI)
    root = f"BSK.{settings.PLC_ASSET}"
    out = [(n.nodeid.Identifier[len(root) + 1:], n.nodeid) for n in await _variables(client, client.get_node(ua.NodeId(root, idx)))]
    for name in ("ExtruderSim", "Plc"):
        out += [(n.nodeid.Identifier, n.nodeid) for n in await _variables(client, client.get_node(ua.NodeId(name, sim)))]
    return out


def read_all() -> dict:
    """Every variable with its value and OPC UA data type, in one session. The list of variables is browsed once and
    kept BROWSE_KEEP_SECONDS (browsing the 774 variables takes seconds; reading them, a fraction of one)."""
    async def work(client: Client):
        fresh = not _browsed.get("at") or time.monotonic() - _browsed["at"] > BROWSE_KEEP_SECONDS
        if fresh:
            _browsed.update(at=time.monotonic(), variables=await _browse(client))
        variables = _browsed["variables"]
        values = await client.read_attributes([client.get_node(nid) for _, nid in variables], ua.AttributeIds.Value)
        return {"browsed_now": fresh, "variables": [
            {"path": path, "value": dv.Value.Value if dv.Value is not None else None,
             "type": dv.Value.VariantType.name if dv.Value is not None else None,
             "good": dv.StatusCode is None or dv.StatusCode.is_good()}
            for (path, _), dv in zip(variables, values)]}

    try:
        return asyncio.run(_session(work, 60))         # the first browse is the slow part
    except OpcError:
        _browsed.clear()                 # the server may have restarted with other nodes
        raise


def read_values() -> dict:
    """Every node the page needs, in one session."""
    return asyncio.run(_session(_read))


def call(command: str) -> None:
    """Call ExtruderSim.Start or ExtruderSim.Pause. Nothing else can be called through here."""
    if command not in COMMANDS:
        raise ValueError(f"not a command: {command}")

    async def work(client: Client):
        sim = await client.get_namespace_index(SIM_NAMESPACE_URI)
        await client.get_node(ua.NodeId("ExtruderSim", sim)).call_method(ua.NodeId(f"ExtruderSim.{command}", sim))

    asyncio.run(_session(work))

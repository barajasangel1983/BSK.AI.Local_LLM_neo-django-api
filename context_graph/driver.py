"""Neo4j access for the Context Graph.

The graph is optional: with GRAPH_ENABLED=false (or no NEO4J_PASSWORD) the
rest of the application runs unchanged and graph endpoints report it.
One driver per process (the driver pools connections and is thread-safe).
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from django.conf import settings
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired

_driver: Driver | None = None
_lock = threading.Lock()


class GraphUnavailable(RuntimeError):
    """The graph is disabled, not configured or unreachable."""


def is_enabled() -> bool:
    return bool(settings.GRAPH_ENABLED)


def is_configured() -> bool:
    return bool(settings.NEO4J_URI and settings.NEO4J_PASSWORD)


def get_driver() -> Driver:
    """Return the process-wide driver, creating it on first use."""
    global _driver
    if not is_enabled():
        raise GraphUnavailable("Context Graph is disabled (GRAPH_ENABLED=false)")
    if not is_configured():
        raise GraphUnavailable("Context Graph is not configured (NEO4J_URI / NEO4J_PASSWORD)")
    with _lock:
        if _driver is None:
            _driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                connection_timeout=settings.NEO4J_CONNECTION_TIMEOUT,
                # Queries name relationship types the graph may not have yet (e.g. no
                # DOCUMENTED_BY before a document is linked); those warnings are noise.
                notifications_disabled_categories=["UNRECOGNIZED"],
            )
        return _driver


def close_driver() -> None:
    global _driver
    with _lock:
        if _driver is not None:
            _driver.close()
            _driver = None


@contextmanager
def session(**kwargs):
    """Session on the configured database (an unreachable server raises GraphUnavailable)."""
    try:
        with get_driver().session(database=settings.NEO4J_DATABASE, **kwargs) as s:
            yield s
    except (ServiceUnavailable, SessionExpired) as exc:
        raise GraphUnavailable(f"Neo4j is unreachable: {exc}") from exc


def health() -> dict:
    """Connectivity check for /api/graph/health/ and the Model Health page.

    status: "online" | "offline" | "disabled" | "not_configured"
    """
    base = {"uri": settings.NEO4J_URI, "database": settings.NEO4J_DATABASE}
    if not is_enabled():
        return {**base, "status": "disabled"}
    if not is_configured():
        return {**base, "status": "not_configured"}

    start = time.time()
    try:
        with session() as s:
            record = s.run(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions[0] AS version, edition"
            ).single()
            nodes = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        return {
            **base,
            "status": "online",
            "latency_ms": round((time.time() - start) * 1000),
            "server": f"{record['name']} {record['version']} ({record['edition']})",
            "node_count": nodes,
        }
    except Exception as exc:
        return {**base, "status": "offline", "latency_ms": 0, "error": f"{type(exc).__name__}: {exc}"}

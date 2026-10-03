"""Unit normalization (P8b): one spelling per unit for imported and extracted properties."""

from __future__ import annotations

import re

_UNITS = {
    "°c": "°C", "degc": "°C", "deg c": "°C", "c": "°C", "celsius": "°C", "ºc": "°C",
    "°f": "°F", "degf": "°F", "f": "°F",
    "bar": "bar", "bar(g)": "bar", "barg": "bar", "bar g": "bar",
    "kg/h": "kg/h", "kg/hr": "kg/h", "kgh": "kg/h", "kg per hour": "kg/h",
    "l/h": "L/h", "l/hr": "L/h", "lph": "L/h",
    "t/h": "t/h", "t/hr": "t/h",
    "rpm": "rpm", "1/min": "rpm",
    "kw": "kW", "w": "W", "a": "A", "amp": "A", "amps": "A", "v": "V",
    "%": "%", "pct": "%", "percent": "%",
    "mm": "mm", "m/min": "m/min", "g/l": "g/L", "min": "min", "s": "s", "sec": "s",
}


def normalize_unit(value):
    """Canonical spelling of a unit (unknown units are returned unchanged, trimmed)."""
    if not isinstance(value, str):
        return value
    key = re.sub(r"\s+", " ", value.strip().lower())
    return _UNITS.get(key, value.strip())

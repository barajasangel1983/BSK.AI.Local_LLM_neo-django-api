"""Canonical IDs for Context Graph entities.

Format: ``bsk:<type>:<key>`` where <type> is the kebab-case entity type and
<key> is one or more path parts joined by "/", e.g.

    bsk:asset:EXTR01
    bsk:component:EXTR01/barrel-zone-1
    bsk:signal:EXTR01/barrel_zone_1_temp_c
    bsk:alarm:EXTR01/DIE_PLUG

IDs are stable identifiers, not display names: case is preserved (tags such as
EXTR01 or DIE_PLUG keep theirs), anything outside [A-Za-z0-9_.-] becomes "-".
"""

from __future__ import annotations

import re

PREFIX = "bsk"
_KEY_PART = re.compile(r"[^A-Za-z0-9_.-]+")
ID_PATTERN = re.compile(r"^bsk:[a-z][a-z0-9-]*:[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$")


def type_slug(entity_type: str) -> str:
    """PascalCase entity type -> kebab-case: OperatingLimit -> operating-limit, OPCNode -> opc-node."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", entity_type)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", s)
    return s.lower()


def clean_part(part: str) -> str:
    cleaned = _KEY_PART.sub("-", str(part).strip()).strip("-")
    if not cleaned:
        raise ValueError(f"empty ID part from {part!r}")
    return cleaned


def make_id(entity_type: str, *parts: str) -> str:
    if not parts:
        raise ValueError("make_id needs at least one key part")
    return f"{PREFIX}:{type_slug(entity_type)}:{'/'.join(clean_part(p) for p in parts)}"


def is_valid_id(value: str) -> bool:
    return bool(ID_PATTERN.match(value or ""))


def parse_id(value: str) -> tuple[str, list[str]]:
    """Return (type_slug, key_parts); raises ValueError for malformed IDs."""
    if not is_valid_id(value):
        raise ValueError(f"not a canonical ID: {value!r}")
    _, kind, key = value.split(":", 2)
    return kind, key.split("/")

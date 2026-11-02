"""Catalog-driven OpenRouter provider-routing preference (TASK-1915).

The model catalog's ``serving_config.openrouter_provider`` selects which
upstream providers OpenRouter may serve a model from.  This module is the
single normalization boundary shared by app-side catalog validation, bridge
command parsing, the proxy request-metadata header, and the OpenRouter
adapter.  Only a small, safe subset of OpenRouter's ``provider`` request
object is supported; unknown keys are dropped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

OPENROUTER_PROVIDER_LIST_KEYS = ("order", "only", "ignore")
OPENROUTER_PROVIDER_BOOL_KEYS = ("allow_fallbacks",)
OPENROUTER_PROVIDER_KEYS = OPENROUTER_PROVIDER_LIST_KEYS + OPENROUTER_PROVIDER_BOOL_KEYS
MAX_PROVIDER_NAMES = 16
MAX_PROVIDER_NAME_CHARS = 64
# Upper bound on the serialized form (command field / proxy header).  The
# largest canonical value the caps above allow is ~3.3 KB.
MAX_SERIALIZED_CHARS = 4096

_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9 ._/-]+$")


def _normalize_names(value: object) -> list[str] | None:
    """Return de-duplicated, stripped names, or ``None`` when invalid."""
    if not isinstance(value, (list, tuple)) or len(value) > MAX_PROVIDER_NAMES:
        return None
    names: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        name = item.strip()
        if (
            not name
            or len(name) > MAX_PROVIDER_NAME_CHARS
            or not _PROVIDER_NAME_RE.fullmatch(name)
        ):
            return None
        if name not in names:
            names.append(name)
    return names


def normalize_openrouter_provider(value: object) -> dict[str, Any] | None:
    """Return the canonical provider preference, or ``None`` when invalid/empty.

    Unknown keys are dropped.  Any malformed supported key invalidates the
    whole value (a half-applied routing policy is worse than none).  Empty
    lists are omitted; a result with no remaining keys is ``None``.
    """
    if not isinstance(value, Mapping):
        return None
    canonical: dict[str, Any] = {}
    for key in OPENROUTER_PROVIDER_LIST_KEYS:
        if key not in value:
            continue
        names = _normalize_names(value[key])
        if names is None:
            return None
        if names:
            canonical[key] = names
    for key in OPENROUTER_PROVIDER_BOOL_KEYS:
        if key not in value:
            continue
        flag = value[key]
        if not isinstance(flag, bool):
            return None
        canonical[key] = flag
    return canonical or None


def serialize_openrouter_provider(value: object) -> str | None:
    """Compact, deterministic JSON for a command field / header, or ``None``."""
    canonical = normalize_openrouter_provider(value)
    if canonical is None:
        return None
    encoded = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return encoded if len(encoded) <= MAX_SERIALIZED_CHARS else None


def parse_openrouter_provider(raw: object) -> dict[str, Any] | None:
    """Parse untrusted serialized JSON (command field or header)."""
    if isinstance(raw, Mapping):
        return normalize_openrouter_provider(raw)
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or len(text) > MAX_SERIALIZED_CHARS:
        return None
    try:
        decoded = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return normalize_openrouter_provider(decoded)

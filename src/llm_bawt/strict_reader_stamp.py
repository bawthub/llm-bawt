"""The strict-reader receipt stamp on a durable delivery row (TASK-2590).

A row admitted while strict mode selected its bot carries the selection in
``metadata_json["strict_reader"]``. Such a row only ever runs as its own
fresh turn, where the drain validates the stamp; no claim lane (steer,
inbox, forced interrupt) may inject it into another run first.
"""
from __future__ import annotations

from typing import Any, Mapping

STAMP_KEY = "strict_reader"


def strict_stamp(row: Mapping[str, Any] | None) -> dict | None:
    metadata = (row or {}).get("metadata_json") or {}
    stamp = metadata.get(STAMP_KEY) if isinstance(metadata, Mapping) else None
    return dict(stamp) if isinstance(stamp, Mapping) else None


def is_strict_stamped(row: Mapping[str, Any] | None) -> bool:
    return strict_stamp(row) is not None

"""Read-only report of where the live ops catalog differs from the code seeds (TASK-1861).

Seeding is insert-only (``OpsStore.seed_operation_if_missing``): once a slug
exists, later seed changes never reach it. That keeps operator edits safe, but
a field a newer release needs (TASK-1801's ``health_probe``) can be silently
absent from a live row. This report shows every difference per slug, with
nested JSON paths, and never writes, syncs, enables or runs anything.

Each seeded slug gets one status:

* ``clean``: every seeded field matches.
* ``missing``: no live row (soft-deleted rows count as present, flagged).
* ``seed_changed``: differs, and the row was last written by the seeder, so
  the code seed moved on after this row was inserted.
* ``install_override``: differs, last written by the tenant installer.
* ``operator_override``: differs, last written by anyone else (a person, a
  bot or a migration); treat as intentional until someone decides otherwise.

``missing_paths`` (seeded keys absent live) is reported for every status; it
is the case that breaks a run. ``enabled`` is operational state, not part of
the definition, so it is not compared. Live rows with no seed are listed as
``live_only``. Values are what ``GET /v1/ops/operations/{slug}`` already
shows, so the report exposes nothing new.
"""
from __future__ import annotations

import json
from typing import Any

from .tenant_catalog import ACTOR as INSTALL_ACTOR
from .tenant_catalog import SEED_ACTOR

#: Not part of an operation's definition: operators flip it, seeds ship off.
IGNORED_FIELDS = frozenset({"slug", "enabled"})
_ABSENT = object()


def _parsed(value: Any) -> Any:
    """A JSON object/array stored as text is compared structurally."""
    # json.loads (the executor's parser) accepts leading whitespace too.
    if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _diff(seed: Any, live: Any, path: str, out: list[dict[str, Any]]) -> None:
    if isinstance(seed, dict) and isinstance(live, dict):
        for key in sorted(set(seed) | set(live), key=str):
            _diff(seed.get(key, _ABSENT), live.get(key, _ABSENT), f"{path}.{key}", out)
        return
    if seed == live:
        return
    if live is _ABSENT:
        out.append({"path": path, "kind": "missing", "seed": seed})
    elif seed is _ABSENT:
        out.append({"path": path, "kind": "added", "live": live})
    else:
        out.append({"path": path, "kind": "changed", "seed": seed, "live": live})


def _status(row: dict[str, Any], differences: list[dict[str, Any]]) -> str:
    if not differences:
        return "clean"
    by = row.get("updated_by")
    if by == SEED_ACTOR:
        return "seed_changed"
    if by == INSTALL_ACTOR:
        return "install_override"
    return "operator_override"


def compare(seeds: list[dict[str, Any]], live_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The drift report for ``seeds`` against catalog rows in ``to_api()`` form."""
    live = {row["slug"]: row for row in live_rows}
    seeded = {seed["slug"] for seed in seeds}
    entries = []
    for seed in seeds:
        slug = seed["slug"]
        row = live.get(slug)
        if row is None:
            entries.append({"slug": slug, "status": "missing", "differences": [], "missing_paths": []})
            continue
        differences: list[dict[str, Any]] = []
        for field in sorted(set(seed) - IGNORED_FIELDS):
            _diff(_parsed(seed[field]), _parsed(row.get(field, _ABSENT)), field, differences)
        entries.append({
            "slug": slug,
            "status": _status(row, differences),
            "version": row.get("version"),
            "script_hash": row.get("script_hash"),
            "updated_by": row.get("updated_by"),
            "updated_at": row.get("updated_at"),
            "soft_deleted": bool(row.get("soft_deleted_at")),
            "differences": differences,
            "missing_paths": [d["path"] for d in differences if d["kind"] == "missing"],
        })
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return {
        "operations": entries,
        "live_only": sorted(slug for slug in live if slug not in seeded),
        "counts": counts,
        "rows_missing_seeded_keys": sum(1 for e in entries if e["missing_paths"]),
    }


def report(store) -> dict[str, Any]:
    """Read the whole catalog (soft-deleted included) and compare it to SEEDS."""
    from .seeds import SEEDS

    rows = store.list_operations(include_disabled=True, include_soft_deleted=True)
    return compare(SEEDS, [row.to_api(include_script=True) for row in rows])


__all__ = ["IGNORED_FIELDS", "compare", "report"]

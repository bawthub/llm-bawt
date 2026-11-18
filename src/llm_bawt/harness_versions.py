"""Running vs latest agent harness versions for Campus (TASK-2470).

Running: each bridge reports its versions (:mod:`agent_bridge.versions`). A
house record counts as ``running`` only while its epoch equals the live lease
epoch for that exact (house, backend); otherwise the house shows ``unknown``
with no versions, so a stale record from a stopped or replaced bridge is never
presented as current. The mansion has no lease: its record is
``last_reported``. The snapshot and the watcher's frames both use
:func:`classify`, so they agree.

Latest: the newest successful ``harness_version_check`` runs, merged per
package; never the pins or the ``behind`` list.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .campus_events import HARNESS_LATEST_CHANGED, HARNESS_VERSIONS_CHANGED
from .house_watcher import LEASE_RE

log = logging.getLogger(__name__)

MANSION = "mansion"
RUNNING, UNKNOWN, LAST_REPORTED = "running", "unknown", "last_reported"
HOUSE_VERSIONS_RE = re.compile(r"^agent:house:([a-z0-9][a-z0-9-]{0,30}):versions:([a-z0-9_-]+)$")
MANSION_VERSIONS_RE = re.compile(r"^agent:mansion:versions:([a-z0-9_-]+)$")
LATEST_RUN_WINDOW = 20


def _parse(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) and isinstance(value.get("versions"), dict) else None


def classify(slot: str, backend: str, raw: str | None, lease_epoch: str | None) -> dict:
    """One row: ``{slot, backend, status, versions, started_at}``."""
    record = _parse(raw)
    if slot == MANSION:
        ok = record is not None
        status = LAST_REPORTED if ok else UNKNOWN
    else:
        ok = (record is not None and lease_epoch is not None
              and str(record.get("epoch")) == str(lease_epoch))
        status = RUNNING if ok else UNKNOWN
    versions = record["versions"] if ok else None
    return {"slot": slot, "backend": backend, "status": status, "versions": versions,
            "started_at": versions.get("started_at") if versions else None}


async def read_one(client: Any, slot: str, backend: str) -> dict:
    from agent_bridge.houses import lease_key
    from agent_bridge.versions import house_versions_key, mansion_versions_key

    if slot == MANSION:
        return classify(slot, backend, await client.get(mansion_versions_key(backend)), None)
    raw = await client.get(house_versions_key(slot, backend))
    epoch = await client.get(lease_key(slot, backend))
    return classify(slot, backend, raw, epoch)


async def read_all(client: Any) -> dict[tuple[str, str], dict]:
    """Every reported (slot, backend), plus houses holding a lease without one."""
    slots: set[tuple[str, str]] = set()
    async for key in client.scan_iter(match="agent:house:*:versions:*", count=200):
        if m := HOUSE_VERSIONS_RE.match(key):
            slots.add((m.group(1), m.group(2)))
    async for key in client.scan_iter(match="agent:house:*:lease:*", count=200):
        if m := LEASE_RE.match(key):
            slots.add((m.group(1), m.group(2)))
    async for key in client.scan_iter(match="agent:mansion:versions:*", count=200):
        if m := MANSION_VERSIONS_RE.match(key):
            slots.add((MANSION, m.group(1)))
    return {slot: await read_one(client, *slot) for slot in sorted(slots)}


def slot_of_key(key: str) -> tuple[str, str] | None:
    """The (slot, backend) a lease or versions key change affects."""
    if m := HOUSE_VERSIONS_RE.match(key) or LEASE_RE.match(key):
        return m.group(1), m.group(2)
    if m := MANSION_VERSIONS_RE.match(key):
        return MANSION, m.group(1)
    return None


def row_frame(row: dict) -> dict:
    return {"_type": HARNESS_VERSIONS_CHANGED, **row}


# ------------------------------------------------------------------- latest


def merge_latest(results: list[tuple[Any, Any]]) -> dict:
    """``results``: (finished_at, result_json) newest first, SUCCESS runs only.

    Each package takes the newest run that reported a usable ``latest``, so a
    subset run or a malformed entry never erases what an older run observed.
    """
    packages: dict[str, dict] = {}
    cli: dict | None = None
    for finished_at, raw in results:
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            entries = data.get("packages") if isinstance(data, dict) else None
        except ValueError:
            entries = None
        if not isinstance(entries, list):
            continue
        observed = finished_at.isoformat() if hasattr(finished_at, "isoformat") else finished_at
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name, latest = entry.get("name"), entry.get("latest")
            if isinstance(name, str) and isinstance(latest, str) and latest and name not in packages:
                packages[name] = {"version": latest, "observed_at": observed}
            if (cli is None and name == "claude-agent-sdk"
                    and isinstance(entry.get("latest_cli"), str) and entry["latest_cli"]):
                cli = {"version": entry["latest_cli"], "observed_at": observed}
    return {"packages": packages, "claude_cli": cli}


def read_latest(engine: Any) -> dict:
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT r.finished_at, r.result_json FROM job_runs r "
            "JOIN scheduled_jobs j ON j.id = r.job_id "
            "WHERE j.job_type = 'HARNESS_VERSION_CHECK' AND r.status = 'SUCCESS' "
            "AND r.result_json IS NOT NULL "
            "ORDER BY r.finished_at DESC NULLS LAST LIMIT :n"), {"n": LATEST_RUN_WINDOW}).all()
    return merge_latest([(r[0], r[1]) for r in rows])


def publish_latest(engine: Any, config: Any) -> None:
    """After a successful check commits: tell open Campus views. Never raises."""
    from .campus_events import publish_frames

    try:
        publish_frames([{"_type": HARNESS_LATEST_CHANGED, "latest": read_latest(engine)}], config)
    except Exception:
        log.warning("harness latest publish failed", exc_info=True)

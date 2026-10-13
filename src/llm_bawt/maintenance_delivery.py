"""Publication barriers for explicit compaction and the existing durable FIFO.

The turn log can become terminal before its normal event publisher runs. A
successor therefore needs an acknowledged terminal snapshot, not just ended_at.
Enqueue captures a read-only snapshot without taking the target lock (scheduler
callers already hold job locks). Updates to existing rows use the target lock.
"""

from __future__ import annotations

import json

from sqlalchemy import text

from agent_bridge.maintenance import is_compact_command


def as_dict(value):
    if isinstance(value, str):
        value = json.loads(value)
    return dict(value) if isinstance(value, dict) else {}


class MaintenanceBarrier:
    def __init__(self, conn, target: str):
        self.conn = conn
        self.target = target

    def _save(self, head, metadata):
        expression = "CAST(:metadata AS jsonb)" if self.conn.dialect.name == "postgresql" else ":metadata"
        self.conn.execute(text(
            f"UPDATE inter_bot_deliveries SET metadata_json={expression} WHERE id=:id"
        ), {"metadata": json.dumps(metadata), "id": head["id"]})

    def stamp_enqueue(self, metadata):
        """Capture the boundary at arrival, before a newer turn can hide it."""
        result = as_dict(metadata)
        latest = self.conn.execute(text(
            "SELECT id, user_prompt FROM turn_logs WHERE bot_id=:target "
            "ORDER BY created_at DESC LIMIT 1"
        ), {"target": self.target}).mappings().first()
        if latest and is_compact_command(latest.get("user_prompt")):
            result["maintenance_barrier"] = {"turn_id": latest["id"], "published": False}
        return json.dumps(result)

    def prepare(self, head):
        """Return an unreleased source row (possibly still running), or None.

        Existing/ambiguous steer RPCs and accepted dispatches retain their exact
        recovery path. No maintenance policy may turn those into fresh work.
        """
        if head.get("delivery_mode") == "steer" or head.get("transport_accepted_at"):
            return None
        metadata = as_dict(head.get("metadata_json"))
        barrier = as_dict(metadata.get("maintenance_barrier"))
        source_id = barrier.get("turn_id")
        latest = self.conn.execute(text(
            "SELECT * FROM turn_logs WHERE bot_id=:target AND id != :own "
            "ORDER BY created_at DESC LIMIT 1"
        ), {"target": self.target, "own": head["turn_id"]}).mappings().first()
        if latest and is_compact_command(latest.get("user_prompt")):
            source_id = latest["id"]
        if not source_id:
            return None
        if barrier.get("turn_id") == source_id and barrier.get("published"):
            return None
        source = self.conn.execute(text(
            "SELECT * FROM turn_logs WHERE id=:id AND bot_id=:target"
        ), {"id": source_id, "target": self.target}).mappings().first()
        if source is None:
            raise RuntimeError("Maintenance source turn is missing; refusing to release queued work")
        metadata["maintenance_barrier"] = {"turn_id": source_id, "published": False}
        self._save(head, metadata)
        return dict(source)

    def released(self, head, source_id: str):
        metadata = as_dict(head.get("metadata_json"))
        barrier = as_dict(metadata.get("maintenance_barrier"))
        if barrier.get("turn_id") != source_id:
            return  # A new maintenance boundary won the race; do not release it.
        metadata["maintenance_barrier"] = {"turn_id": source_id, "published": True}
        self._save(head, metadata)

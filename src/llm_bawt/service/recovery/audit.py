"""The durable trace of a recovery sweep or purge run (TASK-1232, Snark ruling a).

There is no audit table: logs aren't durable, so every row a run writes
carries the run's stamp, the ``actor`` and one ``sweep_run_id`` per run, and
``rows_for_run`` rebuilds the run's report from the rows alone.

Where the stamp lives (one place per table):

- ``turn_logs``: ``request_json.recovery`` (a settle) or
  ``request_json.recovery.tools_interrupted`` (open tool calls interrupted).
  ``recovery`` is a preserved audit key, so it survives a later re-arm.
- ``tool_approval_requests``: ``continuation_last_error`` (text; every row a
  purge or sweep writes), and the text reasons a purge already sets.
- ``ops_jobs``: ``error_text`` (text).
- ``inter_bot_deliveries``: ``metadata_json.recovery_releases`` (one entry per
  reservation release, oldest dropped past ``RELEASE_STAMP_LIMIT``);
  ``metadata_json.recovery`` repeats the latest.

A ``turn_logs`` row whose ``request_json`` isn't valid JSON can't carry a
stamp; the queries skip it instead of failing on the cast.

A released reservation row is deleted, so its stamp is on its producer's row
(the approval or the delivery), with the released turn id.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text as sa_text


@dataclass(frozen=True)
class RunStamp:
    actor: str
    run_id: str

    @classmethod
    def new(cls, actor: str) -> "RunStamp":
        return cls(actor=actor, run_id=uuid.uuid4().hex)

    def as_dict(self) -> dict[str, str]:
        return {"actor": self.actor, "sweep_run_id": self.run_id}

    def text(self, message: str) -> str:
        """``message`` tagged for a text column; ``rows_for_run`` matches the tag."""
        return f"{message} [actor={self.actor} sweep_run_id={self.run_id}]"


def _tag(run_id: str) -> str:
    return f"%sweep_run_id={run_id}]%"


def rows_for_run(engine, run_id: str) -> dict[str, list[dict[str, Any]]]:
    """Every row run ``run_id`` wrote, by table (the run's durable report)."""
    params = {"run": run_id, "tag": _tag(run_id)}
    with engine.connect() as conn:
        turns = conn.execute(sa_text("""
            SELECT id, bot_id, status, rj->'recovery' AS recovery
            FROM (SELECT id, bot_id, status, CAST(request_json AS jsonb) AS rj
                  FROM turn_logs WHERE pg_input_is_valid(request_json, 'jsonb')) t
            WHERE rj->'recovery'->>'sweep_run_id' = :run
               OR rj->'recovery'->'tools_interrupted'->>'sweep_run_id' = :run
            ORDER BY id
        """), params).mappings().all()
        approvals = conn.execute(sa_text("""
            SELECT id, bot_id, status, continuation_state, continuation_last_error
            FROM tool_approval_requests
            WHERE continuation_last_error LIKE :tag OR execution_error LIKE :tag OR resolution_message LIKE :tag
            ORDER BY id
        """), params).mappings().all()
        jobs = conn.execute(sa_text(
            "SELECT id, state, error_text FROM ops_jobs WHERE error_text LIKE :tag ORDER BY id"
        ), params).mappings().all()
        deliveries = conn.execute(sa_text("""
            SELECT d.id, d.status, s.stamp AS recovery
            FROM inter_bot_deliveries d
            CROSS JOIN LATERAL (
                SELECT e AS stamp FROM jsonb_array_elements(
                    CASE WHEN jsonb_typeof(d.metadata_json->'recovery_releases') = 'array'
                         THEN d.metadata_json->'recovery_releases'
                         ELSE jsonb_build_array(d.metadata_json->'recovery') END) e
                WHERE e->>'sweep_run_id' = :run LIMIT 1) s
            ORDER BY d.id
        """), params).mappings().all()
    return {"turns": [dict(r) for r in turns], "approvals": [dict(r) for r in approvals],
            "jobs": [dict(r) for r in jobs], "deliveries": [dict(r) for r in deliveries]}

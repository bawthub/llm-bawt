"""Which task each agent turn is working on (TASK-1754).

One resolver for the BawtHub fleet chips, the board's stale-Doing marker and
MCP ``bots_list_available``. A turn's task comes from structured, signed
records the app already keeps, never from prose, in this order:

1. ``association``: the turn linked itself (``tasks_associate_current``, or a
   task tool called with ``associate_current_turn=true``);
2. admission accounting snapshot (starting delivery or sole active assignment);
3. ``delivery``: the inter-bot delivery that STARTED the turn carried a
   ``task_id`` (``turn_logs.id = 'turn-' || delivery.id``). Deliveries steered
   into a running turn are messages about other work, not assignments;
4. ``mutation``: the turn's latest task-mutating tool call. Read-only calls
   (``tasks_get``, ``tasks_get_context``, ``tasks_reviews_list``…) never count:
   reviewers and coordinators read many tasks.

Failed tool calls are ignored. Claude (``mcp__<server>__tasks_update``) and
Codex (``tasks_update``) names match on the bare suffix. ``TASK-N`` refs are
normalized to upper case; a cuid is returned as given for BawtHub to resolve.
PostgreSQL only.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text

ASSOCIATION, DELIVERY, MUTATION = "association", "delivery", "mutation"
SOURCES = (ASSOCIATION, "snapshot", DELIVERY, MUTATION)
ASSOCIATING_TOOLS = ("tasks_associate_current",)
# Tools that link the turn when called with associate_current_turn=true.
LINKABLE_TOOLS = ("tasks_update",)
MUTATING_TOOLS = (
    "tasks_update", "steps_add", "steps_update", "steps_set", "steps_delete",
    "tasks_reviews_add", "tasks_reviews_resolve_finding",
)
_TASK_N = re.compile(r"^task-(\d+)$", re.IGNORECASE)


def normalize_task_ref(value: Any) -> str | None:
    """``TASK-N`` for any case of it, a cuid as given, None for blank."""
    ref = str(value or "").strip()
    if not ref:
        return None
    match = _TASK_N.fullmatch(ref)
    return f"TASK-{int(match.group(1))}" if match else ref


@dataclass(frozen=True)
class TurnTask:
    turn_id: str
    bot_id: str
    task_ref: str
    source: str
    started_at: str | None
    ended_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Candidates per turn, then the best per turn (lowest rank, newest).
# ``turns`` is narrowed first so tool calls are only read for those turns.
_RESOLVE = text("""
    WITH turns AS (
        SELECT id, bot_id, created_at, ended_at, task_ref, task_attribution_source FROM turn_logs
        WHERE (CAST(:turn_ids AS text[]) IS NULL OR id = ANY(CAST(:turn_ids AS text[])))
          AND (CAST(:bots AS text[]) IS NULL OR bot_id = ANY(CAST(:bots AS text[])))
          AND (CAST(:since AS timestamptz) IS NULL
               OR created_at >= CAST(:since AS timestamptz) OR ended_at IS NULL
               OR ended_at >= CAST(:since AS timestamptz))
    ),
    calls AS (
        SELECT c.turn_id, c.created_at, c.id,
               regexp_replace(c.tool_name, '^mcp__.*__', '') AS tool,
               c.arguments_json::jsonb AS args
        FROM tool_call_records AS c JOIN turns AS t ON t.id = c.turn_id
        WHERE c.tool_name ~ '(^|__)(tasks|steps)_'
          AND c.is_error IS NOT TRUE
          AND pg_input_is_valid(c.arguments_json, 'jsonb')
    ),
    candidates AS (
        SELECT turn_id, 1 AS rank, created_at AS at, args->>'task_id' AS ref FROM calls
        WHERE tool = ANY(CAST(:associating AS text[]))
           OR (tool = ANY(CAST(:linkable AS text[]))
               AND args->'associate_current_turn' = 'true'::jsonb)
        UNION ALL
        SELECT id, 2, created_at, task_ref FROM turns
        UNION ALL
        SELECT t.id, 3, d.created_at, d.task_id
        FROM turns AS t JOIN inter_bot_deliveries AS d ON t.id = 'turn-' || d.id
        UNION ALL
        SELECT turn_id, 4, created_at, args->>'task_id' FROM calls
        WHERE tool = ANY(CAST(:mutating AS text[]))
    ),
    best AS (
        SELECT DISTINCT ON (turn_id) turn_id, rank, ref FROM candidates
        WHERE NULLIF(btrim(ref), '') IS NOT NULL
        ORDER BY turn_id, rank, at DESC
    )
    SELECT t.id AS turn_id, t.bot_id, t.created_at, t.ended_at, b.rank, b.ref,
           t.task_attribution_source
    FROM best AS b JOIN turns AS t ON t.id = b.turn_id
    ORDER BY t.created_at DESC
""")


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else value


def resolve_turn_tasks(
    conn,
    *,
    turn_ids: list[str] | None = None,
    bots: list[str] | None = None,
    since: datetime | None = None,
) -> list[TurnTask]:
    """Every matching turn that resolves to a task, newest first.

    Filters combine: ``turn_ids`` (exact turns), ``bots``, and ``since``
    (turns started, ended or still open since then).
    """
    if turn_ids is not None and not turn_ids:
        return []
    if bots is not None and not bots:
        return []
    rows = conn.execute(_RESOLVE, {
        "turn_ids": turn_ids, "bots": bots, "since": since,
        "associating": list(ASSOCIATING_TOOLS), "linkable": list(LINKABLE_TOOLS),
        "mutating": list(MUTATING_TOOLS),
    }).mappings().all()
    resolved = []
    for row in rows:
        ref = normalize_task_ref(row["ref"])
        if ref is None:
            continue
        resolved.append(TurnTask(
            turn_id=row["turn_id"], bot_id=row["bot_id"], task_ref=ref,
            source=(row["task_attribution_source"] if row["rank"] == 2
                    else SOURCES[row["rank"] - 1]),
            started_at=_iso(row["created_at"]), ended_at=_iso(row["ended_at"]),
        ))
    return resolved

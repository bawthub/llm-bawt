"""Bounded SQL reads for the context-waste report (TASK-1585).

Every query is scoped on *both* tables: the caller's user and bot set are
predicates on ``tool_call_records`` and on the joined ``turn_logs`` row, so a
tool row can never surface through another user's turn or vice versa. Result
bodies and blobs are never read; argument JSON is read truncated, only for
the handful of example rows whose redacted key shape the report shows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import bindparam, text
from sqlmodel import Session

#: Trusted rows were measured by a producer that stamps accounting v2+.
TRUSTED_ACCOUNTING = 2
MAX_REPEAT_ROWS = 50_000
MAX_OVERSIZED_ROWS = 5_000
MAX_TURN_ROWS = 50_000
MAX_ARG_CHARS = 4_000


@dataclass(frozen=True)
class Scope:
    user_id: str
    bots: tuple[str, ...]
    since: datetime
    until: datetime

    def params(self, **extra: Any) -> dict[str, Any]:
        return {"user_id": self.user_id, "bots": list(self.bots),
                "since": self.since, "until": self.until, **extra}


# Both tables carry the scope: the turn's user/bot must match the tool row's.
_JOIN = """
    FROM tool_call_records r
    JOIN turn_logs t ON t.id = r.turn_id
    WHERE r.user_id = :user_id AND t.user_id = :user_id
      AND r.bot_id IN :bots AND t.bot_id IN :bots AND t.bot_id = r.bot_id
      AND r.created_at >= :since AND r.created_at < :until
"""
_TRUSTED = "r.result_accounting >= :trusted"


def _run(engine, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    statement = text(sql).bindparams(bindparam("bots", expanding=True))
    with Session(engine) as session:
        return [dict(row) for row in session.execute(statement, params).mappings().all()]


def load_tool_totals(engine, scope: Scope) -> list[dict[str, Any]]:
    """Calls and result tokens per bot and tool, split trusted vs unverified."""
    return _run(engine, f"""
        SELECT r.bot_id AS bot_id, r.tool_name AS tool_name,
               CASE WHEN {_TRUSTED} THEN 1 ELSE 0 END AS trusted,
               COUNT(*) AS calls,
               COALESCE(SUM(r.result_total_tokens), 0) AS tokens,
               COALESCE(MAX(r.result_total_tokens), 0) AS max_tokens
        {_JOIN}
        GROUP BY r.bot_id, r.tool_name, CASE WHEN {_TRUSTED} THEN 1 ELSE 0 END
    """, scope.params(trusted=TRUSTED_ACCOUNTING))


def load_accounting_cutover(engine, scope: Scope) -> datetime | None:
    """First trusted row in scope and window, so the UI can show the cutover."""
    rows = _run(engine, f"SELECT MIN(r.created_at) AS first_trusted {_JOIN} AND {_TRUSTED}",
                scope.params(trusted=TRUSTED_ACCOUNTING))
    return rows[0]["first_trusted"] if rows else None


def load_repeat_candidates(engine, scope: Scope, *, min_tokens: int) -> list[dict[str, Any]]:
    """Eligible rows whose (bot, sha) occurs more than once, oldest first.

    Eligible means trusted, complete, no media blocks (two different images
    with identical text share a sha), above ``min_tokens``, and a known
    session. Grouping by (bot, sha) here, not (bot, tool, sha), lets the
    report also find cross-tool identical output from the same rows.
    """
    return _run(engine, f"""
        WITH eligible AS (
            SELECT r.id AS id, r.bot_id AS bot_id, r.tool_name AS tool_name,
                   r.result_sha256 AS sha, r.result_total_tokens AS tokens,
                   r.created_at AS created_at, r.turn_id AS turn_id,
                   t.agent_session_key AS session_key
            {_JOIN}
              AND {_TRUSTED} AND r.result_complete = :complete
              AND r.result_media_blocks = 0 AND r.result_sha256 IS NOT NULL
              AND r.result_total_tokens > :min_tokens
              AND t.agent_session_key IS NOT NULL
        ), repeated AS (
            SELECT bot_id, sha FROM eligible GROUP BY bot_id, sha HAVING COUNT(*) > 1
        )
        SELECT e.* FROM eligible e
        JOIN repeated g ON g.bot_id = e.bot_id AND g.sha = e.sha
        ORDER BY e.bot_id, e.sha, e.created_at, e.id
        LIMIT :row_limit
    """, scope.params(trusted=TRUSTED_ACCOUNTING, complete=True,
                      min_tokens=min_tokens, row_limit=MAX_REPEAT_ROWS + 1))


def load_oversized_candidates(engine, scope: Scope, *, min_tokens: int) -> list[dict[str, Any]]:
    """Largest trusted results above the smallest configured threshold."""
    return _run(engine, f"""
        SELECT r.id AS id, r.bot_id AS bot_id, r.tool_name AS tool_name,
               r.result_total_tokens AS tokens, r.created_at AS created_at,
               r.turn_id AS turn_id,
               SUBSTR(r.arguments_json, 1, {MAX_ARG_CHARS}) AS arguments_json
        {_JOIN}
          AND {_TRUSTED} AND r.result_total_tokens > :min_tokens
        ORDER BY r.result_total_tokens DESC, r.id
        LIMIT :row_limit
    """, scope.params(trusted=TRUSTED_ACCOUNTING, min_tokens=min_tokens,
                      row_limit=MAX_OVERSIZED_ROWS + 1))


def load_arguments(engine, scope: Scope, ids: Sequence[int]) -> dict[int, str | None]:
    """Truncated argument JSON for example rows, still scoped to the caller."""
    if not ids:
        return {}
    statement = text(f"""
        SELECT r.id AS id, SUBSTR(r.arguments_json, 1, {MAX_ARG_CHARS}) AS arguments_json
        {_JOIN} AND r.id IN :ids
    """).bindparams(bindparam("bots", expanding=True), bindparam("ids", expanding=True))
    with Session(engine) as session:
        rows = session.execute(statement, scope.params(ids=list(ids))).mappings().all()
    return {row["id"]: row["arguments_json"] for row in rows}


def load_turn_usage(engine, scope: Scope) -> list[dict[str, Any]]:
    """Every finished turn in scope; usage JSON is small (sub-3k chars)."""
    return _run(engine, """
        SELECT t.id AS id, t.bot_id AS bot_id, t.created_at AS created_at,
               t.token_usage_json AS token_usage_json
        FROM turn_logs t
        WHERE t.user_id = :user_id AND t.bot_id IN :bots
          AND t.created_at >= :since AND t.created_at < :until
          AND t.ended_at IS NOT NULL
        ORDER BY t.created_at, t.id
        LIMIT :row_limit
    """, scope.params(row_limit=MAX_TURN_ROWS + 1))

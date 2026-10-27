"""UI-only listing of the records one recall read (TASK-1519).

Reads the reference ledger written by ``RecallService`` and resolves a page of
snippets live, under the CURRENT policy. It never feeds a bot: the result goes
to the BawtHub tool card only, so the recall payload doesn't grow (TASK-1555).

Authority is re-applied on every read:
- the ledger row must belong to the asking bot and user and be unexpired;
- the owner bot must still be searchable, else the whole list is unavailable;
- entries whose source bot (or delivering bot) is no longer searchable are
  dropped and counted in ``withheld_now``;
- tasks pass the same shared-task authority as recall, else they are withheld;
- a message, memory or tool receipt that no longer passes its source's
  authority query is listed as ``available: false`` with no content; so is a
  tool_args call, which is shown only as its re-projected command facts.
"""
from __future__ import annotations

from typing import Any

from .budgets import DATABASE_WORKER
from .contracts import TASK_SOURCES, RecallError, RecallScope
from .ledger import LedgerRecord, PostgresLedger
from .postgres import (
    MESSAGE_AUTHORITY, PUBLIC_HISTORY_AUTHORITY, TOOL_ARG_CHARS, RecallDatabase, delivery_authors,
    partition_id, tool_arg_facts,
)
from .task_source import TaskSource

MAX_PAGE = 50
SNIPPET_CHARS = 280
# Entry fields the UI may see; everything is a reference or server-owned flag.
PUBLIC_FIELDS = ("source", "id", "source_bot", "session_id", "at", "speaker", "follows",
                 "admitted", "cited")

_MESSAGE_SNIPPETS = f"""
    SELECT m.id, substr(m.content, p.start + 1, :chars) AS snippet,
           char_length(m.content) > p.start + :chars AS more
    FROM unnest(CAST(:ids AS text[]), CAST(:parts AS text[]), CAST(:bots AS text[]),
                CAST(:starts AS int[])) AS p(id, partition, source_bot, start)
    JOIN messages m ON m.id = p.id AND m.bot_id = p.partition
    JOIN sessions s ON s.id = m.session_id
    WHERE {PUBLIC_HISTORY_AUTHORITY}
"""
_MEMORY_SNIPPETS = f"""
    SELECT mem.id, substr(mem.content, w.start + 1, :chars) AS snippet,
           char_length(mem.content) > w.start + :chars AS more
    FROM unnest(CAST(:ids AS text[]), CAST(:starts AS int[])) AS w(id, start)
    JOIN memories mem ON mem.id = w.id AND mem.bot_id = :partition
    WHERE jsonb_typeof(mem.source_message_ids) = 'array'
      AND jsonb_array_length(mem.source_message_ids) BETWEEN 1 AND 64
      AND NOT EXISTS (
        SELECT 1 FROM jsonb_array_elements_text(mem.source_message_ids) src(id)
        WHERE NOT EXISTS (
            SELECT 1 FROM messages m JOIN sessions s ON s.id = m.session_id
            WHERE m.id = src.id AND {MESSAGE_AUTHORITY}
        )
      )
"""
_TOOL_SNIPPETS = f"""
    SELECT t.id::text AS id,
           left(t.tool_name, 128) || ' · ' || coalesce(left(t.approval_status, 32), 'unknown') AS snippet,
           FALSE AS more
    FROM tool_call_records t
    JOIN turn_logs turns ON turns.id = t.turn_id
      AND turns.bot_id = t.bot_id AND turns.user_id = t.user_id
    JOIN messages m ON m.id = turns.trigger_message_id AND m.bot_id = :partition
    JOIN sessions s ON s.id = m.session_id
    WHERE t.id::text = ANY(CAST(:ids AS text[]))
      AND t.bot_id = :bot AND t.user_id = :user AND {MESSAGE_AUTHORITY}
"""
# Command facts are re-projected from the call under the current trigger-message
# authority; the arguments themselves never leave this module (TASK-1557).
_TOOL_ARG_ROWS = f"""
    SELECT t.id::text AS id, left(t.arguments_json, :argument_chars) AS arguments
    FROM unnest(CAST(:ids AS text[]), CAST(:parts AS text[]), CAST(:bots AS text[]))
         AS p(id, partition, source_bot)
    JOIN tool_call_records t ON t.id::text = p.id AND t.bot_id = p.source_bot
      AND t.user_id = :user AND t.tool_name = 'Bash'
    JOIN turn_logs turns ON turns.id = t.turn_id
      AND turns.bot_id = t.bot_id AND turns.user_id = t.user_id
    JOIN messages m ON m.id = turns.trigger_message_id AND m.bot_id = p.partition
    JOIN sessions s ON s.id = m.session_id
    WHERE {PUBLIC_HISTORY_AUTHORITY}
"""


def _searchable(scope: RecallScope) -> frozenset[str]:
    return scope.allowed_bot_ids - scope.excluded_bot_ids


def visible(entry: dict, scope: RecallScope) -> bool:
    """Current policy: no entry from, or delivered by, a non-searchable bot."""
    allowed = _searchable(scope)
    source_bot = entry.get("source_bot")
    if source_bot is not None and source_bot not in allowed:
        return False
    speaker = entry.get("speaker")
    if isinstance(speaker, str) and speaker.startswith("bot:") and speaker[4:] not in allowed:
        return False
    return True


def public(entry: dict) -> dict[str, Any]:
    return {field: entry.get(field) for field in PUBLIC_FIELDS}


class EvidenceDrilldown:
    def __init__(self, database: RecallDatabase, ledger: PostgresLedger, tasks: TaskSource):
        self.database = database
        self.ledger = ledger
        self.tasks = tasks

    async def _load(self, handle: str, bot_id: str, user_id: str) -> tuple[LedgerRecord, RecallScope] | None:
        record = await DATABASE_WORKER.run(self.ledger.read, handle, bot_id, user_id)
        if record is None:
            return None
        try:
            scope = await self.database.owner_scope(record.bot_id, record.user_id, record.session_id)
        except RecallError as error:
            if error.code == "unauthorized":
                return None
            raise
        return record, scope

    async def summary(self, handle: str, bot_id: str, user_id: str) -> dict | None:
        """Per-source counts and the cited entries' provenance (speaker for findings)."""
        loaded = await self._load(handle, bot_id, user_id)
        if loaded is None:
            return None
        record, scope = loaded
        sources: dict[str, dict] = {}
        shown = []
        for entry in record.entries:
            counts = sources.setdefault(entry["source"], {"recorded": 0, "withheld_now": 0})
            counts["recorded"] += 1
            if visible(entry, scope):
                shown.append(entry)
            else:
                counts["withheld_now"] += 1
        # Tasks pass the same current task authority as page(), so a task that
        # became protected is neither cited nor counted as readable here.
        tasks = [entry for entry in shown if entry["source"] in TASK_SOURCES]
        if tasks:
            authorized = {id(entry) for entry, _ in await self._task_rows(tasks, scope)}
            for entry in tasks:
                if id(entry) not in authorized:
                    sources[entry["source"]]["withheld_now"] += 1
            shown = [entry for entry in shown
                     if entry["source"] not in TASK_SOURCES or id(entry) in authorized]
        cited = [public(entry) for entry in shown if entry.get("cited")]
        return {"coverage": record.coverage, "sources": sources, "cited": cited,
                "omitted": record.omitted}

    async def page(self, handle: str, bot_id: str, user_id: str, source: str,
                   offset: int, limit: int) -> dict | None:
        loaded = await self._load(handle, bot_id, user_id)
        if loaded is None:
            return None
        record, scope = loaded
        limit = max(1, min(limit, MAX_PAGE))
        offset = max(0, offset)
        recorded = [entry for entry in record.entries if entry.get("source") == source]
        listed = [entry for entry in recorded if visible(entry, scope)]
        if source in TASK_SOURCES:
            # Few by construction (at most three tasks per recall): authorize every
            # one so totals never depend on which page was asked for.
            listed = await self._authorized_tasks(listed, scope)
            window = listed[offset:offset + limit]
        else:
            window = listed[offset:offset + limit]
            window = await self._resolve(source, window, scope)
        return {"source": source, "coverage": record.coverage.get(source),
                "recorded": len(recorded), "withheld_now": len(recorded) - len(listed),
                "total": len(listed), "offset": offset, "limit": limit,
                "entries": window, "omitted": record.omitted}

    async def _authorized_tasks(self, entries: list[dict], scope: RecallScope) -> list[dict]:
        return [{**public(entry), "available": True,
                 "title": row["title"][:300], "status": row["status"][:32],
                 "snippet": row["title"][:SNIPPET_CHARS], "more": False}
                for entry, row in await self._task_rows(entries, scope)]

    async def _task_rows(self, entries: list[dict], scope: RecallScope) -> list[tuple[dict, dict]]:
        """Entries whose task passes the current shared-task authority, with its row."""
        listed = []
        rows: dict[str, dict | None] = {}
        async with self.tasks.reader() as reader:
            for entry in entries:
                # Part evidence ("TASK-N#review:…", TASK-1557) is authorized as
                # its task, once per task.
                ref = str(entry["id"]).split("#", 1)[0]
                if ref not in rows:
                    try:
                        rows[ref] = await self.tasks.authorized_task(reader, ref, scope)
                    except RecallError:
                        rows[ref] = None
                row = rows[ref]
                if row is not None:
                    listed.append((entry, row))
        return listed

    async def _resolve(self, source: str, window: list[dict], scope: RecallScope) -> list[dict]:
        if not window:
            return []
        ids = [str(entry["id"]) for entry in window]
        starts = [max(0, int(entry.get("start_char") or 0)) for entry in window]
        base = {"bot": scope.bot_id, "user": scope.user_id, "partition": partition_id(scope.bot_id),
                "chars": SNIPPET_CHARS, "ids": ids, "starts": starts}
        if source == "messages":
            bots = [str(entry.get("source_bot") or "") for entry in window]
            statement, params = _MESSAGE_SNIPPETS, {
                **base, "bots": bots, "parts": [partition_id(bot) if bot else "" for bot in bots],
                "delivery_authors": delivery_authors(scope)}
        elif source == "memories":
            statement, params = _MEMORY_SNIPPETS, base
        elif source == "tool_evidence":
            statement, params = _TOOL_SNIPPETS, base
        elif source == "tool_args":
            bots = [str(entry.get("source_bot") or "") for entry in window]
            statement, params = _TOOL_ARG_ROWS, {
                **base, "bots": bots, "parts": [partition_id(bot) if bot else "" for bot in bots],
                "delivery_authors": delivery_authors(scope), "argument_chars": TOOL_ARG_CHARS}
        else:
            raise RecallError("invalid_request")
        try:
            rows = await DATABASE_WORKER.run(self.database.read, statement, params)
        except RecallError:
            raise
        except Exception as error:
            raise RecallError("source_unavailable") from error
        if source == "tool_args":
            # A call whose command no longer projects a fact is unavailable.
            projected = ((row["id"], tool_arg_facts(row["arguments"])) for row in rows)
            rows = [{"id": row_id, "snippet": facts[:SNIPPET_CHARS], "more": len(facts) > SNIPPET_CHARS}
                    for row_id, facts in projected if facts]
        found = {str(row["id"]): row for row in rows}
        resolved = []
        for entry in window:
            row = found.get(str(entry["id"]))
            item = public(entry)
            if row is None:
                resolved.append({**item, "available": False})
            else:
                resolved.append({**item, "available": True, "snippet": row["snippet"],
                                 "more": bool(row["more"])})
        return resolved

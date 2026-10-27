"""Shared-task search with bounded provenance checks, never task turns.

Two providers read the same shared tasks under the same authority (the task as
``tasks_get`` returns it, after ``authorized_task``):

- ``tasks`` (depth ``key_fields``): title, status and description, plus the
  task's response and outcome;
- ``task_details`` (depth ``details``): the task's peer reviews and its step
  outputs (TASK-1557).

Each searchable passage is one evidence record ("part"): ``TASK-N`` for the
title/status/description, ``TASK-N#response``, ``TASK-N#outcome``,
``TASK-N#review:<cuid>`` and ``TASK-N#step:<n>``. A part is a window of its
original text, opened where the most query terms are, so an answer deep in a
long response still reaches the model. A review written by an actor the caller
may not read is dropped and counted as withheld, like a withheld task.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
import re

import httpx

from llm_bawt.bawthub_auth import service_auth
from llm_bawt.mcp_server.task_api import API_PREFIX, BASE_URL
from llm_bawt.mcp_server.task_association import _INTERNAL_URL

from .contracts import (
    CUID, MAX_SOURCE_RECORDS, TASK_REF, Evidence, EvidenceBatch, RecallError, RecallScope, SourceProvider,
)
from .excerpts import focused_window, term_coverage
from .http import BoundedHTTP
from .postgres import utc_iso

MAX_TASKS = 3
# A task record, after the provider's own windowing, is still clipped to a
# token prefix by the service and shares the global input budget with every
# other source; a short part keeps six of them affordable.
PART_CHARS = 1000
MAX_PARTS_PER_TASK = 3
# BawtHub's review list caps a page at 50; more means coverage is partial.
REVIEW_PAGE = 50
# A full task row (steps included) and a page of reviews each run to tens of
# KiB on busy tasks; the aggregate covers three tasks and their authority reads.
RESPONSE_BYTES = 262_144
TOTAL_BYTES = 1_048_576
# Search (1) + per task: row, project, associations, audit, reviews (5) x 3.
MAX_CALLS = 18
TASK_SEARCH_SECONDS = 4.0
_QUERY_TASK = re.compile(r"(?<![A-Za-z0-9])TASK-([1-9][0-9]{0,9})(?![0-9])", re.IGNORECASE)
# Failures inside one task's reads are that task's problem: it is withheld,
# never authorized, and other tasks and sources continue.
_TASK_FAILURES = {"source_unavailable", "retrieval_budget", "task_provenance_incomplete", "timeout", "busy"}


@dataclass(frozen=True)
class Part:
    """One searchable passage of an authorized task, before windowing.

    ``header`` is server-built context (task, field, review stage/verdict);
    ``body`` is the stored text the window is cut from. A part that isn't the
    task's own record is evidence only when its body matches a query term.
    """
    suffix: str
    header: str
    body: str
    at: str | None = None
    speaker: str | None = None
    needs_match: bool = True


@dataclass(frozen=True)
class Ranked:
    task: str
    coverage: int
    at: str | None
    order: int
    evidence: Evidence


def _best_first(items: list[Ranked]) -> list[Ranked]:
    """Query-term coverage first, then newest (undated last), then task order."""
    items = sorted(items, key=lambda item: item.order)
    items.sort(key=lambda item: (item.at is not None, item.at or ""), reverse=True)
    items.sort(key=lambda item: -item.coverage)
    return items


class TaskSource(SourceProvider):
    source = "tasks"
    depth = "key_fields"
    # Searched by default, so a slow BawtHub must not spend the recall deadline
    # the other sources and the utility model need (synthesis allows 22 s of 30).
    max_seconds = TASK_SEARCH_SECONDS

    @staticmethod
    @asynccontextmanager
    async def reader():
        """One bounded reader per recall: search plus three full authority checks."""
        async with httpx.AsyncClient(timeout=4.0, follow_redirects=False,
                                     auth=service_auth()) as client:
            yield BoundedHTTP(client, max_calls=MAX_CALLS, max_bytes=TOTAL_BYTES)

    async def search(self, query: str, scope: RecallScope, task_ids: tuple[str, ...]) -> list[Evidence]:
        async with self.reader() as reader:
            return await self.search_with(reader, query, scope, task_ids)

    async def candidates(self, reader, query: str, task_ids: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
        """Explicit ids, else tasks the question names, then semantic matches."""
        if task_ids:
            return task_ids, "explicit"
        named = tuple(dict.fromkeys(f"TASK-{number}" for number in _QUERY_TASK.findall(query)))
        if len(named) >= MAX_TASKS:
            return named[:MAX_TASKS], "explicit"
        # Existing semantic search only embeds the query and SELECTs rows;
        # it does not call the task-embedding reconciler or persist anything.
        payload = await reader.json("POST", f"{BASE_URL}{API_PREFIX}/tasks/search/semantic",
                                    json={"query": query, "limit": MAX_TASKS})
        if (not isinstance(payload, dict) or payload.get("mode") not in {"semantic", "keyword"}
                or not isinstance(payload.get("results"), list) or len(payload["results"]) > MAX_TASKS):
            raise RecallError("source_unavailable")
        found = tuple(row.get("id") if isinstance(row, dict) else None for row in payload["results"])
        if any(not isinstance(ref, str) or not TASK_REF.fullmatch(ref) for ref in found):
            raise RecallError("source_unavailable")
        return named + found, payload["mode"]

    async def search_with(self, reader, query: str, scope: RecallScope,
                          task_ids: tuple[str, ...]) -> EvidenceBatch:
        try:
            refs, mode = await self.candidates(reader, query, task_ids)
        except RecallError as error:
            if error.code in _TASK_FAILURES:
                # Discovery failed: the source is unavailable, never "no match".
                raise RecallError("source_unavailable" if error.code not in {"timeout", "busy"}
                                  else error.code) from error
            raise
        result = EvidenceBatch(search_mode=mode)
        ranked: list[Ranked] = []
        seen: set[str] = set()
        for ref in refs:
            if len(seen) >= MAX_TASKS:
                break
            try:
                row = await self.authorized_task(reader, ref, scope)
                if row is None:
                    result.withheld += 1
                    continue
                if row["id"] in seen:
                    continue
                seen.add(row["id"])
                parts = await self.parts(reader, row, scope, result)
            except RecallError as error:
                if error.code not in _TASK_FAILURES:
                    raise
                # Busy/large tasks whose full provenance cannot be bounded are
                # withheld, never silently authorized. Preserve other sources;
                # coverage says incomplete, not 'no matching tasks exist'.
                result.withheld += 1
                continue
            ranked.extend(self.rank(row, parts, query, scope))
        result.extend(item.evidence for item in self.select(ranked))
        return result

    async def parts(self, reader, row: dict, scope: RecallScope, result: EvidenceBatch) -> list[Part]:
        """Key fields: the task's own record, response and outcome."""
        short_id = row["shortId"]
        updated = utc_iso_text(row.get("updatedAt"))
        head = f"{short_id}: {row['title'][:300]}\nStatus: {row['status'][:32]}\n"
        parts = [Part("", head, _text(row.get("description")), updated, needs_match=False)]
        for field in ("response", "outcome"):
            body = _text(row.get(field))
            if body:
                parts.append(Part(f"#{field}", f"{short_id} {field}:\n", body, updated))
        return parts

    def rank(self, row: dict, parts: list[Part], query: str, scope: RecallScope) -> list[Ranked]:
        """Window each part where its query terms are; drop parts that match none."""
        ranked = []
        for order, part in enumerate(parts):
            coverage = term_coverage(part.body, query)
            if part.needs_match and coverage == 0:
                continue
            window, start, truncated, kind = focused_window(
                part.body, query, max(200, PART_CHARS - len(part.header)))
            evidence = Evidence(self.source, f"{row['shortId']}{part.suffix}", scope.bot_id, scope.user_id,
                                part.header + window, start, truncated, kind, part.at, part.speaker)
            ranked.append(Ranked(row["id"], coverage, part.at, order, evidence))
        return ranked

    @staticmethod
    def select(ranked: list[Ranked]) -> list[Ranked]:
        """Each task's best part, then the best of the rest, within both caps."""
        ordered = _best_first(ranked)
        picked: list[Ranked] = []
        counts: dict[str, int] = {}
        for item in ordered:
            if item.task not in counts and len(picked) < MAX_SOURCE_RECORDS:
                counts[item.task] = 1
                picked.append(item)
        first = {id(item) for item in picked}
        for item in ordered:
            if len(picked) >= MAX_SOURCE_RECORDS:
                break
            if id(item) not in first and counts.get(item.task, 0) < MAX_PARTS_PER_TASK:
                counts[item.task] = counts.get(item.task, 0) + 1
                picked.append(item)
        return _best_first(picked)

    async def authorized_task(self, reader, ref: str, scope: RecallScope) -> dict | None:
        """The task row if the caller may read it now, else None (TASK-1519 shares this).

        Raises RecallError when authority can't be established; callers treat
        that as withheld, never as authorized.
        """
        # Semantic candidates intentionally lack assignment. Fetch it
        # before admitting any task content to the utility model.
        row = await reader.json("GET", f"{BASE_URL}{API_PREFIX}/tasks/{ref}", byte_limit=RESPONSE_BYTES)
        if not isinstance(row, dict) or not CUID.fullmatch(str(row.get("id", ""))):
            raise RecallError("source_unavailable")
        if not await self._authorized(reader, row, scope):
            return None
        short_id = row.get("shortId")
        if not isinstance(short_id, str) or not TASK_REF.fullmatch(short_id):
            raise RecallError("source_unavailable")
        if not all(isinstance(row.get(k), str) for k in ("title", "status")):
            raise RecallError("source_unavailable")
        return row

    async def _authorized(self, reader, row: dict, scope: RecallScope) -> bool:
        # Tasks are shared by the existing API. Assignment is not read authority.
        # Check protected contributors without traversing their conversations.
        owner = row.get("agentBotId")
        if owner is not None and owner not in scope.allowed_bot_ids:
            return False
        if row.get("projectId"):
            project_id = str(row["projectId"])
            if not CUID.fullmatch(project_id):
                return False
            project = await reader.json("GET", f"{BASE_URL}{API_PREFIX}/projects/{project_id}")
            if not isinstance(project, dict):
                raise RecallError("source_unavailable")
            if project.get("agentBotId") is not None and project["agentBotId"] not in scope.allowed_bot_ids:
                return False
        associations = await reader.json("GET", f"{_INTERNAL_URL}/internal/tasks/{row['id']}/associations")
        if not isinstance(associations, dict) or not isinstance(associations.get("sessions"), list):
            raise RecallError("source_unavailable")
        for session in associations["sessions"]:
            if (not isinstance(session, dict) or session.get("botId") not in scope.allowed_bot_ids
                    or session.get("userId") != scope.user_id):
                return False
        # The audit isn't an authorization grant (actor headers are attribution).
        # It supplies an additional protected-contributor deny check. System,
        # cron, and missing human actor IDs are normal shared-task events.
        audit = await reader.json("GET", f"{BASE_URL}{API_PREFIX}/activity",
                                  params={"taskId": row["id"], "limit": 100, "offset": 0})
        if not isinstance(audit, dict) or not isinstance(audit.get("activities"), list):
            raise RecallError("source_unavailable")
        activities = audit["activities"]
        if audit.get("total") != len(activities) or len(activities) > 100:
            raise RecallError("task_provenance_incomplete")
        for activity in activities:
            if not isinstance(activity, dict):
                return False
            actor_type, actor_id = activity.get("actorType"), activity.get("actorId")
            if actor_type == "bot" and actor_id not in scope.allowed_bot_ids:
                return False
            if actor_type == "user" and actor_id is not None and actor_id != scope.user_id:
                return False
            if actor_type not in {"user", "bot", "system", "cron"}:
                return False
        return True


class TaskDetailSource(TaskSource):
    """Reviews and step outputs of the same authorized tasks (opt-in, deeper)."""
    source = "task_details"
    depth = "details"

    async def parts(self, reader, row: dict, scope: RecallScope, result: EvidenceBatch) -> list[Part]:
        short_id = row["shortId"]
        parts = []
        for step in row.get("steps") if isinstance(row.get("steps"), list) else ():
            if not isinstance(step, dict) or not isinstance(step.get("orderIndex"), int):
                continue
            body = "\n".join(text for text in (_text(step.get("title")), _text(step.get("output"))) if text)
            if body:
                number = step["orderIndex"]
                parts.append(Part(f"#step:{number}", f"{short_id} step {number} ({_text(step.get('status'))[:16]}):\n",
                                  body, utc_iso_text(step.get("completedAt") or step.get("updatedAt"))))
        parts.extend(await self.reviews(reader, row, scope, result))
        return parts

    async def reviews(self, reader, row: dict, scope: RecallScope, result: EvidenceBatch) -> list[Part]:
        short_id = row["shortId"]
        try:
            page = await reader.json("GET", f"{BASE_URL}{API_PREFIX}/tasks/{row['id']}/reviews",
                                     params={"limit": REVIEW_PAGE}, byte_limit=RESPONSE_BYTES)
        except RecallError as error:
            if error.code not in _TASK_FAILURES:
                raise
            # The task and its steps stay searchable; its reviews didn't fit.
            result.partial.add("reviews_unavailable")
            return []
        if not isinstance(page, dict) or not isinstance(page.get("reviews"), list):
            result.partial.add("reviews_unavailable")
            return []
        if page.get("nextCursor") is not None or len(page["reviews"]) > REVIEW_PAGE:
            # Newest first: the older reviews past one page aren't searched.
            result.partial.add("reviews_truncated")
        parts = []
        searchable = scope.allowed_bot_ids - scope.excluded_bot_ids
        for review in page["reviews"][:REVIEW_PAGE]:
            if not isinstance(review, dict):
                result.withheld += 1
                continue
            actor_type, actor_id, review_id = review.get("actorType"), review.get("actorId"), review.get("id")
            if actor_type == "bot" and actor_id in searchable:
                speaker = f"bot:{actor_id}"
            elif actor_type == "user" and actor_id == scope.user_id:
                speaker = "user"
            else:
                # A review is read under its author's authority as well as the
                # task's: an excluded or unknown author is withheld, not shown.
                result.withheld += 1
                continue
            if not isinstance(review_id, str) or not CUID.fullmatch(review_id):
                result.withheld += 1
                continue
            stage, verdict = _text(review.get("stage"))[:16], _text(review.get("verdict"))[:24]
            target = _text(review.get("targetRevision"))[:80]
            header = f"{short_id} {stage} review {verdict}" + (f" of {target}" if target else "") + ":\n"
            parts.append(Part(f"#review:{review_id}", header, _text(review.get("body")),
                              utc_iso_text(review.get("createdAt")), speaker))
        return parts


def _text(value) -> str:
    return value if isinstance(value, str) else ""


def utc_iso_text(value) -> str | None:
    """BawtHub returns ISO strings; recall orders by fixed-width UTC strings."""
    if not isinstance(value, str):
        return None
    try:
        return utc_iso(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None

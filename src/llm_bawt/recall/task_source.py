"""Explicit shared-task search with bounded provenance checks, never task turns."""
from __future__ import annotations

import httpx

from llm_bawt.bawthub_auth import service_auth
from llm_bawt.mcp_server.task_api import API_PREFIX, BASE_URL
from llm_bawt.mcp_server.task_association import _INTERNAL_URL

from .contracts import CUID, TASK_REF, Evidence, EvidenceBatch, RecallError, RecallScope, SourceProvider
from .excerpts import excerpt_window
from .http import BoundedHTTP


class TaskSource(SourceProvider):
    source = "tasks"

    async def search(self, query: str, scope: RecallScope, task_ids: tuple[str, ...]) -> list[Evidence]:
        async with httpx.AsyncClient(timeout=4.0, follow_redirects=False,
                                     auth=service_auth()) as client:
            reader = BoundedHTTP(client, max_calls=15, max_bytes=262_144)
            return await self.search_with(reader, query, scope, task_ids)

    async def search_with(self, reader, query: str, scope: RecallScope,
                          task_ids: tuple[str, ...]) -> EvidenceBatch:
        mode = "explicit"
        refs = task_ids
        if not refs:
            # Existing semantic search only embeds the query and SELECTs rows;
            # it does not call the task-embedding reconciler or persist anything.
            payload = await reader.json("POST", f"{BASE_URL}{API_PREFIX}/tasks/search/semantic",
                                        json={"query": query, "limit": 3})
            if (not isinstance(payload, dict) or payload.get("mode") not in {"semantic", "keyword"}
                    or not isinstance(payload.get("results"), list) or len(payload["results"]) > 3):
                raise RecallError("source_unavailable")
            refs = tuple(row.get("id", "") for row in payload["results"] if isinstance(row, dict))
            if len(refs) != len(payload["results"]):
                raise RecallError("source_unavailable")
            mode = payload["mode"]
        if any(not isinstance(ref, str) or not TASK_REF.fullmatch(ref) for ref in refs):
            raise RecallError("invalid_request")
        result = EvidenceBatch(search_mode=mode)
        for ref in tuple(dict.fromkeys(refs))[:3]:
            try:
                # Semantic candidates intentionally lack assignment. Fetch it
                # before admitting any task content to the utility model.
                row = await reader.json("GET", f"{BASE_URL}{API_PREFIX}/tasks/{ref}")
                if not isinstance(row, dict) or not CUID.fullmatch(str(row.get("id", ""))):
                    raise RecallError("source_unavailable")
                if not await self._authorized(reader, row, scope):
                    result.withheld += 1
                    continue
                short_id = row.get("shortId")
                if not isinstance(short_id, str) or not TASK_REF.fullmatch(short_id):
                    raise RecallError("source_unavailable")
                title, status, description = (row.get(k) for k in ("title", "status", "description"))
                if not all(isinstance(v, str) for v in (title, status)):
                    raise RecallError("source_unavailable")
                canonical = f"{short_id}: {title[:300]}\nStatus: {status[:32]}\n{str(description or '')}"
                excerpt, start, truncated, window_kind = excerpt_window(canonical, query)
                result.append(Evidence("tasks", short_id, scope.bot_id, scope.user_id, excerpt,
                                       start, truncated, window_kind))
            except RecallError:
                # Busy/large tasks whose full provenance cannot be bounded are
                # withheld, never silently authorized. Preserve other sources;
                # coverage says incomplete, not 'no matching tasks exist'.
                result.withheld += 1
        return result

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

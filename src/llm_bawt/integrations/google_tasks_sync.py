"""Google Tasks import runs (TASK-905 slice B).

llm-bawt holds the Google grant and fetches pages; BawtHub decides what each
task becomes. A run takes BawtHub's per-source lease (``startRun``, which also
sweeps pending bot dispatches), lists the source's Google tasks including
completed, hidden and deleted ones since its watermark, and posts each page to
the ingest route with the run fence. Manual and scheduled sync share that
lease, so concurrent app workers cannot double-import.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable
from urllib.parse import quote

import httpx

from llm_bawt.integrations.connections import TASKS_LISTS_URL, IntegrationConnections, IntegrationError, access_token

logger = logging.getLogger(__name__)

INTEGRATION_ID = "google-tasks"
INTERVAL_SECONDS = 600
MAX_PAGES = 50
TASK_FIELDS = ("id", "title", "notes", "status", "updated", "due", "deleted", "hidden", "parent", "webViewLink", "selfLink")
SOURCES_PATH = "/internal/integrations/google-tasks/sources"
INGEST_PATH = "/internal/integrations/google-tasks/ingest"
ROUTE_PATH = "/internal/integrations/google-tasks/route"
MAX_LIST_PAGES = 10


def tasks_url(tasklist_id: str) -> str:
    return f"https://tasks.googleapis.com/tasks/v1/lists/{quote(tasklist_id, safe='')}/tasks"


def bawthub_client() -> httpx.Client:
    from llm_bawt.bawthub_auth import service_auth
    from llm_bawt.integrations.notify_bell import bell_base_url
    return httpx.Client(base_url=bell_base_url(), timeout=30, auth=service_auth())


def _item(raw: Any) -> dict | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
        return None
    return {key: raw[key] for key in TASK_FIELDS if key in raw}


class GoogleTasksSync:
    def __init__(self, manager: IntegrationConnections, *, google_factory: Callable[..., httpx.Client] = httpx.Client,
                 bawthub_factory: Callable[[], httpx.Client] = bawthub_client,
                 decide: Callable[..., dict] | None = None, roster: Callable[[], list[dict]] | None = None):
        from llm_bawt.integrations import google_tasks_route
        self.manager = manager
        self.decide = decide or google_tasks_route.decide
        self.roster = roster or google_tasks_route.roster
        self.google_factory = google_factory
        self.bawthub_factory = bawthub_factory

    def connected(self) -> bool:
        record = self.manager.store.load(INTEGRATION_ID)
        return bool(record and record.status == "connected")

    def sync_all(self) -> dict:
        """One pass over every enabled source. Bounded per-source errors, never Google text."""
        summary = {"sources": 0, "created": 0, "updated": 0, "skipped": 0, "errors": []}
        if not self.connected():
            raise IntegrationError("not_connected")
        with self.bawthub_factory() as hub, self.google_factory(timeout=20, follow_redirects=False) as google:
            response = hub.get(SOURCES_PATH)
            if not response.is_success:
                raise IntegrationError("bawthub_unavailable")
            sources = response.json().get("sources", [])
            for source in sources:
                summary["sources"] += 1
                try:
                    result = self.sync_source(hub, google, source)
                except IntegrationError as exc:
                    summary["errors"].append({"source": source.get("id"), "error": str(exc)})
                    continue
                except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError):
                    summary["errors"].append({"source": source.get("id"), "error": "sync_failed"})
                    continue
                if result is None:
                    summary["skipped"] += 1
                else:
                    summary["created"] += result["created"]
                    summary["updated"] += result["updated"]
        return summary

    def task_lists(self) -> list[dict]:
        """The connected account's task lists (id and title only)."""
        lists: list[dict] = []
        with self.google_factory(timeout=20, follow_redirects=False) as google:
            token, _subject = access_token(self.manager, INTEGRATION_ID, google)
            page_token = None
            for _ in range(MAX_LIST_PAGES):
                params = {"maxResults": 100, **({"pageToken": page_token} if page_token else {})}
                response = google.get(TASKS_LISTS_URL, params=params, headers={"Authorization": f"Bearer {token}"})
                if response.status_code != 200:
                    from llm_bawt.integrations.connections import keep_failure_code
                    raise IntegrationError(keep_failure_code(response, "tasks"))
                page = response.json()
                if not isinstance(page, dict) or not isinstance(page.get("items", []), list):
                    raise IntegrationError("tasks_invalid_response")
                lists += [{"id": item["id"], "title": str(item.get("title") or "")[:1024]}
                          for item in page.get("items", []) if isinstance(item, dict) and isinstance(item.get("id"), str)]
                page_token = page.get("nextPageToken")
                if not isinstance(page_token, str) or not page_token:
                    return lists
        raise IntegrationError("tasks_too_many_pages")

    def sync_source(self, hub: httpx.Client, google: httpx.Client, source: dict) -> dict | None:
        """Run one source; a run that fails after BawtHub's lease reports its bounded code."""
        run: dict = {}
        try:
            return self._sync_source(hub, google, source, run)
        except Exception as exc:
            if run.get("runId"):
                code = str(exc) if isinstance(exc, IntegrationError) else "sync_failed"
                try:
                    hub.post(f"{SOURCES_PATH}/{quote(source['id'], safe='')}/failure", json={"runId": run["runId"], "error": code})
                except httpx.HTTPError:
                    pass
            raise

    def _sync_source(self, hub: httpx.Client, google: httpx.Client, source: dict, current: dict) -> dict | None:
        """Run one source to its final page. None when BawtHub declined the run (busy, disabled)."""
        started = hub.post(f"{SOURCES_PATH}/{quote(source['id'], safe='')}/runs")
        if started.status_code in (404, 409):
            return None
        if not started.is_success:
            raise IntegrationError("bawthub_unavailable")
        run = started.json()
        fence = run["fence"]
        current["runId"] = fence["runId"]
        token, subject = access_token(self.manager, INTEGRATION_ID, google)
        if subject != fence["subject"]:
            return None  # account changed after BawtHub's lease; its generation fence rejects it anyway
        page_token, totals = run.get("pageToken"), {"created": 0, "updated": 0}
        for _ in range(MAX_PAGES):
            params = {"showCompleted": "true", "showHidden": "true", "showDeleted": "true", "maxResults": 100}
            if run.get("updatedMin"):
                params["updatedMin"] = run["updatedMin"]
            if page_token:
                params["pageToken"] = page_token
            response = google.get(tasks_url(source["tasklistId"]), params=params,
                                  headers={"Authorization": f"Bearer {token}"})
            if response.status_code != 200:
                from llm_bawt.integrations.connections import keep_failure_code
                raise IntegrationError(keep_failure_code(response, "tasks"))
            page = response.json()
            if not isinstance(page, dict) or not isinstance(page.get("items", []), list):
                raise IntegrationError("tasks_invalid_response")
            next_token = page.get("nextPageToken") or None
            if next_token is not None and (not isinstance(next_token, str) or next_token == page_token):
                raise IntegrationError("tasks_invalid_response")
            items = [item for item in map(_item, page.get("items", [])) if item]
            ingested = hub.post(INGEST_PATH, json={**fence, "tasks": items, "nextPageToken": next_token})
            if ingested.status_code in (404, 409):
                return None  # fenced: connection or source changed mid-run
            if not ingested.is_success:
                raise IntegrationError("bawthub_unavailable")
            result = ingested.json()
            totals["created"] += result.get("created", 0)
            totals["updated"] += result.get("updated", 0)
            self._route(hub, result)
            if result.get("final") or next_token is None:
                return totals
            page_token = next_token
        # The stored cursor lets the next run resume where this one stopped.
        raise IntegrationError("tasks_too_many_pages")


    def _route(self, hub: httpx.Client, result: dict) -> None:
        """Classify each new import once; BawtHub holds its delivery until answered or timed out."""
        pending = [i for i in result.get("toRoute") or [] if isinstance(i, dict) and isinstance(i.get("linkId"), str)]
        if not pending:
            return
        try:
            bots = self.roster()
        except Exception:  # noqa: BLE001 -- without a roster every item keeps its default
            logger.warning("google task routing: bot roster unavailable")
            bots = []
        projects = [p for p in result.get("projects") or [] if isinstance(p, str)]
        for item in pending:
            decision = {"linkId": item["linkId"], "botId": None, "project": None,
                        "reason": "No bots or projects to route to; used the list's default"}
            if bots or projects:
                try:
                    decision = self.decide(item, bots, projects)
                except Exception:  # noqa: BLE001 -- one bad item never stops the page or the run
                    decision = {**decision, "reason": "Routing unavailable; used the list's default"}
            try:
                hub.post(ROUTE_PATH, json=decision)
            except httpx.HTTPError:
                logger.warning("google task routing: BawtHub unavailable; the route will time out to the default")


async def run_once(service: Any) -> dict:
    from llm_bawt.service import state_events
    sync = GoogleTasksSync(IntegrationConnections(service.config))
    try:
        summary = await asyncio.to_thread(sync.sync_all)
    finally:
        # Every run commits status (baseline, last result, failure) even with no
        # imports, so the open panel always hears about it.
        await state_events.publish_state_changed(service, [state_events.GOOGLE_TASKS])
    return summary


async def run_scheduler(service: Any, interval: float = INTERVAL_SECONDS) -> None:
    """Every ``interval`` seconds while connected; BawtHub lists only enabled sources."""
    while True:
        await asyncio.sleep(interval)
        try:
            connected = await asyncio.to_thread(GoogleTasksSync(IntegrationConnections(service.config)).connected)
            if connected:
                summary = await run_once(service)
                if summary["errors"]:
                    logger.warning("Google Tasks sync errors: %s", summary["errors"])
        except IntegrationError as exc:
            logger.warning("Google Tasks sync skipped: %s", exc)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Google Tasks sync pass failed")

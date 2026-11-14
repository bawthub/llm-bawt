"""Weekly agent-harness version check (TASK-1975).

Compares the harness packages pinned in ``docker/agent-bridge-constraints.txt``
with their latest PyPI releases. When any pin is behind, it keeps exactly one
open BawtHub task describing the available upgrades: it updates the existing
open task (matched by title prefix) or creates one as ``QUEUED``. It never
upgrades or restarts anything itself.

For ``claude-agent-sdk`` the bundled Claude Code CLI version is read from the
SDK changelog ("Updated bundled Claude CLI to version X").

A network failure raises, so the job run is marked failed and no task is
written from partial data.

Result payload
--------------
``{"packages": [{"name", "pinned", "latest", ...}], "behind": [...],
   "task": {"action": "created"|"updated"|"none", "id"}}``
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_PACKAGES = (
    "claude-agent-sdk",
    "openai-codex",
    "openai-codex-cli-bin",
    "openai-codex-sdk",
)
TITLE_PREFIX = "Agent harness upgrades available"
SDK_CHANGELOG_URL = (
    "https://raw.githubusercontent.com/anthropics/claude-agent-sdk-python/main/CHANGELOG.md"
)
PYPI_URL = "https://pypi.org/pypi/{name}/json"
# BawtHub taskStatuses.ts: closed + hidden groups.
CLOSED_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "DELETED"}
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONSTRAINTS = _REPO_ROOT / "docker" / "agent-bridge-constraints.txt"


def read_pins(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    """Return ``{name: version}`` for ``name==version`` lines in *path*."""
    wanted = {n.lower().replace("_", "-") for n in names}
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" not in line:
            continue
        name, version = (part.strip() for part in line.split("==", 1))
        key = name.lower().replace("_", "-")
        if key in wanted:
            pins[key] = version
    return pins


def bundled_cli_versions(changelog: str) -> dict[str, str]:
    """Map SDK version -> bundled CLI version from the SDK changelog."""
    result: dict[str, str] = {}
    current: str | None = None
    for line in changelog.splitlines():
        heading = re.match(r"^##\s+v?(\d[\w.\-]*)\s*$", line)
        if heading:
            current = heading.group(1)
            continue
        cli = re.search(r"Updated bundled Claude CLI to version\s+v?(\d[\w.\-]*)", line)
        if current and cli and current not in result:
            result[current] = cli.group(1)
    return result


def _version_key(version: str) -> tuple:
    try:
        from packaging.version import Version

        return (0, Version(version))
    except Exception:
        return (1, version)


def collect(client: httpx.Client, pins: dict[str, str]) -> list[dict[str, Any]]:
    """Look up the latest release for every pinned package."""
    packages = []
    for name, pinned in sorted(pins.items()):
        resp = client.get(PYPI_URL.format(name=name))
        resp.raise_for_status()
        latest = str(resp.json()["info"]["version"])
        packages.append(
            {
                "name": name,
                "pinned": pinned,
                "latest": latest,
                "behind": _version_key(latest) > _version_key(pinned),
            }
        )
    sdk = next((p for p in packages if p["name"] == "claude-agent-sdk"), None)
    if sdk:
        resp = client.get(SDK_CHANGELOG_URL)
        resp.raise_for_status()
        cli = bundled_cli_versions(resp.text)
        sdk["pinned_cli"] = cli.get(sdk["pinned"])
        sdk["latest_cli"] = cli.get(sdk["latest"])
    return packages


def describe(behind: list[dict[str, Any]]) -> str:
    lines = [
        "The weekly harness version check found newer releases than the pins in "
        "`docker/agent-bridge-constraints.txt`:",
        "",
    ]
    for p in behind:
        line = f"- `{p['name']}`: pinned {p['pinned']}, latest {p['latest']}"
        if p.get("pinned_cli") or p.get("latest_cli"):
            line += (
                f" (bundled Claude CLI {p.get('pinned_cli') or '?'} -> "
                f"{p.get('latest_cli') or '?'})"
            )
        lines.append(line)
    lines += [
        "",
        "Review the changelogs, bump the pins and lock, rebuild the bridge image, "
        "run the SDK probe and full suites, then activate through an approved "
        "bridge restart. Nothing was upgraded automatically.",
    ]
    return "\n".join(lines)


def _task_list(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("tasks", "items", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def upsert_task(
    client: httpx.Client,
    behind: list[dict[str, Any]],
    *,
    agent_bot_id: str | None,
    priority: str,
) -> dict[str, Any]:
    """Update the one open harness-upgrade task, or create it as QUEUED."""
    title = f"{TITLE_PREFIX}: " + ", ".join(f"{p['name']} {p['latest']}" for p in behind)
    description = describe(behind)
    resp = client.get("/api/tasks/tasks", params={"q": TITLE_PREFIX, "archived": "false", "limit": "20"})
    resp.raise_for_status()
    open_tasks = [
        t
        for t in _task_list(resp.json())
        if str(t.get("title", "")).startswith(TITLE_PREFIX)
        and str(t.get("status", "")).upper() not in CLOSED_STATUSES
    ]
    if open_tasks:
        task = open_tasks[0]
        task_id = task.get("id") or task.get("shortId")
        if task.get("title") == title and task.get("description") == description:
            return {"action": "none", "id": task_id}
        client.patch(
            f"/api/tasks/tasks/{task_id}", json={"title": title, "description": description}
        ).raise_for_status()
        return {"action": "updated", "id": task_id}
    body: dict[str, Any] = {
        "title": title,
        "description": description,
        "status": "QUEUED",
        "priority": priority,
    }
    if agent_bot_id:
        body["agentBotId"] = agent_bot_id
    resp = client.post("/api/tasks/tasks", json=body)
    resp.raise_for_status()
    created = resp.json() if resp.content else {}
    return {"action": "created", "id": created.get("shortId") or created.get("id")}


def _task_client() -> httpx.Client:
    from llm_bawt.bawthub_auth import service_auth
    from llm_bawt.mcp_server.task_api import BASE_URL

    return httpx.Client(base_url=BASE_URL, timeout=30.0, auth=service_auth())


def run_harness_version_check(
    config: Any = None,
    *,
    constraints_path: str | None = None,
    packages: tuple[str, ...] | list[str] | None = None,
    agent_bot_id: str | None = None,
    priority: str = "MEDIUM",
    http_client: httpx.Client | None = None,
    task_client: httpx.Client | None = None,
) -> dict[str, Any]:
    path = Path(constraints_path) if constraints_path else DEFAULT_CONSTRAINTS
    names = tuple(packages or DEFAULT_PACKAGES)
    pins = read_pins(path, names)
    http = http_client or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        found = collect(http, pins)
    finally:
        if http_client is None:
            http.close()
    behind = [p for p in found if p["behind"]]
    task: dict[str, Any] = {"action": "none", "id": None}
    if behind:
        tasks = task_client or _task_client()
        try:
            task = upsert_task(tasks, behind, agent_bot_id=agent_bot_id, priority=priority)
        finally:
            if task_client is None:
                tasks.close()
    logger.info(
        "Harness version check: %d pinned, %d behind, task %s",
        len(found),
        len(behind),
        task["action"],
    )
    return {"packages": found, "behind": [p["name"] for p in behind], "task": task}

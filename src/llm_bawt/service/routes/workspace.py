"""App facade for the bridge-owned workspace; frontend mounts stay read-only."""

from __future__ import annotations

import json
import os
from urllib.parse import urlsplit

import httpx
from agent_bridge.workspace import CloneInput, WorkspaceError, clone_input
from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from .. import bridge_gate
from ..dependencies import get_runtime_settings_store, get_service

router = APIRouter()


def _bridge_url() -> str:
    config = get_service().config
    store = get_runtime_settings_store(config)
    settings = store.get_scope_settings("global", "*")
    url = settings.get("workspace.bridge_url", getattr(config, "WORKSPACE_BRIDGE_URL", ""))
    if not isinstance(url, str) or not url.strip():
        raise WorkspaceError(503, "Workspace bridge is not configured.", "bridge_unavailable")
    url = url.strip().rstrip("/")
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"} or not parsed.port and parsed.netloc.endswith(":")
        ):
            raise ValueError()
    except ValueError:
        raise WorkspaceError(503, "Workspace bridge configuration is invalid.", "bridge_unavailable") from None
    return url


def _error(status: int, detail: str, code: str) -> JSONResponse:
    return JSONResponse({"detail": detail, "code": code}, status_code=status, headers={"Cache-Control": "no-store"})


def _github_disconnected() -> bool:
    from ..providers.registry import get_adapter
    from ..providers.base import STATUS_CONNECTED
    adapter = get_adapter(get_service().config, "github")
    record = adapter.store.load("github") if adapter is not None else None
    return record is None or record.status != STATUS_CONNECTED


async def _proxy(method: str, item: CloneInput | None = None):
    try:
        url = await run_in_threadpool(_bridge_url)
        # Reuse the broker identity for app -> bridge on private networks.
        # Legacy tenants share the env secret; current bridges read its derived
        # replacement from the app-exported broker file.
        derived = bridge_gate.bridge_gates().derived_token(bridge_gate.BROKER)
        legacy = os.getenv("BRIDGE_CLAUDE_TOKEN_SECRET")
        token = (derived if os.getenv("BRIDGE_BROKER_TOKEN_EXPORT_FILE") else legacy) or derived or legacy
        async with httpx.AsyncClient(timeout=125.0, follow_redirects=False) as client:
            response = await client.request(
                method, url + ("/workspace/clone" if item else "/workspace"),
                headers={"X-Bridge-Token": token} if token else {},
                json={"url": item.url, "name": item.name, **({"branch": item.branch} if item.branch else {})} if item else None,
            )
        # Never pass through arbitrary bridge diagnostics (Git URLs may hold secrets).
        if response.status_code == 400:
            return _error(400, "Invalid repository URL, folder name or branch.", "invalid_clone")
        if response.status_code == 409:
            return _error(409, "That workspace folder already exists.", "folder_exists")
        if response.status_code == 504:
            return _error(504, "Clone timed out.", "clone_timeout")
        if response.status_code == 502:
            if item and item.github and await run_in_threadpool(_github_disconnected):
                return _error(503, "Connect GitHub in Providers before cloning private repositories.", "github_not_connected")
            return _error(502, "Git could not clone the repository. Check URL, branch and access.", "clone_failed")
        if response.status_code != 200:
            return _error(503, "Workspace bridge is unavailable.", "bridge_unavailable")
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError()
        if item:
            if result.get("path") != item.name or result.get("name") != item.name:
                raise ValueError()
            result = {"name": item.name, "path": item.name, **({"branch": item.branch} if item.branch else {})}
        elif not isinstance(result.get("entries"), list):
            raise ValueError()
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except WorkspaceError as exc:
        return _error(exc.status, exc.detail, exc.code)
    except httpx.TimeoutException:
        return _error(504, "Workspace request timed out. Check folders before retrying.", "clone_timeout")
    except Exception:
        return _error(503, "Workspace bridge is unavailable.", "bridge_unavailable")


@router.get("/v1/workspace")
async def workspace_list():
    return await _proxy("GET")


@router.post("/v1/workspace/clone")
async def workspace_clone(request: Request):
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 4096:
            return _error(400, "Invalid clone request.", "invalid_clone")
    try:
        item = clone_input(json.loads(raw))
    except WorkspaceError as exc:
        return _error(exc.status, exc.detail, exc.code)
    except (ValueError, UnicodeError):
        return _error(400, "Invalid clone request.", "invalid_clone")
    return await _proxy("POST", item)

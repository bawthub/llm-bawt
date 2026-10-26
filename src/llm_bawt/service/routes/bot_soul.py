"""OpenClaw SOUL.md sync routes (split out of ``routes/settings.py``, TASK-1521).

A bot on the OpenClaw backend keeps its system prompt in the gateway agent's
SOUL.md as well; these routes copy it either way over the bridge RPC, and
``push_soul_background`` is the best-effort push a profile save triggers.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException

from ..dependencies import get_bot_profile_store, get_service

logger = logging.getLogger(__name__)

router = APIRouter()

SOUL_FILE = "SOUL.md"


def openclaw_agent_id(profile) -> str:
    """Gateway agent id from the session key (``agent:<id>:<session>``), else ``main``."""
    session_key = (profile.agent_backend_config or {}).get("session_key", "")
    if session_key.startswith("agent:"):
        parts = session_key.split(":")
        if len(parts) >= 2:
            return parts[1]
    return "main"


async def push_soul_background(slug: str, agent_id: str, system_prompt: str) -> None:
    """Best-effort push of system_prompt to SOUL.md on the gateway."""
    try:
        from ...agent_backends.agent_bridge import get_agent_subscriber
        subscriber = get_agent_subscriber()
        if subscriber is None:
            logger.warning("Auto-push SOUL.md skipped for '%s': no subscriber", slug)
            return

        result = await subscriber.send_rpc(
            method="agents.files.set",
            params={"agentId": agent_id, "name": SOUL_FILE, "content": system_prompt},
            request_id=f"rpc_{uuid.uuid4().hex}",
            timeout_s=10,
            backend="openclaw",
        )
        if result.get("ok"):
            logger.info("Auto-pushed SOUL.md for bot '%s' (agent=%s)", slug, agent_id)
        else:
            logger.warning("Auto-push SOUL.md failed for '%s': %s", slug, result.get("error"))
    except Exception as e:
        logger.warning("Auto-push SOUL.md failed for '%s': %s", slug, e)


def _openclaw_profile(slug: str):
    service = get_service()
    store = get_bot_profile_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Bot profiles DB unavailable")
    profile = store.get(slug)
    if profile is None:
        raise HTTPException(status_code=404, detail="Bot profile not found")
    if profile.agent_backend != "openclaw":
        raise HTTPException(status_code=400, detail="Bot is not an OpenClaw agent backend")
    return service, store, profile


async def _soul_rpc(method: str, params: dict) -> dict:
    from ...agent_backends.agent_bridge import get_agent_subscriber
    subscriber = get_agent_subscriber()
    if subscriber is None:
        raise HTTPException(status_code=503, detail="OpenClaw bridge not connected")
    try:
        result = await subscriber.send_rpc(
            method=method,
            params=params,
            request_id=f"rpc_{uuid.uuid4().hex}",
            timeout_s=15,
            backend="openclaw",
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Bridge RPC timed out")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Bridge RPC failed: {e}")
    if not result.get("ok"):
        raise HTTPException(
            status_code=502,
            detail=f"Gateway error: {result.get('error', 'unknown')}",
        )
    return result


@router.post("/v1/bots/{slug}/sync-soul", tags=["System"])
async def sync_soul(slug: str):
    """Sync bot system_prompt from OpenClaw agent's SOUL.md via the bridge."""
    service, store, profile = _openclaw_profile(slug)
    agent_id = openclaw_agent_id(profile)
    result = await _soul_rpc("agents.files.get", {"agentId": agent_id, "name": SOUL_FILE})

    file_info = result.get("payload", {}).get("file", {})
    if file_info.get("missing"):
        raise HTTPException(status_code=404, detail=f"SOUL.md not found for agent '{agent_id}'")

    soul_content = file_info.get("content", "")
    if not soul_content.strip():
        raise HTTPException(status_code=404, detail=f"SOUL.md is empty for agent '{agent_id}'")

    # Imported here: settings imports this module for push_soul_background.
    from .settings import _invalidate_bot_instance_cache, _reload_bot_registry

    store.upsert({"slug": slug, "system_prompt": soul_content})
    _reload_bot_registry()
    _invalidate_bot_instance_cache(service, slug)

    return {
        "success": True,
        "slug": slug,
        "agent_id": agent_id,
        "system_prompt_length": len(soul_content),
    }


@router.post("/v1/bots/{slug}/push-soul", tags=["System"])
async def push_soul(slug: str):
    """Push bot system_prompt to the OpenClaw agent's SOUL.md via the bridge."""
    _service, _store, profile = _openclaw_profile(slug)
    content = profile.system_prompt or ""
    if not content.strip():
        raise HTTPException(status_code=400, detail="Bot system_prompt is empty — nothing to push")

    agent_id = openclaw_agent_id(profile)
    result = await _soul_rpc(
        "agents.files.set", {"agentId": agent_id, "name": SOUL_FILE, "content": content})

    file_info = result.get("payload", {}).get("file", {})
    return {
        "success": True,
        "slug": slug,
        "agent_id": agent_id,
        "file_size": file_info.get("size", len(content)),
    }

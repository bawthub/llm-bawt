"""Snapshot and fence owner-owned active threads for personal routed continuations.

A route never follows the active pointer later: both the durable thread and its
provider resume binding are fixed at admission, then checked before acceptance.
"""
from __future__ import annotations

import uuid

from sqlalchemy import text

from ..model_catalog import ModelEndpoint
from .routes.sessions import agent_key_name, resolve_agent_session_key


def active_session(conn, bot_id: str, owner: str, *, lock: bool = False):
    rows = conn.execute(text("""
        SELECT id, session_metadata FROM sessions
        WHERE bot_id=:bot AND user_id=:owner AND status='active' AND ended_at IS NULL
        ORDER BY started_at DESC LIMIT 2
    """ + (" FOR UPDATE" if lock else "")),
        {"bot": bot_id, "owner": owner}).mappings().all()
    if len(rows) > 1:
        raise ValueError("Multiple active owner threads; cannot safely continue")
    return rows[0] if rows else None


def stored_provider_model(meta: dict, backend: str) -> str:
    key = agent_key_name(backend)
    bound = str((meta.get("agent_session_key_models") or {}).get(key) or "").strip()
    if not bound and str(meta.get("provider") or "").strip() == backend:
        bound = str(meta.get("provider_session_model") or "").strip()
    return bound


def check_provider_binding(config, bot, endpoint: ModelEndpoint, metadata: dict | None) -> str | None:
    """Reject model/provider switches that the persisted SDK transcript cannot resume."""
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("Thread provider metadata is invalid; start a new session")
    meta = metadata or {}
    backend = bot.agent_backend
    key = agent_key_name(backend)
    keys = meta.get("agent_session_keys") or {}
    if not isinstance(keys, dict) or not isinstance(meta.get("agent_session_key_models") or {}, dict):
        raise ValueError("Thread provider metadata is invalid; start a new session")
    if any(value for name, value in keys.items() if name != key):
        raise ValueError("Thread belongs to another agent backend; start a new session")
    raw = str(keys.get(key) or "").strip()
    if raw and ":" in raw:
        raise ValueError("Thread contains an invalid provider resume key; start a new session")
    bound_model = stored_provider_model(meta, backend)
    if not raw:
        return None
    if not bound_model:
        raise ValueError("Thread provider model is unknown; start a new session")
    catalog = config.ensure_model_catalog()
    current_model = catalog.resolve(endpoint.ref, harness=bot.harness).get("model_id")
    if backend == "codex":
        if bound_model not in {endpoint.ref, current_model} or resolve_agent_session_key(meta, backend, bound_model) != raw:
            raise ValueError("Codex cannot resume this thread on a different model; start a new session")
    else:
        previous = [candidate for candidate in catalog.endpoints(harness=bot.harness)
                    if bound_model in {candidate.ref, catalog.resolve(candidate.ref, harness=bot.harness).get("model_id")}]
        if len({candidate.access_path.id for candidate in previous}) != 1:
            raise ValueError("Thread provider binding is unknown or ambiguous; start a new session")
        if previous[0].access_path.id != endpoint.access_path.id:
            raise ValueError("Continuing across model providers is unsupported; start a new session")
    return raw


def snapshot_session(conn, config, bot, owner: str, endpoint: ModelEndpoint) -> dict:
    """Capture an owned active thread inside the locked route admission."""
    session = active_session(conn, bot.slug, owner, lock=True)
    if session is None:
        # Match get_or_create_active_session's lost-race behavior without
        # leaving the caller's route receipt/delivery transaction. PostgreSQL's
        # one-active-(bot,user) index arbitrates direct-chat writers that do not
        # hold this route's target lock.
        sid = str(uuid.uuid4())
        conn.execute(text("""
            INSERT INTO sessions(id, bot_id, user_id, started_at, status)
            VALUES (:id, :bot, :owner, now(), 'active')
            ON CONFLICT (bot_id, user_id) WHERE status='active' DO NOTHING
        """), {"id": sid, "bot": bot.slug, "owner": owner})
        session = active_session(conn, bot.slug, owner, lock=True)
        if session is None:
            raise ValueError("Owner active thread could not be bound")
    resume = check_provider_binding(config, bot, endpoint, session["session_metadata"])
    return {"session_id": session["id"], "resume_id": resume,
            "provider_model": stored_provider_model(session["session_metadata"] or {}, bot.agent_backend)}


def check_session_fence(conn, config, bot, owner: str, model: str, binding: dict) -> None:
    """Fail closed if active identity, SDK key or provider model changed."""
    session = active_session(conn, bot.slug, owner)
    if not session or session["id"] != binding.get("session_id"):
        raise ValueError("Routed owner thread changed before acceptance")
    endpoint = config.ensure_model_catalog().resolve_endpoint(model, harness=bot.harness)
    resume = check_provider_binding(config, bot, endpoint, session["session_metadata"])
    if (resume != binding.get("resume_id") or
            stored_provider_model(session["session_metadata"] or {}, bot.agent_backend) != binding.get("provider_model")):
        raise ValueError("Routed provider session changed before acceptance")

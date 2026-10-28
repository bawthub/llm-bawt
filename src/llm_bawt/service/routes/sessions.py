"""Thread (session) management routes — TASK-284.

Exposes the durable `sessions` registry as a user-scoped thread API for a
threaded UI: list/read/create/activate over the one-active-per-(bot,user)
invariant enforced in the DB (idx_sessions_one_active_per_bot_user).

All routes resolve an effective `(bot_id, user_id)` and NEVER cross the user
boundary — a session that belongs to a different user is treated as absent
(404), so users sharing a bot cannot see, read, activate, or hydrate each
other's threads.

The lifecycle primitives (rotate/activate/get_or_create) live in
`PostgreSQLShortTermManager` and are surfaced here through the async
`MemoryStorage` wrappers; this module owns only HTTP shape + user scoping.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...mcp_server.storage import get_storage
from ...working_memory.session_state import mode_from_flag
from ..dependencies import get_effective_bot_id, get_service
from ..logging import get_service_logger
from ..schemas_history_memory import MessageAuthor, ScheduledMessageOrigin
from .history_scheduler import hydrate_scheduler_for_page
from .session_search import router as session_search_router

router = APIRouter()
# Static /search must precede /{session_id}.
router.include_router(session_search_router)
log = get_service_logger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SessionInfo(BaseModel):
    id: str
    bot_id: str
    user_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    archived_at: str | None = None
    last_activity_at: str | None = None
    first_message_at: str | None = None
    message_count: int = 0
    turn_count: int = 0
    first_user_message: str | None = None
    last_user_message: str | None = None
    last_assistant_message: str | None = None
    last_assistant_at: str | None = None
    summary: str | None = None
    status: str
    session_metadata: dict | None = None


class SessionListResponse(BaseModel):
    bot_id: str
    user_id: str
    sessions: list[SessionInfo]
    total_count: int


class SessionMessage(BaseModel):
    id: str | None = None
    role: str
    content: str
    timestamp: float | None = None
    session_id: str | None = None
    author: MessageAuthor
    scheduler: ScheduledMessageOrigin | None = None


class SessionTranscriptResponse(BaseModel):
    session_id: str
    bot_id: str
    messages: list[SessionMessage]
    total_count: int


class ForkFrom(BaseModel):
    """TASK-250: fork provenance — both fields required when forking."""

    session_id: str
    at_message_id: str


class CreateSessionRequest(BaseModel):
    """Optional JSON body for POST /v1/sessions (TASK-250).

    Back-compat: the route still accepts bare query-param usage (TASK-284
    callers send no body at all).
    """

    bot_id: str | None = None
    user: str | None = None
    title: str | None = None
    fork_from: ForkFrom | None = None
    activate: bool = Field(
        default=True,
        description=(
            "TASK-702: when true (default), opening a thread rotates "
            "(closes) the prior active thread for (bot,user) — the historic "
            "behavior. When false, mint a session that is BORN archived: "
            "no rotation, no activation, no impact on the one-active-per-"
            "(bot,user) invariant. Reachable via explicit session_id for "
            "background task threads that must not disturb the user's live "
            "conversation."
        ),
    )
    include_working_memory: bool | None = Field(
        default=None,
        description=(
            "With activate=true: false starts the new thread with working "
            "memory suppressed (no section, no change notice) for its "
            "lifetime; omitted or true pins it. A non-activating thread is "
            "not a rotation and always pins, so false with activate=false "
            "is refused (422)."
        ),
    )


class UpdateSessionRequest(BaseModel):
    """PATCH /v1/sessions/{id} body (TASK-250)."""

    title: str | None = None
    status: str | None = Field(
        default=None,
        description="'archived' or 'deleted'. Reactivation goes through "
        "POST /v1/sessions/{id}/activate, not PATCH.",
    )


class AgentSessionKeyRequest(BaseModel):
    """PUT /v1/sessions/{id}/agent-session-key body (TASK-252).

    The bridge write-back: persist the SDK session id a thread's turns run
    under so re-opening the thread later resumes the right transcript.
    """

    backend: str = Field(description="Agent backend, e.g. 'claude-code'")
    session_key: str = Field(description="Provider/SDK session id (no routing keys)")
    model: str | None = Field(
        default=None,
        description="Model the SDK session was created with (drives resume-vs-reset)",
    )


class CreateSessionResponse(BaseModel):
    session_id: str
    bot_id: str
    user_id: str
    status: str = "active"
    rotated: bool = Field(
        default=True,
        description=(
            "True when the default activating path ran (prior active thread "
            "closed, new one opened active). False for TASK-702 non-activating "
            "creation (activate=false): a thread was minted archived without "
            "touching the active-thread invariant."
        ),
    )


class ActivateSessionResponse(BaseModel):
    session_id: str
    bot_id: str
    user_id: str
    activated: bool
    provider_session: str | None = Field(
        default=None,
        description=(
            "Agent bots only (TASK-284 step 15): 'resumed' when the thread's "
            "stored provider session id was restored for the next turn, "
            "'reseed' when none was stored so the provider will cold-start "
            "and re-seed from this thread's context. None for chat bots."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_user(user: str | None) -> str:
    """Resolve the effective user id or 400 (mirrors routes/config.py)."""
    service = get_service()
    user_id = (user or getattr(service.config, "DEFAULT_USER", "") or "").strip()
    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="user is required (no DEFAULT_USER configured)",
        )
    return user_id


def agent_key_name(backend: str) -> str:
    """Canonical ``agent_session_keys`` key for a backend (TASK-252).

    Underscore style per the M1c spec shape: ``claude-code`` → ``claude_code``.
    """
    return (backend or "").strip().replace("-", "_")


def resolve_agent_session_key(
    meta: dict, backend: str, current_model: str | None = None
) -> str | None:
    """Resolve the SDK session id stored for a thread + backend.

    Reads from ``session_metadata.agent_session_keys[<backend>]``. Claude SDK
    transcripts are model-agnostic, so Claude model metadata is informational
    and never blocks resume. Other backends retain their existing model guard.

    A value containing ``:`` is a routing key (legacy bug), never an
    SDK session id.
    """
    meta = meta or {}
    key_name = agent_key_name(backend)
    keys = meta.get("agent_session_keys") or {}
    val = str(keys.get(key_name) or "").strip()
    if not val or ":" in val:
        return None
    if current_model and key_name != "claude_code":
        key_models = meta.get("agent_session_key_models") or {}
        stored_model = str(key_models.get(key_name) or "").strip()
        if not stored_model and str(meta.get("provider") or "").strip() == (
            backend or ""
        ).strip():
            stored_model = str(meta.get("provider_session_model") or "").strip()
        if stored_model and stored_model != current_model:
            return None
    return val


def _normalize_metadata(row: dict) -> dict:
    """Backfill-on-read: coerce ``session_metadata`` to the standard shape
    without force-writing anything back.

    Target shape::

        {title, title_source: user|auto|default,
         agent_session_keys: {<backend>: sdk-id},
         forked_from: {session_id, at_message_id} | None,
         archived_at}
    """
    meta = dict(row.get("session_metadata") or {})
    meta.setdefault("title", None)
    meta.setdefault("title_source", "user" if meta.get("title") else "default")
    meta.setdefault("agent_session_keys", {})
    meta.setdefault("forked_from", None)
    meta.setdefault("archived_at", row.get("archived_at"))
    return meta


def _to_info(row: dict) -> SessionInfo:
    return SessionInfo(
        id=row["id"],
        bot_id=row.get("bot_id", ""),
        user_id=row.get("user_id"),
        started_at=str(row["started_at"]) if row.get("started_at") is not None else None,
        ended_at=str(row["ended_at"]) if row.get("ended_at") is not None else None,
        archived_at=str(row["archived_at"]) if row.get("archived_at") is not None else None,
        last_activity_at=row.get("last_activity_at"),
        first_message_at=row.get("first_message_at"),
        message_count=int(row.get("message_count") or 0),
        turn_count=int(row.get("turn_count") or 0),
        first_user_message=row.get("first_user_message"),
        last_user_message=row.get("last_user_message"),
        last_assistant_message=row.get("last_assistant_message"),
        last_assistant_at=row.get("last_assistant_at"),
        summary=row.get("summary"),
        status=row.get("status", ""),
        session_metadata=_normalize_metadata(row),
    )


async def _owned_session_or_404(
    session_id: str,
    bot_id: str,
    user_id: str,
    allow_deleted: bool = False,
) -> dict:
    """Fetch a session and enforce (bot, user) ownership; 404 otherwise.

    TASK-250: a soft-deleted session's deep-link answers 410 Gone (the row
    exists, the resource is intentionally unavailable) unless the caller asks
    for it explicitly via ``allow_deleted``.
    """
    storage = get_storage()
    row = await storage.get_session(session_id, bot_id=bot_id)
    if not row or row.get("bot_id") != bot_id or row.get("user_id") != user_id:
        # Cross-user / cross-bot / missing all look identical to the caller.
        raise HTTPException(status_code=404, detail="Session not found")
    if row.get("status") == "deleted" and not allow_deleted:
        raise HTTPException(status_code=410, detail="Session deleted")
    return row


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("/v1/sessions", response_model=SessionListResponse, tags=["Sessions"])
async def list_sessions(
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
    status: str | None = Query(
        None, description="Filter by status: active | archived | deleted"
    ),
    include_deleted: bool = Query(
        False,
        description="TASK-250: soft-deleted threads are excluded by default; "
        "pass true (admin) to include them.",
    ),
    limit: int = Query(50, ge=1, le=200),
):
    """List a user's threads for a bot, newest first (excludes deleted)."""
    effective_bot = get_effective_bot_id(bot_id)
    user_id = _resolve_user(user)
    rows = await get_storage().list_sessions(
        bot_id=effective_bot,
        user_id=user_id,
        status=status,
        limit=limit,
        include_deleted=include_deleted,
    )
    return SessionListResponse(
        bot_id=effective_bot,
        user_id=user_id,
        sessions=[_to_info(r) for r in rows],
        total_count=len(rows),
    )


@router.get("/v1/sessions/active", response_model=SessionInfo, tags=["Sessions"])
async def get_active_session(
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
):
    """Return the current active thread for (bot, user)."""
    effective_bot = get_effective_bot_id(bot_id)
    user_id = _resolve_user(user)
    row = await get_storage().get_active_session(bot_id=effective_bot, user_id=user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No active session")
    return _to_info(row)


@router.get("/v1/sessions/{session_id}", response_model=SessionInfo, tags=["Sessions"])
async def get_session(
    session_id: str,
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
):
    """Return one thread's metadata (user-scoped)."""
    effective_bot = get_effective_bot_id(bot_id)
    user_id = _resolve_user(user)
    row = await _owned_session_or_404(session_id, effective_bot, user_id)
    return _to_info(row)


@router.get(
    "/v1/sessions/{session_id}/messages",
    response_model=SessionTranscriptResponse,
    tags=["Sessions"],
)
async def get_session_messages(
    session_id: str,
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
    limit: int = Query(500, ge=1, le=2000),
):
    """Return a thread's raw transcript (user-scoped)."""
    effective_bot = get_effective_bot_id(bot_id)
    user_id = _resolve_user(user)
    await _owned_session_or_404(session_id, effective_bot, user_id)
    rows = await get_storage().get_messages(
        bot_id=effective_bot, session_id=session_id, limit=limit
    )
    scheduler_by_id = hydrate_scheduler_for_page(get_service(), effective_bot, rows, user_id)
    return SessionTranscriptResponse(
        session_id=session_id,
        bot_id=effective_bot,
        messages=[
            SessionMessage(
                id=str(r["id"]) if r.get("id") is not None else None,
                role=r.get("role", "?"),
                content=r.get("content", ""),
                timestamp=r.get("timestamp"),
                session_id=r.get("session_id"),
                scheduler=scheduler_by_id.get(str(r.get("id"))) if r.get("role") == "user" else None,
                author=(
                    r.get("author")
                    or {"entity_type": None, "entity_id": None, "status": "unknown"}
                ),
            )
            for r in rows
        ],
        total_count=len(rows),
    )


@router.post("/v1/sessions", response_model=CreateSessionResponse, tags=["Sessions"])
async def create_session(
    body: CreateSessionRequest | None = None,
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
):
    """Open a fresh thread for (bot, user), rotating (closing) the prior active
    one. Non-destructive: old rows are preserved under the closed session.

    TASK-250: optional JSON body ``{bot_id?, title?, fork_from?}``. ``title``
    is stored as a user-sourced title; ``fork_from`` records fork provenance
    (``at_message_id`` required — enforced by the schema) after verifying the
    source thread is owned by the same (bot, user).

    TASK-702: ``activate=false`` mints a session that is BORN archived — it
    does not rotate the active thread and does not become active. Background
    task dispatch uses this to open a dedicated task thread without disturbing
    the user's currently active conversation.
    """
    body = body or CreateSessionRequest()
    effective_bot = get_effective_bot_id(body.bot_id or bot_id)
    user_id = _resolve_user(body.user or user)
    storage = get_storage()
    if body.include_working_memory is False and not body.activate:
        raise HTTPException(
            status_code=422,
            detail="include_working_memory=false applies only to an activating session",
        )

    # Validate fork provenance BEFORE rotating anything.
    if body.fork_from is not None:
        await _owned_session_or_404(
            body.fork_from.session_id, effective_bot, user_id
        )

    meta: dict = {}
    if body.title and body.title.strip():
        meta["title"] = body.title.strip()
        meta["title_source"] = "user"
    if body.fork_from is not None:
        meta["forked_from"] = {
            "session_id": body.fork_from.session_id,
            "at_message_id": body.fork_from.at_message_id,
        }

    if body.activate:
        new_id = await storage.rotate_session(
            bot_id=effective_bot, user_id=user_id,
            working_memory_mode=mode_from_flag(body.include_working_memory),
        )
        if meta:
            await storage.update_session_metadata(
                new_id, meta, bot_id=effective_bot,
            )
        return CreateSessionResponse(
            session_id=new_id,
            bot_id=effective_bot,
            user_id=user_id,
            status="active",
            rotated=True,
        )

    # Non-activating: seed metadata at creation time so a follow-up write
    # is not required, and mark the row so operators can distinguish
    # background/task threads from a UI-archived one on read.
    seed_meta = dict(meta)
    seed_meta.setdefault("created_inactive", True)
    new_id = await storage.create_inactive_session(
        bot_id=effective_bot,
        user_id=user_id,
        session_metadata=seed_meta,
    )
    return CreateSessionResponse(
        session_id=new_id,
        bot_id=effective_bot,
        user_id=user_id,
        status="archived",
        rotated=False,
    )


@router.patch("/v1/sessions/{session_id}", response_model=SessionInfo, tags=["Sessions"])
async def update_session(
    session_id: str,
    body: UpdateSessionRequest,
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
):
    """Rename and/or archive/soft-delete a thread (TASK-250, user-scoped).

    - ``title``: stored with ``title_source='user'`` (a user rename locks out
      future auto-titling).
    - ``status``: ``archived`` or ``deleted`` only. Reactivation goes through
      ``POST /v1/sessions/{id}/activate`` (it owns the one-active invariant).
      ``archived`` on a deleted thread restores it (undelete).
    """
    effective_bot = get_effective_bot_id(bot_id)
    user_id = _resolve_user(user)
    # allow_deleted so a soft-deleted thread can be restored via PATCH.
    await _owned_session_or_404(
        session_id, effective_bot, user_id, allow_deleted=True
    )
    storage = get_storage()

    if body.status is not None:
        if body.status not in ("archived", "deleted"):
            raise HTTPException(
                status_code=422,
                detail="status must be 'archived' or 'deleted' "
                "(use POST /v1/sessions/{id}/activate to reactivate)",
            )
        ok = await storage.set_session_status(
            session_id, body.status, bot_id=effective_bot
        )
        if not ok:
            raise HTTPException(status_code=500, detail="Status update failed")

    if body.title is not None:
        title = body.title.strip()
        if not title:
            raise HTTPException(status_code=422, detail="title must be non-empty")
        await storage.update_session_metadata(
            session_id,
            {"title": title, "title_source": "user"},
            bot_id=effective_bot,
        )

    row = await storage.get_session(session_id, bot_id=effective_bot)
    return _to_info(row)


@router.delete("/v1/sessions/{session_id}", tags=["Sessions"])
async def delete_session(
    session_id: str,
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
):
    """Soft-delete a thread (TASK-250, user-scoped).

    Sets ``status='deleted'``; messages are retained. The thread leaves
    default listings and its deep-links answer 410 Gone. Restore with
    ``PATCH {status: 'archived'}``. Idempotent: deleting an already-deleted
    thread is a no-op success.
    """
    effective_bot = get_effective_bot_id(bot_id)
    user_id = _resolve_user(user)
    row = await _owned_session_or_404(
        session_id, effective_bot, user_id, allow_deleted=True
    )
    if row.get("status") != "deleted":
        ok = await get_storage().set_session_status(
            session_id, "deleted", bot_id=effective_bot
        )
        if not ok:
            raise HTTPException(status_code=500, detail="Delete failed")
    return {"session_id": session_id, "status": "deleted", "deleted": True}


@router.put(
    "/v1/sessions/{session_id}/agent-session-key",
    tags=["Sessions"],
)
async def put_agent_session_key(
    session_id: str,
    body: AgentSessionKeyRequest,
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
):
    """Persist an SDK session id onto its thread (TASK-252, bridge write-back).

    Writes the canonical per-backend entries in
    ``session_metadata.agent_session_keys`` and
    ``session_metadata.agent_session_key_models``. Both maps merge so one
    backend can never overwrite another backend's resume metadata.

    Bot-scoped, not user-scoped: the caller is a trusted bridge on the LAN
    that knows the thread id from the dispatch command; the originating turn
    already passed user-ownership validation at /v1/chat/completions.
    """
    effective_bot = get_effective_bot_id(bot_id)
    backend = (body.backend or "").strip()
    session_key = (body.session_key or "").strip()
    if not backend or not session_key:
        raise HTTPException(status_code=422, detail="backend and session_key required")
    if ":" in session_key:
        # Routing keys ("agent:main:main", legacy "bot:user") are
        # never SDK session ids — refuse rather than poison the thread.
        raise HTTPException(status_code=422, detail="session_key looks like a routing key")

    storage = get_storage()
    row = await storage.get_session(session_id, bot_id=effective_bot)
    if not row or row.get("bot_id") != effective_bot:
        raise HTTPException(status_code=404, detail="Session not found")

    meta = row.get("session_metadata") or {}
    key_name = agent_key_name(backend)
    keys = dict(meta.get("agent_session_keys") or {})
    keys[key_name] = session_key
    models = dict(meta.get("agent_session_key_models") or {})
    if body.model:
        models[key_name] = body.model
    else:
        # A replaced key must not inherit stale model metadata from its
        # predecessor. No model means resume without a model gate.
        models.pop(key_name, None)
    patch: dict = {
        "agent_session_keys": keys,
        "agent_session_key_models": models,
    }
    ok = await storage.update_session_metadata(session_id, patch, bot_id=effective_bot)
    if not ok:
        raise HTTPException(status_code=500, detail="Metadata update failed")
    log.info(
        "Agent session key stored: bot=%s thread=%s backend=%s sid=%s",
        effective_bot, session_id, backend, session_key,
    )
    return {
        "session_id": session_id,
        "backend": backend,
        "agent_session_keys": keys,
        "stored": True,
    }


@router.post(
    "/v1/sessions/{session_id}/activate",
    response_model=ActivateSessionResponse,
    tags=["Sessions"],
)
async def activate_session(
    session_id: str,
    bot_id: str = Query(None, description="Bot slug; defaults to service default bot"),
    user: str | None = Query(None, description="User id; defaults to DEFAULT_USER"),
):
    """Switch the active thread to an existing one (user-scoped)."""
    effective_bot = get_effective_bot_id(bot_id)
    user_id = _resolve_user(user)
    # Ownership check first so a cross-user id can't be activated (404, not 403,
    # to avoid confirming the session exists for another user).
    row = await _owned_session_or_404(session_id, effective_bot, user_id)
    ok = await get_storage().activate_session(
        session_id, bot_id=effective_bot, user_id=user_id
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    # TASK-638: the scalar agent_backend_config.session_key is retired.
    # The bridge now resolves per-thread SDK keys directly from the thread's
    # agent_session_keys. Thread activation just switches which thread is
    # active; the next turn's _resolve_active_thread_binding picks it up.
    return ActivateSessionResponse(
        session_id=session_id,
        bot_id=effective_bot,
        user_id=user_id,
        activated=True,
        provider_session=None,
    )

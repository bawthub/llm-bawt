"""Agent context-health and safe session-reset policy.

The app owns policy; bridges only execute the explicit fresh-thread binding and
optional seed they receive. Resident context is derived from the latest
per-iteration usage snapshot, never cumulative billing totals.
"""

from __future__ import annotations

import json
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import text as sa_text

from .utils.config import Config
from .utils.db import get_shared_engine


class SessionPolicy(StrEnum):
    CONTINUE = "continue"
    RESET_RETAIN_HISTORY = "reset_retain_history"
    RESET_WITHOUT_HISTORY = "reset_without_history"


@dataclass(frozen=True)
class AgentContextCapabilities:
    backend: str
    inspect: bool
    compact: bool
    reset_retain_history: bool
    reset_without_history: bool

    def to_api(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "inspect": self.inspect,
            "compact": self.compact,
            "reset_retain_history": self.reset_retain_history,
            "reset_without_history": self.reset_without_history,
        }


def capabilities_for_backend(backend: str | None) -> AgentContextCapabilities:
    name = (backend or "").strip().lower()
    if name == "claude-code":
        return AgentContextCapabilities(name, True, True, True, True)
    if name == "codex":
        # No context-health or compact API. Retained history is seeded into a
        # fresh thread through app-server ``thread/inject_items`` (TASK-1579),
        # whichever transport then runs the turn.
        return AgentContextCapabilities(name, False, False, True, True)
    if name == "openclaw":
        return AgentContextCapabilities(name, False, False, False, False)
    return AgentContextCapabilities(name or "unknown", False, False, False, False)


def normalize_session_policy(
    session_policy: str | SessionPolicy | None = None,
    *,
    reset_session_before_delivery: bool | None = None,
    retain_history: bool | None = None,
) -> SessionPolicy:
    """Normalize the enum and compatibility booleans, rejecting contradictions."""
    raw = str(session_policy or SessionPolicy.CONTINUE).strip().lower().replace("-", "_")
    try:
        selected = SessionPolicy(raw)
    except ValueError as exc:
        allowed = ", ".join(p.value for p in SessionPolicy)
        raise ValueError(f"Unknown session_policy {session_policy!r}; expected one of: {allowed}") from exc

    compat: SessionPolicy | None = None
    if reset_session_before_delivery is False:
        if retain_history is not None:
            raise ValueError("retain_history requires reset_session_before_delivery=true")
        compat = SessionPolicy.CONTINUE
    elif reset_session_before_delivery is True:
        compat = (
            SessionPolicy.RESET_RETAIN_HISTORY
            if retain_history is not False
            else SessionPolicy.RESET_WITHOUT_HISTORY
        )
    elif retain_history is not None:
        raise ValueError("retain_history requires reset_session_before_delivery")

    if compat is not None and session_policy is not None and compat != selected:
        raise ValueError(
            f"session_policy={selected.value!r} contradicts compatibility reset fields"
        )
    return compat or selected


def validate_session_policy(backend: str | None, policy: SessionPolicy) -> None:
    if policy == SessionPolicy.CONTINUE:
        return
    caps = capabilities_for_backend(backend)
    allowed = (
        caps.reset_retain_history
        if policy == SessionPolicy.RESET_RETAIN_HISTORY
        else caps.reset_without_history
    )
    if not allowed:
        raise ValueError(
            f"Backend {caps.backend!r} does not support session_policy={policy.value!r}"
        )


def preventive_session_policy(
    *,
    requested: SessionPolicy,
    health_state: str,
    configured_critical_policy: str | None,
    backend: str | None,
) -> tuple[SessionPolicy, str | None]:
    """Select one persisted pre-delivery action from health + capabilities."""
    if requested != SessionPolicy.CONTINUE or health_state != "critical":
        return requested, None
    try:
        candidate = normalize_session_policy(configured_critical_policy)
    except ValueError:
        candidate = SessionPolicy.RESET_RETAIN_HISTORY
    if candidate == SessionPolicy.CONTINUE:
        return candidate, "critical context policy explicitly continues"
    caps = capabilities_for_backend(backend)
    if candidate == SessionPolicy.RESET_RETAIN_HISTORY and caps.reset_retain_history:
        return candidate, "critical context threshold reached"
    if caps.reset_without_history:
        return SessionPolicy.RESET_WITHOUT_HISTORY, (
            "critical context threshold reached; backend cannot retain seed history"
        )
    return SessionPolicy.CONTINUE, (
        "critical context threshold reached; backend has no safe reset capability"
    )


def rotate_delivery_session(
    conn,
    *,
    head: Any,
    payload: dict[str, Any],
    target_bot_id: str,
    user_id: str,
) -> tuple[dict[str, Any], str | None, str | None]:
    """Compatibility shim: the shared lifecycle owns policy rotation."""
    from .service.session_lifecycle.lifecycle import NewSessionLifecycle

    return NewSessionLifecycle.rotate_delivery_claim(
        conn, head=head, payload=payload,
        target_bot_id=target_bot_id, user_id=user_id,
    )


def recover_delivery_after_overflow(
    conn,
    *,
    head: Any,
    backend: str | None,
    user_id: str,
    error: str,
) -> dict[str, Any] | None:
    """Rotate and requeue one logical delivery after one confirmed overflow."""
    if int(head.get("overflow_recovery_count") or 0) >= 1:
        return None
    caps = capabilities_for_backend(backend)
    if caps.reset_retain_history:
        policy = SessionPolicy.RESET_RETAIN_HISTORY
    elif caps.reset_without_history:
        policy = SessionPolicy.RESET_WITHOUT_HISTORY
    else:
        return None
    target = head["target_bot_id"]
    conn.execute(
        sa_text("SELECT pg_advisory_xact_lock(hashtext(:target))"),
        {"target": target},
    )
    other_open = conn.execute(sa_text("""
        SELECT id FROM turn_logs
        WHERE bot_id=:target AND ended_at IS NULL AND id != :turn_id
        ORDER BY created_at DESC LIMIT 1
    """), {"target": target, "turn_id": head["turn_id"]}).scalar_one_or_none()
    if other_open:
        return None
    side_effects = conn.execute(sa_text("""
        SELECT
            EXISTS(SELECT 1 FROM tool_call_records WHERE turn_id=:turn_id),
            EXISTS(
                SELECT 1 FROM turn_logs
                WHERE id=:turn_id AND length(COALESCE(response_text, '')) > 0
            )
    """), {"turn_id": head["turn_id"]}).one()
    if bool(side_effects[0]) or bool(side_effects[1]):
        # A later tool-loop iteration can overflow after earlier tools already
        # executed. Replaying the whole logical turn would duplicate side effects.
        return None
    old_session = conn.execute(sa_text("""
        SELECT id FROM sessions
        WHERE bot_id=:bot AND user_id=:user
          AND status='active' AND ended_at IS NULL
        ORDER BY started_at DESC LIMIT 1 FOR UPDATE
    """), {"bot": target, "user": user_id}).scalar_one_or_none()
    new_session = str(uuid.uuid5(
        uuid.NAMESPACE_URL, f"inter-bot-overflow:{head['id']}:1"
    ))
    from .memory.session_reset_lineage import SessionResetLineage
    from .working_memory.session_state import PINNED

    # Keep the existing overflow namespace; a persisted delivery must not
    # acquire a different successor when recovered after a process restart.
    # Overflow recovery is not an explicit suppression request, so it pins (N1).
    SessionResetLineage.rotate(
        None, connection=conn, bot_id=target, user_id=user_id,
        new_id=new_session, idempotency_key=f"{head['id']}:overflow:1",
        accept_pointer=True, working_memory_mode=PINNED,
        context_lifecycle={
            "action": policy.value,
            "reason": "confirmed context overflow",
            "delivery_id": head["id"],
            "old_session_id": old_session,
        },
    )
    from .memory.postgresql import _sanitize_table_name

    conn.execute(sa_text("""
        UPDATE messages SET session_id=:new_session
        WHERE bot_id=:storage_bot AND id=:message_id
    """), {
        "new_session": new_session,
        "storage_bot": _sanitize_table_name(target),
        "message_id": head["user_message_id"],
    })
    payload = dict(head.get("payload_json") or {})
    payload["session_id"] = new_session
    payload["inter_bot_session_policy"] = policy.value
    payload["inter_bot_seed_session_id"] = old_session
    payload["inter_bot_bridge_request_id"] = (
        f"req_delivery_{head['id'].removeprefix('delivery-')}_recovery_1"
    )
    conn.execute(sa_text("DELETE FROM turn_logs WHERE id=:id"), {"id": head["turn_id"]})
    row = conn.execute(sa_text("""
        UPDATE inter_bot_deliveries SET
            status='QUEUED', available_at=now(), next_retry_at=now(),
            claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL,
            transport_accepted_at=NULL, target_turn_id=NULL, delivery_mode=NULL,
            session_policy=:policy, reset_status='ROTATED',
            reset_reason='confirmed context overflow', old_session_id=:old_session,
            new_session_id=:new_session, reset_at=now(),
            overflow_recovery_count=1, payload_json=CAST(:payload AS jsonb),
            last_error=:error, updated_at=now()
        WHERE id=:id AND status='DISPATCHING'
        RETURNING *
    """), {
        "id": head["id"],
        "policy": policy.value,
        "old_session": old_session,
        "new_session": new_session,
        "payload": json.dumps(payload, ensure_ascii=False, default=str),
        "error": error[:4000],
    }).mappings().first()
    return dict(row) if row else None


class AgentContextStore:
    """Context-health projection and idle-only non-destructive reset primitive."""

    def __init__(self, config: Config):
        self.config = config
        self.engine = get_shared_engine(config)

    @staticmethod
    def _usage(value: str | None) -> dict[str, Any]:
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def health(
        self,
        *,
        bot_id: str,
        user_id: str,
        backend: str | None,
        configured_ceiling: int | None,
        warning_ratio: float = 0.75,
        critical_ratio: float = 0.90,
        connection=None,
    ) -> dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("Context-health database unavailable")
        with (nullcontext(connection) if connection is not None else self.engine.connect()) as conn:
            session = conn.execute(sa_text("""
                SELECT id, started_at, session_metadata
                FROM sessions
                WHERE bot_id=:bot AND user_id=:user
                  AND status='active' AND ended_at IS NULL
                ORDER BY started_at DESC LIMIT 1
            """), {"bot": bot_id, "user": user_id}).mappings().first()
            turn = None
            snapshot = (session.get("session_metadata") or {}).get("context_stats") if session else None
            if (isinstance(snapshot, dict) and snapshot.get("turn_id")
                    and snapshot.get("created_at") and "ended_at" in snapshot):
                # Terminal writer stamps the owning bot/user session atomically
                # with the turn log. Older sessions without a snapshot use the
                # historical session-bound lookup until their next completion.
                turn = {
                    "id": snapshot["turn_id"],
                    "created_at": datetime.fromisoformat(snapshot["created_at"]),
                    "ended_at": (datetime.fromisoformat(snapshot["ended_at"])
                                 if snapshot["ended_at"] else None),
                    "token_usage_json": json.dumps(snapshot.get("usage")),
                }
            elif session:
                # A turn belongs to a durable thread through its trigger user
                # message. Never carry a previous thread's critical gauge across
                # a reset merely because its turn log is newest. Query the parent
                # table with its partition key instead of interpolating a bot table.
                from .memory.postgresql import _sanitize_table_name

                turn = conn.execute(sa_text("""
                    SELECT t.id, t.created_at, t.ended_at, t.token_usage_json, t.error_text
                    FROM turn_logs t
                    JOIN messages m
                      ON m.bot_id=:storage_bot AND m.id=t.trigger_message_id
                    WHERE t.bot_id=:bot AND t.user_id=:user
                      AND m.session_id=:session_id
                    ORDER BY t.created_at DESC, t.id DESC LIMIT 1
                """), {
                    "bot": bot_id,
                    "storage_bot": _sanitize_table_name(bot_id),
                    "user": user_id,
                    "session_id": session["id"],
                }).mappings().first()

        # A later turn without reliable terminal usage invalidates an earlier
        # gauge. Creation time is not an inactivity timestamp.
        usage = self._usage(turn.get("token_usage_json") if turn and turn.get("ended_at") else None)
        observed_ceiling = int(usage.get("context_window") or 0) or None
        ceiling = observed_ceiling or configured_ceiling
        ceiling_source = "provider" if observed_ceiling else ("catalog" if configured_ceiling else "unknown")
        resident = None
        resident_source = None
        if usage:
            native_resident = int(usage.get("resident_tokens") or 0)
            if native_resident:
                resident = native_resident
                resident_source = str(
                    usage.get("resident_source") or "provider_context"
                )
            else:
                # Provider usage schemas disagree about whether input_tokens is
                # uncached-only or total resident input. It is still the only
                # defensible fallback exposed by every backend. Cache read/write
                # fields are accounting/cache telemetry and may be cumulative;
                # summing them produced impossible multi-million-token gauges.
                final_input = int(usage.get("input_tokens") or 0)
                # Some backends expose only a turn-wide aggregate across tool
                # iterations.  A fallback larger than the model window cannot
                # be resident context; treating it as such manufactures a
                # critical gauge from billing telemetry.  Native resident
                # counters above remain authoritative and are not clamped.
                if final_input and (ceiling is None or final_input <= ceiling):
                    resident = final_input
                    resident_source = "final_iteration_input"
        ratio = (resident / ceiling) if resident is not None and ceiling else None
        state = (
            "unknown"
            if ratio is None
            else "critical"
            if ratio >= critical_ratio
            else "warning"
            if ratio >= warning_ratio
            else "healthy"
        )
        metadata = dict(session.get("session_metadata") or {}) if session else {}
        lifecycle = metadata.get("context_lifecycle")
        latest_action = lifecycle if isinstance(lifecycle, dict) else None
        if usage.get("context_action"):
            latest_action = {
                "action": usage.get("context_action"),
                "outcome": usage.get("context_action_outcome"),
                "error": usage.get("context_action_error"),
                "pre_tokens": usage.get("context_action_pre_tokens"),
                "post_tokens": usage.get("context_action_post_tokens"),
                "turn_id": turn.get("id") if turn else None,
                "at": turn.get("created_at").isoformat() if turn else None,
            }
        disagreement = bool(
            observed_ceiling
            and configured_ceiling
            and observed_ceiling != configured_ceiling
        )
        return {
            "bot_id": bot_id,
            "user_id": user_id,
            "session_id": session.get("id") if session else None,
            "backend": (backend or "unknown"),
            "capabilities": capabilities_for_backend(backend).to_api(),
            "state": state,
            "resident_prompt_tokens": resident,
            "resident_source": resident_source,
            "effective_ceiling_tokens": ceiling,
            "ceiling_source": ceiling_source,
            "configured_ceiling_tokens": configured_ceiling,
            "provider_ceiling_tokens": observed_ceiling,
            "ceiling_disagreement": disagreement,
            "remaining_tokens": max(0, ceiling - resident) if ceiling and resident is not None else None,
            "usage_ratio": ratio,
            "warning_ratio": warning_ratio,
            "critical_ratio": critical_ratio,
            "usage_turn_id": turn.get("id") if turn else None,
            "usage_observed_at": turn.get("ended_at").isoformat() if turn and turn.get("ended_at") else None,
            "last_lifecycle_action": latest_action,
        }

    def reset_idle_session(
        self,
        *,
        bot_id: str,
        user_id: str,
        backend: str | None,
        policy: SessionPolicy,
        reason: str,
        include_working_memory: bool | None = None,
    ) -> dict[str, Any]:
        """Rotate one idle durable thread; prior messages remain untouched.

        Only an explicit ``include_working_memory=False`` suppresses working
        memory for the successor; omitted pins it (docs/WORKING_MEMORY.md 10.1).
        """
        from .working_memory.session_state import mode_from_flag

        working_memory_mode = mode_from_flag(include_working_memory)
        validate_session_policy(backend, policy)
        if policy == SessionPolicy.CONTINUE:
            raise ValueError("continue does not reset a session")
        if self.engine is None:
            raise RuntimeError("Context-health database unavailable")
        with self.engine.begin() as conn:
            from .service.turn_admission import classify_target, lock_target

            lock_target(conn, bot_id)
            state = classify_target(conn, bot_id, exclude_turn_id=None)
            if not state.idle:
                raise RuntimeError(f"target bot is active in turn {state.blocking_id()}")
            return self.rotate_locked(
                conn, bot_id=bot_id, user_id=user_id, policy=policy, reason=reason,
                working_memory_mode=working_memory_mode,
            )

    @staticmethod
    def rotate_locked(
        conn, *, bot_id: str, user_id: str, policy: SessionPolicy,
        reason: str, working_memory_mode: str,
        expected_session_id: str | None = None,
        new_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Rotate under the caller's per-bot lock, with /new lineage and a receipt.

        An automated caller supplies the expected old thread and a deterministic
        successor; a changed active thread must never be reset by a stale sweep.
        """
        from .memory.session_reset_lineage import SessionResetLineage

        if conn.dialect.name == "postgresql":
            conn.execute(sa_text("SELECT pg_advisory_xact_lock(hashtext(:bot), hashtext(:user))"),
                         {"bot": bot_id, "user": user_id})
        old = conn.execute(sa_text("""
            SELECT id FROM sessions
            WHERE bot_id=:bot AND user_id=:user AND status='active' AND ended_at IS NULL
            ORDER BY started_at DESC LIMIT 1 FOR UPDATE
        """), {"bot": bot_id, "user": user_id}).mappings().first()
        if old is None or (expected_session_id is not None and old["id"] != expected_session_id):
            raise RuntimeError("active context session changed or is missing")
        old_session = old["id"]
        new_session = new_session_id or str(uuid.uuid4())
        SessionResetLineage.rotate(
            None, connection=conn, bot_id=bot_id, user_id=user_id,
            new_id=new_session, accept_pointer=True,
            working_memory_mode=working_memory_mode,
            context_lifecycle={
                "action": policy.value, "reason": reason,
                "old_session_id": old_session,
            },
        )
        return {
            "reset": True, "session_policy": policy.value,
            "retained_history": policy == SessionPolicy.RESET_RETAIN_HISTORY,
            "old_session_id": old_session, "new_session_id": new_session,
            "reason": reason,
        }

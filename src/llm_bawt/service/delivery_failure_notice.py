"""One durable initiating-session notice per failed delegation (TASK-1946).

Reconciles canonical delivery/turn terminal state on the existing leader sweep;
no new detector or scheduler. A DELIVERED steer is not a failed delegation just
because the shared turn failed: command lifecycle does not prove that. Only a
delivery that owns the turn's trigger can inherit that turn's failure.

The source decision and idempotent notice commit together under sorted target
locks then the delivery row lock. Failure rolls back both; a restarted pump can
resume. Prospective signed origins only, never a broadcast/backfill of history.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable

from sqlalchemy import text as sa_text

from ..bot_identities import SYSTEM_BOT_ID
from ..bot_tags import BehaviorTag, sql_allowed
from ..bot_types import normalize_bot_type
from ..inter_bot_failure import DECISION, NOTICE_KEY_PREFIX, NOTICE_KIND, ORIGIN, FailureOrigin, failure_code
from ..memory.postgresql import _sanitize_table_name
from ..utils.db import is_postgresql
from .delivery_cancel_notice import PassResult

logger = logging.getLogger(__name__)
_BATCH_SIZE = 20
# Fixed SQL identifiers only; caller-shaped values remain bound parameters.
_PENDING = f"""
    d.metadata_json ? '{ORIGIN}' AND NOT (d.metadata_json ? '{DECISION}')
    AND d.author_entity_type='bot' AND d.sender_bot_id <> d.target_bot_id
    AND d.message_kind IS DISTINCT FROM '{NOTICE_KIND}'
    AND d.message_kind IS DISTINCT FROM 'DELIVERY_CANCELLED'
    AND NOT (d.metadata_json ? 'prompt_schedule')
    AND (d.status='FAILED' OR (
        d.status='DELIVERED' AND d.delivery_mode='turn'
        AND t.id=d.turn_id AND t.bot_id=d.target_bot_id
        AND t.trigger_message_id=d.user_message_id
        AND t.user_id=d.payload_json->>'user'
        AND t.ended_at IS NOT NULL AND t.status IN ('error','timeout')))
"""
_SENDER_ALLOWED, _SENDER_PARAMS = sql_allowed(
    "d.sender_bot_id", {BehaviorTag.PRIVATE}, prefix="failure_sender")
_TARGET_ALLOWED, _TARGET_PARAMS = sql_allowed(
    "d.target_bot_id", {BehaviorTag.PRIVATE}, prefix="failure_target")
_ALLOWED = f"{_SENDER_ALLOWED} AND {_TARGET_ALLOWED}"
_PARAMS = {**_SENDER_PARAMS, **_TARGET_PARAMS}
# Never select the request message, payload or response/tool body.
_SELECT = """
    SELECT d.id, d.sender_bot_id, d.target_bot_id, d.author_entity_type,
           d.author_entity_id, d.sender_turn_id, d.turn_id, d.target_turn_id,
           d.user_message_id, d.delivery_mode, d.status, d.metadata_json,
           d.task_id, d.project_id, d.last_error, d.payload_json->>'user' AS delivery_user,
           t.id AS execution_turn_id, t.status AS execution_status,
           t.bot_id AS execution_bot_id, t.user_id AS execution_user_id,
           t.trigger_message_id AS execution_trigger_id, t.ended_at AS execution_ended_at,
           t.error_text AS execution_error, t.request_json AS execution_request
    FROM inter_bot_deliveries d
    LEFT JOIN turn_logs t ON t.id=CASE WHEN d.delivery_mode='steer'
                                     THEN d.target_turn_id ELSE d.turn_id END
"""
_SAFE_TASK = re.compile(r"(?:TASK-[0-9]+|c[a-z0-9]{20,40})\Z")
_SAFE_DELIVERY = re.compile(r"delivery-[0-9a-f]{32}\Z")


class FailureNoticePump:
    def __init__(self, store: Any, *, batch_size: int = _BATCH_SIZE,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.batch_size = batch_size
        self._clock = clock
        self._backoff: dict[str, tuple[int, float]] = {}

    def run_pass(self) -> PassResult:
        result = PassResult()
        if not is_postgresql(self.store.engine):
            return result
        waiting = [key for key, (_, until) in self._backoff.items() if until > self._clock()]
        with self.store.engine.connect() as conn:
            candidates = conn.execute(sa_text(f"""
                SELECT d.id, d.sender_bot_id, d.target_bot_id FROM inter_bot_deliveries d
                LEFT JOIN turn_logs t ON t.id=d.turn_id
                WHERE {_PENDING} AND {_ALLOWED} AND d.id <> ALL(:waiting)
                ORDER BY d.updated_at, d.ordinal LIMIT :limit
            """), {**_PARAMS, "waiting": waiting, "limit": self.batch_size}).mappings().all()
        for candidate in candidates:
            try:
                notice, skipped = self._decide(candidate)
            except Exception as exc:  # rollback; never log exception bodies/SQL parameters
                failures = self._backoff.get(candidate["id"], (0, 0))[0] + 1
                delay = min(900, 30 * 2 ** min(failures - 1, 5))
                self._backoff[candidate["id"]] = (failures, self._clock() + delay)
                logger.warning("Failure notice for %s deferred (%s)", candidate["id"], type(exc).__name__)
                result.errors += 1
                continue
            self._backoff.pop(candidate["id"], None)
            if notice is not None:
                result.notified.append(notice)
            result.skipped += int(skipped)
        return result

    def _decide(self, candidate):
        from .turn_admission import lock_target

        with self.store.engine.begin() as conn:
            for bot in sorted({candidate["sender_bot_id"], candidate["target_bot_id"]}):
                lock_target(conn, bot)
            row = conn.execute(sa_text(
                f"{_SELECT} WHERE d.id=:id AND {_PENDING} AND {_ALLOWED} FOR UPDATE OF d"
            ), {**_PARAMS, "id": candidate["id"]}).mappings().first()
            if row is None:
                return None, False
            origin = FailureOrigin.from_row(row)
            if origin is None or not self._owned_origin(conn, origin) or self._user_stopped(row):
                self._stamp(conn, row["id"], "ineligible")
                return None, True
            # A shared turn's terminal state does not confer failure on every
            # accepted steer. FAILED here is this delivery's own send failure.
            variables = self._variables(row, origin)
            # Retry markers in caller metadata are not authority. Only an
            # actual server-produced child with its deterministic key counts.
            existing_retry = conn.execute(sa_text("""
                SELECT id FROM inter_bot_deliveries
                WHERE sender_bot_id=:sender AND target_bot_id=:target
                  AND metadata_json->>'_request_identity_producer'='retry'
                  AND ((idempotency_key=:manual AND metadata_json->>'manual_retry_of'=:id)
                    OR (idempotency_key=:restart AND metadata_json->>'restart_retry_of'=:id))
                ORDER BY ordinal LIMIT 1
            """), {"sender": row["sender_bot_id"], "target": row["target_bot_id"], "id": row["id"],
                   "manual": f"manual-retry:{row['id']}", "restart": f"restart-retry:{row['id']}"}).scalar_one_or_none()
            variables["existing_retry_delivery_id"] = existing_retry or "none"
            from ..prompt_registry import get_prompt_resolver
            from .delivery_failure_notice_prompt import PROMPT_KEY

            message = get_prompt_resolver().render(
                key=PROMPT_KEY, variables=variables, scope_type="bot", scope_id=origin.bot_id)
            if not (message or "").strip():
                raise ValueError("Failure notice prompt rendered blank")
            notice, _ = self.store.enqueue(
                sender_bot_id=SYSTEM_BOT_ID, target_bot_id=origin.bot_id,
                message=message, message_kind=NOTICE_KIND,
                idempotency_key=f"{NOTICE_KEY_PREFIX}{row['id']}",
                project_id=row["project_id"], task_id=row["task_id"],
                metadata={"failed_delivery_id": row["id"], **variables},
                payload={
                    "bot_id": origin.bot_id, "user": origin.user_id,
                    "session_id": origin.session_id,
                    "messages": [{"role": "user", "content": message}],
                    "prefer_steer": False, "reply_mode": "async", "stream": True,
                    "extract_memory": False, "augment_memory": False,
                },
                on_gate="hold", connection=conn,
            )
            self._stamp(conn, row["id"], "enqueued", notice.id)
            return notice, False

    @staticmethod
    def _stamp(conn, delivery_id, outcome, notice_id=None):
        conn.execute(sa_text(f"""
            UPDATE inter_bot_deliveries
            SET metadata_json=metadata_json || jsonb_build_object('{DECISION}', CAST(:decision AS jsonb))
            WHERE id=:id
        """), {"id": delivery_id, "decision": json.dumps({"outcome": outcome, "notice_delivery_id": notice_id})})

    @staticmethod
    def _owned_origin(conn, origin: FailureOrigin) -> bool:
        # Revalidate against the actual initiating turn + trigger + session, not
        # a callback's caller-shaped metadata, target user, or current default.
        row = conn.execute(sa_text("""
            SELECT p.bot_type, p.agent_backend
            FROM sessions s
            JOIN turn_logs t ON t.id=:turn AND t.bot_id=s.bot_id AND t.user_id=s.user_id
            JOIN messages m ON m.bot_id=:partition AND m.id=t.trigger_message_id AND m.session_id=s.id
            JOIN bot_profiles p ON p.slug=s.bot_id
            WHERE s.id=:session AND s.bot_id=:bot AND s.user_id=:user
              AND s.status IN ('active','archived') AND t.trigger_message_id=:trigger
        """), {"session": origin.session_id, "bot": origin.bot_id, "user": origin.user_id,
               "partition": _sanitize_table_name(origin.bot_id), "turn": origin.turn_id,
               "trigger": origin.trigger_message_id}).mappings().first()
        return row is not None and normalize_bot_type(row["bot_type"], row["agent_backend"]) == "agent"

    @staticmethod
    def _user_stopped(row) -> bool:
        if (row["metadata_json"] or {}).get("stopped_after_start") is True:
            return True
        raw = row.get("execution_request")
        try:
            request = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            request = None
        # Any explicit abort request is conservative Stop evidence; do not turn
        # it into a failure/restart request, regardless of final status spelling.
        return isinstance(request, dict) and bool(request.get("abort_request"))

    @staticmethod
    def _variables(row, origin):
        owns_turn = (row["delivery_mode"] == "turn" and row.get("execution_turn_id") == row["turn_id"]
                     and row.get("execution_bot_id") == row["target_bot_id"]
                     and row.get("execution_user_id") == row.get("delivery_user")
                     and row.get("execution_trigger_id") == row["user_message_id"])
        metadata = row["metadata_json"] or {}
        original = ((metadata.get("manual_retry_of") or metadata.get("restart_retry_of"))
                    if metadata.get("_request_identity_producer") == "retry" else None)
        task = row["task_id"] or ""
        return {
            "worker_bot_id": row["target_bot_id"],
            "task_id": task if _SAFE_TASK.fullmatch(task) else "unavailable",
            "delivery_id": row["id"],
            "original_delivery_id": original if isinstance(original, str) and _SAFE_DELIVERY.fullmatch(original) else row["id"],
            # Stable fallback identity remains inspectable even before a turn
            # was reserved. A steer has only a shared target, not a failed
            # command-specific turn, and is explicitly labelled unattributed.
            "failed_turn_id": (row.get("target_turn_id") or "unavailable") if row["delivery_mode"] == "steer" else row["turn_id"],
            "initiating_turn_id": origin.turn_id,
            "delivery_status": row["status"],
            "turn_status": row["execution_status"] if owns_turn else "unattributed",
            "error_code": failure_code(row["last_error"] or (row["execution_error"] if owns_turn else None)),
            "existing_retry_delivery_id": "none",
        }

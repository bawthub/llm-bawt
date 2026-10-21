"""Owner-scoped, default-off routing into the existing durable delivery outbox.

Lock order: routing receipt row -> selected target advisory lock -> delivery identity.
The routing receipt and delivery commit together; retries never rerank. The
read-only preview is advisory and performs no admission or reservation.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from ..agent_context import SessionPolicy, validate_session_policy
from ..bot_types import durable_agent_backend
from ..bots import BotManager
from ..inter_bot_delivery_record import DeliveryRecord
from ..message_authorship import AuthorReference
from ..model_catalog import bot_model_ref
from ..utils.schema import SchemaBootstrapGuard
from .jobs.idle_context_candidates import PROTECTED_BOT_IDS
from .turn_admission import classify_target, lock_target


class RouteConflict(ValueError):
    """An immutable route key/intent was reused with different data."""


class SelectionChanged(RuntimeError):
    """The selected target lost admission before its lock was obtained."""


def route_opted_in(conn, owner: str, bot_id: str) -> bool:
    """Authoritative owner-specific consent; absent membership is disabled."""
    return bool(conn.execute(text("""
        SELECT enabled FROM message_route_eligibility
        WHERE owner_user_id=:owner AND bot_id=:bot
    """), {"owner": owner, "bot": bot_id}).scalar_one_or_none())


def lock_routing_pool(conn, owner: str) -> None:
    """Serialize owner selection across routes and schedule preparations."""
    if conn.dialect.name == "postgresql":
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:pool))"),
                     {"pool": f"message-routing-pool:{owner}"})


@dataclass(frozen=True)
class RouteRequest:
    user_id: str
    candidate_bot_ids: list[str] | None = None
    model_endpoint_id: int | None = None
    requested_model: str | None = None  # schedule adapter uses canonical endpoint ref
    session_policy: str = "continue"
    scope: str = "personal"


class MessageRouter:
    """One selection/admission authority for API, MCP and schedule producers."""

    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, service):
        self.service = service
        dispatcher = getattr(service, "_inter_bot_dispatcher", None)
        if dispatcher is None or dispatcher.store.engine is None:
            raise RuntimeError("Durable delivery database unavailable")
        self.dispatcher = dispatcher
        self.engine = dispatcher.store.engine
        self._ensure_schema()

    def _ensure_schema(self):
        # Static DDL, not a fleet enablement: absent membership rows are OFF.
        def create(conn):
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS message_route_eligibility (
                    owner_user_id VARCHAR(128) NOT NULL,
                    bot_id VARCHAR(128) NOT NULL,
                    enabled BOOLEAN NOT NULL DEFAULT FALSE,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (owner_user_id, bot_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS message_routes (
                    id VARCHAR(96) PRIMARY KEY,
                    owner_user_id VARCHAR(128) NOT NULL,
                    actor_entity_type VARCHAR(16) NOT NULL,
                    actor_entity_id VARCHAR(128) NOT NULL,
                    idempotency_key VARCHAR(256) NOT NULL,
                    intent_hash VARCHAR(64) NOT NULL,
                    selected_bot_id VARCHAR(128),
                    model_endpoint_id INTEGER,
                    model_ref VARCHAR(256),
                    session_policy VARCHAR(32) NOT NULL,
                    scope VARCHAR(16) NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    ranking_json JSONB,
                    delivery_id VARCHAR(96),
                    state VARCHAR(24) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (owner_user_id, actor_entity_type, actor_entity_id, idempotency_key)
                )
            """))
            conn.execute(text("""
                ALTER TABLE message_routes ADD COLUMN IF NOT EXISTS ranking_json JSONB
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_message_routes_pending
                ON message_routes (selected_bot_id) WHERE state='queued'
            """))

        self._schema_guard.run(self.engine, "message-routing-v1", create)

    @staticmethod
    def _validate(request: RouteRequest) -> tuple[str, set[str] | None]:
        owner = request.user_id.strip().lower()
        if not owner or len(owner) > 128:
            raise ValueError("user_id must identify one owner")
        if request.scope not in {"personal", "schedule"}:
            raise ValueError("scope must be personal or schedule")
        SessionPolicy(request.session_policy)
        if request.scope == "schedule" and request.session_policy != "continue":
            raise ValueError("Scheduled prompts keep isolated threads; reset policy is not supported")
        if request.model_endpoint_id is not None and request.requested_model is not None:
            raise ValueError("Provide a model endpoint ID or a canonical reference, not both")
        if request.model_endpoint_id is not None and (
            isinstance(request.model_endpoint_id, bool) or request.model_endpoint_id < 1
        ):
            raise ValueError("model_endpoint_id must be a positive integer")
        narrowed = None
        if request.candidate_bot_ids is not None:
            narrowed = {slug.strip().lower() for slug in request.candidate_bot_ids}
            if "" in narrowed or len(narrowed) != len(request.candidate_bot_ids):
                raise ValueError("candidate_bot_ids must contain unique, nonblank bot IDs")
        return owner, narrowed

    def set_enabled(self, user_id: str, bot_id: str, enabled: bool) -> dict:
        owner, _ = self._validate(RouteRequest(user_id=user_id))
        slug = bot_id.strip().lower()
        bot = BotManager(self.service.config).get_bot(slug)
        if not bot or slug in PROTECTED_BOT_IDS or not durable_agent_backend(bot):
            raise ValueError("This bot cannot be configured for routing")
        with self.engine.begin() as conn:
            # A route binds under the same target lock; disabling wins or waits.
            lock_target(conn, slug)
            conn.execute(text("""
                INSERT INTO message_route_eligibility(owner_user_id, bot_id, enabled)
                VALUES (:owner, :bot, :enabled)
                ON CONFLICT (owner_user_id, bot_id)
                DO UPDATE SET enabled=EXCLUDED.enabled, updated_at=now()
            """), {"owner": owner, "bot": slug, "enabled": enabled})
        return {"user_id": owner, "bot_id": slug, "enabled": enabled}

    def _model(self, conn, bot, owner: str, request: RouteRequest):
        backend = durable_agent_backend(bot)
        if backend is None:
            raise ValueError("Unsupported durable backend")
        validate_session_policy(backend, SessionPolicy(request.session_policy))
        # Scheduled prompt execution currently requires a Claude explicit-thread
        # backend. Never advertise a capability its executor cannot deliver.
        if request.scope == "schedule" and (
            bot.agent_backend != "claude-code" or bot.harness not in {"claude-code", "claude-proxy"}
        ):
            raise ValueError("Scheduled prompts require Claude Code explicit thread isolation")
        catalog = self.service.config.ensure_model_catalog()
        if catalog is None:
            raise ValueError("Model catalog unavailable")
        model = (request.model_endpoint_id if request.model_endpoint_id is not None
                 else request.requested_model if request.requested_model is not None
                 else bot_model_ref(self.service.config, bot))
        endpoint = catalog.resolve_endpoint(model, harness=bot.harness)
        if request.scope == "personal" and request.session_policy == "continue":
            from .route_session_binding import active_session, check_provider_binding

            session = active_session(conn, bot.slug, owner)
            if session:
                check_provider_binding(self.service.config, bot, endpoint,
                                       session["session_metadata"])
        from ..bot_types import agent_backend_for_model_def
        from ..model_catalog import resolve_model_config
        if agent_backend_for_model_def(resolve_model_config(
            self.service.config, endpoint.ref, harness=bot.harness, default={}
        )) != backend:
            raise ValueError("Model endpoint is incompatible with this agent backend")
        return endpoint

    def _candidates(self, conn, request: RouteRequest, *, all_bots: bool) -> list[dict]:
        owner, narrowed = self._validate(request)
        bots = BotManager(self.service.config).list_bots()
        membership = dict(conn.execute(text("""
            SELECT bot_id, enabled FROM message_route_eligibility WHERE owner_user_id=:owner
        """), {"owner": owner}).all())
        schedule_table = (conn.execute(text("SELECT to_regclass('prompt_occurrences')")).scalar()
                          if conn.dialect.name == "postgresql" else None)
        rows = []
        for bot in bots:
            slug = bot.slug
            # Protection must be evaluated before ANY activity/session/turn query.
            protected = slug in PROTECTED_BOT_IDS
            configurable = not protected and bool(durable_agent_backend(bot))
            enabled = bool(membership.get(slug, False)) if configurable else False
            if not all_bots and (not enabled or narrowed is not None and slug not in narrowed):
                continue
            reasons = []
            if protected:
                reasons.append("protected bot")
            elif not configurable:
                reasons.append("unsupported backend")
            elif not getattr(bot, "enabled", True):
                reasons.append("bot disabled")
            if not enabled:
                reasons.append("not opted in")
            if narrowed is not None and slug not in narrowed:
                reasons.append("not in requested pool")
            endpoint = None
            if configurable and enabled and (narrowed is None or slug in narrowed):
                try:
                    endpoint = self._model(conn, bot, owner, request)
                except ValueError as exc:
                    reasons.append(str(exc))
            # Excluded/protected bots never enter the timestamp or occupancy reads.
            last_at = None
            last_iso = None
            pending = 0
            available = False
            if not reasons:
                try:
                    # A savepoint lets the next bot remain inspectable when a
                    # partition/history read fails; unknown is NOT never-used.
                    with conn.begin_nested():
                        last_at = conn.execute(text("""
                            SELECT max(m.timestamp) FROM messages m
                            JOIN sessions s ON s.id=m.session_id
                            WHERE m.bot_id=:bot AND s.bot_id=:bot AND s.user_id=:owner
                              AND COALESCE(m.recalled_history, false) = false
                              AND ((m.role='assistant' AND COALESCE(m.author_entity_type, 'bot')='bot') OR
                                   (m.role='user' AND COALESCE(m.author_entity_type, 'user') IN ('user','bot')))
                        """), {"bot": slug, "owner": owner}).scalar_one()
                        last_iso = (datetime.fromtimestamp(float(last_at), timezone.utc).isoformat()
                                    if last_at is not None else None)
                        pending = int(conn.execute(text("""
                            SELECT count(*) FROM inter_bot_deliveries
                            WHERE target_bot_id=:bot AND status IN ('QUEUED','STEERING','DISPATCHING')
                        """), {"bot": slug}).scalar_one())
                        if schedule_table:
                            pending += int(conn.execute(text("""
                                SELECT count(*) FROM prompt_occurrences
                                WHERE state='pending' AND delivery_id IS NULL
                                  AND snapshot_json->>'target_mode'='auto'
                                  AND snapshot_json->>'bot_id'=:bot
                            """), {"bot": slug}).scalar_one())
                        available = pending == 0 and classify_target(
                            conn, slug, exclude_turn_id=None,
                        ).idle
                except Exception:
                    last_iso = None
                    available = False
                    reasons.append("activity or admission state unavailable")
            rows.append({
                "bot_id": slug, "configurable": configurable, "enabled": enabled,
                "eligible": not reasons, "available": available,
                "last_message_at": last_iso,
                "pending_routes": pending, "exclusion_reasons": reasons,
                "model_endpoint_id": endpoint.id if endpoint else None,
                "model_ref": endpoint.ref if endpoint else None,
            })
        rows.sort(key=lambda row: (
            not row["eligible"], not row["available"],
            row["last_message_at"] is not None,
            row["last_message_at"] or "", row["pending_routes"], row["bot_id"],
        ))
        return rows

    def preview(self, request: RouteRequest) -> dict:
        with self.engine.connect() as conn:
            rows = self._candidates(conn, request, all_bots=False)
        selected = next((row for row in rows if row["eligible"]), None)
        return {
            "state": "selected" if selected else "not_routed",
            "selected_bot_id": selected["bot_id"] if selected else None,
            "model_endpoint_id": selected["model_endpoint_id"] if selected else None,
            "reason": self._reason(selected) if selected else "No opted-in compatible candidates",
            "candidates": rows,
        }

    def candidates(self, request: RouteRequest) -> dict:
        with self.engine.connect() as conn:
            return {"candidates": self._candidates(conn, request, all_bots=True)}

    @staticmethod
    def _reason(candidate: dict) -> str:
        age = "no prior messages" if candidate["last_message_at"] is None else "oldest last message"
        return ("idle; " if candidate["available"] else "all compatible bots busy; queued; ") + age

    @staticmethod
    def ranking_evidence(selected: dict, rows: list[dict]) -> dict:
        return {
            "version": 1, "selected_last_message_at": selected["last_message_at"],
            "selected_available": selected["available"],
            "selected_pending_routes": selected["pending_routes"],
            "compatible_candidate_ids": [item["bot_id"] for item in rows if item["eligible"]],
        }

    def route(self, request: RouteRequest, *, message: str, idempotency_key: str,
              project_id: str | None = None, task_id: str | None = None,
              metadata: dict | None = None,
              actor: AuthorReference | None = None) -> dict:
        for _ in range(3):
            try:
                return self._route_once(
                    request, message=message, idempotency_key=idempotency_key,
                    project_id=project_id, task_id=task_id, metadata=metadata,
                    actor=actor,
                )
            except SelectionChanged:
                # The entire transaction rolled back, including its provisional
                # route row; retry with the same key and no stale target lock.
                continue
        raise RuntimeError("Routing targets changed during admission; retry later")

    def _route_once(self, request: RouteRequest, *, message: str, idempotency_key: str,
                    project_id: str | None = None, task_id: str | None = None,
                    metadata: dict | None = None,
                    actor: AuthorReference | None = None) -> dict:
        owner, narrowed = self._validate(request)
        author = actor or AuthorReference.user(owner)
        if author.entity_type not in {"user", "bot"} or not author.entity_id:
            raise ValueError("Routing requires a trusted caller actor")
        if author.entity_type == "user" and author.entity_id != owner:
            raise ValueError("User author cannot differ from routing owner")
        key = idempotency_key.strip()
        if not key or len(key) > 256 or not message.strip():
            raise ValueError("message and idempotency_key are required")
        if request.scope != "personal":
            raise ValueError("Schedule occurrences use their isolated prompt outbox")
        intent = {
            "user_id": owner, "actor": author.to_dict(),
            "candidates": sorted(narrowed) if narrowed is not None else None,
            "model_endpoint_id": request.model_endpoint_id, "session_policy": request.session_policy,
            "scope": request.scope, "message": message.strip(),
            "project_id": project_id, "task_id": task_id,
        }
        digest = hashlib.sha256(json.dumps(intent, sort_keys=True).encode()).hexdigest()
        route_id = "route-" + uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"message-route:{owner}:{author.entity_type}:{author.entity_id}:{key}",
        ).hex
        sender = (author.entity_id if author.entity_type == "bot" else
                  "router:" + hashlib.sha256(owner.encode()).hexdigest())
        with self.engine.begin() as conn:
            lock_routing_pool(conn, owner)
            conn.execute(text("""
                INSERT INTO message_routes(id, owner_user_id, actor_entity_type, actor_entity_id,
                    idempotency_key, intent_hash, session_policy, scope, state, reason)
                VALUES (:id,:owner,:actor_type,:actor_id,:key,:digest,:policy,:scope,'pending','')
                ON CONFLICT (owner_user_id, actor_entity_type, actor_entity_id, idempotency_key) DO NOTHING
            """), {"id": route_id, "owner": owner, "actor_type": author.entity_type,
                    "actor_id": author.entity_id, "key": key, "digest": digest,
                    "policy": request.session_policy, "scope": request.scope})
            row = conn.execute(text("""
                SELECT * FROM message_routes WHERE owner_user_id=:owner
                  AND actor_entity_type=:actor_type AND actor_entity_id=:actor_id
                  AND idempotency_key=:key FOR UPDATE
            """), {"owner": owner, "actor_type": author.entity_type,
                   "actor_id": author.entity_id, "key": key}).mappings().one()
            if row["intent_hash"] != digest:
                raise RouteConflict("Idempotency key belongs to a different routing intent")
            if row["state"] != "pending":
                return self._receipt(row, duplicate=True)
            candidates = self._candidates(conn, request, all_bots=False)
            selected = next((item for item in candidates if item["eligible"]), None)
            if selected is None:
                result = conn.execute(text("""
                    UPDATE message_routes SET state='not_routed',reason=:reason
                    WHERE id=:id RETURNING *
                """), {"id": route_id, "reason": "No opted-in compatible candidates"}).mappings().one()
                return self._receipt(result)
            slug = selected["bot_id"]
            lock_target(conn, slug)
            # Recheck selection *after* target lock; competing admissions cannot
            # both decide this bot is idle based on the same earlier snapshot.
            candidates = self._candidates(conn, request, all_bots=False)
            selected = next((item for item in candidates if item["eligible"]), None)
            if not selected or selected["bot_id"] != slug:
                # No second target lock in this transaction: let the caller
                # retry a fresh request rather than violate sorted lock order.
                raise SelectionChanged("Routing candidates changed during admission")
            endpoint_id = selected["model_endpoint_id"]
            model_ref = selected["model_ref"]
            binding = None
            if request.session_policy == "continue":
                from .route_session_binding import snapshot_session

                bot = BotManager(self.service.config).get_bot(slug)
                endpoint = self.service.config.ensure_model_catalog().resolve_endpoint(
                    endpoint_id, harness=bot.harness,
                )
                binding = snapshot_session(conn, self.service.config, bot, owner, endpoint)
            payload = {
                "bot_id": slug, "user": owner,
                "messages": [{"role": "user", "content": message.strip()}],
                "model": model_ref, "prefer_steer": False, "stream": True,
                "augment_memory": True, "extract_memory": True,
            }
            if binding:
                payload["session_id"] = binding["session_id"]
            provenance = {**(metadata or {}), "message_route": {
                "route_id": route_id, "owner_user_id": owner,
                "model_endpoint_id": endpoint_id, "model_ref": model_ref,
                "session_binding": binding,
            }}
            if request.session_policy == "continue":
                record, _ = self.dispatcher.store.enqueue(
                    sender_bot_id=sender, target_bot_id=slug, author=author,
                    message=message.strip(), payload=payload, idempotency_key=route_id,
                    project_id=project_id, task_id=task_id, metadata=provenance,
                    session_policy=request.session_policy, connection=conn, producer="bot",
                )
            else:
                from .session_lifecycle.intent import Admission, Continuity, NewSessionIntent, Origin
                from .session_lifecycle.lifecycle import NewSessionLifecycle
                intent_reset = NewSessionIntent(
                    bot_id=slug, user_id=owner, author=author,
                    continuity=(Continuity.RETAIN if request.session_policy == "reset_retain_history" else Continuity.CLEAN),
                    admission=Admission.WAIT, origin=Origin.BOT_DELIVERY,
                    idempotency_key=route_id, follow_up=message.strip(),
                )
                record, _ = NewSessionLifecycle(self.service).admit_bot_delivery(
                    intent_reset, payload=payload, metadata=provenance,
                    sender_bot_id=sender, project_id=project_id, task_id=task_id,
                    producer="bot", connection=conn,
                )
            reason = self._reason(selected)
            ranking = self.ranking_evidence(selected, candidates)
            result = conn.execute(text("""
                UPDATE message_routes SET selected_bot_id=:bot, model_endpoint_id=:endpoint,
                  model_ref=:model, delivery_id=:delivery, state='queued', reason=:reason,
                  ranking_json=CAST(:ranking AS jsonb)
                WHERE id=:id RETURNING *
            """), {"id": route_id, "bot": slug, "endpoint": endpoint_id,
                    "model": model_ref, "delivery": record.id, "reason": reason,
                    "ranking": json.dumps(ranking)}).mappings().one()
        self.dispatcher.announce_admitted(record)
        return self._receipt(result)

    def _receipt(self, row, *, duplicate=False) -> dict:
        result = {"route_id": row["id"],
                  "author": {"entity_type": row["actor_entity_type"],
                             "entity_id": row["actor_entity_id"]},
                  "selected_bot_id": row["selected_bot_id"],
                  "model_endpoint_id": row["model_endpoint_id"], "model_ref": row["model_ref"],
                  "session_policy": row["session_policy"], "scope": row["scope"],
                  "delivery_id": row["delivery_id"], "state": row["state"],
                  "reason": row["reason"], "ranking": row["ranking_json"],
                  "duplicate": duplicate}
        if row["delivery_id"]:
            record = self.dispatcher.store.get(row["delivery_id"])
            if record:
                result["delivery_status"] = record.status
                result["turn_id"] = record.turn_id
                result["state"] = {
                    "QUEUED": "queued", "STEERING": "running", "DISPATCHING": "running",
                    "DELIVERED": "succeeded", "FAILED": "failed", "CANCELLED": "cancelled",
                }.get(record.status, "unknown")
        return result

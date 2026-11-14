"""Reference-only continuity provenance on the existing sessions JSONB record."""
from __future__ import annotations

import json
import time
from sqlalchemy import bindparam, text
from agent_bridge.continuity import continuity_digest
from llm_bawt.utils.bot_partition import sanitize_table_name

# Every app-authored /new receipt starts with this (agent_bridge.continuity).
_RESET_RECEIPT_PREFIX = "Session reset."


def _is_reset_bookkeeping(message: dict) -> bool:
    """A bare ``/new`` or the receipt the app stored for a reset.

    Receipts are committed just before their command (receipt.py), so order
    is not a signal. ``/new <task>`` is a real request: neither it nor its
    answer is bookkeeping.
    """
    role, content = message.get("role"), (message.get("content") or "").strip()
    if role == "user":
        return content == "/new"
    return role == "assistant" and content.startswith(_RESET_RECEIPT_PREFIX)


def carries_prior_context(messages: list[dict]) -> bool:
    """True when the seed holds a summary or any row besides reset bookkeeping.

    A bare /new and its receipt are not conversation: a seed of only those
    rows restores nothing (TASK-851).
    """
    return any(m.get("role") in ("summary", "user", "assistant") and not _is_reset_bookkeeping(m)
               for m in messages)


def seed_empty_reason(messages: list[dict], settings: dict) -> str | None:
    """Why a seed restored nothing while continuity was on, else None (TASK-851).

    ``settings`` holds the effective ``history_scope`` and
    ``history_max_age_hours`` the seed was built with. Scope ``none`` is an
    explicit choice to carry nothing, so it has no reason to report.
    """
    if carries_prior_context(messages):
        return None
    scope = str(settings.get("history_scope") or "")
    if scope not in ("inline", "summaries", "inline+summaries"):
        return None
    parts = []
    if "inline" in scope:
        try:
            max_age = int(settings.get("history_max_age_hours") or 0)
        except (TypeError, ValueError):
            max_age = 0
        parts.append(f"no messages from the last {max_age}h (history_max_age_hours)"
                     if max_age > 0 else "no earlier messages")
    else:
        parts.append("history is off")
    parts.append("no summaries yet" if "summaries" in scope else "summaries are off")
    return "; ".join(parts)


class SessionContinuity:
    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def manifest(messages: list[dict], *, settings: dict, model: str | None,
                 report_empty: bool = True) -> dict:
        timestamps = [m["timestamp"] for m in messages if m.get("timestamp")]
        return {
            "version": 1,
            "status": "prepared" if messages else "empty",
            "captured_at": time.time(),
            "payload_digest": continuity_digest(messages),
            "model": model,
            "settings": settings,
            # Null preserves the position of a legacy source without a durable ID.
            "source_message_ids": [m.get("source_id") for m in messages],
            "shortened_indices": [i for i, m in enumerate(messages) if m.get("shortened")],
            "summary_count": sum(m.get("role") == "summary" for m in messages),
            "message_count": sum(m.get("role") in ("user", "assistant") for m in messages),
            "approx_tokens": sum(len(m.get("content") or "") for m in messages) // 4,
            "oldest_timestamp": min(timestamps) if timestamps else None,
            "newest_timestamp": max(timestamps) if timestamps else None,
            "empty_reason": seed_empty_reason(messages, settings) if report_empty else None,
        }

    def record(self, session_id: str, bot_id: str, user_id: str, manifest: dict) -> bool:
        """Merge one field atomically; never overwrite an established restoration."""
        with self.engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE sessions SET session_metadata =
                    COALESCE(session_metadata, '{}'::jsonb) || jsonb_build_object('continuity', CAST(:value AS jsonb))
                WHERE id=:session AND bot_id=:bot AND user_id=:user AND status <> 'deleted'
                  AND (session_metadata->'continuity' IS NULL
                    OR session_metadata->'continuity'->>'status' IN ('prepared', 'failed'))
            """), {"session": session_id, "bot": bot_id, "user": user_id, "value": json.dumps(manifest)})
            return bool(result.rowcount)

    def acknowledge(self, session_id: str, bot_id: str, digest: str, *, restored: bool) -> bool:
        """Bridge receipt: a stale/different payload cannot confirm this capture.

        Internal service boundary, like agent-session-key writeback. The digest
        binds the callback to the actual payload, not a caller-supplied user ID.
        """
        patch = {"status": "restored" if restored else "failed", "completed_at": time.time()}
        with self.engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE sessions SET session_metadata = jsonb_set(session_metadata, '{continuity}',
                    (session_metadata->'continuity') || CAST(:patch AS jsonb))
                WHERE id=:session AND bot_id=:bot AND status <> 'deleted'
                  AND session_metadata->'continuity'->>'payload_digest'=:digest
                  AND session_metadata->'continuity'->>'status' IN ('prepared', 'failed')
            """), {"session": session_id, "bot": bot_id, "digest": digest, "patch": json.dumps(patch)})
            return bool(result.rowcount)

    def resolve(self, manifest: dict, *, bot_id: str, user_id: str, offset: int, limit: int) -> dict:
        """Resolve a page of references; source-session ownership is mandatory."""
        ids = manifest.get("source_message_ids")
        ids = ids if isinstance(ids, list) else []
        page = ids[offset:offset + limit]
        valid_ids = [item for item in page if isinstance(item, str)]
        rows = {}
        if valid_ids:
            query = text("""
                SELECT m.id, m.role, m.content, m.timestamp, m.session_id
                FROM messages m JOIN sessions s ON s.id=m.session_id AND s.bot_id=:bot
                WHERE m.bot_id=:partition AND s.user_id=:user AND s.status <> 'deleted'
                  AND m.id IN :ids
            """).bindparams(bindparam("ids", expanding=True))
            with self.engine.connect() as conn:
                params = {"bot": bot_id, "partition": sanitize_table_name(bot_id),
                          "user": user_id, "ids": valid_ids}
                rows = {r["id"]: dict(r) for r in conn.execute(query, params).mappings()}
        shortened = set(manifest.get("shortened_indices") or [])
        return {
            "messages": [
                {"index": offset + i, "id": source_id, "available": source_id in rows if isinstance(source_id, str) else False,
                 "shortened": offset + i in shortened, **(rows.get(source_id, {}) if isinstance(source_id, str) else {})}
                for i, source_id in enumerate(page)
            ],
            "total_count": len(ids), "offset": offset,
            "has_more": offset + len(page) < len(ids),
        }

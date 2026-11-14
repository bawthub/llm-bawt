"""Guarded execution of an exact, fresh bot purge preview (TASK-2045).

The caller confirms the ``manifest_digest`` of a ``purge-preview`` for the same
slug and action. Inside one PostgreSQL REPEATABLE READ transaction this module
fences profile/tag/prompt writers, recomputes the preview, refuses a changed
digest, refuses protected or active bots, then deletes exactly the rows the
recomputed preview selected (plus the profile and its tag rows for the combined
action) and commits once. Any refusal or database error rolls everything back:
there is never a partial profile delete.

Turn and delivery admission for the bot is fenced for the whole purge by its
admission lock, taken before the snapshot. Limits, stated rather than hidden:
other message/memory writers are not table-locked
(that would stall every bot). A row a writer commits after the snapshot is not
seen by the DELETE and survives; for a deleted profile it becomes orphan
storage that System → Clean up orphans covers. Stores outside local PostgreSQL
are reported ``unknown`` and are untouched.
"""
from __future__ import annotations

import re
import time
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.exc import DBAPIError, IntegrityError, SQLAlchemyError

from .bot_identities import RESERVED_JOB_BOT_IDS
from .bot_purge_preview import collect_preview, normalize_slug

_DIGEST = re.compile(r"^[a-f0-9]{64}$")
LOCK_TIMEOUT = "3s"
ADMISSION_WAIT_SECONDS = 3.0
_ADMISSION_POLL_SECONDS = 0.05


class PurgeRefused(RuntimeError):
    """Nothing was deleted. ``status`` is the HTTP mapping of the 1897 contract."""

    status = 409
    code = "refused"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        if code:
            self.code = code


class PurgeStale(PurgeRefused):
    status = 409
    code = "stale_preview"


class PurgeLocked(PurgeRefused):
    """Protected identity, active work or a busy writer fence (423)."""

    status = 423
    code = "locked"


class PurgeUnavailable(PurgeRefused):
    status = 503
    code = "unavailable"


PARTITION_PARENTS = ("messages", "memories", "forgotten_messages")


def delete_scoped_rows(conn, slug: str, prompt_purge) -> dict[str, int]:
    """Delete exactly the standalone-purge selection on the caller's transaction.

    ``prompt_purge.guard`` must already have run on this connection. The
    predicates mirror ``bot_purge_inventory.DELETED``; partition rows use the
    sanitized storage key. Profile, tags, receipts and audit rows are retained.
    """
    storage = re.sub(r"[^a-z0-9_]", "", slug) or "default"
    deleted: dict[str, int] = {}

    def run(key: str, sql: str, value: str = slug) -> None:
        deleted[key] = conn.execute(text(sql), {"s": value}).rowcount

    # Partition parents (TASK-571): DELETE rows, never DROP the partition.
    for parent in PARTITION_PARENTS:
        run(parent, f"DELETE FROM {parent} WHERE bot_id=:s", storage)
    # Messages hold a non-cascading session FK: a foreign reference must roll
    # back the whole purge rather than erase another bot's rows.
    if "sessions" in prompt_purge.tables:
        run("sessions", "DELETE FROM sessions WHERE bot_id=:s")
    if "session_resumption_receipts" in prompt_purge.tables:
        run("session_resumption_receipts", "DELETE FROM session_resumption_receipts WHERE bot_id=:s")
    run("runtime_settings", "DELETE FROM runtime_settings WHERE scope_type='bot' AND scope_id=:s")
    # Attributes first: there is no FK cascade between the entity tables.
    for table in ("entity_profile_attributes", "entity_profiles"):
        run(table, f"DELETE FROM {table} WHERE CAST(entity_type AS TEXT) IN ('BOT','bot') AND entity_id=:s")
    deleted.update(prompt_purge.delete_jobs(slug))
    for table in ("prompt_template_versions", "prompt_templates"):
        run(table, f"DELETE FROM {table} WHERE scope_type='bot' AND scope_id=:s")
    # Legacy overflow payloads have no FK; select them through owned records.
    if "tool_call_result_payloads" in prompt_purge.tables:
        run("tool_call_result_payloads", "DELETE FROM tool_call_result_payloads WHERE tool_call_record_id IN "
            "(SELECT id FROM tool_call_records WHERE bot_id=:s)")
    run("tool_call_records", "DELETE FROM tool_call_records WHERE bot_id=:s")
    if "llm_response_records" in prompt_purge.tables:
        run("llm_response_records", "DELETE FROM llm_response_records WHERE bot_id=:s")
    if "worker_lifecycle_records" in prompt_purge.tables:
        run("worker_lifecycle_records", "DELETE FROM worker_lifecycle_records WHERE bot_id=:s")
    run("turn_logs", "DELETE FROM turn_logs WHERE bot_id=:s")
    return deleted


def evict_partition_caches(slug: str) -> None:
    """Drop cached partition Table objects so a later write re-resolves them."""
    storage = re.sub(r"[^a-z0-9_]", "", slug) or "default"
    try:
        from .memory.postgresql import _memory_table_cache
        from .memory.message_store import _forgotten_table_cache, _message_table_cache
    except ImportError:  # pragma: no cover - memory backend not importable
        return
    keys = {f"messages_p_{storage}", f"memories_p_{storage}", f"forgotten_messages_p_{storage}"}
    for cache in (_memory_table_cache, _message_table_cache, _forgotten_table_cache):
        for key in list(cache):
            if key in keys:
                cache.pop(key, None)


def _check_protected(conn, slug: str, preview: dict) -> bool:
    """Refuse protected/unavailable scopes; return whether partition rows are owned."""
    from . import bot_slug_aliases

    if slug in RESERVED_JOB_BOT_IDS:
        raise PurgeLocked(f"{slug} is a reserved system identity and cannot be purged", code="protected")
    retired = bot_slug_aliases.aliases_for(conn, slug)
    if retired:
        raise PurgeRefused(f"{slug} still has retired slug aliases ({', '.join(retired)}); remove them first",
                           code="aliases")
    if preview["storage_ownership"]["status"] == "incomplete":
        raise PurgeUnavailable("A store needed to prove ownership is unavailable", code="unavailable")
    for entry in preview["inventory"]:
        if entry["store"] == "local_postgresql" and entry["coverage"] == "missing_table_or_columns" \
                and entry["key"] in _REQUIRED_DELETED_TABLES:
            raise PurgeUnavailable(f"Store {entry['key']} is unavailable", code="unavailable")
    # Delete partition rows only when the preview itself claimed them. Shared or
    # profile-less storage is reported unknown and left alone.
    rows = {e["key"]: e for e in preview["inventory"]}
    owned = all(rows[p]["disposition"] == "deleted" for p in PARTITION_PARENTS)
    return owned and not bot_slug_aliases.storage_id_reserved(conn, preview["storage_id"])


# Executed unconditionally by delete_scoped_rows; other deleted tables are optional.
_REQUIRED_DELETED_TABLES = frozenset({
    "messages", "memories", "forgotten_messages", "runtime_settings", "entity_profile_attributes",
    "entity_profiles", "prompt_template_versions", "prompt_templates", "tool_call_records", "turn_logs",
})


def _acquire_admission_fence(raw, slug: str) -> None:
    """Hold the bot's turn-admission lock for the whole purge.

    Same key as ``turn_admission.lock_target`` and the delivery claim, so no
    turn or delivery can be admitted for this bot until the purge ends. It is
    taken at session level *before* the REPEATABLE READ transaction begins: a
    lock acquired by a statement inside that transaction would fix the snapshot
    before the wait, hiding a turn admitted while we waited. Non-blocking polls
    keep the session's lock_timeout untouched.
    """
    raw.execution_options(isolation_level="AUTOCOMMIT")
    deadline = time.monotonic() + ADMISSION_WAIT_SECONDS
    while True:
        with raw.begin():
            got = raw.execute(text("SELECT pg_try_advisory_lock(hashtext(:target))"), {"target": slug}).scalar_one()
        if got:
            return
        if time.monotonic() >= deadline:
            raise PurgeLocked(f"{slug} is starting or finishing a turn; try again in a moment", code="busy")
        time.sleep(_ADMISSION_POLL_SECONDS)


def _release_admission_fence(raw, slug: str) -> None:
    """Release the session lock; a connection that can't is discarded, which also frees it."""
    try:
        raw.execution_options(isolation_level="AUTOCOMMIT")
        with raw.begin():
            released = raw.execute(text("SELECT pg_advisory_unlock(hashtext(:target))"), {"target": slug}).scalar_one()
        if not released:
            raw.invalidate()
    except Exception:  # noqa: BLE001 — never return a pooled connection that may still hold it
        raw.invalidate()


def _check_active(conn, slug: str, tables: set[str]) -> None:
    from .service.turn_admission import classify_target

    if {"turn_logs"} <= tables:
        # The admission authority's own view, under its lock: any open turn row
        # (running, reserved or unrecovered) blocks. Age is not proof of
        # termination; an abandoned turn must be recovered first.
        state = classify_target(conn, slug, exclude_turn_id=None)
        if not state.idle:
            raise PurgeLocked(f"{slug} has an open turn ({state.describe()}); stop it or wait for it to finish, "
                              "then preview again", code="active")
    if "inter_bot_deliveries" in tables:
        pending = conn.execute(text(
            "SELECT count(*) FROM inter_bot_deliveries WHERE target_bot_id=:s "
            "AND (status IS NULL OR status NOT IN ('DELIVERED','FAILED','CANCELLED'))"), {"s": slug}).scalar_one()
        if pending:
            raise PurgeLocked(f"{slug} has {pending} undelivered message(s) addressed to it; cancel them or wait, "
                              "then preview again", code="active")


def _purge_in_snapshot(raw, slug: str, manifest_digest: str, delete_profile: bool,
                       tables: set[str]) -> tuple[dict[str, Any], dict[str, int]]:
    """The confirmed purge in one REPEATABLE READ transaction (admission already fenced)."""
    from .prompt_purge import PromptPurgeGuard

    conn = raw.execution_options(isolation_level="REPEATABLE READ")
    with conn.begin():
        conn.execute(text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'"))
        conn.execute(text("LOCK TABLE bot_profiles, bot_profile_tags IN SHARE ROW EXCLUSIVE MODE"))
        prompt_purge = PromptPurgeGuard(conn, tables)
        try:
            prompt_purge.guard({slug})
        except ValueError as exc:
            raise PurgeLocked(str(exc), code="active") from exc
        current = collect_preview(conn, slug, delete_profile=delete_profile)
        if current["manifest_digest"] != manifest_digest:
            raise PurgeStale("The bot's data changed since the preview; preview again")
        if delete_profile and current["profile_version"]["state"] != "present":
            raise PurgeRefused(f"Bot profile {slug} not found", code="not_found")
        owned = _check_protected(conn, slug, current)
        _check_active(conn, slug, prompt_purge.tables)

        if not owned:
            # Unattributable message storage cannot be purged, and its rows
            # still reference this bot's sessions, so a partial purge would
            # either fail on the FK or misreport sessions.
            raise PurgeLocked(f"Message storage for {slug} can't be tied to this bot alone, so its data "
                              "can't be purged. Delete the profile only, or resolve the shared storage "
                              "first.", code="protected")
        deleted = delete_scoped_rows(conn, slug, prompt_purge)
        if delete_profile:
            deleted["bot_profile_tags"] = conn.execute(
                text("DELETE FROM bot_profile_tags WHERE bot_slug=:s"), {"s": slug}).rowcount
            deleted["bot_profiles"] = conn.execute(
                text("DELETE FROM bot_profiles WHERE slug=:s"), {"s": slug}).rowcount
            from .bot_policy_events import policy_changed
            policy_changed(conn)
    return current, deleted


def execute_bot_purge(engine, slug: str, *, manifest_digest: str | None, delete_profile: bool) -> dict[str, Any]:
    """Run one confirmed purge; raise ``PurgeRefused`` (nothing deleted) otherwise."""
    slug = normalize_slug(slug)
    if not manifest_digest or not _DIGEST.match(manifest_digest):
        raise PurgeStale("A manifest_digest from a fresh purge preview is required", code="preview_required")
    if engine is None or engine.dialect.name != "postgresql":
        raise PurgeUnavailable("PostgreSQL is unavailable")
    try:
        # Discover tables first, on its own connection: any query in the purge
        # transaction would fix its REPEATABLE READ snapshot before the locks.
        with engine.connect() as probe:
            tables = set(inspect(probe).get_table_names())
        with engine.connect() as raw:
            _acquire_admission_fence(raw, slug)
            try:
                current, deleted = _purge_in_snapshot(raw, slug, manifest_digest, delete_profile, tables)
            finally:
                _release_admission_fence(raw, slug)
    except PurgeRefused:
        raise
    except IntegrityError as exc:
        # Never echo SQL/parameters; the transaction is already rolled back.
        raise PurgeRefused("Purge blocked by dependent records owned by another bot", code="dependent") from exc
    except DBAPIError as exc:
        if getattr(exc.orig, "pgcode", None) in {"55P03", "40001", "40P01"}:
            raise PurgeLocked("Another change to this bot is in progress; preview again", code="busy") from exc
        raise PurgeUnavailable("PostgreSQL is unavailable") from exc
    except SQLAlchemyError as exc:
        raise PurgeUnavailable("PostgreSQL is unavailable") from exc

    evict_partition_caches(slug)
    return {
        "slug": slug,
        "action": current["scope"],
        "manifest_digest": manifest_digest,
        "profile_deleted": bool(delete_profile),
        "deleted_rows": deleted,
        "retained": [
            {k: e.get(k) for k in ("key", "store", "count", "reason")}
            for e in current["inventory"] if e["disposition"] != "deleted"
        ],
    }

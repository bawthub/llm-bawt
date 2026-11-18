"""Read-only classification of user-role history by trusted execution origin.

A user-role row is not proof a human typed it: schedules, wake-ups, inter-bot
deliveries and approval continuations are all persisted as ``role='user'``.
The trusted signal is the app-written delivery record for the message (its
``_request_identity_producer`` is overwritten by the app on admission), joined
by user message ID, plus durable authorship. Message text never raises
confidence; legacy frame matching only lowers it for rows with no origin.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import bindparam, text

from llm_bawt.utils.bot_partition import sanitize_table_name

HUMAN, BOT, AUTOMATION, UNKNOWN = "HUMAN", "BOT", "AUTOMATION", "UNKNOWN"
DIRECT_HUMAN, SCHEDULED, DELIVERY, CONTINUATION = (
    "DIRECT_HUMAN", "SCHEDULED", "DELIVERY", "CONTINUATION")
HIGH, MEDIUM, LOW = "high", "medium", "low"

# App-owned frames that legacy (origin-less) automation rows start with.
# Anchored at the start of content so a human quoting one mid-message is not
# matched.
_LEGACY_FRAME = re.compile(
    r"^\s*(\[Scheduled (wake-up|automation|prompt)|\[Reply channel:"
    r"|Message from bot '[^']+':|\[Approval (result|continuation))",
)
_COMMAND = re.compile(r"^\s*/[a-z][\w-]*(\s|$)")


@dataclass(frozen=True)
class Classification:
    message_id: str
    bot_id: str
    original_author: str | None  # "user:<id>" | "bot:<id>" | "system:<id>" | None
    occurrence: str
    cls: str
    confidence: str
    reason: str
    command: bool = False  # typed slash command: human, but not a preference
    source_ids: tuple[str, ...] = field(default_factory=tuple)


def _author(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    kind = (row.get("author_entity_type") or "").lower() or None
    ident = (row.get("author_entity_id") or "").lower() or None
    return kind, ident


def _producer(meta: Mapping[str, Any]) -> str:
    # Same precedence as RequestIdentityStore._validate.
    return meta.get("_request_identity_producer") or (
        "ordinary" if "ordinary_chat" in meta else
        "maintenance" if "maintenance_user_queue" in meta else
        "schedule" if "prompt_schedule" in meta else
        "retry" if "manual_retry_of" in meta else "bot")


def classify(row: Mapping[str, Any]) -> Classification:
    """Classify one projected row (message + optional delivery origin)."""
    kind, ident = _author(row)
    author = f"{kind}:{ident}" if kind else None
    content = row.get("content") or ""
    sources = tuple(s for s in (row.get("id"), row.get("delivery_id")) if s)

    def out(occ: str, cls: str, conf: str, reason: str, command: bool = False):
        return Classification(row["id"], row["bot_id"], author, occ, cls, conf,
                              reason, command, sources)

    if kind == "system":
        return out(CONTINUATION, AUTOMATION, HIGH, "app-owned continuation author")

    if row.get("delivery_id"):
        meta = row.get("delivery_metadata") or {}
        producer = _producer(meta)
        self_send = row.get("sender_bot_id") == row.get("target_bot_id")
        if producer == "schedule":
            return out(SCHEDULED, AUTOMATION, HIGH, "trusted schedule origin")
        if producer == "maintenance":
            return out(SCHEDULED, AUTOMATION, HIGH, "trusted maintenance origin")
        if producer == "ordinary" or (producer == "retry" and "ordinary_chat" in meta):
            if kind != "user":
                return out(DIRECT_HUMAN, UNKNOWN, LOW, "direct-chat origin without user author")
            conf = HIGH if producer == "ordinary" else MEDIUM
            if _COMMAND.match(content):
                return out(DIRECT_HUMAN, HUMAN, conf, "typed command", command=True)
            return out(DIRECT_HUMAN, HUMAN, conf, "trusted direct-chat origin")
        if self_send:
            return out(SCHEDULED, AUTOMATION, HIGH, "self-addressed wake-up")
        if kind == "bot":
            return out(DELIVERY, BOT, HIGH, "inter-bot delivery")
        return out(DELIVERY, UNKNOWN, MEDIUM, "user-attributed relay by a bot producer")

    # No origin record: durable authorship only, never high-confidence human.
    if kind == "bot":
        return out(UNKNOWN, BOT, HIGH, "durable bot authorship")
    if kind == "user":
        if _LEGACY_FRAME.match(content):
            return out(UNKNOWN, UNKNOWN, LOW, "legacy app frame without origin")
        if _COMMAND.match(content):
            return out(UNKNOWN, HUMAN, MEDIUM, "typed command without origin", command=True)
        return out(UNKNOWN, HUMAN, MEDIUM, "user authorship without origin")
    return out(UNKNOWN, UNKNOWN, LOW, "legacy row without authorship or origin")


# Excluded bots are filtered on the messages scan itself, so no content or
# delivery row of theirs is read or joined.
_QUERY = text("""
    SELECT m.id, m.bot_id, m.content, m.timestamp,
           m.author_entity_type, m.author_entity_id,
           d.id AS delivery_id, d.sender_bot_id, d.target_bot_id,
           d.metadata_json AS delivery_metadata
    FROM messages m
    LEFT JOIN inter_bot_deliveries d
      ON d.user_message_id = m.id AND d.target_bot_id IN :targets
    WHERE m.role = 'user'
      AND m.bot_id IN :parts
      AND m.bot_id NOT IN :excluded
      AND m.timestamp >= :since AND m.timestamp < :until
    ORDER BY m.timestamp
    LIMIT :limit
""").bindparams(bindparam("parts", expanding=True), bindparam("targets", expanding=True),
              bindparam("excluded", expanding=True))


def _partitions(bot_ids: Sequence[str], excluded_bot_ids: Sequence[str]) -> dict[str, str]:
    """Map storage partition -> canonical slug for the authorized, unprotected bots.

    messages.bot_id is the sanitized partition; deliveries use the canonical
    slug. Ambiguous aliases (two slugs on one partition, or an authorized slug
    sharing a protected bot's partition) are refused, not guessed.
    """
    if not excluded_bot_ids:
        raise ValueError("excluded_bot_ids must name the protected bots")
    excluded_canon = {b.lower() for b in excluded_bot_ids}
    excluded_parts = {sanitize_table_name(b) for b in excluded_canon}
    out: dict[str, str] = {}
    for slug in sorted({b.lower() for b in bot_ids} - excluded_canon):
        part = sanitize_table_name(slug)
        if part in excluded_parts:
            raise ValueError(f"bot {slug!r} shares a protected bot's storage partition")
        if part in out and out[part] != slug:
            raise ValueError(f"bots {out[part]!r} and {slug!r} share storage partition {part!r}")
        out[part] = slug
    if not out:
        raise ValueError("bot_ids must name at least one authorized bot")
    return out


def fetch_rows(conn: Any, *, bot_ids: Sequence[str], excluded_bot_ids: Sequence[str],
               since: float, until: float, limit: int = 5000) -> list[Mapping[str, Any]]:
    """Operator-only scan of the caller's authorized bots (canonical slugs).

    Not an authorized-history reader: rows are not scoped to a session owner.
    Protected bots and their partitions are removed before the query, so their
    rows are never read. A delivery only counts as a row's origin when its
    canonical target maps to that row's partition.
    """
    parts = _partitions(bot_ids, excluded_bot_ids)
    excluded_parts = sorted({sanitize_table_name(b.lower()) for b in excluded_bot_ids})
    rows = conn.execute(_QUERY, {
        "parts": sorted(parts), "targets": sorted(parts.values()),
        "excluded": excluded_parts,
        "since": since, "until": until, "limit": limit,
    }).mappings()
    # messages is keyed (bot_id, id): the same ID may exist in two partitions.
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        row = dict(r)
        canonical = parts[row["bot_id"]]
        matched = bool(row.get("delivery_id")) and row.get("target_bot_id") == canonical
        if not matched:
            for k in ("delivery_id", "sender_bot_id", "target_bot_id", "delivery_metadata"):
                row[k] = None
        row["bot_id"] = canonical
        key = (canonical, row["id"])
        prev = out.get(key)
        if prev is None or (matched and not prev.get("delivery_id")):
            out[key] = row
    return list(out.values())


def report(rows: Iterable[Mapping[str, Any]], samples: int = 3) -> dict[str, Any]:
    """Per-class counts with sample IDs and the coverage actually observed."""
    counts: dict[str, dict[str, Any]] = {}
    first = last = None
    n = 0
    for row in rows:
        c = classify(row)
        n += 1
        ts = row.get("timestamp")
        if ts is not None:
            first = ts if first is None else min(first, ts)
            last = ts if last is None else max(last, ts)
        key = f"{c.cls}/{c.occurrence}/{c.confidence}" + ("/command" if c.command else "")
        bucket = counts.setdefault(key, {"count": 0, "sample_ids": []})
        bucket["count"] += 1
        if len(bucket["sample_ids"]) < samples:
            bucket["sample_ids"].append(c.message_id)
    return {"rows_read": n, "first_ts": first, "last_ts": last,
            "classes": dict(sorted(counts.items(), key=lambda kv: -kv[1]["count"]))}

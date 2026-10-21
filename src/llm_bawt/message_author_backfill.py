"""Audited one-time backfill of canonical authorship on legacy message rows.

Rows written before ``author_entity_*`` existed (and composer steers written
before that path set an author) carry NULL authorship, which strict consumers
such as isolated recall treat as unknown provenance. This resolves them with
the same deterministic order as :class:`LegacyAuthorResolver` -- delivery
record, then role, then the owning session's single user -- and never parses
content into authority. Content that merely *claims* to come from a bot is the
one place it is stricter than the display resolver: such rows stay NULL.

Every change is recorded in ``message_author_backfill_audit`` inside the same
transaction as the update, so a batch is reversible and re-runs are no-ops.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Engine

AUDIT_TABLE = "message_author_backfill_audit"

# Content that claims bot authorship without a delivery record proving it.
# Anyone can type this, so it never establishes authority (stays NULL).
CLAIMED_BOT_PREFIX = r"^\s*\[?\s*(message|reply|response|handoff)\s+from\s+(bot|agent)\M"
# Legacy harness-injected user-role rows that no person or bot authored.
INJECTED_PREFIXES = ("SYSTEM PROMPT — APPLY", "Read HEARTBEAT.md")

_CLASSIFY = """
    SELECT m.bot_id, m.id,
      CASE
        WHEN d.user_message_id IS NOT NULL THEN 'delivery'
        WHEN m.role = 'assistant' THEN 'assistant'
        WHEN m.content ~* :claimed_prefix
          OR position('Message from bot ''' IN left(m.content, 300)) > 0 THEN 'claimed_bot'
        WHEN left(m.content, 32) LIKE ANY(CAST(:injected AS text[])) THEN 'injected'
        ELSE 'session_user'
      END AS rule,
      CASE
        WHEN d.user_message_id IS NOT NULL THEN d.author_entity_type
        WHEN m.role = 'assistant' THEN 'bot'
        ELSE 'user'
      END AS new_type,
      CASE
        WHEN d.user_message_id IS NOT NULL THEN lower(d.author_entity_id)
        WHEN m.role = 'assistant' THEN m.bot_id
        ELSE lower(s.user_id)
      END AS new_id
    FROM messages m
    JOIN sessions s ON s.id = m.session_id AND s.bot_id = m.bot_id AND s.user_id IS NOT NULL
    LEFT JOIN inter_bot_deliveries d
      ON d.user_message_id = m.id AND d.target_bot_id = m.bot_id
      AND d.author_entity_type IN ('user', 'bot') AND d.author_entity_id <> ''
    WHERE m.author_entity_type IS NULL AND m.author_entity_id IS NULL
      AND m.role IN ('user', 'assistant')
      AND NOT (m.bot_id = ANY(CAST(:excluded AS text[])))
"""
# Rules that are written; claimed_bot and injected are counted but stay NULL.
WRITTEN_RULES = ("delivery", "assistant", "session_user")


@dataclass(frozen=True)
class BackfillPlan:
    counts: dict[str, int]

    @property
    def writable(self) -> int:
        return sum(self.counts.get(rule, 0) for rule in WRITTEN_RULES)


class MessageAuthorBackfill:
    def __init__(self, engine: Engine, *, excluded_partitions: Iterable[str]):
        self.engine = engine
        self.excluded = sorted({p.strip().lower() for p in excluded_partitions if p.strip()})

    def _params(self) -> dict:
        return {
            "claimed_prefix": CLAIMED_BOT_PREFIX,
            "injected": [prefix + "%" for prefix in INJECTED_PREFIXES],
            "excluded": self.excluded,
        }

    def plan(self) -> BackfillPlan:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT rule, count(*) FROM ({_CLASSIFY}) c GROUP BY rule"), self._params()
            ).all()
        return BackfillPlan({rule: int(count) for rule, count in rows})

    def ensure_audit_table(self, conn) -> None:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {AUDIT_TABLE} (
                batch_id VARCHAR(36) NOT NULL,
                bot_id VARCHAR(100) NOT NULL,
                message_id VARCHAR(36) NOT NULL,
                rule VARCHAR(16) NOT NULL,
                old_type VARCHAR(16),
                old_id VARCHAR(100),
                new_type VARCHAR(16) NOT NULL,
                new_id VARCHAR(100) NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                reverted_at TIMESTAMPTZ,
                PRIMARY KEY (batch_id, bot_id, message_id)
            )
        """))

    def apply(self, *, expected: int) -> tuple[str, int]:
        """Write one audited batch; roll back unless exactly ``expected`` rows change."""
        batch_id = str(uuid.uuid4())
        with self.engine.begin() as conn:
            self.ensure_audit_table(conn)
            audited = conn.execute(text(f"""
                INSERT INTO {AUDIT_TABLE} (batch_id, bot_id, message_id, rule, new_type, new_id)
                SELECT :batch, c.bot_id, c.id, c.rule, c.new_type, c.new_id
                FROM ({_CLASSIFY}) c
                WHERE c.rule = ANY(CAST(:rules AS text[])) AND c.new_id IS NOT NULL
            """), {**self._params(), "batch": batch_id, "rules": list(WRITTEN_RULES)}).rowcount
            updated = conn.execute(text(f"""
                UPDATE messages m
                SET author_entity_type = a.new_type, author_entity_id = a.new_id
                FROM {AUDIT_TABLE} a
                WHERE a.batch_id = :batch AND m.bot_id = a.bot_id AND m.id = a.message_id
                  AND m.author_entity_type IS NULL AND m.author_entity_id IS NULL
            """), {"batch": batch_id}).rowcount
            if not (audited == updated == expected):
                raise RuntimeError(
                    f"backfill count mismatch: expected={expected} audited={audited} updated={updated}"
                )
        return batch_id, updated

    def revert(self, batch_id: str) -> int:
        """Restore authorship for rows this batch set that nothing has since changed.

        Only rows actually restored are marked reverted, and excluded partitions
        are never written even if an audit batch names them.
        """
        with self.engine.begin() as conn:
            return conn.execute(text(f"""
                WITH restored AS (
                    UPDATE messages m
                    SET author_entity_type = a.old_type, author_entity_id = a.old_id
                    FROM {AUDIT_TABLE} a
                    WHERE a.batch_id = :batch AND a.reverted_at IS NULL
                      AND NOT (m.bot_id = ANY(CAST(:excluded AS text[])))
                      AND m.bot_id = a.bot_id AND m.id = a.message_id
                      AND m.author_entity_type = a.new_type AND m.author_entity_id = a.new_id
                    RETURNING m.bot_id, m.id
                )
                UPDATE {AUDIT_TABLE} a SET reverted_at = now()
                FROM restored r
                WHERE a.batch_id = :batch AND a.bot_id = r.bot_id AND a.message_id = r.id
            """), {"batch": batch_id, "excluded": self.excluded}).rowcount

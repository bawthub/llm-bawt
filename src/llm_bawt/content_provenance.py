"""Durable per-turn exposure and data-minimal bot-write admission receipts."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, MetaData, String, Table, func, text

from .utils.schema import SchemaBootstrapGuard

_metadata = MetaData()
_turns = Table(
    "agent_content_turns", _metadata,
    Column("id", String(64), primary_key=True),
    Column("bot_id", String(128), nullable=False),
    Column("user_id", String(128), nullable=False),
    Column("turn_id", String(128), nullable=False),
    Column("trigger_message_id", String(128), nullable=False),
    Column("untrusted_seen", Boolean, nullable=False),
)
_edges = Table(
    "agent_content_lineage", _metadata,
    Column("target_id", String(64), primary_key=True),
    Column("source_id", String(64), primary_key=True),
)
_writes = Table(
    "agent_content_write_decisions", _metadata,
    Column("id", String(36), primary_key=True),
    Column("source_turn", String(128), nullable=False),
    Column("bot_id", String(128), nullable=False),
    Column("user_id", String(128), nullable=False),
    Column("trigger_message_id", String(128), nullable=False),
    Column("tool_name", String(128), nullable=False),
    Column("write_kind", String(32), nullable=False),
    Column("action", String(32), nullable=False),
    Column("invocation_hash", String(64), nullable=False),
    Column("untrusted_seen", Boolean, nullable=False),
    Column("decision", String(16), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)
_guard = SchemaBootstrapGuard()


def _ensure_schema(conn) -> None:
    if conn.dialect.name != "postgresql":
        _metadata.create_all(conn)
        return
    from .bot_tags.catalog import owned, require_owned

    for table in _metadata.sorted_tables:
        if not owned(conn, table.name):
            table.create(conn)
        require_owned(conn, table.name)


@dataclass(frozen=True)
class ContentTurn:
    bot_id: str
    user_id: str
    turn_id: str
    trigger_message_id: str

    @property
    def key(self) -> str:
        values = (self.bot_id, self.user_id, self.turn_id, self.trigger_message_id)
        if any(not v or len(v) > 128 for v in values):
            raise ValueError("complete server-owned content turn identity required")
        return sha256("\0".join(values[:3]).encode()).hexdigest()


@dataclass(frozen=True)
class WriteAdmission:
    receipt_id: str
    source_turn: str
    untrusted_seen: bool

    @property
    def held(self) -> bool:
        return self.untrusted_seen


class ContentProvenanceStore:
    """No proposed write text, release grants or automatic replay is stored."""

    def __init__(self, engine):
        if engine is None:
            raise RuntimeError("content provenance database unavailable")
        self.engine = engine
        _guard.run(engine, "agent-content-provenance", _ensure_schema)

    def _turn(self, conn, turn: ContentTurn) -> str:
        key = turn.key
        conn.execute(text("""
            INSERT INTO agent_content_turns
                (id,bot_id,user_id,turn_id,trigger_message_id,untrusted_seen)
            VALUES (:id,:bot,:user,:turn,:trigger,false)
            ON CONFLICT (id) DO NOTHING
        """), {"id": key, "bot": turn.bot_id, "user": turn.user_id,
               "turn": turn.turn_id, "trigger": turn.trigger_message_id})
        query = "SELECT trigger_message_id FROM agent_content_turns WHERE id=:id"
        if conn.dialect.name == "postgresql":
            query += " FOR UPDATE"
        trigger = conn.execute(text(query), {"id": key}).scalar_one()
        if trigger != turn.trigger_message_id:
            raise ValueError("content turn trigger identity changed")
        return key

    def expose(self, turn: ContentTurn) -> None:
        """Commit before dispatching an external read, including error paths."""
        with self.engine.begin() as conn:
            key = self._turn(conn, turn)
            conn.execute(text("""
                UPDATE agent_content_turns SET untrusted_seen=true WHERE id=:id
            """), {"id": key})

    def inherit(self, source: ContentTurn, target: ContentTurn) -> None:
        """Link only server-verified delivery/continuation lineage."""
        with self.engine.begin() as conn:
            # Stable lock ordering also handles reciprocal bot deliveries.
            for turn in sorted((source, target), key=lambda t: t.key):
                self._turn(conn, turn)
            conn.execute(text("""
                INSERT INTO agent_content_lineage (source_id,target_id)
                VALUES (:source,:target) ON CONFLICT DO NOTHING
            """), {"source": source.key, "target": target.key})

    @staticmethod
    def _seen(conn, key: str) -> bool:
        # UNION (not ALL) terminates cycles; only server-authored edges exist.
        return bool(conn.execute(text("""
            WITH RECURSIVE origins(id) AS (
                SELECT CAST(:id AS TEXT)
                UNION
                SELECT CAST(e.source_id AS TEXT) FROM agent_content_lineage e
                JOIN origins o ON e.target_id=o.id
            )
            SELECT EXISTS(
                SELECT 1 FROM agent_content_turns t
                JOIN origins o ON t.id=o.id WHERE t.untrusted_seen=true
            )
        """), {"id": key}).scalar_one())

    def seen(self, turn: ContentTurn) -> bool:
        with self.engine.begin() as conn:
            return self._seen(conn, self._turn(conn, turn))

    def resolve(self, bot_id: str, user_id: str, turn_id: str) -> ContentTurn:
        """Recover a previously verified turn for approved server execution."""
        with self.engine.connect() as conn:
            row = conn.execute(
                _turns.select().where(
                    _turns.c.bot_id == bot_id, _turns.c.user_id == user_id,
                    _turns.c.turn_id == turn_id,
                )
            ).mappings().one_or_none()
        if row is None:
            raise ValueError("verified content turn unavailable")
        return ContentTurn(bot_id, user_id, turn_id, row["trigger_message_id"])

    def admit(
        self, turn: ContentTurn, *, tool_name: str, write_kind: str,
        invocation_hash: str, action: str = "",
    ) -> WriteAdmission:
        """Linearize the decision against exposure; receipt is not execution."""
        receipt_id = str(uuid4())
        with self.engine.begin() as conn:
            key = self._turn(conn, turn)
            exposed = self._seen(conn, key)
            conn.execute(_writes.insert().values(
                id=receipt_id, source_turn=turn.turn_id, bot_id=turn.bot_id,
                user_id=turn.user_id, trigger_message_id=turn.trigger_message_id,
                tool_name=tool_name, write_kind=write_kind, action=action[:32],
                invocation_hash=invocation_hash, untrusted_seen=exposed,
                decision="held" if exposed else "allowed",
            ))
        return WriteAdmission(receipt_id, turn.turn_id, exposed)

"""Durable home-audio FIFO and one global playback lease (including groups)."""
from __future__ import annotations

import hashlib
import json
import time
import uuid

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text, select, update
from sqlalchemy.exc import IntegrityError

from llm_bawt.service.commit_wake import notify_in_transaction
from llm_bawt.utils.schema import SchemaBootstrapGuard

_metadata = MetaData()
_jobs = Table(
    "home_audio_jobs", _metadata,
    Column("sequence", Integer, primary_key=True, autoincrement=True),
    Column("id", String(32), unique=True, nullable=False),
    Column("request_key", String(64), unique=True, nullable=False),
    Column("payload", Text, nullable=False),
    Column("status", String(32), nullable=False),
    Column("created_at", Float, nullable=False),
    Column("expires_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    Column("asset_id", String(64)),
    Column("error", Text),
)
_lane = Table(
    "home_audio_lane", _metadata,
    Column("id", Integer, primary_key=True),
    Column("owner", String(32)),
    Column("lease_until", Float, nullable=False),
)
TERMINAL = {"completed", "failed", "cancelled", "expired", "interrupted"}


class HomeAudioStore:
    """SQL transactions only; never hold a transaction across network work."""

    _guard = SchemaBootstrapGuard()

    def __init__(self, engine):
        self.engine = engine
        self._guard.run(engine, "home-audio-v1", lambda conn: _metadata.create_all(conn))
        try:
            with engine.begin() as conn:
                conn.execute(_lane.insert().values(id=1, owner=None, lease_until=0))
        except IntegrityError:
            pass

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        result = dict(row)
        result["request"] = json.loads(result.pop("payload"))
        result.pop("request_key")
        result.pop("sequence")
        return result

    def enqueue(self, payload: dict, key: str, ttl: int = 300) -> dict:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        request_key = hashlib.sha256(f"{payload['bot_id']}\0{key}".encode()).hexdigest()
        now = time.time()
        values = dict(id=uuid.uuid4().hex, request_key=request_key, payload=canonical,
                      status="queued", created_at=now, updated_at=now, expires_at=now + ttl)
        try:
            with self.engine.begin() as conn:
                conn.execute(_jobs.insert().values(**values))
                notify_in_transaction(conn, "home_audio")  # wakes the app worker at commit (TASK-2108)
        except IntegrityError:
            with self.engine.connect() as conn:
                row = conn.execute(select(_jobs).where(_jobs.c.request_key == request_key)).mappings().one()
            if row["payload"] != canonical:
                raise ValueError("Idempotency key already used for a different announcement")
            return self._decode(row)
        return self.get(values["id"])

    def get(self, job_id: str) -> dict | None:
        with self.engine.connect() as conn:
            return self._decode(conn.execute(select(_jobs).where(_jobs.c.id == job_id)).mappings().first())

    def cancel(self, job_id: str) -> dict | None:
        with self.engine.begin() as conn:
            conn.execute(update(_jobs).where(_jobs.c.id == job_id, _jobs.c.status == "queued")
                         .values(status="cancelled", updated_at=time.time()))
        return self.get(job_id)

    def claim(self, owner: str) -> dict | None:
        now = time.time()
        with self.engine.begin() as conn:
            acquired = conn.execute(update(_lane).where(_lane.c.id == 1, _lane.c.lease_until < now)
                                    .values(owner=owner, lease_until=now + 60)).rowcount
            if not acquired:
                return None
            # A crashed worker may have dispatched already. Never replay uncertain speech.
            uncertain_playback = conn.execute(select(_jobs.c.id).where(_jobs.c.status == "playing").limit(1)).first()
            conn.execute(update(_jobs).where(_jobs.c.status.in_(["preparing", "playing"]))
                         .values(status="interrupted", error="Worker lease expired; playback outcome unknown; not replayed", updated_at=now))
            if uncertain_playback:
                # Groups can overlap individual speakers without exposing membership.
                # Quarantine the whole lane for the maximum clip length after a crash.
                conn.execute(update(_lane).where(_lane.c.id == 1).values(owner=None, lease_until=now + 330))
                return None
            conn.execute(update(_jobs).where(_jobs.c.status == "queued", _jobs.c.expires_at <= now)
                         .values(status="expired", updated_at=now))
            row = conn.execute(select(_jobs).where(_jobs.c.status == "queued")
                               .order_by(_jobs.c.sequence).limit(1)).mappings().first()
            if row is None:
                conn.execute(update(_lane).where(_lane.c.id == 1).values(owner=None, lease_until=0))
                return None
            conn.execute(update(_jobs).where(_jobs.c.id == row["id"])
                         .values(status="preparing", updated_at=now))
            result = self._decode(row)
            result["status"] = "preparing"
            return result

    def while_owned(self, owner: str, action):
        """Fence a mutable upload against takeover by another queue worker.

        A cancelled asyncio.to_thread upload can continue running. Holding the
        lane row lock for this bounded write prevents a successor from replacing
        the slot and playing while that old upload is still in flight.
        """
        now = time.time()
        with self.engine.begin() as conn:
            if not conn.execute(update(_lane).where(
                _lane.c.id == 1, _lane.c.owner == owner, _lane.c.lease_until > now,
            ).values(lease_until=now + 60)).rowcount:
                raise RuntimeError("Playback lease lost before replacing audio")
            result = action()
            conn.execute(update(_lane).where(_lane.c.id == 1).values(lease_until=time.time() + 60))
            return result

    def renew(self, owner: str) -> bool:
        now = time.time()
        with self.engine.begin() as conn:
            return bool(conn.execute(update(_lane).where(
                _lane.c.id == 1, _lane.c.owner == owner, _lane.c.lease_until > now,
            ).values(lease_until=now + 60)).rowcount)

    def save(self, owner: str, job_id: str, status: str, *, asset_id=None, error=None) -> bool:
        now = time.time()
        with self.engine.begin() as conn:
            # Lock/refresh lease before changing job state; stale workers cannot write.
            if not conn.execute(update(_lane).where(
                _lane.c.id == 1, _lane.c.owner == owner, _lane.c.lease_until > now,
            ).values(lease_until=now + 60)).rowcount:
                return False
            values = dict(status=status, updated_at=now, error=error)
            if asset_id is not None:
                values["asset_id"] = asset_id
            conn.execute(update(_jobs).where(_jobs.c.id == job_id, ~_jobs.c.status.in_(TERMINAL)).values(**values))
            if status in TERMINAL:
                conn.execute(update(_lane).where(_lane.c.id == 1).values(owner=None, lease_until=0))
            return True

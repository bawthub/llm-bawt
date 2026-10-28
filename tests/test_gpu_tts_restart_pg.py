"""TASK-1171: the TTS restart fence and its job commit as one PostgreSQL transaction.

Concurrent creates race on the handoff row's FOR UPDATE lock: one request
begins the fence; a replay of the same key reads that job, and any other key
is refused with no job row.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import select

from llm_bawt.media.gpu_handoff_store import (
    TTS_RESTART_OPERATION, GpuHandoffFence, GpuHandoffStore, HandoffConflict, _state,
)
from llm_bawt.ops import OpsService
from llm_bawt.ops.models import OpsJob
from tests.test_gpu_tts_restart import TTS_SPEC, FakeExecutor

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig(private_app_pg):
    from llm_bawt.ops.store import OpsStore

    store = OpsStore(private_app_pg)
    if store.get_operation_by_slug(TTS_RESTART_OPERATION) is None:
        store.create_operation({"slug": TTS_RESTART_OPERATION, "enabled": True,
                                "command_script": json.dumps(TTS_SPEC),
                                "args_schema_json": {"type": "object", "properties": {},
                                                     "additionalProperties": False}}, actor="t")
    handoff = GpuHandoffStore(store.engine)

    def reset():
        with store.engine.begin() as conn:
            conn.execute(_state.update().where(_state.c.id == 1).values(
                owner="voice", phase="idle", generation=0, tts_restart_key=None))
            conn.execute(OpsJob.__table__.delete().where(OpsJob.operation_slug == TTS_RESTART_OPERATION))

    reset()
    yield store, handoff, OpsService(store, executor=FakeExecutor())
    reset()


def _race(service, keys):
    barrier = threading.Barrier(len(keys))

    def create(key):
        barrier.wait()
        try:
            return service.dispatch_job(operation_slug=TTS_RESTART_OPERATION, args={},
                                        idempotency_key=key, caller_backend="http-operator",
                                        create_fence=GpuHandoffFence.tts_restart(key))["id"]
        except HandoffConflict as exc:
            return exc

    with ThreadPoolExecutor(len(keys)) as pool:
        return list(pool.map(create, keys))


def _jobs(store):
    with store.engine.connect() as conn:
        return conn.execute(select(OpsJob.idempotency_key).where(
            OpsJob.operation_slug == TTS_RESTART_OPERATION)).scalars().all()


def test_same_key_race_begins_the_fence_once(rig):
    store, handoff, service = rig
    results = _race(service, ["tts-restart:a"] * 4)
    assert all(isinstance(r, str) for r in results) and len(set(results)) == 1
    state = handoff.status()
    assert (state.phase, state.tts_restart_key, state.generation) == ("tts_restarting", "tts-restart:a", 1)
    assert _jobs(store) == ["tts-restart:a"]


def test_different_keys_race_admits_exactly_one(rig):
    store, handoff, service = rig
    keys = [f"tts-restart:{n}" for n in range(4)]
    results = _race(service, keys)
    winners = [key for key, r in zip(keys, results, strict=True) if isinstance(r, str)]
    assert len(winners) == 1
    assert all(r.code == "tts_restart_in_progress" for r in results if not isinstance(r, str))
    assert _jobs(store) == winners and handoff.status().tts_restart_key == winners[0]

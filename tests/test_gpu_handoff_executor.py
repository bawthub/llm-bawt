"""Forward handoff contract: no containers, network, or GPU work in these tests."""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from llm_bawt.media.gpu_handoff import (
    CALIBRATION_PROFILE, START_OPERATIONS, STOP_OPERATIONS, GpuHandoff, HandoffAdapter,
)
from llm_bawt.media.gpu_handoff_store import GpuHandoffStore, HandoffConflict
from llm_bawt.service.routes import media_handoff


class FakeAdapter(HandoffAdapter):
    def __init__(self):
        self.voice = {"active_count": 0, "active_gpu_work": 0, "parked": False, "park_id": None}
        self.stopped = []
        self.effects = []
        self.denied = False
        self.failure = None
        self.stale = False
        self.retain_memory = False
        self.lose_lease = False
        self.block_park = False
        self.failed_job = False
        self.jobs = {}
        self.render_active = False
        self.unhealthy = False

    async def observe(self):
        timestamp = datetime.now(UTC) - timedelta(seconds=60 if self.stale else 0)
        free = 15000 if len(self.stopped) == 2 and not self.retain_memory else 3800
        return {"gpu": {"ready": True, "observed_at": timestamp.isoformat(), "uuid": "GPU-one",
                        "total_mib": 16303, "free_mib": free},
                "voice": deepcopy(self.voice),
                "model": {"installed": True, "worker_ready": True, "resident": False,
                          "worker_uncertain": False, "active": self.render_active, "queued": 0},
                "services": {s: {"id": s, "running": s not in self.stopped,
                                 "status": "exited" if s in self.stopped else "running",
                                 "health": "unhealthy" if self.unhealthy and s not in self.stopped else "healthy"}
                             for s in ("stt", "tts")}}

    async def prepare_operations(self, slugs=STOP_OPERATIONS):
        return [{"operation": {"slug": slug}} for slug in slugs]

    async def check_operations(self, snapshots, slugs=STOP_OPERATIONS):
        if self.denied:
            raise HandoffConflict("Denied")
        assert [snapshot["operation"]["slug"] for snapshot in snapshots] == list(slugs)

    async def unpark(self, lease_id):
        assert lease_id == self.voice["park_id"]
        self.effects.append("unpark")
        self.voice.update(parked=False, park_id=None)

    async def reset_video(self):
        self.effects.append("reset_video")

    async def park(self):
        if self.block_park:
            raise HandoffConflict("Voice arrived before park")
        self.effects.append("park")
        self.voice.update(parked=True, park_id="a" * 32)
        return "a" * 32

    async def dispatch(self, snapshot, *, user, generation):
        slug = snapshot["operation"]["slug"]
        self.effects.append(slug)
        if self.failure:
            raise self.failure
        service = slug.rsplit("-", 1)[1]
        if "start-" in slug:
            if service in self.stopped:
                self.stopped.remove(service)
        else:
            self.stopped.append(service)
        job = {"id": service, "operation": slug, "terminal": True,
               "state": "failed" if self.failed_job else "succeeded", "exit_code": 1 if self.failed_job else 0}
        self.jobs[service] = job
        if self.lose_lease:
            self.voice.update(parked=False, park_id=None)
        return job

    async def job_status(self, job_id):
        return self.jobs[job_id]


@pytest.fixture
def setup(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/handoff.db", connect_args={"check_same_thread": False})
    store, adapter = GpuHandoffStore(engine), FakeAdapter()
    yield store, adapter, GpuHandoff(store, adapter, poll_interval=0, job_timeout=0.1)
    engine.dispose()


async def offer(controller):
    return await controller.offer(user="nick", expected_generation=0, calibration=True)


async def confirm(controller, offered, **overrides):
    args = dict(user="nick", token=offered["token"], expected_generation=offered["generation"], approved=True)
    return await controller.confirm(**(args | overrides))


def test_scoped_calibration_success_and_normal_render_gate(setup):
    store, adapter, controller = setup

    async def run():
        offered = await offer(controller)
        assert adapter.effects == []
        assert offered["calibration_profile"] == CALIBRATION_PROFILE
        result = await confirm(controller, offered)
        assert result["owner"] == "video_calibration"
        assert result["ordinary_render_ready"] is False
        with pytest.raises(HandoffConflict):
            await confirm(controller, offered)
        return offered

    offered = asyncio.run(run())
    assert adapter.effects == ["park", *STOP_OPERATIONS]
    assert store.plan(offered["generation"])["voice_lease"] == "a" * 32
    assert store.status().owner == "video_calibration"
    with pytest.raises(HandoffConflict):
        store.claim_video("ordinary")
    assert store.recover_orphaned_video(worker_restarted=True).phase == "recovery_required"


@pytest.mark.parametrize("changes", [{"approved": False}, {"user": "other"}, {"token": "wrong"}, {"expected_generation": 99}])
def test_invalid_or_denied_confirmation_has_no_effects(setup, changes):
    store, adapter, controller = setup

    async def run():
        offered = await offer(controller)
        with pytest.raises(HandoffConflict):
            await confirm(controller, offered, **changes)
    asyncio.run(run())
    assert not adapter.effects
    assert store.status().phase == "offered"


@pytest.mark.parametrize("blocker", ["active", "gpu_work", "stale", "policy", "parked"])
def test_fresh_confirm_preflight_blocks_without_mutations(setup, blocker):
    store, adapter, controller = setup

    async def run():
        offered = await offer(controller)
        if blocker == "active":
            adapter.voice["active_count"] = 1
        if blocker == "gpu_work":
            adapter.voice["active_gpu_work"] = 1
        if blocker == "stale":
            adapter.stale = True
        if blocker == "policy":
            adapter.denied = True
        if blocker == "parked":
            adapter.voice["parked"] = True
        with pytest.raises(HandoffConflict):
            await confirm(controller, offered)
    asyncio.run(run())
    assert not adapter.effects
    assert store.status().phase == "offered"


@pytest.mark.parametrize("failure", ["ops", "memory", "lease", "park_race", "failed_receipt", "cancel"])
def test_partial_failure_retains_recovery_evidence_and_never_replays(setup, failure):
    store, adapter, controller = setup

    async def run():
        offered = await offer(controller)
        if failure == "ops":
            adapter.failure = RuntimeError("lost dispatch response")
        if failure == "memory":
            adapter.retain_memory = True
        if failure == "lease":
            adapter.lose_lease = True
        if failure == "park_race":
            adapter.block_park = True
        if failure == "failed_receipt":
            adapter.failed_job = True
        if failure == "cancel":
            adapter.failure = asyncio.CancelledError()
        with pytest.raises((RuntimeError, HandoffConflict, asyncio.CancelledError)):
            await confirm(controller, offered)
        effects = list(adapter.effects)
        with pytest.raises(HandoffConflict):
            await confirm(controller, offered)
        assert adapter.effects == effects
    asyncio.run(run())
    assert store.status().owner == "unknown"
    assert store.status().phase == "recovery_required"
    assert store.status().pending_action is not None
    if failure not in ("ops", "cancel", "park_race"):
        assert store.status().last_job_id


def test_expired_and_replaced_offers_do_not_run(setup, monkeypatch):
    store, adapter, controller = setup

    async def run():
        old = await offer(controller)
        new = await controller.offer(user="nick", expected_generation=old["generation"], calibration=True)
        with pytest.raises(HandoffConflict):
            await confirm(controller, old)
        monkeypatch.setattr("llm_bawt.media.gpu_handoff_store.time.time", lambda: new["expires_at"] + 1)
        with pytest.raises(HandoffConflict):
            await confirm(controller, new)
    asyncio.run(run())
    assert not adapter.effects


def test_ordinary_offer_does_not_invent_a_memory_budget(setup):
    store, adapter, controller = setup
    with pytest.raises(HandoffConflict, match="measured profile"):
        asyncio.run(controller.offer(user="nick", expected_generation=0, calibration=False))
    assert store.status().generation == 0
    assert not adapter.effects


def test_concurrent_confirmation_has_one_executor(setup):
    store, adapter, controller = setup

    async def run():
        offered = await offer(controller)
        results = await asyncio.gather(confirm(controller, offered), confirm(controller, offered), return_exceptions=True)
        assert sum(isinstance(result, dict) for result in results) == 1
        assert sum(isinstance(result, HandoffConflict) for result in results) == 1
    asyncio.run(run())
    assert adapter.effects == ["park", *STOP_OPERATIONS]
    assert store.status().owner == "video_calibration"


def test_operator_routes_expose_the_tested_contract(setup, monkeypatch):
    store, adapter, controller = setup

    @asynccontextmanager
    async def factory():
        yield controller
    monkeypatch.setattr(media_handoff, "_controller", factory)
    app = FastAPI()
    app.include_router(media_handoff.router)
    with TestClient(app) as client:
        path = "/v1/media/local-video/handoff"
        assert client.post(path + "/offer", json={"user": "nick", "expected_generation": 0}).status_code == 409
        offered = client.post(path + "/offer", json={"user": "nick", "expected_generation": 0, "calibration": True})
        assert offered.status_code == 200
        data = offered.json()
        payload = {"user": "nick", "expected_generation": data["generation"], "token": data["token"], "approved": True}
        assert client.post(path + "/confirm", json=payload).json()["owner"] == "video_calibration"
        assert client.post(path + "/confirm", json=payload).status_code == 409
        assert client.post(path + "/confirm", json=payload | {"approved": "true"}).status_code == 422


async def restore(controller, store):
    offered = await controller.offer_restore(user="nick", expected_generation=store.status().generation)
    generation = await controller.accept_restore(user="nick", token=offered["token"],
                                                 expected_generation=offered["generation"], approved=True)
    await controller.execute_restore(user="nick", generation=generation)
    return offered


def test_restore_voice_after_video_switch(setup):
    store, adapter, controller = setup

    async def run():
        await confirm(controller, await offer(controller))
        adapter.effects.clear()
        offered = await restore(controller, store)
        assert offered["operations"] == list(START_OPERATIONS)

    asyncio.run(run())
    assert adapter.effects == ["reset_video", *START_OPERATIONS, "unpark"]
    state = store.status()
    assert (state.owner, state.phase) == ("voice", "idle")
    assert adapter.voice["parked"] is False and adapter.stopped == []


def test_restore_recovers_fenced_state_and_clears_stale_claims(setup):
    store, adapter, controller = setup

    async def run():
        await confirm(controller, await offer(controller))
        store.claim_calibration("job1", generation=store.status().generation,
                                profile={**CALIBRATION_PROFILE, "image_conditioned": False})
        store.recover_orphaned_video(worker_restarted=True)
        assert store.status().phase == "recovery_required"
        # Worker reset (the adapter's job) is what clears claims live.
        adapter.reset_video = lambda: asyncio.to_thread(store.release_all_video_claims)
        await restore(controller, store)

    asyncio.run(run())
    state = store.status()
    assert (state.owner, state.phase) == ("voice", "idle")
    assert store.active_video_jobs() == 0
    # And the forward switch is available again.
    asyncio.run(controller.offer(user="nick", expected_generation=state.generation, calibration=True))


def test_restore_refuses_while_rendering(setup):
    store, adapter, controller = setup
    asyncio.run(confirm(controller, asyncio.run(offer(controller))))
    adapter.render_active = True
    with pytest.raises(HandoffConflict, match="still running"):
        asyncio.run(controller.offer_restore(user="nick", expected_generation=store.status().generation))
    assert store.status().owner == "video_calibration"


def test_unhealthy_moshi_fences_with_reason(setup):
    store, adapter, controller = setup
    asyncio.run(confirm(controller, asyncio.run(offer(controller))))
    adapter.unhealthy = True
    with pytest.raises(HandoffConflict, match="unhealthy"):
        asyncio.run(restore(controller, store))
    state = store.status()
    assert state.phase == "recovery_required"
    assert "Switch to voice failed" in state.last_error and "unhealthy" in state.last_error
    # Voice stays parked: nothing reopened admission on a failed start.
    assert adapter.voice["parked"] is True and "unpark" not in adapter.effects


def _fenced_after_calibration_switch(setup, monkeypatch):
    """Voice parked + Moshi stopped under lease 'a'*32, then a worker restart fence."""
    store, adapter, controller = setup
    asyncio.run(confirm(controller, asyncio.run(offer(controller))))
    store.recover_orphaned_video(worker_restarted=True)
    monkeypatch.setattr(store, "calibration", lambda: {"supported": True})
    state = store.status()
    assert state.phase == "recovery_required"
    return store, adapter, controller, state


def test_resume_video_from_recovery_verifies_live_state_without_side_effects(setup, monkeypatch):
    store, adapter, controller, fenced = _fenced_after_calibration_switch(setup, monkeypatch)
    effects = list(adapter.effects)
    result = asyncio.run(controller.resume_video(user="nick", expected_generation=fenced.generation))
    assert result["owner"] == "video"
    assert adapter.effects == effects  # no park/stop/start/reset
    state = store.status()
    assert (state.owner, state.phase, state.last_error) == ("video", "idle", None)
    store.claim_video("after-resume")


@pytest.mark.parametrize("drift", ["moshi_running", "lease_lost", "stale_generation", "claims", "uncalibrated"])
def test_resume_video_refuses_when_live_state_is_not_videos(setup, monkeypatch, drift):
    store, adapter, controller, fenced = _fenced_after_calibration_switch(setup, monkeypatch)
    generation = fenced.generation
    if drift == "moshi_running":
        adapter.stopped.remove("tts")
    elif drift == "lease_lost":
        adapter.voice.update(parked=False, park_id=None)
    elif drift == "stale_generation":
        generation -= 1
    elif drift == "claims":
        with store.engine.begin() as conn:
            from llm_bawt.media.gpu_handoff_store import _video_jobs
            conn.execute(_video_jobs.insert().values(id="orphan", generation=generation))
    else:
        monkeypatch.setattr(store, "calibration", lambda: None)
    with pytest.raises(HandoffConflict):
        asyncio.run(controller.resume_video(user="nick", expected_generation=generation))
    assert store.status().phase == "recovery_required"

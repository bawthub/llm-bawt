"""Scoped GPU handoff between Moshi voice and local Wan video.

Forward: park voice, stop Moshi STT/TTS, verify released VRAM. The first
acquisition is a calibration reservation, not ordinary render admission.
Restore: reset the Wan worker, start Moshi, wait for health, unpark voice. It
is also the recovery path: it may start from ``recovery_required`` because every
step converges on a known state (docker start is idempotent, reset ends the
worker). Nothing retries automatically; each run needs fresh consent.
Resume: leave recovery back to video with no side effects, when live state
still is video's (voice parked under our lease, Moshi stopped, Wan idle).
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from .gpu_handoff_store import GpuHandoffStore, HandoffConflict
from .gpu_profile import CALIBRATION_PROFILE

STOP_OPERATIONS = ("bawthub.stop-moshi-stt", "bawthub.stop-moshi-tts")
START_OPERATIONS = ("bawthub.start-moshi-stt", "bawthub.start-moshi-tts")
ACTIONS = ("park_voice", "stop_moshi_stt", "stop_moshi_tts", "verify_video_capacity")
RESTORE_ACTIONS = ("reset_video_worker", "start_moshi_stt", "start_moshi_tts", "verify_voice", "unpark_voice")
SERVICES = ("stt", "tts")

logger = logging.getLogger(__name__)


class HandoffAdapter(ABC):
    """External observations/actions; tests substitute controlled adapters."""

    @abstractmethod
    async def observe(self) -> dict: ...

    @abstractmethod
    async def prepare_operations(self, slugs: tuple[str, ...] = STOP_OPERATIONS) -> list[dict]: ...

    @abstractmethod
    async def check_operations(self, snapshots: list[dict], slugs: tuple[str, ...] = STOP_OPERATIONS) -> None: ...

    @abstractmethod
    async def park(self) -> str: ...

    @abstractmethod
    async def unpark(self, lease_id: str) -> None: ...

    @abstractmethod
    async def reset_video(self) -> None: ...

    @abstractmethod
    async def dispatch(self, snapshot: dict, *, user: str, generation: int) -> dict: ...

    @abstractmethod
    async def job_status(self, job_id: str) -> dict: ...


class GpuHandoff:
    def __init__(self, store: GpuHandoffStore, adapter: HandoffAdapter, *, poll_interval: float = 1,
                 job_timeout: float = 150, health_timeout: float = 300):
        self.store, self.adapter = store, adapter
        self.poll_interval, self.job_timeout = poll_interval, job_timeout
        self.health_timeout = health_timeout

    @staticmethod
    def _check(observed: dict, *, stopped: tuple[str, ...] = (), lease: str | None = None,
               original: dict | None = None) -> None:
        gpu, model, voice, services = (observed[k] for k in ("gpu", "model", "voice", "services"))
        if gpu.get("ready") is not True or not isinstance(gpu.get("uuid"), str) or not gpu["uuid"].startswith("GPU-"):
            raise HandoffConflict("GPU driver telemetry unavailable")
        try:
            age = (datetime.now(UTC) - datetime.fromisoformat(gpu["observed_at"])).total_seconds()
            total, free = gpu["total_mib"], gpu["free_mib"]
            valid = (0 <= age <= 15 and type(total) is int and type(free) is int and 0 <= free <= total and total > 0)
        except (TypeError, ValueError, KeyError):
            valid = False
        if not valid:
            raise HandoffConflict("GPU telemetry is stale or invalid")
        if (model.get("installed") is not True or model.get("worker_ready") is not True
                or model.get("resident") is not False or model.get("worker_uncertain") is not False
                or model.get("active") is not False or type(model.get("queued")) is not int or model["queued"] != 0):
            raise HandoffConflict("Wan installation/worker is unavailable, busy or uncertain")
        if (type(voice.get("active_count")) is not int or voice["active_count"] != 0
                or type(voice.get("active_gpu_work")) is not int or voice["active_gpu_work"] != 0):
            raise HandoffConflict("Active or unverified voice work blocks handoff")
        if lease is None:
            if voice.get("parked") is not False:
                raise HandoffConflict("Voice admission is already parked")
        elif voice.get("parked") is not True or voice.get("park_id") != lease:
            raise HandoffConflict("Voice park lease was lost; do not stop services")
        for service in ("stt", "tts"):
            current = services[service]
            if not current.get("id"):
                raise HandoffConflict("Moshi container identity is unavailable")
            if service in stopped:
                if current.get("running") is not False or current.get("status") not in ("exited", "created"):
                    raise HandoffConflict("Moshi stop has not been verified")
            elif current.get("running") is not True or current.get("health") != "healthy":
                raise HandoffConflict("Moshi source services are not healthy")
            if original and current["id"] != original["services"][service]["id"]:
                raise HandoffConflict("Moshi container changed since confirmation")
        if original and (gpu["uuid"] != original["gpu"]["uuid"] or total != original["gpu"]["total_mib"]):
            raise HandoffConflict("GPU changed since confirmation")
        # Calibration is experimental, NOT a guessed peak. Require essentially
        # exclusive capacity before reserving it; never test under Moshi's load.
        if len(stopped) == 2 and (free < total - 2048 or (original and free <= original["gpu"]["free_mib"])):
            raise HandoffConflict("GPU memory was not sufficiently released for isolated calibration")

    async def offer(self, *, user: str, expected_generation: int, calibration: bool) -> dict:
        if calibration is not True:
            raise HandoffConflict("Ordinary video needs a measured profile; explicitly request first-render calibration")
        state = await asyncio.to_thread(self.store.status)
        if state.owner not in ("unknown", "voice") or state.phase not in ("idle", "offered"):
            raise HandoffConflict("GPU ownership requires reconciliation")
        if await asyncio.to_thread(self.store.active_video_jobs):
            raise HandoffConflict("Active video claims block switching")
        observed = await self.adapter.observe()
        self._check(observed)
        snapshots = await self.adapter.prepare_operations()
        await self.adapter.check_operations(snapshots)
        plan = {"user": user, "target": "video_calibration", "observed": observed, "operations": snapshots,
                "calibration_profile": CALIBRATION_PROFILE}
        token, offered = await asyncio.to_thread(
            self.store.offer, user=user, target="video_calibration", actions=ACTIONS,
            expected_generation=expected_generation, plan=plan,
        )
        prior = await asyncio.to_thread(self.store.calibration)
        measured = ("The first render re-measures memory against the last passing calibration."
                    if prior and prior.get("supported") else
                    "First-render memory is unmeasured; calibration may fail.")
        return {"token": token, "generation": offered.generation, "expires_at": offered.expires_at,
                "target": "video_calibration", "actions": list(ACTIONS),
                "operations": list(STOP_OPERATIONS), "calibration_profile": CALIBRATION_PROFILE,
                "warning": f"Stops Moshi speech until a separate voice handoff. {measured} "
                           "Ordinary renders unlock once it passes."}

    async def confirm(self, *, user: str, token: str, expected_generation: int, approved: bool) -> dict:
        if approved is not True:
            raise HandoffConflict("Switch not approved; no service actions performed")
        plan = await asyncio.to_thread(self.store.plan, expected_generation)
        # Validate all policies and fresh observations BEFORE consuming consent or
        # touching voice. confirm's DB row lock arbitrates concurrent executors.
        await self.adapter.check_operations(plan["operations"])
        observed = await self.adapter.observe()
        self._check(observed, original=plan["observed"])
        state = await asyncio.to_thread(self.store.confirm, user=user, token=token,
                                        expected_generation=expected_generation)
        generation = state.generation
        try:
            if state.actions != ACTIONS or state.target != "video_calibration" or plan["user"] != user:
                raise HandoffConflict("Unexpected transition plan")
            await asyncio.to_thread(self.store.begin_action, generation=generation, action="park_voice")
            lease = await self.adapter.park()
            await asyncio.to_thread(self.store.record_voice_lease, generation=generation, lease_id=lease)
            await asyncio.to_thread(self.store.finish_action, generation=generation, action="park_voice")
            for index, service in enumerate(("stt", "tts")):
                fresh = await self.adapter.observe()
                self._check(fresh, stopped=("stt",)[:index], lease=lease, original=observed)
                await self.adapter.check_operations(plan["operations"])
                action = f"stop_moshi_{service}"
                await self._run_operation(generation=generation, action=action, user=user,
                                          snapshot=plan["operations"][index], slug=STOP_OPERATIONS[index])
                # Receipt success alone isn't proof of released services/VRAM.
                fresh = await self.adapter.observe()
                self._check(fresh, stopped=("stt", "tts")[:index + 1], lease=lease, original=observed)
                await asyncio.to_thread(self.store.finish_action, generation=generation, action=action)
            await asyncio.to_thread(self.store.begin_action, generation=generation, action="verify_video_capacity")
            fresh = await self.adapter.observe()
            self._check(fresh, stopped=("stt", "tts"), lease=lease, original=observed)
            await asyncio.to_thread(self.store.finish_action, generation=generation, action="verify_video_capacity")
            completed = await asyncio.to_thread(self.store.complete, generation=generation, target="video_calibration")
            return {"owner": completed.owner, "generation": completed.generation,
                    "calibration_profile": plan["calibration_profile"], "ordinary_render_ready": False}
        except BaseException as exc:
            # Preserve parked voice, plan, cursor and job ID; never silently
            # replay a stop/start or reopen admission after an ambiguous outcome.
            try:
                await asyncio.shield(asyncio.to_thread(self.store.recover, generation=generation,
                    reason=f"Switch to video failed: {_reason(exc)}"))
            except HandoffConflict:
                pass  # Startup/another observer has already fenced the generation.
            raise

    async def _run_operation(self, *, generation: int, action: str, user: str, snapshot: dict, slug: str) -> None:
        """Record intent, dispatch the consented snapshot once, await its receipt."""
        await asyncio.to_thread(self.store.begin_action, generation=generation, action=action)
        job = await self.adapter.dispatch(snapshot, user=user, generation=generation)
        job_id = job["id"]
        await asyncio.to_thread(self.store.record_job, generation=generation, action=action, job_id=job_id)
        async with asyncio.timeout(self.job_timeout):
            while True:
                receipt = await self.adapter.job_status(job_id)
                if receipt.get("operation") != slug or receipt.get("id") != job_id:
                    raise HandoffConflict("Operation receipt does not match the planned action")
                if type(receipt.get("terminal")) is not bool:
                    raise HandoffConflict("Operation receipt has an invalid terminal state")
                if receipt["terminal"]:
                    if (receipt.get("state") != "succeeded" or type(receipt.get("exit_code")) is not int
                            or receipt["exit_code"] != 0):
                        raise HandoffConflict(f"{slug} failed; inspect ops job {job_id}")
                    return
                await asyncio.sleep(self.poll_interval)

    async def resume_video(self, *, user: str, expected_generation: int) -> dict:
        """Recovery toward video. Touches nothing: it re-verifies, with the same
        live check the forward switch ends on, that the GPU is still video's."""
        lease = await asyncio.to_thread(self.store.voice_lease)
        if not lease:
            raise HandoffConflict("No recorded voice pause to resume under; restore voice first")
        calibration = await asyncio.to_thread(self.store.calibration)
        if not (calibration and calibration.get("supported")):
            raise HandoffConflict("No passing memory calibration; restore voice, then switch to video")
        observed = await self.adapter.observe()
        self._check(observed, stopped=SERVICES, lease=lease)
        state = await asyncio.to_thread(self.store.resume_video, expected_generation=expected_generation)
        logger.info("GPU resumed to video for %s after verification (generation %s)", user, state.generation)
        return {"owner": state.owner, "generation": state.generation}

    @staticmethod
    def _check_restorable(observed: dict) -> None:
        """Restore needs only an idle video worker and identifiable Moshi containers."""
        model, voice, services = observed["model"], observed["voice"], observed["services"]
        if model.get("active") is not False or model.get("queued") != 0:
            raise HandoffConflict("A local video render is still running; wait for it to finish")
        if type(voice.get("active_count")) is not int or voice["active_count"] != 0:
            raise HandoffConflict("Active voice work blocks the switch")
        if any(not services[service].get("id") for service in SERVICES):
            raise HandoffConflict("Moshi container identity is unavailable")

    async def offer_restore(self, *, user: str, expected_generation: int) -> dict:
        state = await asyncio.to_thread(self.store.status)
        if state.phase == "transitioning":
            raise HandoffConflict("A GPU switch is already running")
        observed = await self.adapter.observe()
        self._check_restorable(observed)
        snapshots = await self.adapter.prepare_operations(START_OPERATIONS)
        await self.adapter.check_operations(snapshots, START_OPERATIONS)
        plan = {"user": user, "target": "voice", "observed": observed, "operations": snapshots}
        token, offered = await asyncio.to_thread(
            self.store.offer, user=user, target="voice", actions=RESTORE_ACTIONS,
            expected_generation=expected_generation, plan=plan,
        )
        return {"token": token, "generation": offered.generation, "expires_at": offered.expires_at,
                "target": "voice", "actions": list(RESTORE_ACTIONS), "operations": list(START_OPERATIONS),
                "warning": "Unloads Wan from the GPU and restarts Moshi speech. Moshi takes a minute or two to load."}

    async def accept_restore(self, *, user: str, token: str, expected_generation: int, approved: bool) -> int:
        """Validate and consume consent synchronously; execution runs separately."""
        if approved is not True:
            raise HandoffConflict("Switch not approved; no service actions performed")
        plan = await asyncio.to_thread(self.store.plan, expected_generation)
        if plan.get("target") != "voice" or plan["user"] != user:
            raise HandoffConflict("Unexpected transition plan")
        await self.adapter.check_operations(plan["operations"], START_OPERATIONS)
        self._check_restorable(await self.adapter.observe())
        state = await asyncio.to_thread(self.store.confirm, user=user, token=token,
                                        expected_generation=expected_generation)
        if state.actions != RESTORE_ACTIONS or state.target != "voice":
            await asyncio.to_thread(self.store.recover, generation=state.generation,
                                    reason="Unexpected restore plan")
            raise HandoffConflict("Unexpected transition plan")
        return state.generation

    async def execute_restore(self, *, user: str, generation: int) -> None:
        plan = await asyncio.to_thread(self.store.plan, generation - 1)
        try:
            await asyncio.to_thread(self.store.begin_action, generation=generation, action="reset_video_worker")
            await self.adapter.reset_video()
            await asyncio.to_thread(self.store.finish_action, generation=generation, action="reset_video_worker")
            for index, service in enumerate(SERVICES):
                action = f"start_moshi_{service}"
                await self._run_operation(generation=generation, action=action, user=user,
                                          snapshot=plan["operations"][index], slug=START_OPERATIONS[index])
                await asyncio.to_thread(self.store.finish_action, generation=generation, action=action)
            await asyncio.to_thread(self.store.begin_action, generation=generation, action="verify_voice")
            observed = await self._await_moshi_healthy(plan["observed"])
            await asyncio.to_thread(self.store.finish_action, generation=generation, action="verify_voice")
            await asyncio.to_thread(self.store.begin_action, generation=generation, action="unpark_voice")
            if observed["voice"].get("parked") is True:
                await self.adapter.unpark(observed["voice"]["park_id"])
            await asyncio.to_thread(self.store.finish_action, generation=generation, action="unpark_voice")
            await asyncio.to_thread(self.store.complete, generation=generation, target="voice")
        except BaseException as exc:
            try:
                await asyncio.shield(asyncio.to_thread(self.store.recover, generation=generation,
                    reason=f"Switch to voice failed: {_reason(exc)}"))
            except HandoffConflict:
                pass
            raise

    async def _await_moshi_healthy(self, original: dict) -> dict:
        """Moshi loads its models after start; health, not the receipt, is proof."""
        try:
            async with asyncio.timeout(self.health_timeout):
                while True:
                    observed = await self.adapter.observe()
                    services = observed["services"]
                    if any(services[s]["id"] != original["services"][s]["id"] for s in SERVICES):
                        raise HandoffConflict("Moshi container changed during the switch")
                    if all(services[s].get("running") is True and services[s].get("health") == "healthy"
                           for s in SERVICES):
                        return observed
                    if any(services[s].get("health") == "unhealthy" or services[s].get("status") == "exited"
                           for s in SERVICES):
                        raise HandoffConflict("Moshi started but reports unhealthy; check its container logs")
                    await asyncio.sleep(self.poll_interval * 5)
        except TimeoutError as exc:
            raise HandoffConflict(f"Moshi did not become healthy within {int(self.health_timeout)}s") from exc


def _reason(exc: BaseException) -> str:
    detail = str(exc).strip()
    return f"{detail[:400]}" if detail else type(exc).__name__

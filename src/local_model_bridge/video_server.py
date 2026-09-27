"""Private GPU video API with serialized, resident subprocess rendering.

The bridge keeps the video pipeline outside the embedding process. Durable
claims fence queue admission and block handoff until outcomes are known.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from llm_bawt.media.gpu_handoff_store import GpuHandoffStore, HandoffConflict
from llm_bawt.media.gpu_profile import video_profile
from llm_bawt.utils.config import Config
from llm_bawt.utils.db import get_shared_engine

from .gpu_telemetry import GpuTelemetry
from .video_models import VideoModelManager
from .video_residency import ResidentVideoWorker, WorkerBusy
from .video_worker import DIMENSIONS

logger = logging.getLogger(__name__)
MAX_PENDING_VIDEO_JOBS = 8


class VideoRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    source_image: str | None = None
    aspect_ratio: str = "16:9"
    duration: float = Field(default=5, ge=1, le=15)
    resolution: str = "480p"
    calibration_generation: int | None = Field(default=None, ge=0, strict=True)


async def verify_voice_park(ledger: GpuHandoffStore) -> None:
    from llm_bawt.integrations.home_audio import HomeAudioSettings

    lease = await asyncio.to_thread(ledger.voice_lease)
    if not lease:
        raise HandoffConflict("Video ownership has no recorded voice lease")
    settings = await asyncio.to_thread(HomeAudioSettings.load, Config())
    async with httpx.AsyncClient() as client:
        response = await client.get(settings.tts_url.rstrip("/") + "/v1/internal/voice/sessions", timeout=3)
        response.raise_for_status()
        voice = response.json()
    if (voice.get("parked") is not True or voice.get("park_id") != lease
            or type(voice.get("active_count")) is not int or voice["active_count"] != 0
            or type(voice.get("active_gpu_work")) is not int or voice["active_gpu_work"] != 0):
        raise HandoffConflict("Voice lease was lost or voice work is active; reconcile before rendering")


def failure_message(exc: BaseException) -> str:
    """A user-facing reason; the traceback stays in the bridge log."""
    detail = str(exc).strip()
    if isinstance(exc, (HandoffConflict, RuntimeError)) and detail:
        return detail[:600]
    return f"Video worker failed ({type(exc).__name__}); reset it from the GPU panel"


class VideoJobs:
    def __init__(self, directory: Path):
        self.directory = directory
        self.jobs: dict[str, dict] = {}
        self.lock = asyncio.Lock()
        self.worker = ResidentVideoWorker()
        self.queue: asyncio.Queue[tuple[str, Path, Path, GpuHandoffStore]] = asyncio.Queue()
        self._drain_task: asyncio.Task | None = None
        self._active_job: str | None = None
        self._blocked = False
        self._worker_reserved_mib = 0
        self._calibrations: set[str] = set()

    async def submit(self, request: VideoRequest) -> dict:
        if request.aspect_ratio not in DIMENSIONS.get(request.resolution, {}):
            raise HTTPException(status_code=400, detail="Unsupported video size or aspect ratio")
        if request.source_image and (not request.source_image.startswith("data:image/") or len(request.source_image) > 30_000_000):
            raise HTTPException(status_code=400, detail="Source image must be an embedded image under 22MB")
        async with self.lock:
            if not models.status()["worker_ready"]:
                raise HTTPException(status_code=503, detail="GPU video worker is not installed yet")
            if not models.installed():
                raise HTTPException(status_code=409, detail="Install Wan 2.2 weights in Studio before generating")
            if self._blocked or self.worker.uncertain:
                raise HTTPException(status_code=409, detail="Wan worker requires manual reconciliation")
            if self.queue.qsize() >= MAX_PENDING_VIDEO_JOBS:
                raise HTTPException(status_code=429, detail="Local video queue is full")
            job_id = uuid.uuid4().hex
            try:
                ledger = await asyncio.to_thread(GpuHandoffStore, get_shared_engine(Config()))
                if request.calibration_generation is not None:
                    await asyncio.to_thread(ledger.claim_calibration, job_id,
                        generation=request.calibration_generation, profile=video_profile(request))
                    self._calibrations.add(job_id)
                else:
                    await asyncio.to_thread(ledger.claim_video, job_id)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            try:
                await verify_voice_park(ledger)
                if request.calibration_generation is None:
                    gpu = await asyncio.to_thread(GpuTelemetry().observe)
                    await asyncio.to_thread(ledger.validate_video_profile, video_profile(request), gpu, self._worker_reserved_mib)
                self.directory.mkdir(parents=True, exist_ok=True)
                input_path = self.directory / f"{job_id}.input.json"
                output_path = self.directory / f"{job_id}.mp4"
                input_path.write_text(request.model_dump_json(), encoding="utf-8")
                self.queue.put_nowait((job_id, input_path, output_path, ledger))
                self.jobs[job_id] = {"status": "pending", "progress": 0}
                if self._drain_task is None or self._drain_task.done():
                    self._drain_task = asyncio.create_task(self._drain())
            except BaseException as exc:
                if job_id not in self.jobs:
                    if job_id in self._calibrations:
                        await asyncio.shield(asyncio.to_thread(ledger.recover_orphaned_video))
                    else:
                        await asyncio.shield(asyncio.to_thread(ledger.release_video, job_id))
                if isinstance(exc, ValueError):
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                raise
            return {"id": job_id, **self.jobs[job_id]}

    async def _drain(self) -> None:
        while True:
            async with self.lock:
                if self.queue.empty():
                    # Serialize task retirement with submit(); otherwise a new
                    # request can enqueue as this task exits and never drain.
                    self._drain_task = None
                    return
                job_id, input_path, output_path, ledger = self.queue.get_nowait()
            self._active_job = job_id
            self.jobs[job_id] = {"status": "processing", "progress": 10}
            try:
                await verify_voice_park(ledger)
                metadata = await self.worker.render(input_path, output_path)
                # Keep the claim until the worker confirms an output and the
                # ledger acknowledges release; a failed DB write is not a
                # completed handoff-safe render.
                reserved = metadata.get("worker_reserved_mib", 0)
                self._worker_reserved_mib = reserved if type(reserved) is int and reserved >= 0 else 0
                if job_id in self._calibrations:
                    self._calibrations.discard(job_id)
                    try:
                        metadata["calibration"] = await asyncio.shield(asyncio.to_thread(
                            ledger.finish_calibration, job_id, metadata.get("gpu_measurement")))
                    except HandoffConflict as exc:
                        # The video exists and the worker is healthy; only the
                        # memory evidence was rejected. Deliver the video, keep
                        # ordinary rendering fenced, and say why.
                        logger.warning("Calibration evidence for %s rejected: %s", job_id, exc)
                        await asyncio.shield(asyncio.to_thread(ledger.abandon_calibration, job_id, reason=str(exc)))
                        metadata["warning"] = f"Video rendered, but its GPU measurement was not recorded: {exc}"
                else:
                    await asyncio.shield(asyncio.to_thread(ledger.release_video, job_id))
                self.jobs[job_id] = {"status": "completed", "progress": 100, **metadata}
                try:
                    input_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove completed video input %s", job_id, exc_info=True)
            except BaseException as exc:
                # Worker death, timeout, cancelled drain or failed claim release:
                # leave ALL outstanding claims for read-only reconciliation.
                self._blocked = True
                logger.exception("Wan job %s requires reconciliation", job_id)
                self.jobs[job_id] = {"status": "failed", "progress": 0, "error": failure_message(exc)}
                try:
                    await asyncio.shield(asyncio.to_thread(ledger.recover_orphaned_video))
                except BaseException:
                    logger.exception("Could not mark video claims for recovery")
                while not self.queue.empty():
                    queued_id, _, _, _ = self.queue.get_nowait()
                    self.jobs[queued_id] = {"status": "failed", "progress": 0,
                                            "error": "Queued work blocked by worker recovery"}
                    self.queue.task_done()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                break
            finally:
                self._active_job = None
                self.queue.task_done()

    def status(self, job_id: str) -> dict:
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail="Unknown local video job")
        job = self.jobs[job_id]
        if job["status"] == "processing":
            job = {**job, "progress": self._denoise_progress(job_id, job["progress"])}
        return {"id": job_id, **job}

    def _denoise_progress(self, job_id: str, fallback: int) -> int:
        """Map worker denoising steps to 10-95%; loading and export sit outside."""
        try:
            data = json.loads(self.directory.joinpath(f"{job_id}.progress").read_text(encoding="utf-8"))
            return 10 + int(85 * min(1, data["step"] / data["total"]))
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
            return fallback

    async def reset(self) -> dict:
        """Operator recovery: end the worker and clear claims it can no longer own.

        Caller holds ``self.lock``. Refuses while a render is running or queued;
        those outcomes are real and must finish or fail on their own.
        """
        if self._active_job is not None or not self.queue.empty():
            raise HTTPException(status_code=409, detail="A local video render is still running; wait for it to finish")
        try:
            await self.worker.reset()
        except WorkerBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        ledger = await asyncio.to_thread(GpuHandoffStore, get_shared_engine(Config()))
        released = await asyncio.to_thread(ledger.release_all_video_claims)
        for job_id, job in self.jobs.items():
            if job["status"] in ("pending", "processing"):
                self.jobs[job_id] = {"status": "failed", "progress": 0, "error": "Video worker was reset during recovery"}
        self._blocked = False
        self._calibrations.clear()
        self._worker_reserved_mib = 0
        return {"reset": True, "released_claims": released}

    def file(self, job_id: str) -> Path:
        if self.status(job_id)["status"] != "completed":
            raise HTTPException(status_code=404, detail="Video not ready")
        return self.directory / f"{job_id}.mp4"


jobs = VideoJobs(Path(tempfile.gettempdir()) / "llm-bawt-video-jobs")
models = VideoModelManager()
app = FastAPI(title="Local GPU video API", docs_url=None, redoc_url=None)


@app.get("/models/wan2.2-ti2v-5b")
def model_status() -> dict:
    return {**models.status(active=jobs._active_job is not None or not jobs.queue.empty()),
            "resident": jobs.worker.resident, "worker_uncertain": jobs.worker.uncertain or jobs._blocked,
            "queued": jobs.queue.qsize()}


@app.post("/models/wan2.2-ti2v-5b/install")
async def install_model() -> dict:
    return await models.install()


@app.delete("/models/wan2.2-ti2v-5b")
async def remove_model() -> dict:
    async with jobs.lock:
        ledger = await asyncio.to_thread(GpuHandoffStore, get_shared_engine(Config()))
        claims = await asyncio.to_thread(ledger.active_video_jobs)
        return await models.remove(active=bool(claims) or jobs.worker.running or jobs.worker.uncertain
                                   or jobs._blocked or jobs._active_job is not None or not jobs.queue.empty())


@app.get("/health")
def health() -> dict:
    return {"ok": True, "worker": "wan2.2-ti2v-5b"}


@app.get("/gpu/telemetry")
def gpu_telemetry() -> dict:
    """Fresh driver observation; a healthy worker is not proof of CUDA capacity."""
    return GpuTelemetry().observe()


@app.post("/videos")
async def submit(request: VideoRequest) -> dict:
    return await jobs.submit(request)


@app.get("/videos/{job_id}")
def status(job_id: str) -> dict:
    return jobs.status(job_id)


@app.get("/videos/{job_id}/content")
def content(job_id: str) -> FileResponse:
    return FileResponse(jobs.file(job_id), media_type="video/mp4")


@app.delete("/videos/{job_id}")
def remove(job_id: str) -> dict:
    if jobs.status(job_id)["status"] == "processing":
        raise HTTPException(status_code=409, detail="Generation in progress")
    jobs.jobs.pop(job_id, None)
    jobs.directory.joinpath(f"{job_id}.mp4").unlink(missing_ok=True)
    jobs.directory.joinpath(f"{job_id}.json").unlink(missing_ok=True)
    jobs.directory.joinpath(f"{job_id}.progress").unlink(missing_ok=True)
    return {"deleted": True}


@app.post("/worker/reset")
async def reset_worker() -> dict:
    async with jobs.lock:
        return await jobs.reset()


async def serve_video(port: int) -> None:
    import uvicorn

    # Restart loses process-local residency even if no render claim survived.
    # Do not silently grant the prior video owner a newly started worker.
    ledger = await asyncio.to_thread(GpuHandoffStore, get_shared_engine(Config()))
    state = await asyncio.to_thread(ledger.recover_orphaned_video, worker_restarted=True)
    if state.phase == "recovery_required":
        logger.warning("GPU video requires manual recovery: %s", state.last_error)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning", access_log=False)
    await uvicorn.Server(config).serve()

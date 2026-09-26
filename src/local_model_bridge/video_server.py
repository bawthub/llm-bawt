"""Private GPU video API with serialized, resident subprocess rendering.

The bridge keeps the video pipeline outside the embedding process. Durable
claims fence queue admission and block handoff until outcomes are known.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from llm_bawt.media.gpu_handoff_store import GpuHandoffStore, HandoffConflict
from llm_bawt.utils.config import Config
from llm_bawt.utils.db import get_shared_engine

from .gpu_telemetry import GpuTelemetry
from .video_models import VideoModelManager
from .video_residency import ResidentVideoWorker
from .video_worker import DIMENSIONS

logger = logging.getLogger(__name__)
MAX_PENDING_VIDEO_JOBS = 8


class VideoRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    source_image: str | None = None
    aspect_ratio: str = "16:9"
    duration: float = Field(default=5, ge=1, le=15)
    resolution: str = "480p"


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
                await asyncio.to_thread(ledger.claim_video, job_id)
            except HandoffConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                input_path = self.directory / f"{job_id}.input.json"
                output_path = self.directory / f"{job_id}.mp4"
                input_path.write_text(request.model_dump_json(), encoding="utf-8")
                self.queue.put_nowait((job_id, input_path, output_path, ledger))
                self.jobs[job_id] = {"status": "pending", "progress": 0}
                if self._drain_task is None or self._drain_task.done():
                    self._drain_task = asyncio.create_task(self._drain())
            except BaseException:
                if job_id not in self.jobs:
                    await asyncio.shield(asyncio.to_thread(ledger.release_video, job_id))
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
                metadata = await self.worker.render(input_path, output_path)
                # Keep the claim until the worker confirms an output and the
                # ledger acknowledges release; a failed DB write is not a
                # completed handoff-safe render.
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
                self.jobs[job_id] = {"status": "failed", "progress": 0,
                                     "error": f"Worker outcome uncertain: {type(exc).__name__}"}
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
        return {"id": job_id, **self.jobs[job_id]}

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
    return {"deleted": True}


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

"""FastAPI routes for the unified media generation API.

Endpoints:
  POST   /v1/media/generations          — Submit a generation job
  GET    /v1/media/generations           — List generations (paginated)
  GET    /v1/media/generations/{id}      — Get status/result
  DELETE /v1/media/generations/{id}      — Delete generation + files
  GET    /v1/media/generations/{id}/content   — Stream raw media file
  GET    /v1/media/generations/{id}/thumbnail — Serve poster frame
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from ...media.clients import MediaClient, media_provider_registry
from ...media.db import MediaGenerationStore
from ...media.generation_service import MediaGenerationService
from ...media.gpu_handoff_store import GpuHandoffStore
from ...media.gpu_preflight import GpuPreflight
from ...media.object_store import BlobBackendUnavailable
from ...media.schemas import (
    MediaGenerationListResponse,
    MediaGenerationRequest,
    MediaGenerationResponse,
    MediaOutput,
    MediaProviderCapabilitiesResponse,
    MediaProviderListResponse,
    PromptExpansionRequest,
    PromptExpansionResponse,
)
from ...media.storage import MediaStorage
from ...utils.db import get_shared_engine
from ..dependencies import get_media_generation_store, get_ops_service, get_service
from .media_handoff import router as handoff_router

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Media"])
router.include_router(handoff_router)

# Module-level non-DB helpers initialised lazily
_storage: MediaStorage | None = None
_generation_service = MediaGenerationService()
_video_clients: dict[str, MediaClient] = {}
# Track active polling tasks so they can be cancelled on shutdown
_poll_tasks: dict[str, asyncio.Task] = {}


def _get_store() -> MediaGenerationStore:
    service = get_service()
    return get_media_generation_store(service.config)


def _get_storage() -> MediaStorage:
    global _storage
    if _storage is None:
        _storage = MediaStorage()
    return _storage


def _get_video_client(provider: str) -> MediaClient:
    canonical = media_provider_registry.canonical_provider(provider)
    client = _video_clients.get(canonical)
    if client is None:
        client = media_provider_registry.create_client(canonical, "video")
        _video_clients[canonical] = client
    return client


def _sniff_image_storage(raw: bytes) -> tuple[str, str]:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return ".webp", "image/webp"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif", "image/gif"
    return ".jpg", "image/jpeg"


async def _stored_image_as_data_url(row: dict, storage: MediaStorage) -> str:
    file_path = row.get("file_path")
    if not file_path or row.get("status") != "completed":
        raise ValueError(f"Generation {row.get('id')} has no completed image output")
    if row.get("media_type") != "image":
        raise ValueError(f"Generation {row.get('id')} is not an image")
    raw = await asyncio.to_thread(storage.read, file_path)
    mime = row.get("mime_type") or _sniff_image_storage(raw)[1]
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _gen_id() -> str:
    """Generate a prefixed unique ID for a media generation."""
    return f"gen_{uuid.uuid4().hex[:20]}"


# ------------------------------------------------------------------
# Helpers: row -> response
# ------------------------------------------------------------------

def _row_to_response(row: dict) -> MediaGenerationResponse:
    """Convert a database row dict to a MediaGenerationResponse."""
    outputs: list[MediaOutput] = []
    if row.get("status") == "completed" and row.get("file_path"):
        gen_id = row["id"]
        output = MediaOutput(
            file_path=row.get("file_path"),
            thumbnail_path=row.get("thumbnail_path"),
            file_size_bytes=row.get("file_size_bytes"),
            mime_type=row.get("mime_type"),
            content_url=f"/v1/media/generations/{gen_id}/content",
            thumbnail_url=f"/v1/media/generations/{gen_id}/thumbnail" if row.get("thumbnail_path") else None,
            width=row.get("width"),
            height=row.get("height"),
            actual_duration=row.get("actual_duration"),
        )
        outputs.append(output)

    return MediaGenerationResponse(
        id=row["id"],
        status=row["status"],
        media_type=row["media_type"],
        prompt=row["prompt"],
        original_prompt=row.get("original_prompt"),
        negative_prompt=row.get("negative_prompt"),
        revised_prompt=row.get("revised_prompt"),
        progress=row.get("progress", 0),
        outputs=outputs,
        provider=media_provider_registry.canonical_provider(row.get("provider")),
        model=row.get("model"),
        aspect_ratio=row.get("aspect_ratio"),
        duration=row.get("duration"),
        resolution=row.get("resolution"),
        error=row.get("error"),
        created_at=row["created_at"].isoformat() if isinstance(row.get("created_at"), datetime) else str(row.get("created_at", "")),
        completed_at=row["completed_at"].isoformat() if isinstance(row.get("completed_at"), datetime) else (str(row["completed_at"]) if row.get("completed_at") else None),
    )


# ------------------------------------------------------------------
# Background polling
# ------------------------------------------------------------------

async def _poll_video_job(gen_id: str) -> None:
    """Background task that polls xAI for video completion.

    Runs every 5 seconds until the job completes or fails, then
    downloads the video, stores it, and extracts a poster frame.
    """
    store = _get_store()
    storage = _get_storage()

    row = store.get_by_id(gen_id)
    if not row:
        logger.error("Poll task: generation %s not found", gen_id)
        return

    client = _get_video_client(row.get("provider") or "grok")
    provider_job_id = row["provider_job_id"]
    if not provider_job_id:
        logger.error("Poll task: no provider_job_id for %s", gen_id)
        store.update(gen_id, {"status": "failed", "error": "Missing provider job ID"})
        return

    logger.info("Starting poll loop for generation %s (job %s)", gen_id, provider_job_id)

    max_polls = 720 if row.get("provider") == "local-video" else 120  # Local offloading can take up to an hour.
    transient_errors = 0
    max_transient_errors = 5

    try:
        for poll_num in range(max_polls):
            await asyncio.sleep(5)

            try:
                result = await client.poll_status(provider_job_id)
                transient_errors = 0  # Reset on success
            except Exception as e:
                transient_errors += 1
                logger.warning("Poll error for %s (%d/%d): %s", gen_id, transient_errors, max_transient_errors, e)
                if transient_errors >= max_transient_errors:
                    store.update(gen_id, {"status": "failed", "error": f"Too many poll errors: {e}"})
                    return
                continue

            # Update progress
            store.update(gen_id, {
                "status": result.status,
                "progress": result.progress,
                "revised_prompt": result.revised_prompt,
            })

            if result.status == "failed":
                store.update(gen_id, {"error": result.error or "Generation failed"})
                logger.warning("Generation %s failed: %s", gen_id, result.error)
                return

            if result.status == "completed":
                break
        else:
            # Exhausted all polls
            store.update(gen_id, {"status": "failed", "error": f"Generation timed out after {max_polls * 5 // 60} minutes"})
            logger.error("Generation %s timed out", gen_id)
            return

        # Download the completed video
        if not result.media_url:
            store.update(gen_id, {"status": "failed", "error": "No media URL in completed response"})
            logger.error("Generation %s completed but no media URL", gen_id)
            return

        logger.info("Downloading video for %s from %s", gen_id, result.media_url[:80])
        video_data = await client.download(result.media_url)

        # Determine extension from media type
        media_type = row.get("media_type", "video")
        ext = ".mp4" if media_type == "video" else ".png"
        subdir = "videos" if media_type == "video" else "images"
        mime = "video/mp4" if media_type == "video" else "image/png"

        # Store to filesystem
        rel_path, sha256 = await storage.write_async(video_data, subdir, ext)

        updates: dict = {
            "status": "completed",
            "progress": 100,
            "file_path": rel_path,
            "file_size_bytes": len(video_data),
            "mime_type": mime,
            "completed_at": datetime.now(timezone.utc),
            "width": result.width,
            "height": result.height,
            "actual_duration": result.actual_duration,
            "revised_prompt": result.revised_prompt,
        }

        # Extract poster frame for videos
        if media_type == "video":
            thumb_path = await storage.extract_poster_frame(rel_path)
            if thumb_path:
                updates["thumbnail_path"] = thumb_path

        store.update(gen_id, updates)
        if row.get("provider") == "local-video" and hasattr(client, "remove_job"):
            try:
                await client.remove_job(provider_job_id)
            except Exception:
                logger.warning("Could not remove local video staging file for %s", gen_id, exc_info=True)
        logger.info("Generation %s completed: %s (%d bytes)", gen_id, rel_path, len(video_data))

    except asyncio.CancelledError:
        logger.info("Poll task cancelled for %s", gen_id)
        raise
    except Exception as e:
        logger.exception("Unexpected error polling generation %s", gen_id)
        store.update(gen_id, {"status": "failed", "error": str(e)})
    finally:
        _poll_tasks.pop(gen_id, None)


async def _store_image_job(gen_id: str, img_data: bytes) -> None:
    """Persist normalized image bytes returned by any synchronous provider."""
    store = _get_store()
    storage = _get_storage()
    try:
        extension, mime_type = _sniff_image_storage(img_data)
        rel_path, _sha256 = await storage.write_async(img_data, "images", extension)
        store.update(gen_id, {
            "status": "completed",
            "progress": 100,
            "file_path": rel_path,
            "thumbnail_path": rel_path,
            "file_size_bytes": len(img_data),
            "mime_type": mime_type,
            "completed_at": datetime.now(timezone.utc),
        })
        logger.info("Image generation %s completed: %s (%d bytes)", gen_id, rel_path, len(img_data))
    except Exception as e:
        logger.exception("Failed to store image for %s", gen_id)
        store.update(gen_id, {"status": "failed", "error": str(e)})
    finally:
        _poll_tasks.pop(gen_id, None)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.get("/v1/media/providers", response_model=MediaProviderListResponse)
async def list_media_providers(media_type: str | None = Query(default=None)):
    """Return provider/model/capability metadata for clients."""
    providers = [
        MediaProviderCapabilitiesResponse.model_validate(capability.to_dict())
        for capability in media_provider_registry.list_capabilities(media_type)
    ]
    return MediaProviderListResponse(providers=providers, default_provider="grok")


@router.get("/v1/media/local-video/model")
async def local_video_model_status():
    return await _get_video_client("local-video").model_status()


@router.get("/v1/media/local-video/gpu")
async def local_video_gpu_status():
    """Observations, not a switch permit. Stale/failed telemetry is explicit."""
    ledger = await asyncio.to_thread(
        GpuHandoffStore, get_shared_engine(get_service().config),
    )
    state = await asyncio.to_thread(ledger.status)
    try:
        active_video_jobs = await asyncio.to_thread(ledger.active_video_jobs)
    except Exception:
        active_video_jobs = None
    client = _get_video_client("local-video")
    try:
        gpu = await client.gpu_telemetry()
    except Exception as exc:
        gpu = {"ready": False, "error": f"GPU bridge telemetry unavailable: {type(exc).__name__}"}
    try:
        model = await client.model_status()
    except Exception:
        model = None
    calibration = await asyncio.to_thread(ledger.calibration)
    import httpx
    async with httpx.AsyncClient() as http:
        def operations():
            return get_ops_service(get_service().config).list_operations_for_agent(include_disabled=True)

        preflight = await GpuPreflight(
            config=get_service().config, http=http, ops=lambda: asyncio.to_thread(operations),
        ).assess(state=state, gpu=gpu, model=model, active_video_jobs=active_video_jobs, calibration=calibration)
    return {
        "owner": state.owner, "phase": state.phase, "generation": state.generation,
        "pending_target": state.target, "pending_actions": state.actions,
        "next_action": state.next_action, "pending_action": state.pending_action,
        "last_job_id": state.last_job_id, "last_error": state.last_error,
        "gpu": gpu, "model": model, "active_video_jobs": active_video_jobs,
        "preflight": preflight,
        "handoff_available": True,
        "calibration": calibration,
    }


@router.post("/v1/media/local-video/model/install")
async def install_local_video_model():
    return await _get_video_client("local-video").install_model()


@router.delete("/v1/media/local-video/model")
async def remove_local_video_model():
    return await _get_video_client("local-video").remove_model()


def _original_prompt(request: MediaGenerationRequest) -> str | None:
    """Keep the pre-expansion text only when it differs from what was sent."""
    original = (request.original_prompt or "").strip()
    return original if original and original != request.prompt.strip() else None


@router.post("/v1/media/prompts/expand", response_model=PromptExpansionResponse)
async def expand_media_prompt(request: PromptExpansionRequest):
    """Rewrite a short idea into a detailed prompt with the registry template
    and the global maintenance_model. Nothing is stored; Studio shows the result
    for editing and sends it back with ``original_prompt`` on submit."""
    from ...media.prompt_expansion import PromptExpansionError, expand_prompt
    from .llm import complete_utility

    try:
        expanded = await asyncio.to_thread(
            expand_prompt,
            prompt=request.prompt,
            media_type=request.media_type,
            duration=request.duration,
            aspect_ratio=request.aspect_ratio,
            source_image=request.source_image,
            complete=complete_utility,
        )
    except PromptExpansionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PromptExpansionResponse(
        prompt=expanded.prompt,
        original_prompt=expanded.original_prompt,
        model=expanded.model,
        template_key=expanded.template_key,
    )


@router.post("/v1/media/generations", response_model=MediaGenerationResponse)
async def create_generation(request: MediaGenerationRequest):
    """Submit a provider-routed media generation job."""
    store = _get_store()
    storage = _get_storage()

    source_image = request.source_image
    if request.reference_generation_id:
        if source_image:
            raise HTTPException(
                status_code=400,
                detail="Provide source_image or reference_generation_id, not both",
            )
        reference = store.get_by_id(request.reference_generation_id)
        if not reference:
            raise HTTPException(
                status_code=404,
                detail=f"Reference generation {request.reference_generation_id} not found",
            )
        try:
            source_image = await _stored_image_as_data_url(reference, storage)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        provider, model, aspect_ratio, resolution = _generation_service.resolve_request(
            provider=request.provider,
            media_type=request.media_type,
            model=request.model,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            has_source_image=bool(source_image),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if provider == "local-video":
        ledger = await asyncio.to_thread(GpuHandoffStore, get_shared_engine(get_service().config))
        handoff = await asyncio.to_thread(ledger.status)
        calibration = request.calibration_generation
        required_owner = "video_calibration" if calibration is not None else "video"
        if (handoff.phase != "idle" or handoff.owner != required_owner
                or (calibration is not None and handoff.generation != calibration)):
            raise HTTPException(status_code=409, detail="GPU does not belong to video; a consented handoff is required")
        from ...media.gpu_profile import require_calibration_profile
        try:
            require_calibration_profile({"resolution": resolution, "aspect_ratio": aspect_ratio,
                "duration": request.duration or 5, "num_outputs": request.num_outputs,
                "image_conditioned": bool(source_image)})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if calibration is None and not (await asyncio.to_thread(ledger.calibration) or {}).get("supported"):
            raise HTTPException(status_code=409, detail="A measured local video profile is required")
    elif request.calibration_generation is not None:
        raise HTTPException(status_code=400, detail="Calibration is only available for local video")
    negative_prompt = (request.negative_prompt or "").strip() or None
    if negative_prompt and not media_provider_registry.capabilities(provider).negative_prompt:
        raise HTTPException(status_code=400, detail=f"{provider} does not support a negative prompt")

    gen_id = _gen_id()
    store.insert({
        "id": gen_id,
        "status": "pending",
        "media_type": request.media_type,
        "prompt": request.prompt,
        "original_prompt": _original_prompt(request),
        "negative_prompt": negative_prompt,
        "provider": provider,
        "model": model,
        "aspect_ratio": aspect_ratio,
        "duration": request.duration if request.media_type == "video" else None,
        "resolution": resolution,
        "progress": 0,
        "created_at": datetime.now(timezone.utc),
    })

    try:
        if request.media_type == "image":
            generated = await _generation_service.generate_image(
                prompt=request.prompt,
                provider=provider,
                model=model,
                source_image=source_image,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                num_outputs=request.num_outputs,
            )
            store.update(gen_id, {
                "status": "processing",
                "provider_job_id": generated.result.provider_job_id,
                "progress": 75,
                "revised_prompt": generated.result.revised_prompt,
            })
            task = asyncio.create_task(_store_image_job(gen_id, generated.data))
            _poll_tasks[gen_id] = task
        else:
            client = _get_video_client(provider)
            result = await client.generate(
                prompt=request.prompt,
                media_type="video",
                model=model,
                source_image=source_image,
                aspect_ratio=aspect_ratio,
                duration=request.duration or 5,
                resolution=resolution,
                num_outputs=request.num_outputs,
                **({"calibration_generation": request.calibration_generation}
                   if request.calibration_generation is not None else {}),
                **({"negative_prompt": negative_prompt} if negative_prompt else {}),
            )
            store.update(gen_id, {
                "status": result.status,
                "provider_job_id": result.provider_job_id,
                "progress": result.progress,
                "revised_prompt": result.revised_prompt,
            })
            if result.status in ("pending", "processing"):
                task = asyncio.create_task(_poll_video_job(gen_id))
                _poll_tasks[gen_id] = task
    except Exception as exc:
        logger.exception("Failed to submit generation %s", gen_id)
        store.update(gen_id, {"status": "failed", "error": str(exc)})

    db_row = store.get_by_id(gen_id)
    if not db_row:
        raise HTTPException(status_code=500, detail="Failed to retrieve generation after creation")
    return _row_to_response(db_row)


@router.get("/v1/media/generations/{gen_id}", response_model=MediaGenerationResponse)
async def get_generation(gen_id: str):
    """Get the status and result of a media generation."""
    store = _get_store()
    row = store.get_by_id(gen_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Generation {gen_id} not found")
    return _row_to_response(row)


@router.get("/v1/media/generations", response_model=MediaGenerationListResponse)
async def list_generations(
    media_type: str | None = Query(default=None, description="Filter by media type"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List media generations with optional filtering and pagination."""
    store = _get_store()
    rows, total = store.list_generations(media_type=media_type, limit=limit, offset=offset)
    items = [_row_to_response(r) for r in rows]
    return MediaGenerationListResponse(items=items, total=total)


@router.delete("/v1/media/generations/{gen_id}")
async def delete_generation(gen_id: str):
    """Delete a media generation and its associated files."""
    store = _get_store()
    storage = _get_storage()

    row = store.get_by_id(gen_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Generation {gen_id} not found")

    # Cancel any active polling task
    poll_task = _poll_tasks.pop(gen_id, None)
    if poll_task and not poll_task.done():
        poll_task.cancel()

    # Delete files from disk
    if row.get("file_path"):
        await storage.delete_async(row["file_path"])
    if row.get("thumbnail_path"):
        await storage.delete_async(row["thumbnail_path"])

    # Delete DB record
    store.delete(gen_id)

    return {"deleted": True, "id": gen_id}


@router.get("/v1/media/generations/{gen_id}/content")
async def stream_content(gen_id: str, request: Request):
    """Serve the raw media file with HTTP Range support.

    Range passthrough (TASK-266): the client's ``Range`` header is
    forwarded verbatim to the storage backend. Garage / boto3 honors
    standard ``bytes=…`` ranges, and the FS backend has a matching
    implementation in :class:`FsBlobBackend`. Both produce a
    :class:`BlobRange` we map to either HTTP 200 (full body) or 206
    (partial content) here.

    Backend-down → 503 JSON; missing blob → 404; bad/unsatisfiable
    range → 416.
    """
    store = _get_store()
    storage = _get_storage()

    row = store.get_by_id(gen_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Generation {gen_id} not found")

    file_path = row.get("file_path")
    if not file_path:
        raise HTTPException(status_code=404, detail="Media file not yet available (generation may still be in progress)")

    range_header = request.headers.get("range")

    # Pull the (possibly Range-bounded) body via the storage backend.
    # asyncio.to_thread because the FS backend may issue blocking I/O on
    # an NFS mount and we don't want to stall the event loop.
    try:
        blob_range = await asyncio.to_thread(storage.read_range, file_path, range_header)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Media file not found")
    except ValueError:
        # Unsatisfiable Range — RFC 7233 says we should also include a
        # Content-Range: bytes */<size> header, but we don't have the
        # size to hand here without an extra HEAD. 416 alone is enough
        # for browsers / video players.
        raise HTTPException(status_code=416, detail="Requested range not satisfiable")
    except BlobBackendUnavailable as e:
        logger.error("media content backend unavailable for %s: %s", gen_id, e)
        raise HTTPException(
            status_code=503,
            detail="Media storage backend unavailable; try again shortly",
        )

    mime_type = blob_range.content_type or row.get("mime_type", "application/octet-stream")

    if blob_range.partial:
        headers = {
            "Content-Range": f"bytes {blob_range.start}-{blob_range.end}/{blob_range.total_size}",
            "Accept-Ranges": "bytes",
        }
        return Response(
            content=blob_range.data,
            status_code=206,
            media_type=mime_type,
            headers=headers,
        )

    return Response(
        content=blob_range.data,
        status_code=200,
        media_type=mime_type,
        headers={"Accept-Ranges": "bytes"},
    )


@router.get("/v1/media/generations/{gen_id}/thumbnail")
async def serve_thumbnail(gen_id: str):
    """Serve the poster frame / thumbnail image.

    Backend-aware: same error mapping as ``stream_content``
    (backend down → 503, missing blob → 404). Thumbnails are small
    enough to serve as a single Response body — no Range needed.
    """
    store = _get_store()
    storage = _get_storage()

    row = store.get_by_id(gen_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Generation {gen_id} not found")

    thumb_path = row.get("thumbnail_path")
    if not thumb_path:
        raise HTTPException(status_code=404, detail="Thumbnail not available")

    try:
        data = await asyncio.to_thread(storage.read, thumb_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Thumbnail file not found")
    except BlobBackendUnavailable as e:
        logger.error("thumbnail backend unavailable for %s: %s", gen_id, e)
        raise HTTPException(
            status_code=503,
            detail="Media storage backend unavailable; try again shortly",
        )

    return Response(
        content=data,
        status_code=200,
        media_type=thumbnail_mime_type(thumb_path),
    )


def thumbnail_mime_type(thumb_path: str) -> str:
    """The thumbnail's own type. A video's row mime is video/mp4, but its poster
    frame is a JPEG; Safari refuses to paint an <img> labelled video/mp4."""
    guessed, _ = mimetypes.guess_type(thumb_path)
    return guessed if guessed and guessed.startswith("image/") else "image/jpeg"

"""Application-owned durable announcement worker, independent of MCP requests."""
from __future__ import annotations

import asyncio
import logging
import uuid

import httpx

from .home_audio import HomeAudioClient, HomeAudioSettings, wav_duration
from .home_audio_store import HomeAudioStore

logger = logging.getLogger(__name__)


class HomeAudioWorker:
    def __init__(self, config, store: HomeAudioStore, *, hub=None):
        self.config = config
        self.store = store
        self.hub = hub
        self.task: asyncio.Task | None = None

    def start(self):
        self.task = asyncio.create_task(self.run(), name="home-audio-worker")

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def run(self):
        from ..service.commit_wake import BACKSTOP_SECONDS, wait_for_wake

        # TASK-2108: an enqueue's commit wakes this loop; the timer is a
        # backstop for a missed notification or a quarantined lane lapsing.
        wake = self.hub.subscribe() if self.hub is not None else None
        interval = BACKSTOP_SECONDS if wake is not None else 1
        while True:
            job = None
            try:
                owner = uuid.uuid4().hex
                job = await asyncio.to_thread(self.store.claim, owner)
                if job:
                    await self._run_claim(owner, job)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Home audio worker iteration failed")
                await asyncio.sleep(1)
            if job and wake is not None:
                continue  # the next queued clip plays now
            await wait_for_wake(wake, interval)

    async def _run_claim(self, owner: str, job: dict):
        work = asyncio.create_task(self._perform(owner, job))
        heartbeat = asyncio.create_task(self._heartbeat(owner))
        try:
            done, _ = await asyncio.wait({work, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except asyncio.CancelledError:
            # Retain lease and active state: recovery records uncertainty, never replays.
            raise
        except Exception as exc:
            logger.warning("Home audio job %s failed: %s", job["id"], type(exc).__name__)
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            current = await asyncio.to_thread(self.store.get, job["id"])
            if current and current["status"] == "playing":
                # Dispatch may have succeeded. Keep the lane leased while any
                # five-minute clip can still be audible, including Cast groups.
                await asyncio.sleep(330)
            await asyncio.to_thread(self.store.save, owner, job["id"], "failed",
                                    error=f"{type(exc).__name__}: {str(exc)[:400]}")
        finally:
            work.cancel()
            heartbeat.cancel()
            await asyncio.gather(work, heartbeat, return_exceptions=True)

    async def _heartbeat(self, owner: str):
        while True:
            await asyncio.sleep(10)
            if not await asyncio.to_thread(self.store.renew, owner):
                raise RuntimeError("Playback lease lost; stopping worker")

    async def _perform(self, owner: str, job: dict):
        from llm_bawt.media import get_media_store

        settings = await asyncio.to_thread(HomeAudioSettings.load, self.config)
        request = job["request"]
        target = request["target"]
        async with httpx.AsyncClient() as http:
            client = HomeAudioClient(self.config, settings, http)
            await client.wait_idle(target, job["expires_at"])
            store = get_media_store()
            asset_id = request.get("asset_id")
            if asset_id:
                source_asset = await asyncio.to_thread(store.stat, asset_id)
                if source_asset is not None and source_asset.storage_key:
                    raise ValueError("Playback slots are replaceable, not saved clips; enqueue text or a speech_generate asset")
                raw, mime = await asyncio.to_thread(store.read_variant, asset_id, "original")
                if mime not in {"audio/wav", "audio/x-wav"}:
                    raise ValueError("Only generated PCM WAV assets are supported")
            else:
                raw = await client.render(request["text"], request["voice"])
            duration = wav_duration(raw)
            # Synthesis can take time. Recheck busy state and expiration before casting.
            await client.wait_idle(target, job["expires_at"])
            asset = await asyncio.to_thread(
                self.store.while_owned, owner,
                lambda: store.upload(raw_bytes=raw, original_mime="audio/wav",
                                     source="tool_generated", owner_user_id=request["user_id"],
                                     filename="announcement.wav", replace_key="announcements/current.wav"),
            )
            asset_id = asset.id
            # Upload may have blocked; honor expiry/busy state again before dispatch.
            await client.wait_idle(target, job["expires_at"])
            if not await asyncio.to_thread(self.store.save, owner, job["id"], "playing", asset_id=asset_id):
                raise RuntimeError("Playback lease lost before dispatch")
            # Exactly one dispatch attempt, including ambiguous HTTP failures.
            await client.play(target, asset_id, version=job["id"])
            await client.wait_finished(target, asset_id, duration, version=job["id"])
            await asyncio.to_thread(self.store.save, owner, job["id"], "completed")

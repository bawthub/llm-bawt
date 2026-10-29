"""Focused speech-generation and home-playback tools; the app owns execution."""
from __future__ import annotations

import asyncio
from typing import Annotated

import httpx
from pydantic import Field

from llm_bawt.integrations.home_audio import HomeAudioClient, HomeAudioSettings, wav_duration
from llm_bawt.integrations.home_audio_store import HomeAudioStore
from llm_bawt.utils.config import Config
from .registry import mcp

Text = Annotated[str, Field(min_length=1, max_length=4000)]
Identifier = Annotated[str, Field(min_length=1, max_length=128)]


def _store(config):
    from llm_bawt.utils.db import get_shared_engine
    return HomeAudioStore(get_shared_engine(config))


def _voice(bot_id: str, voice: str | None) -> str:
    if voice and voice.strip():
        return voice.strip()
    from llm_bawt.bots import get_bot
    from llm_bawt.voice_resolution import effective_voice_for_bot
    bot = get_bot(bot_id)
    resolved = effective_voice_for_bot(Config(), bot) if bot is not None else None
    if resolved is None or not resolved.voice:
        raise ValueError(
            "No voice configured for this bot and no global default voice; supply voice explicitly"
        )
    return resolved.voice


@mcp.tool()
async def home_audio_devices() -> dict:
    """List allowed home audio targets and available voice catalog location."""
    config = Config()
    settings = await asyncio.to_thread(HomeAudioSettings.load, config)
    async with httpx.AsyncClient() as http:
        client = HomeAudioClient(config, settings, http)
        return {"devices": await client.devices(),
                "voices_url": settings.tts_url.rstrip("/") + "/v1/tts/voices"}


@mcp.tool()
async def speech_generate(text: Text, bot_id: Identifier, voice: str | None = None,
                          user_id: str = "nick") -> dict:
    """Generate a stored WAV using an explicit or bot-default voice. No playback."""
    from llm_bawt.media import get_media_store
    if not text.strip():
        raise ValueError("Speech text must not be blank")
    config = Config()
    resolved = await asyncio.to_thread(_voice, bot_id, voice)
    settings = await asyncio.to_thread(HomeAudioSettings.load, config)
    async with httpx.AsyncClient() as http:
        client = HomeAudioClient(config, settings, http)
        raw = await client.render(text, resolved)
        asset = await asyncio.to_thread(get_media_store().upload, raw_bytes=raw,
                                       original_mime="audio/wav", source="tool_generated",
                                       owner_user_id=user_id, filename="speech.wav")
        return {"asset_id": asset.id, "voice": resolved, "duration_seconds": wav_duration(raw),
                "playback_url": client.media_url(asset.id),
                "download_path": f"/v1/uploads/{asset.id}"}


@mcp.tool()
async def home_audio_enqueue(target: Identifier, bot_id: Identifier, idempotency_key: Identifier,
                             text: Text | None = None, asset_id: str | None = None,
                             voice: str | None = None, user_id: str = "nick") -> dict:
    """Queue exactly one of text or a generated WAV asset. Audible, asynchronous."""
    if bool(text) == bool(asset_id):
        raise ValueError("Supply exactly one of text or asset_id")
    if text is not None and not text.strip():
        raise ValueError("Speech text must not be blank")
    if asset_id and voice:
        raise ValueError("voice cannot modify an existing asset")
    config = Config()
    settings = await asyncio.to_thread(HomeAudioSettings.load, config)
    if target not in settings.targets:
        raise ValueError("Target is not allowed; use home_audio_devices")
    resolved = await asyncio.to_thread(_voice, bot_id, voice) if text else None
    payload = {"target": target, "bot_id": bot_id, "text": text, "asset_id": asset_id,
               "voice": resolved, "user_id": user_id}
    store = await asyncio.to_thread(_store, config)
    return await asyncio.to_thread(store.enqueue, payload, idempotency_key)


@mcp.tool()
async def home_audio_status(job_id: Identifier) -> dict:
    """Read a durable announcement outcome; queued is not played."""
    store = await asyncio.to_thread(_store, Config())
    job = await asyncio.to_thread(store.get, job_id)
    return job or {"error": "Announcement not found"}


@mcp.tool()
async def home_audio_cancel(job_id: Identifier) -> dict:
    """Cancel only a queued announcement; cannot stop active playback."""
    store = await asyncio.to_thread(_store, Config())
    job = await asyncio.to_thread(store.cancel, job_id)
    return job or {"error": "Announcement not found"}

"""BawtHub speech rendering and Home Assistant Cast transport.

Configuration is DB-backed: global runtime setting `home_audio`. Credentials
reuse the existing HA integration; no keys or tokens belong in this setting.
"""
from __future__ import annotations

import asyncio
import io
import re
import wave
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from llm_bawt.integrations.ha_credentials import HomeAssistantUnavailable, resolve_ha_credentials
from llm_bawt.runtime_settings import RuntimeSettingsStore


@dataclass(frozen=True)
class HomeAudioSettings:
    tts_url: str = "http://10.0.0.101/api"
    media_base_url: str = "http://10.0.0.101:8642"
    # Explicitly scoped Google devices, not every HA media player/receiver.
    targets: tuple[str, ...] = (
        "media_player.kitchen_display", "media_player.bedroom_display",
        "media_player.sunroom_display", "media_player.office_speaker",
        "media_player.bathroom_speaker", "media_player.chromecastaudio2533",
        "media_player.all",
    )

    @classmethod
    def load(cls, config):
        raw = RuntimeSettingsStore(config).get_scope_settings("global", "*").get("home_audio", {})
        if not isinstance(raw, dict):
            raise ValueError("home_audio runtime setting must be an object")
        result = cls(**raw)
        for url in (result.tts_url, result.media_base_url):
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("home_audio URLs must be HTTP(S) base URLs without credentials/query/fragment")
        if not isinstance(result.targets, (tuple, list)) or not result.targets or not all(
            isinstance(target, str) and re.fullmatch(r"media_player\.[a-z0-9_]+", target)
            for target in result.targets
        ):
            raise ValueError("home_audio targets must be a nonempty media_player allowlist")
        return result


def wav_duration(raw: bytes) -> float:
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
            if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2 or wav.getnchannels() not in (1, 2):
                raise ValueError("Expected 16-bit PCM WAV audio")
            expected = wav.getnframes() * wav.getnchannels() * wav.getsampwidth()
            if len(wav.readframes(wav.getnframes())) != expected:
                raise ValueError("Truncated WAV audio")
    except (wave.Error, EOFError, ZeroDivisionError) as exc:
        raise ValueError("Invalid WAV audio") from exc
    if not 0 < duration <= 300:
        raise ValueError("Speech must be between zero and five minutes")
    return duration


class HomeAudioClient:
    def __init__(self, config, settings: HomeAudioSettings, http: httpx.AsyncClient):
        self.settings = settings
        self.http = http
        self.config = config

    async def _ha(self, method: str, path: str, **kwargs):
        # Resolved per request (TASK-1476): the Integrations record first, so
        # a saved or rotated token is used without a restart.
        try:
            credentials = await asyncio.to_thread(resolve_ha_credentials, self.config)
        except HomeAssistantUnavailable as error:
            raise ValueError(f"Home Assistant connection unavailable: {error.code}") from None
        if credentials is None:
            raise ValueError("Home Assistant URL/token are not configured")
        response = await self.http.request(method, credentials.base_url + "/api/" + path,
                                           headers={"Authorization": f"Bearer {credentials.token}"},
                                           timeout=15, **kwargs)
        response.raise_for_status()
        return response.json()

    async def devices(self) -> list[dict]:
        states = await self._ha("GET", "states")
        return [{"entity_id": s["entity_id"], "name": s.get("attributes", {}).get("friendly_name"),
                 "state": s["state"]} for s in states if s["entity_id"] in self.settings.targets]

    async def state(self, target: str) -> dict:
        if target not in self.settings.targets:
            raise ValueError("Target is not in the home_audio allowlist; use home_audio_devices")
        return await self._ha("GET", f"states/{target}")

    async def render(self, text: str, voice: str) -> bytes:
        response = await self.http.post(self.settings.tts_url.rstrip("/") + "/v1/tts/render",
                                        json={"text": text, "voice": voice}, timeout=190)
        response.raise_for_status()
        raw = response.content
        if len(raw) > 60_000_000:
            raise ValueError("Speech response is too large")
        wav_duration(raw)
        return raw

    def media_url(self, asset_id: str, version: str | None = None) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", asset_id):
            raise ValueError("Invalid asset ID")
        if version is not None and not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", version):
            raise ValueError("Invalid playback version")
        url = f"{self.settings.media_base_url.rstrip('/')}/v1/uploads/{asset_id}"
        return f"{url}?v={version}" if version else url

    async def play(self, target: str, asset_id: str, *, version: str | None = None):
        if target not in self.settings.targets:
            raise ValueError("Target is no longer allowed")
        await self._ha("POST", "services/media_player/play_media", json={
            "entity_id": target, "media_content_id": self.media_url(asset_id, version),
            "media_content_type": "audio/wav",
        })

    async def wait_idle(self, target: str, expires_at: float):
        import time
        while time.time() < expires_at:
            state = (await self.state(target))["state"]
            if state in {"off", "idle"}:
                return
            if state in {"unavailable", "unknown"}:
                raise RuntimeError("Target is unavailable")
            # Do not replace music, buffering, or paused media.
            await asyncio.sleep(1)
        raise TimeoutError("Announcement expired waiting for an idle speaker")

    async def wait_finished(self, target: str, asset_id: str, duration: float, *, version: str | None = None):
        loop = asyncio.get_running_loop()
        started = None
        deadline = loop.time() + 20
        expected_url = self.media_url(asset_id, version)
        while loop.time() < deadline:
            state = await self.state(target)
            status = state["state"]
            if status in {"unknown", "unavailable"}:
                raise RuntimeError("Target became unavailable; playback outcome unknown")
            attributes = state.get("attributes", {})
            content_id = attributes.get("media_content_id")
            if status == "playing" and content_id == expected_url and started is None:
                started = loop.time()
                deadline = started + duration + 30
            if started is not None:
                if content_id and content_id != expected_url and status == "playing":
                    raise RuntimeError("Another media item replaced the announcement")
                if status in {"off", "idle"}:
                    if loop.time() - started < max(0, duration - 2):
                        raise RuntimeError("Playback ended early; completion not confirmed")
                    return
            await asyncio.sleep(0.5)
        raise TimeoutError("Playback start/completion not confirmed; announcement will not be replayed")

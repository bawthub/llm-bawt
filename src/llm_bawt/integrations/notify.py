"""Phone push notifications through Home Assistant ``notify`` services (TASK-1531).

Targets, quiet hours and limits come from the global ``notify`` runtime
setting, never from source::

    {"ha_services": ["mobile_app_<device>"],
     "quiet_hours": {"start": "22:00", "end": "07:00", "timezone": "<IANA zone>"},
     "per_bot_interval_seconds": 600, "global_per_hour": 6,
     "dedupe_window_seconds": 3600}

Only ``ha_services`` is required. The Home Assistant URL and token are the
shared integration credentials (:mod:`.ha_credentials`), resolved per call.
Results never carry the token or a Home Assistant response body.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from llm_bawt.integrations.ha_credentials import (
    HACredentials, HomeAssistantUnavailable, resolve_ha_credentials,
)
from llm_bawt.integrations.notify_store import Limits, NotifyStore
from llm_bawt.runtime_settings import RuntimeSettingsStore

logger = logging.getLogger(__name__)

SETTING_KEY = "notify"
URGENCIES = ("normal", "high")
_SERVICE = re.compile(r"[a-z0-9_]+")
_CLOCK = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")
_TIMEOUT = httpx.Timeout(10.0)


class NotifyNotConfigured(ValueError):
    """The ``notify`` setting is absent or unusable; the message says why."""


@dataclass(frozen=True)
class QuietHours:
    start: dtime
    end: dtime
    zone: ZoneInfo

    def active(self, now: datetime) -> bool:
        local = now.astimezone(self.zone).time()
        if self.start <= self.end:
            return self.start <= local < self.end
        return local >= self.start or local < self.end  # wraps past midnight


@dataclass(frozen=True)
class NotifySettings:
    services: tuple[str, ...]
    quiet_hours: QuietHours | None = None
    limits: Limits = field(default_factory=lambda: Limits(600, 6, 3600))

    @classmethod
    def load(cls, config) -> NotifySettings:
        raw = RuntimeSettingsStore(config).get_scope_settings("global", "*").get(SETTING_KEY)
        return cls.parse(raw)

    @classmethod
    def parse(cls, raw) -> NotifySettings:
        if raw is None:
            raise NotifyNotConfigured("notify setting is not set")
        if not isinstance(raw, dict):
            raise NotifyNotConfigured("notify setting must be an object")
        services = raw.get("ha_services")
        if (not isinstance(services, list) or not services
                or not all(isinstance(s, str) and _SERVICE.fullmatch(s) for s in services)):
            raise NotifyNotConfigured("notify.ha_services must be a nonempty list of notify service names")
        return cls(services=tuple(dict.fromkeys(services)),
                   quiet_hours=_quiet_hours(raw.get("quiet_hours")),
                   limits=Limits(_positive(raw, "per_bot_interval_seconds", 600),
                                 _positive(raw, "global_per_hour", 6),
                                 _positive(raw, "dedupe_window_seconds", 3600)))


def _positive(raw: dict, key: str, default: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NotifyNotConfigured(f"notify.{key} must be a positive integer")
    return value


def _quiet_hours(raw) -> QuietHours | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise NotifyNotConfigured("notify.quiet_hours must be an object")
    clocks = []
    for key in ("start", "end"):
        match = _CLOCK.fullmatch(str(raw.get(key, "")))
        if not match:
            raise NotifyNotConfigured(f"notify.quiet_hours.{key} must be HH:MM")
        clocks.append(dtime(int(match[1]), int(match[2])))
    try:
        zone = ZoneInfo(str(raw.get("timezone") or ""))
    except (ZoneInfoNotFoundError, ValueError):
        raise NotifyNotConfigured("notify.quiet_hours.timezone must be an IANA time zone") from None
    if clocks[0] == clocks[1]:
        raise NotifyNotConfigured("notify.quiet_hours start and end must differ")
    return QuietHours(clocks[0], clocks[1], zone)


def resolve_link(link: str | None, origin: str) -> str | None:
    """An app-relative path or a URL on the public origin, as an absolute URL."""
    if not link:
        return None
    origin = (origin or "").rstrip("/")
    if not origin:
        raise ValueError("link needs a public origin (LLM_BAWT_PUBLIC_ORIGIN); none is set")
    if link.startswith("/") and not link.startswith("//"):
        return origin + link
    parts, own = urlsplit(link), urlsplit(origin)
    if parts.scheme == own.scheme and parts.netloc == own.netloc and not parts.username:
        return link
    raise ValueError("link must be an app-relative path or a URL on the public origin")


def build_payload(message: str, *, title: str | None, urgency: str, bot_id: str,
                  url: str | None) -> dict:
    """The Home Assistant companion-app payload; iOS and Android keys side by side."""
    data: dict = {"group": f"bawthub-{bot_id}"}
    if url:
        data["url"] = url          # iOS
        data["clickAction"] = url  # Android
    if urgency == "high":
        data["push"] = {"interruption-level": "time-sensitive"}  # iOS
        data["priority"] = "high"                                 # Android
        data["ttl"] = 0                                           # Android: deliver now
    payload: dict = {"message": message, "data": data}
    if title:
        payload["title"] = title
    return payload


async def send_to_service(http: httpx.AsyncClient, credentials: HACredentials, service: str,
                          payload: dict) -> str:
    """``sent`` or ``ha_error:<status|unreachable>``; never a body or the token."""
    try:
        response = await http.post(
            f"{credentials.base_url}/api/services/notify/{service}", json=payload,
            headers={"Authorization": f"Bearer {credentials.token}"}, timeout=_TIMEOUT)
    except httpx.HTTPError as error:
        logger.warning("notify %s unreachable: %s", service, type(error).__name__)
        return "ha_error:unreachable"
    if response.is_success:
        return "sent"
    logger.warning("notify %s returned HTTP %s", service, response.status_code)
    return f"ha_error:{response.status_code}"


class Notifier:
    """Settings → credentials → suppression → send → audit, for one verified bot."""

    def __init__(self, config, store: NotifyStore, http: httpx.AsyncClient, *,
                 now=lambda: datetime.now(timezone.utc)):
        self.config, self.store, self.http, self.now = config, store, http, now

    async def notify(self, bot_id: str, message: str, *, title: str | None = None,
                     urgency: str = "normal", link: str | None = None,
                     dedupe_key: str | None = None) -> dict:
        if urgency not in URGENCIES:
            raise ValueError("urgency must be normal or high")
        url = resolve_link(link, self.config.PUBLIC_ORIGIN)
        audit = (bot_id, urgency, message, dedupe_key)

        async def done(result: str, **extra) -> dict:
            row = await asyncio.to_thread(self.store.record, *audit, result)
            return {"result": result, "id": row, **extra}

        try:
            settings = await asyncio.to_thread(NotifySettings.load, self.config)
        except NotifyNotConfigured as error:
            return await done("not_configured", detail=str(error))
        try:
            credentials = await asyncio.to_thread(resolve_ha_credentials, self.config)
        except HomeAssistantUnavailable as error:
            return await done(f"ha_error:{error.code}")
        if credentials is None:
            return await done("not_configured", detail="Home Assistant is not connected")
        now = self.now()
        if urgency != "high" and settings.quiet_hours and settings.quiet_hours.active(now):
            return await done("suppressed:quiet_hours")
        row, admitted = await asyncio.to_thread(self.store.admit, *audit, settings.limits,
                                                now.timestamp())
        if admitted != "sending":
            return {"result": admitted, "id": row}
        payload = build_payload(message, title=title, urgency=urgency, bot_id=bot_id, url=url)
        results = {}
        for service in settings.services:
            results[service] = await send_to_service(self.http, credentials, service, payload)
        failed = [r for r in results.values() if r != "sent"]
        result = "sent"
        if failed:
            # Some phone already got it: "partial:" keeps counting toward the
            # limits so a retry can't push to that phone twice.
            result = failed[0] if len(failed) == len(results) else f"partial:{failed[0]}"
        await asyncio.to_thread(self.store.finish, row, result)
        return {"result": result, "id": row, "services": results}

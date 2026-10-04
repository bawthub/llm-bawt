"""Request-local metadata shared by the bridge and proxy adapters.

The Claude CLI supports ``ANTHROPIC_CUSTOM_HEADERS`` for requests sent to a
custom ``ANTHROPIC_BASE_URL``.  The bridge uses that channel to carry opaque,
non-secret routing metadata into the in-process proxy without putting volatile
values in the model prompt.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

CONVERSATION_HEADER = "X-LLM-Bawt-Conversation-ID"
BOT_HEADER = "X-LLM-Bawt-Bot-ID"
REQUEST_HEADER = "X-LLM-Bawt-Request-ID"
RESPONSES_TRANSPORT_HEADER = "X-LLM-Bawt-Responses-Transport"
SKILL_NAMES_HEADER = "X-LLM-Bawt-Skill-Names"

_OPAQUE_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

ProxyStatusCallback = Callable[[str, dict[str, Any]], None]
logger = logging.getLogger(__name__)


class ProxyCancellationRegistry:
    """In-process request tombstones checked before any upstream sampling."""

    def __init__(self) -> None:
        self._cancelled: set[str] = set()

    def cancel(self, request_id: str) -> None:
        if request_id:
            self._cancelled.add(request_id)

    def clear(self, request_id: str) -> None:
        self._cancelled.discard(request_id)

    def is_cancelled(self, request_id: str) -> bool:
        return request_id in self._cancelled


def valid_skill_names(value: str | None) -> tuple[str, ...]:
    """Parse the bounded comma-separated selector catalog from the bridge."""
    if not value or len(value) > 16384:
        return ()
    return tuple(
        dict.fromkeys(
            candidate
            for raw in value.split(",")
            if (candidate := raw.strip()) and _SKILL_NAME_RE.fullmatch(candidate)
        )
    )[:256]


def durable_conversation_identity(
    *, bot_id: str, session_key: str, thread_session_id: str
) -> str:
    """Return a stable opaque identity for one durable conversation.

    ``session_key`` contains the app's bot/user routing scope and
    ``thread_session_id`` is the durable DB thread rotated by ``/new``.  A
    versioned domain separator prevents accidental collisions with unrelated
    hashes.  The 128-bit UUID-shaped result matches the format historically
    accepted by the ChatGPT backend's ``session_id`` header.
    """
    seed = "\0".join(
        ("llm-bawt-proxy-conversation-v1", bot_id, session_key, thread_session_id)
    ).encode("utf-8")
    return uuid.UUID(bytes=hashlib.sha256(seed).digest()[:16]).hex


def valid_conversation_identity(value: str | None) -> str | None:
    """Accept only the opaque format emitted by the bridge."""
    candidate = (value or "").strip().lower()
    return candidate if _OPAQUE_ID_RE.fullmatch(candidate) else None


@dataclass(slots=True)
class ProxyRequestContext:
    """Mutable request telemetry; one instance is never shared across calls."""

    request_id: str
    provider: str
    bot_id: str | None = None
    conversation_id: str | None = None
    responses_transport: str | None = None
    skill_names: tuple[str, ...] = ()
    started_at: float = 0.0
    account_hash: str = "default"
    active_provider: int = 0
    active_account: int = 0
    queue_wait_ms: float = 0.0
    local_setup_ms: float | None = None
    upstream_ttfb_ms: float | None = None
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    attempt: int = 0
    status_callback: ProxyStatusCallback | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = time.perf_counter()

    @property
    def session_hash(self) -> str:
        return (self.conversation_id or "missing")[:12]

    @property
    def cache_hit_pct(self) -> float:
        if not self.input_tokens:
            return 0.0
        return 100.0 * self.cached_tokens / self.input_tokens

    def record_usage(
        self, input_tokens: int, output_tokens: int, cached_tokens: int, _created: int
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_tokens = cached_tokens

    def report_status(self, status: dict[str, Any]) -> None:
        """Publish transient proxy progress without adding assistant text."""
        if self.status_callback is None:
            return
        try:
            self.status_callback(self.request_id, status)
        except Exception:
            # UI progress is best-effort and must never turn successful
            # upstream recovery into a failed model request.
            logger.warning(
                "Proxy status callback failed request_id=%s state=%s",
                self.request_id,
                status.get("state"),
                exc_info=True,
            )


def custom_header_env(context: ProxyRequestContext) -> str:
    """Serialize proxy metadata in Claude CLI's newline header format."""
    lines = [
        f"{CONVERSATION_HEADER}: {context.conversation_id}",
        f"{BOT_HEADER}: {context.bot_id or 'unknown'}",
        f"{REQUEST_HEADER}: {context.request_id}",
    ]
    if context.responses_transport:
        lines.append(
            f"{RESPONSES_TRANSPORT_HEADER}: {context.responses_transport}"
        )
    if context.skill_names:
        names = [name for name in context.skill_names if _SKILL_NAME_RE.fullmatch(name)]
        if names:
            lines.append(f"{SKILL_NAMES_HEADER}: {','.join(names[:256])}")
    return "\n".join(lines)

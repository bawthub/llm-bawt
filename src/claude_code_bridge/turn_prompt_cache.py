"""SDK TTL intent and foreground observations, not provider expiry."""
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from agent_bridge.prompt_cache_clock import prompt_cache_clock, sdk_session_ref

from .context_env import PROMPT_CACHE_TTL_ENV

TTL_SECONDS = {"5m": 300, "1h": 3600}
_CREATION_FIELDS = {"5m": "ephemeral_5m_input_tokens", "1h": "ephemeral_1h_input_tokens"}
_NATIVE_PATHS = {"anthropic-api", "anthropic-oauth"}


def _clock_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass
class TurnPromptCache:
    """One SDK attempt, not a ledger across retries or mutable settings."""

    sdk_ttl: str
    route: str
    access_path: str | None
    requested_model: str
    started_at_ms: int
    resumed: bool
    _session_id: str | None = field(default=None, repr=False)
    _actual_model: str | None = None
    _observation: dict | None = None
    _foreground_starts: int = 0
    _maintenance: bool = False
    _oauth_relay: bool = False
    _protection_write: dict | None = None

    @classmethod
    def from_sdk_env(cls, env: Mapping, *, use_proxy: bool, access_path: str | None,
                     model: str, resumed: bool, oauth_gateway_url: str | None = None) -> TurnPromptCache:
        value = env.get(PROMPT_CACHE_TTL_ENV)
        # This records the option we passed, not the subprocess's inherited
        # defaults, internal forced-TTL rules, or actual provider cache state.
        ttl = "not_applicable" if use_proxy else value if value in TTL_SECONDS else "unset"
        route = "proxy" if use_proxy else "native" if access_path in _NATIVE_PATHS else "legacy_native"
        relay = (route == "native" and access_path == "anthropic-oauth" and bool(oauth_gateway_url)
                 and env.get("ANTHROPIC_BASE_URL") == oauth_gateway_url)
        return cls(ttl, route, access_path, model, _clock_ms(), resumed, _oauth_relay=relay)

    def bind_session(self, session_id: object) -> None:
        if isinstance(session_id, str) and session_id and session_id != self._session_id:
            self._session_id = session_id
            self._observation = None
            self._actual_model = None
            self._protection_write = None

    def suppress_estimate(self) -> None:
        """Compaction can put a maintenance request on the foreground stream."""
        self._maintenance = True
        self._protection_write = None
        if self._observation is not None:
            self._observation.pop("estimated_reuse_until_ms", None)

    def observe_start(self, event: dict, *, session_id: str, parent: str | None) -> None:
        if (parent is not None or event.get("type") != "message_start"
                or not self._session_id or session_id != self._session_id):
            return
        self._foreground_starts += 1
        self._observation = None
        message = event.get("message")
        if not isinstance(message, dict):
            return
        model = message.get("model")
        self._actual_model = model if isinstance(model, str) and model else None
        usage = message.get("usage")
        if (self.route != "native" or self.sdk_ttl not in TTL_SECONDS or not self._session_id
                or not self._actual_model or not isinstance(message.get("id"), str)
                or not isinstance(usage, dict)):
            return
        read, wrote = usage.get("cache_read_input_tokens"), usage.get("cache_creation_input_tokens")
        if (type(read) is not int or read < 0 or type(wrote) is not int or wrote < 0
                or not (read or wrote)):
            return
        observed_ms = _clock_ms()
        self._observation = {"observed_at_ms": observed_ms, "read_tokens": read, "write_tokens": wrote}
        # Reads have no entry identity or applied TTL. In particular a short
        # turn may read an earlier long-lived prefix; never introduce a shorter
        # session estimate on resume. A matching, unmixed write supports only
        # a nominal estimate for the observed foreground request, not a reset.
        buckets = usage.get("cache_creation")
        if (self._maintenance or not isinstance(buckets, dict) or not wrote or
                (self.sdk_ttl == "5m" and (self.resumed or read or self._foreground_starts != 1))):
            return
        expected = _CREATION_FIELDS[self.sdk_ttl]
        if all(type(buckets.get(key)) is int and buckets[key] == (wrote if key == expected else 0)
               for key in _CREATION_FIELDS.values()):
            self._observation["estimated_reuse_until_ms"] = observed_ms + TTL_SECONDS[self.sdk_ttl] * 1000
            if self.sdk_ttl == "1h" and self._oauth_relay:
                clock = prompt_cache_clock()
                if clock is not None:
                    domain, elapsed_ms = clock
                    self._protection_write = {
                        "source": "native_oauth_relay_1h_write", "clock_domain": domain,
                        "observed_ms": elapsed_ms, "model": self._actual_model,
                        "write_tokens": wrote,
                    }

    def snapshot(self) -> dict:
        result = {
            "sdk_ttl": self.sdk_ttl, "route": self.route, "access_path": self.access_path,
            "requested_model": self.requested_model, "started_at_ms": self.started_at_ms,
        }
        if self._session_id:
            result["sdk_session_ref"] = sdk_session_ref(self._session_id)
        if self._protection_write is not None:
            result["protection_write"] = self._protection_write.copy()
        if self._actual_model:
            result["actual_model"] = self._actual_model
        if self._observation is not None:
            result.update(self._observation)
        return result

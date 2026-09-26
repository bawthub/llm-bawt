"""OpenAI Responses API adapter authenticated by the ChatGPT subscription OAuth bundle.

TASK-636 Phase 2: the adapter no longer reads or writes ``~/.codex/auth.json``.
Token resolution goes through the app's broker endpoint
``GET /v1/providers/openai_chatgpt/token`` — the app is the sole owner and
refresher of the OAuth bundle (encrypted at rest in the CredentialStore).
This adapter is a read-only consumer with a small in-memory cache.

The upstream request shaping (header quirks, param stripping, reasoning
defaults) is unchanged from the TASK-270 live-smoke verification.
"""

from __future__ import annotations

import hashlib
import asyncio
import json
import logging
import os
import time
from typing import ClassVar

import httpx

from .base import ProviderAdapter
from .. import retry as retry_mod
from ..request_context import ProxyRequestContext
from ..transport_policy import (
    DEFAULT_RESPONSES_TRANSPORT,
    RESPONSES_TRANSPORT_LITE_WS,
    normalize_responses_transport,
)

logger = logging.getLogger(__name__)


# ── Request shaping constants (unchanged from TASK-270) ─────────────────

# Responses API base. ChatGPT *subscription* OAuth tokens do NOT carry the
# platform ``api.responses.write`` scope, so they 401 against
# ``api.openai.com/v1/responses``. They DO authenticate against the ChatGPT
# backend's codex endpoint (the same surface ``codex exec`` uses).
DEFAULT_API_BASE = "https://chatgpt.com/backend-api/codex"
API_BASE_ENV = "OPENAI_BASE_URL"

# Params the ChatGPT codex backend rejects with HTTP 400 "Unsupported
# parameter" even though the standard Responses API accepts them.
_UNSUPPORTED_PARAMS = ("temperature", "top_p", "max_output_tokens")

# The codex backend hard-requires a non-empty ``instructions`` field.
_FALLBACK_INSTRUCTIONS = "You are a helpful coding assistant."

# Transport defaults must stay model-agnostic. Per-endpoint policy is resolved
# from model_endpoints.serving_config by the app and reaches the proxy through
# the bridge's existing per-turn effort setting. Explicit inbound reasoning and
# the legacy environment override still win; absent configuration fails safe to
# high rather than silently reducing model quality.
DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORT_ENV = "OPENAI_CHATGPT_REASONING_EFFORT"
_VALID_EFFORT = {"none", "low", "medium", "high", "xhigh", "max"}

# Cache buffer — re-fetch from the broker when the token is this close to
# expiry. The app's proactive loop refreshes at exp−20min, so the broker
# almost always returns a fresh token without an upstream refresh.
_CACHE_BUFFER_S = 5 * 60


def _prompt_cache_key(responses_body: dict) -> str:
    """Derive a stable per-conversation cache key from the opening prefix.

    The Claude SDK doesn't pass a conversation/thread id through
    ``/v1/messages``, so for the ChatGPT codex backend we approximate codex
    CLI's stable thread key by hashing the invariant opening: instructions plus
    the first user content item.
    """
    instructions = responses_body.get("instructions") or ""
    first_user_json = ""
    for item in responses_body.get("input") or []:
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        first_user_json = json.dumps(
            item.get("content") or [], sort_keys=True, separators=(",", ":")
        )
        break
    seed = f"{instructions}\n\n{first_user_json}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()


class OpenAIChatGPTAdapter(ProviderAdapter):
    """ChatGPT-subscription-authenticated OpenAI Responses API client.

    TASK-636: tokens come from the app's broker endpoint. The adapter caches
    the token in-memory and re-fetches when near expiry. It NEVER reads or
    writes ``~/.codex/auth.json`` — the app owns the full refresh chain.
    """

    name: ClassVar[str] = "openai_chatgpt"

    def __init__(self) -> None:
        super().__init__()
        self._cached_token: str | None = None
        self._cached_account_id: str | None = None
        self._cached_expires_at: float | None = None
        self._authorize_lock = asyncio.Lock()
        self._chatgpt_transport = None

    @staticmethod
    def _responses_transport(context: ProxyRequestContext | None) -> str:
        configured = (
            context.responses_transport if context is not None else None
        )
        return normalize_responses_transport(configured) or DEFAULT_RESPONSES_TRANSPORT

    async def open_stream(self, *, client, body, headers, bearer, base_url, context):
        if self._responses_transport(context) != RESPONSES_TRANSPORT_LITE_WS:
            return await super().open_stream(
                client=client, body=body, headers=headers, bearer=bearer,
                base_url=base_url, context=context,
            )
        from ..chatgpt_transport import ChatGPTResponsesTransport

        if self._chatgpt_transport is None:
            self._chatgpt_transport = ChatGPTResponsesTransport()
        return await self._chatgpt_transport.open(
            body=body, headers=headers, bearer=bearer, base_url=base_url,
            context=context, http_client=client,
        )

    async def close(self):
        if self._chatgpt_transport is not None:
            await self._chatgpt_transport.close()
            self._chatgpt_transport = None
        await super().close()

    def retry_policy(
        self,
        upstream_model: str,
        context: ProxyRequestContext | None = None,
    ) -> retry_mod.RetryPolicy:
        policy = super().retry_policy(upstream_model, context)
        if self._responses_transport(context) == RESPONSES_TRANSPORT_LITE_WS:
            # The bridge watchdog is 600s. Two attempts preserve one safe
            # reconnect while a dead first-event path remains far below it.
            policy.max_attempts = 2
        return policy

    # ── broker token resolution ──────────────────────────────────────────
    def _cache_valid(self) -> bool:
        if not self._cached_token or self._cached_expires_at is None:
            return False
        return (self._cached_expires_at - time.time()) > _CACHE_BUFFER_S

    def account_hash(self) -> str:
        if not self._cached_account_id:
            return "pending"
        return hashlib.sha256(
            self._cached_account_id.encode("utf-8")
        ).hexdigest()[:12]

    def _fetch_broker_token(self, *, force: bool = False) -> tuple[str, str | None, float | None]:
        """Ask the app for the current ChatGPT access token.

        Returns ``(token, account_id, expires_at)``.
        """
        api_url = (os.environ.get("LLM_BAWT_API_URL") or "").rstrip("/")
        if not api_url:
            raise RuntimeError(
                "LLM_BAWT_API_URL not set — cannot reach the token broker"
            )
        headers = {}
        secret = os.environ.get("BRIDGE_CLAUDE_TOKEN_SECRET")
        if secret:
            headers["X-Bridge-Token"] = secret
        try:
            resp = httpx.get(
                f"{api_url}/v1/providers/openai_chatgpt/token",
                params={"force": "true"} if force else None,
                headers=headers,
                timeout=25.0,
            )
            if resp.is_error:
                raise RuntimeError(
                    f"ChatGPT token broker returned {resp.status_code}: "
                    f"{(resp.text or '')[:200]}"
                )
            payload = resp.json()
            token = payload.get("access_token")
            if not token:
                raise RuntimeError(
                    f"ChatGPT token broker returned state={payload.get('state')} "
                    f"with no access_token"
                )
            logger.info(
                "Fetched ChatGPT access token from app broker (state=%s)",
                payload.get("state"),
            )
            return (
                token,
                payload.get("account_id"),
                payload.get("expires_at"),
            )
        except httpx.HTTPError as e:
            raise RuntimeError(f"ChatGPT token broker unreachable: {e}") from e

    # ── ProviderAdapter ──────────────────────────────────────────────────
    async def authorize(self) -> tuple[str, str]:
        """Resolve the ChatGPT access token via the app broker.

        Uses an in-memory cache to avoid a broker round-trip on every request.
        The app's proactive loop keeps the token fresh, so the broker almost
        always returns without doing an upstream refresh.
        """
        if not self._cache_valid():
            # Single-flight cold/near-expiry resolution. Re-check after taking
            # the lock so a burst of parallel streams performs one broker call.
            async with self._authorize_lock:
                if not self._cache_valid():
                    token, account_id, expires_at = await asyncio.to_thread(
                        self._fetch_broker_token
                    )
                    self._cached_token = token
                    self._cached_account_id = account_id
                    # Broker contract: Unix epoch SECONDS. Be tolerant of an
                    # accidental millisecond producer at this boundary while
                    # keeping seconds as the explicit canonical unit.
                    if isinstance(expires_at, (int, float)):
                        value = float(expires_at)
                        if value >= 100_000_000_000:
                            logger.warning(
                                "ChatGPT broker expires_at looked like epoch-ms; normalizing"
                            )
                            value /= 1000.0
                        self._cached_expires_at = value
                    else:
                        self._cached_expires_at = None

        base_url = os.getenv(API_BASE_ENV) or DEFAULT_API_BASE
        assert self._cached_token is not None
        return self._cached_token, base_url

    def reasoning_codec(self, upstream_model: str, base_url: str):
        from ..reasoning import ReasoningCodec

        # Missing account identity must not collapse unrelated credentials into
        # a shared replay scope. Token refresh does not change account identity.
        if not self._cached_account_id:
            return None
        return ReasoningCodec.for_account(
            self.name, upstream_model, base_url, self._cached_account_id,
        )

    # ── upstream quirks (unchanged from TASK-270) ────────────────────────
    def extra_headers(
        self,
        responses_body: dict,
        context: ProxyRequestContext | None = None,
    ) -> dict[str, str]:
        """Headers the ChatGPT codex backend requires/expects."""
        # Real bridge traffic always carries the durable identity. The
        # deterministic opening-prefix fallback preserves compatibility for
        # direct proxy callers without reintroducing bridge-global affinity.
        session_id = (
            context.conversation_id if context and context.conversation_id
            else _prompt_cache_key(responses_body)
        )
        headers = {
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "session_id": session_id,
        }
        if self._cached_account_id:
            headers["chatgpt-account-id"] = self._cached_account_id
            if context is not None:
                context.account_hash = self.account_hash()
        return headers

    def prepare_request(
        self,
        responses_body: dict,
        context: ProxyRequestContext | None = None,
    ) -> dict:
        """Adapt the translated Responses body to the codex backend's quirks."""
        for key in _UNSUPPORTED_PARAMS:
            responses_body.pop(key, None)
        responses_body["stream"] = True
        responses_body["store"] = False
        if not responses_body.get("instructions"):
            responses_body["instructions"] = _FALLBACK_INSTRUCTIONS
        responses_body["prompt_cache_key"] = (
            context.conversation_id if context and context.conversation_id
            else _prompt_cache_key(responses_body)
        )
        if "reasoning" not in responses_body:
            effort = (os.getenv(REASONING_EFFORT_ENV) or "").strip().lower()
            if effort not in _VALID_EFFORT:
                effort = DEFAULT_REASONING_EFFORT
            responses_body["reasoning"] = {"effort": effort}
        responses_body["reasoning"].setdefault("summary", "auto")
        # Match Codex: standard Responses leaves context to the server default
        # (current_turn); Lite explicitly retains all turns. Native reasoning
        # items are replayed in BOTH modes, including with store:false.
        if self._responses_transport(context) == RESPONSES_TRANSPORT_LITE_WS:
            responses_body["reasoning"]["context"] = "all_turns"
        else:
            responses_body["reasoning"].pop("context", None)
        responses_body.setdefault("include", [])
        if "reasoning.encrypted_content" not in responses_body["include"]:
            responses_body["include"].append("reasoning.encrypted_content")
        return responses_body

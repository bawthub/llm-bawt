"""Own-app ChatGPT SIWC public Responses route; never the Codex backend.

Uses the existing request-local translator, SSE supervisor, tool safety and
reasoning envelopes. The app broker is the only OAuth owner/refresher. No env
credential fallback, ambient OpenAI base override, or automatic provider switch.
"""
from __future__ import annotations

import asyncio
import hashlib
import os

import httpx

from agent_bridge import bridge_token

from .base import ProviderAdapter
from .openai_chatgpt import _prompt_cache_key
from .. import retry as retry_mod
from ..reasoning import ReasoningCodec

BASE_URL = "https://api.openai.com/v1"
TOOL_NAMESPACE = "bawthub"
_UNSUPPORTED = (
    "background", "conversation", "max_output_tokens", "max_tool_calls", "metadata",
    "moderation", "multi_agent", "prompt", "prompt_cache_retention", "safety_identifier",
    "temperature", "top_logprobs", "top_p", "truncation", "user", "previous_response_id",
)


class ChatGPTAppAdapter(ProviderAdapter):
    name = "chatgpt_app"
    capture_codex_usage = False  # Separate plan: never overwrite Codex quota snapshots.

    def __init__(self):
        super().__init__()
        self._account_id = None

    def _fetch_token(self):
        app = (os.getenv("LLM_BAWT_API_URL") or "").rstrip("/")
        if not app:
            raise retry_mod.BrokerAuthorizationError("App token broker is not configured")
        try:
            response = httpx.get(
                f"{app}/v1/providers/chatgpt-app/token", headers=bridge_token.headers(), timeout=30.0,
            )
            if response.status_code != 200:
                raise retry_mod.BrokerAuthorizationError("ChatGPT app connection is unavailable; check Providers")
            payload = response.json()
            if payload.get("state") != "ok" or not payload.get("access_token") or not payload.get("account_id"):
                raise retry_mod.BrokerAuthorizationError("Reconnect ChatGPT (app-authorized) in Providers")
            return payload
        except (httpx.HTTPError, ValueError):
            raise retry_mod.BrokerAuthorizationError("ChatGPT app token broker is unavailable") from None

    async def authorize(self):
        # Resolve every request, so disconnect/reconnect isn't hidden behind a
        # long-lived bridge token cache. No bridge-side refresh or persistence.
        payload = await asyncio.to_thread(self._fetch_token)
        self._account_id = payload["account_id"]
        return payload["access_token"], BASE_URL

    def account_hash(self):
        return hashlib.sha256(self._account_id.encode()).hexdigest()[:12] if self._account_id else "pending"

    def retry_auth_identity(self, base_url):
        return (self.name, base_url, self._account_id) if self._account_id else None

    def reasoning_codec(self, upstream_model, base_url):
        if self._account_id:
            return ReasoningCodec.for_account(self.name, upstream_model, base_url, self._account_id)
        return None

    def extra_headers(self, body, context=None):
        if context is not None:
            context.account_hash = self.account_hash()
        return {}  # No Codex originator, account header, or impersonated client identity.

    def prepare_request(self, body, context=None):
        for key in _UNSUPPORTED:
            body.pop(key, None)
        body.update(stream=True, store=False)
        body.setdefault("reasoning", {"effort": "high"})
        body["reasoning"].pop("context", None)
        body["reasoning"].setdefault("summary", "auto")
        body.setdefault("include", [])
        if "reasoning.encrypted_content" not in body["include"]:
            body["include"].append("reasoning.encrypted_content")
        body["prompt_cache_key"] = (
            context.conversation_id if context and context.conversation_id else _prompt_cache_key(body)
        )
        # Public SIWC accepts function tools in namespaces, not flat top-level
        # functions. Names stay identical for SDK local/MCP dispatch and output
        # translation. Replay carries the same namespace + call_id.
        functions = body.get("tools", [])
        if functions:
            if any(tool.get("type") != "function" for tool in functions):
                raise ValueError("ChatGPT app route supports local function tools only")
            body["tools"] = [{"type": "namespace", "name": TOOL_NAMESPACE,
                              "description": "BawtHub local tools", "tools": functions}]
        for item in body.get("input", []):
            if item.get("type") == "function_call":
                item["namespace"] = TOOL_NAMESPACE
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            kind = choice.get("type")
            if kind in {"tool", "function"}:
                body["tool_choice"] = {"type": "function", "name": choice["name"], "namespace": TOOL_NAMESPACE}
            elif kind in {"any", "auto", "none"}:
                body["tool_choice"] = "required" if kind == "any" else kind
            else:
                body.pop("tool_choice", None)
        return body

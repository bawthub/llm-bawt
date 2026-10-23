"""xAI (Grok) adapter — OpenAI Responses API, API-key auth.

Unlike Z.AI (native Anthropic passthrough) and the ChatGPT codex backend
(a quirky non-standard Responses surface), xAI exposes a **standard
OpenAI-compatible Responses API**:

    https://api.x.ai/v1/responses      Authorization: Bearer $XAI_API_KEY

Verified live against grok-4.5: it accepts the exact body our
``translate.anthropic_to_responses`` emits (``store:false``, ``stream:true``,
``max_output_tokens``, ``temperature``, function ``tools``, and
``reasoning:{effort, summary}``) and streams the same Responses events
(``response.reasoning_summary_text.delta``, ``function_call`` output items)
that ``stream.py`` already converts back to Anthropic SSE.

So this adapter needs **no** ``call``/``prepare_request`` override — the
default Responses path in ``ProviderAdapter`` handles the whole round-trip.
All it supplies is credentials: ``authorize()`` returns
``(bearer, base_url)`` and the shared base class does the rest.

Auth: a standard xAI API key, read on every call from the app's API-key
broker (``/v1/providers/xai/token``, the DB-connected key). Only when the
broker says no key is connected (503) does the legacy bridge env apply:
``XAI_API_KEY`` (``LLM_BAWT_XAI_API_KEY``/``GROK_API_KEY`` as aliases), with
a warning. Models are namespaced ``xai/<model>``
(e.g. ``xai/grok-4.5``, ``xai/grok-4.3``); the proxy strips the ``xai/``
prefix before handing the bare upstream model to ``call``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import ClassVar

from ...provider_key_broker import (
    BrokerKeyError,
    BrokerKeyNotInstalled,
    fetch_provider_api_key,
)
from .base import ProviderAdapter
from ..request_context import ProxyRequestContext

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.x.ai/v1"
BASE_URL_ENV = "XAI_BASE_URL"
API_KEY_ENVS = ("XAI_API_KEY", "LLM_BAWT_XAI_API_KEY", "GROK_API_KEY")


def _env_true(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class XaiAdapter(ProviderAdapter):
    """Grok via xAI's standard OpenAI Responses API (default call path)."""

    name: ClassVar[str] = "xai"

    def _api_key(self) -> str:
        """The DB-connected key from the app broker (TASK-1295 B1b).

        Only an explicit "nothing connected" answer (503) may fall back to the
        legacy bridge-environment key; any other broker failure surfaces so a
        stale env key never masks a broken or rotated connection.
        """
        try:
            return fetch_provider_api_key("xai")
        except BrokerKeyNotInstalled:
            for env in API_KEY_ENVS:
                key = os.getenv(env)
                if key:
                    logger.warning(
                        "Using %s from the bridge environment because no "
                        "DB-connected xAI key is installed (legacy)", env,
                    )
                    return key
            raise RuntimeError(
                "xAI API key required: connect an xAI API key in Settings → "
                "Providers."
            ) from None
        except BrokerKeyError as exc:
            raise RuntimeError(f"xAI API key unavailable: {exc}") from exc

    @staticmethod
    def _base_url() -> str:
        return (os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")

    async def authorize(self) -> tuple[str, str]:
        """Return ``(bearer, base_url)`` for the shared Responses-API path.

        xAI keys are static — no refresh, no OAuth bundle — so this just
        reads the key (off the event loop: the broker read is blocking I/O)
        and hands back the API root the ``openai`` client appends
        ``/responses`` to.
        """
        return await asyncio.to_thread(self._api_key), self._base_url()

    def prepare_request(
        self,
        responses_body: dict,
        context: ProxyRequestContext | None = None,
    ) -> dict:
        """Inject xAI's native **server-side** search tools so Grok can search
        the live web inside its own turn.

        The Claude CLI's ``WebSearch``/``WebFetch`` are Anthropic server tools
        (stripped upstream — they hang on the proxy path). xAI has its own,
        different, server-side ``web_search`` (and ``x_search``) that run on
        api.x.ai and fold results + inline ``[[N]](url)`` citations straight
        into the answer. They're not function tools, so the SDK never sends
        them — we add them here. Grok only invokes them when a query actually
        needs the web (verified: trivial questions fire 0 searches), so the
        per-call cost ($5/1k web-search calls + tokens) is demand-driven.

        Knobs (default: web on, x off):
            XAI_WEB_SEARCH=0   disable native web search
            XAI_X_SEARCH=1     enable X/Twitter search
        """
        tools = list(responses_body.get("tools") or [])
        existing_types = {
            t.get("type") for t in tools if isinstance(t, dict)
        }
        native: list[dict] = []
        if _env_true("XAI_WEB_SEARCH", default=True):
            native.append({"type": "web_search"})
        if _env_true("XAI_X_SEARCH", default=False):
            native.append({"type": "x_search"})
        added = False
        for nt in native:
            if nt["type"] not in existing_types:
                tools.append(nt)
                added = True
        if added:
            responses_body["tools"] = tools
            logger.debug(
                "xAI adapter injected native search tools: %s",
                [t["type"] for t in native],
            )
        return responses_body

"""Upstream model-catalog discovery providers.

The Add Model dialog consumes one normalized contract regardless of whether the
source is a local bridge, an SDK, or a provider HTTP endpoint.  Provider-specific
credential and response handling belongs here rather than in the models route.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import httpx

from ..utils.config import Config


class ModelDiscoveryError(RuntimeError):
    """Expected discovery failure with an HTTP status suitable for the API."""

    def __init__(self, message: str, *, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


# The OpenAI /v1/models catalog is authoritative for availability but does not
# include context windows or display metadata. Enrich only models confirmed by
# that live response, using facts from OpenAI's official model documentation.
_OPENAI_DOCUMENTED_METADATA: dict[str, dict[str, Any]] = {
    "gpt-6-astra": {
        "description": "Astra — OpenAI's flagship model for complex reasoning and coding.",
        "context_length": 1_050_000,
    },
}


class ModelDiscoveryProvider(ABC):
    """Provider-specific source for normalized upstream model metadata."""

    aliases: tuple[str, ...] = ()

    @abstractmethod
    def fetch(self) -> list[dict[str, Any]]:
        """Return normalized ``id`` / ``description`` model entries."""


def _normalize(models: Iterable[object]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("model_id")
        if not model_id:
            continue
        row: dict[str, Any] = {
            "id": str(model_id),
            "description": str(
                item.get("description")
                or item.get("summary")
                or item.get("display_name")
                or item.get("name")
                or ""
            ),
        }
        context_length = item.get("context_length")
        if isinstance(context_length, int) and not isinstance(context_length, bool):
            row["context_length"] = context_length
        normalized.append(row)
    return normalized


class ExistingFetcherProvider(ModelDiscoveryProvider):
    """Adapter for the established model-manager fetch functions."""

    def __init__(
        self,
        *,
        aliases: tuple[str, ...],
        label: str,
        fetcher: Callable[..., tuple[bool, list[dict[str, Any]]]],
        key_envs: tuple[str, ...] = (),
        provider_id: str | None = None,
        config: Config | None = None,
        config_attr: str | None = None,
        missing_key_message: str | None = None,
        pass_key: bool = False,
    ):
        self.aliases = aliases
        self.label = label
        self._fetcher = fetcher
        self._key_envs = key_envs
        self._provider_id = provider_id
        self._config = config
        self._config_attr = config_attr
        self._missing_key_message = missing_key_message
        self._pass_key = pass_key

    def fetch(self) -> list[dict[str, Any]]:
        key = None
        if self._provider_id and self._config is not None:
            from .providers.api_key import resolve_api_key

            key = resolve_api_key(
                self._config,
                self._provider_id,
                env_vars=self._key_envs,
                config_attr=self._config_attr,
            )
        if key is None:
            key = next((os.getenv(name) for name in self._key_envs if os.getenv(name)), None)
        if self._key_envs and not key:
            raise ModelDiscoveryError(
                self._missing_key_message or f"{self.label} discovery requires an API key",
                status_code=503,
            )
        ok, models = self._fetcher(key) if self._pass_key else self._fetcher()
        if not ok:
            raise ModelDiscoveryError(
                f"Failed to fetch {self.label} model catalog from upstream"
            )
        return self._postprocess(_normalize(models))

    def _postprocess(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provider-specific shaping of the normalized rows (default: none)."""
        return rows


class OpenAIDiscoveryProvider(ExistingFetcherProvider):
    """OpenAI API catalog, enriched with documented metadata it omits."""

    def _postprocess(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {**row, **_OPENAI_DOCUMENTED_METADATA.get(row["id"], {})}
            for row in rows
        ]


#: Window the Claude Code CLI assumes for a plain ``claude-*`` model name, and
#: the window it switches to when the name carries the ``[1m]`` suffix. The CLI
#: ignores ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` for canonical Claude names, so the
#: suffix is the only way to get the large window (verified in CLI 2.1.283).
CLAUDE_CODE_STANDARD_WINDOW = 200_000
CLAUDE_CODE_1M_WINDOW = 1_000_000
CLAUDE_CODE_1M_SUFFIX = "[1m]"


class AnthropicDiscoveryProvider(ExistingFetcherProvider):
    """Anthropic catalog expressed as the model ids the Claude Code CLI accepts.

    Every Anthropic catalog endpoint runs through the Claude Code harness
    (``model_catalog`` maps vendor ``anthropic`` to ``claude-code``), so the
    upstream id must be the CLI's slug, not the bare API id. A model whose API
    window exceeds the CLI's standard window is offered twice: the bare id at
    the standard window and the ``[1m]`` id at the large window. Each choice
    carries the window the CLI will actually enforce, so the catalog can't
    claim a window the session never gets.
    """

    def _postprocess(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            api_window = row.get("context_length")
            if not isinstance(api_window, int) or api_window <= CLAUDE_CODE_STANDARD_WINDOW:
                out.append(row)
                continue
            label = row.get("description") or row["id"]
            out.append({
                **row,
                "description": f"{label} · 200K context",
                "context_length": CLAUDE_CODE_STANDARD_WINDOW,
            })
            out.append({
                **row,
                "id": f"{row['id']}{CLAUDE_CODE_1M_SUFFIX}",
                "description": f"{label} · 1M context",
                "context_length": min(api_window, CLAUDE_CODE_1M_WINDOW),
            })
        return out


class CodexBridgeDiscoveryProvider(ModelDiscoveryProvider):
    """Authenticated ChatGPT subscription catalog from the Codex bridge."""

    aliases = ("codex",)

    def fetch(self) -> list[dict[str, Any]]:
        url = os.getenv(
            "CODEX_BRIDGE_MODELS_URL", "http://codex-bridge:8682/models"
        )
        try:
            response = httpx.get(url, timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = None
            try:
                payload = exc.response.json()
                if isinstance(payload, dict):
                    detail = payload.get("detail") or payload.get("error")
            except ValueError:
                pass
            message = str(detail).strip() if detail else f"HTTP {exc.response.status_code}"
            raise ModelDiscoveryError(
                f"ChatGPT subscription model discovery failed: {message}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelDiscoveryError(
                f"ChatGPT subscription model discovery failed: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelDiscoveryError(
                "ChatGPT subscription model discovery returned non-JSON data"
            ) from exc
        data = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ModelDiscoveryError(
                "ChatGPT subscription model discovery returned an invalid payload"
            )
        models = _normalize(
            {
                **item,
                "context_length": item.get("context_window"),
            }
            for item in data
            if isinstance(item, dict)
        )
        if not models:
            raise ModelDiscoveryError(
                "ChatGPT subscription model discovery returned no available models"
            )
        return models


class KimiCodingDiscoveryProvider(ModelDiscoveryProvider):
    """Kimi For Coding subscription catalog from its OpenAI-compatible API."""

    aliases = ("kimi", "kimi_coding", "kimi-code")
    _url = "https://api.kimi.com/coding/v1/models"
    _key_envs = ("KIMI_CODING_API_KEY", "KIMI_API_KEY")

    def fetch(self) -> list[dict[str, Any]]:
        key = next((os.getenv(name) for name in self._key_envs if os.getenv(name)), None)
        if not key:
            raise ModelDiscoveryError(
                "Kimi discovery requires KIMI_CODING_API_KEY in env",
                status_code=503,
            )
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        try:
            response = httpx.get(self._url, headers=headers, timeout=20.0)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                raise ModelDiscoveryError(
                    "Kimi rejected KIMI_CODING_API_KEY while listing models",
                    status_code=503,
                ) from exc
            raise ModelDiscoveryError(
                f"Kimi model catalog returned HTTP {status}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelDiscoveryError(
                f"Network error reaching Kimi model catalog: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelDiscoveryError("Kimi model catalog returned non-JSON data") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ModelDiscoveryError("Kimi model catalog returned an invalid payload")

        # Kimi currently lists Highspeed even for BASIC subscriptions, then
        # rejects inference with 401. Hide that known-unusable choice; upgraded
        # plans keep the full catalog. A usage-read failure is non-fatal and
        # falls back to the provider's model list.
        try:
            usage = httpx.get(
                "https://api.kimi.com/coding/v1/usages",
                headers=headers,
                timeout=20.0,
            )
            usage.raise_for_status()
            usage_payload = usage.json()
            membership = (
                usage_payload.get("user", {}).get("membership", {}).get("level")
                if isinstance(usage_payload, dict)
                else None
            )
            if membership == "LEVEL_BASIC":
                data = [
                    item
                    for item in data
                    if not isinstance(item, dict)
                    or item.get("id") != "kimi-for-coding-highspeed"
                ]
        except (httpx.HTTPError, ValueError, AttributeError):
            pass

        return _normalize(data)


class OpenRouterDiscoveryProvider(ModelDiscoveryProvider):
    """OpenRouter public catalog (TASK-822) — no key needed to list models."""

    aliases = ("openrouter",)
    _url = "https://openrouter.ai/api/v1/models"

    def fetch(self) -> list[dict[str, Any]]:
        try:
            response = httpx.get(
                self._url, headers={"Accept": "application/json"}, timeout=20.0
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ModelDiscoveryError(
                f"OpenRouter model catalog returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelDiscoveryError(
                f"Network error reaching OpenRouter model catalog: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelDiscoveryError(
                "OpenRouter model catalog returned non-JSON data"
            ) from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ModelDiscoveryError(
                "OpenRouter model catalog returned an invalid payload"
            )
        # Prefer the human ``name`` over OpenRouter's long-form ``description``
        # blurbs — the dialog shows one line. _normalize keeps id/context_length.
        rows = [
            {
                "id": item.get("id"),
                "description": item.get("name") or "",
                "context_length": item.get("context_length"),
            }
            for item in data
            if isinstance(item, dict)
        ]
        return _normalize(rows)


def _providers(config: Config | None = None) -> tuple[ModelDiscoveryProvider, ...]:
    # Imports stay lazy so the model-manager's optional provider SDKs remain off
    # the service import path until discovery is actually requested.
    from ..model_manager import (
        fetch_anthropic_api_models,
        fetch_grok_api_models,
        fetch_openai_api_models,
    )

    return (
        CodexBridgeDiscoveryProvider(),
        OpenAIDiscoveryProvider(
            aliases=("openai",),
            label="openai",
            fetcher=fetch_openai_api_models,
            key_envs=("OPENAI_API_KEY",),
            provider_id="openai-api",
            config=config,
            missing_key_message="OpenAI discovery requires a connected OpenAI API key",
            pass_key=True,
        ),
        ExistingFetcherProvider(
            aliases=("grok", "xai"),
            label="Grok",
            fetcher=fetch_grok_api_models,
            key_envs=("XAI_API_KEY", "LLM_BAWT_XAI_API_KEY"),
            provider_id="xai",
            config=config,
            config_attr="XAI_API_KEY",
            missing_key_message="Grok discovery requires a connected xAI API key",
            pass_key=True,
        ),
        AnthropicDiscoveryProvider(
            aliases=("anthropic", "claude-code"),
            label="Anthropic",
            fetcher=fetch_anthropic_api_models,
            key_envs=("LLM_BAWT_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
            provider_id="anthropic-api",
            config=config,
            missing_key_message=(
                "Anthropic discovery requires a connected Anthropic API key "
                "(Claude subscription OAuth does not authenticate /v1/models)."
            ),
            pass_key=True,
        ),
        KimiCodingDiscoveryProvider(),
        OpenRouterDiscoveryProvider(),
    )


def discover_models(
    provider: str, config: Config | None = None
) -> list[dict[str, Any]]:
    """Resolve a provider alias and return its normalized live catalog."""

    provider_key = provider.strip().lower()
    for adapter in _providers(config):
        if provider_key in adapter.aliases:
            return adapter.fetch()
    supported = sorted(alias for adapter in _providers() for alias in adapter.aliases)
    raise ModelDiscoveryError(
        f"Unknown provider '{provider}'. Use {', '.join(supported)}.",
        status_code=400,
    )

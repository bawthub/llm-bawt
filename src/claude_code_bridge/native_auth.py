"""Endpoint-aware credentials for native (non-proxy) Claude turns (TASK-1295).

The app resolves each agent turn to one catalog endpoint and sends its
``access_path`` key on ``chat.send``. A native turn talks to Anthropic
directly, and which credential it uses must follow that endpoint:

* ``anthropic-api``: the connected Anthropic API key from the app's API-key
  broker, sent as ``ANTHROPIC_API_KEY``.
* ``anthropic-oauth``: the app-owned Claude subscription token from the Claude
  broker (``CLAUDE_CODE_OAUTH_TOKEN``), behind the OAuth gateway when the
  bridge runs one.

Both explicit routes overwrite every inherited auth and base-URL value, so a
parent environment or credentials file can't change the account or auth
mode, and both fail closed when their own credential is missing. Neither
substitutes the other credential. Only a command with no ``access_path`` at
all (a legacy producer) keeps the legacy behavior. A path that is unknown, or
that disagrees with the model's proxy routing, is an error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, ClassVar

from ._bridge_helpers import _fetch_broker_token, _get_fresh_oauth_token
from .provider_key_broker import BrokerKeyError, BrokerKeyNotInstalled, fetch_provider_api_key

ANTHROPIC_API_BASE_URL = "https://api.anthropic.com"

# Every inherited value that selects the Claude CLI's auth mode, credential
# or upstream. Explicit routes set each one, so nothing leaks in from the
# bridge process environment (the SDK overlays ``options.env`` on os.environ).
AUTH_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)


class EndpointRouteError(RuntimeError):
    """``access_path`` is unknown or inconsistent with the model's routing."""


class NativeCredentialUnavailable(RuntimeError):
    """The selected endpoint's own credential is not available."""


class NativeAuthRoute(ABC):
    """How a native turn authenticates to Anthropic."""

    access_path: ClassVar[str | None]
    # The OAuth gateway swaps in the subscription bearer on every request, so
    # only routes that authenticate with that subscription may use it.
    oauth_gateway_eligible: ClassVar[bool]
    # A credential failure on this route is the Claude subscription's, so
    # BawtHub may offer the Claude reconnect.
    claude_subscription: ClassVar[bool]

    @abstractmethod
    def env(self, *, force_refresh: bool, gateway_base_url: str | None) -> dict[str, str]:
        """The SDK env for one attempt. ``gateway_base_url`` is set only when
        the bridge runs the OAuth gateway and this route is eligible."""

    @staticmethod
    def _cleared() -> dict[str, str]:
        return {key: "" for key in AUTH_ENV_KEYS}


class LegacyNativeRoute(NativeAuthRoute):
    """A command with no ``access_path``: unchanged pre-TASK-1295 behavior."""

    access_path = None
    oauth_gateway_eligible = True
    claude_subscription = True

    def env(self, *, force_refresh: bool, gateway_base_url: str | None) -> dict[str, str]:
        env: dict[str, str] = {}
        fresh_token = _get_fresh_oauth_token(force_refresh=force_refresh)
        if fresh_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = fresh_token
        if gateway_base_url:
            env["ANTHROPIC_BASE_URL"] = gateway_base_url
        return env


class AnthropicOAuthRoute(NativeAuthRoute):
    """``anthropic-oauth``: the app-owned Claude subscription, broker only."""

    access_path = "anthropic-oauth"
    oauth_gateway_eligible = True
    claude_subscription = True

    def env(self, *, force_refresh: bool, gateway_base_url: str | None) -> dict[str, str]:
        # Broker only: the legacy credentials-file and env-token fallbacks may
        # belong to a different account than the connected subscription.
        token, _expires_at = _fetch_broker_token(force=force_refresh)
        if not token:
            raise NativeCredentialUnavailable(
                "endpoint is anthropic-oauth but no Claude subscription is "
                "connected (the app's Claude token broker returned nothing); "
                "connect Claude in Settings → Providers"
            )
        env = self._cleared()
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
        env["ANTHROPIC_BASE_URL"] = gateway_base_url or ANTHROPIC_API_BASE_URL
        return env


class AnthropicApiKeyRoute(NativeAuthRoute):
    """``anthropic-api``: the connected Anthropic API key, broker only."""

    access_path = "anthropic-api"
    oauth_gateway_eligible = False
    claude_subscription = False

    def env(self, *, force_refresh: bool, gateway_base_url: str | None) -> dict[str, str]:
        # Static key: a post-401 retry simply rereads the broker, which picks
        # up a rotated key. There is deliberately no env-key fallback.
        try:
            key = fetch_provider_api_key("anthropic-api")
        except BrokerKeyNotInstalled:
            raise NativeCredentialUnavailable(
                "endpoint is anthropic-api but no Anthropic API key is "
                "connected; repoint the bot to anthropic-oauth or connect a key"
            ) from None
        except BrokerKeyError as exc:
            raise NativeCredentialUnavailable(
                f"endpoint is anthropic-api but its key is unavailable: {exc}"
            ) from exc
        env = self._cleared()
        env["ANTHROPIC_API_KEY"] = key
        env["ANTHROPIC_BASE_URL"] = ANTHROPIC_API_BASE_URL
        return env


NATIVE_ROUTES: dict[str, Callable[[], NativeAuthRoute]] = {
    AnthropicApiKeyRoute.access_path: AnthropicApiKeyRoute,  # type: ignore[dict-item]
    AnthropicOAuthRoute.access_path: AnthropicOAuthRoute,  # type: ignore[dict-item]
}


def resolve_native_route(
    access_path: str | None,
    *,
    proxy_provider: str | None,
    access_path_provider: str | None = None,
) -> NativeAuthRoute | None:
    """The native auth route for a turn, or ``None`` for a proxy-routed turn.

    ``proxy_provider`` is the proxy adapter the model is routed to (its
    ``<provider>/`` prefix), or None when the bridge sends it to Anthropic.
    ``access_path_provider`` is the proxy provider the app derived from the
    same catalog access path; the bridge can't derive it from the key alone
    because access paths are catalog rows, not a fixed set.

    Trust contract (TASK-1295): ``(access_path, access_path_provider)`` is one
    catalog-issued pair, written together from one endpoint row
    (``ModelCatalog._dispatch_config``, forwarded as a unit by
    ``service.core``). The bridge has no catalog, so it does not check that a
    proxy path exists. It checks that the pair is consistent with how the model
    is routed. On a proxy turn the path selects no credential (the model's
    adapter does), so an inconsistent pair is the failure that matters.

    Raises ``EndpointRouteError`` when an explicit ``access_path`` doesn't
    match how the model is routed: a native path on a proxy-prefixed model or
    paired with a proxy provider, a proxy path with no provider or another
    provider's prefix, or a non-native path on a model the bridge would send
    to Anthropic.
    """
    if access_path is None:
        return None if proxy_provider else LegacyNativeRoute()
    if not access_path:
        raise EndpointRouteError("access_path is present but blank")
    native = NATIVE_ROUTES.get(access_path)
    if proxy_provider:
        if native is not None:
            raise EndpointRouteError(
                f"access_path {access_path!r} is a native Anthropic endpoint "
                "but the model is proxy-routed"
            )
        if access_path_provider != proxy_provider:
            raise EndpointRouteError(
                f"access_path {access_path!r} routes to proxy provider "
                f"{access_path_provider!r} but the model is routed to "
                f"{proxy_provider!r}; refusing to guess a credential"
            )
        return None
    if native is None:
        raise EndpointRouteError(
            f"access_path {access_path!r} is not a native Anthropic endpoint "
            "and the model is not proxy-routed; refusing to guess a credential"
        )
    if access_path_provider:
        raise EndpointRouteError(
            f"access_path {access_path!r} is a native Anthropic endpoint but is "
            f"paired with proxy provider {access_path_provider!r}"
        )
    return native()

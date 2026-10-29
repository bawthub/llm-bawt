"""Background (maintenance) client factories, one per serving protocol (TASK-1726).

Summaries, memory extraction, profile maintenance and utility/recall completions
run on the app's own clients, isolated from bot conversations. Which client
serves a catalog endpoint is decided here by a registry of
:class:`BackgroundClientFactory` subclasses, keyed on the endpoint's resolved
type, protocol and engine, never on a model or provider name.

Endpoints are resolved with the ``background`` harness, which admits
chat-completions, responses and anthropic-messages access paths authorized by an
API key or no auth. Subscription OAuth paths are excluded by design: the
app-owned token broker is their only refresher, and their rate limits belong to
the agent turns they were bought for.

Every factory's client supports both maintenance call shapes: ``query(...,
plaintext_output=True, stream=False, max_tokens=, temperature=)`` and
``query_bounded(...)``. An endpoint no factory accepts is reported as
unsupported; nothing is substituted.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..model_catalog import IncompatibleModelError, ProtocolCompatibility, resolve_model_config

log = logging.getLogger(__name__)

BACKGROUND_HARNESS = "background"

#: ``chat`` keeps priority so a model with both a chat-completions and another
#: endpoint resolves to the one existing jobs already use (no new ambiguity).
_HARNESS_ORDER = ("chat", BACKGROUND_HARNESS)


def resolve_background_model(config: Any, ref: str) -> dict | None:
    """Catalog definition for ``ref`` as a background job model, or None if unknown.

    Raises the catalog's ``ModelResolutionError`` when ``ref`` exists but no
    endpoint can serve background calls (or the choice is ambiguous), so the
    caller can say why.
    """
    last_error: Exception | None = None
    for harness in _HARNESS_ORDER:
        try:
            model_def = resolve_model_config(config, ref, harness=harness, default=None)
        except IncompatibleModelError as error:
            last_error = error
            continue
        # The chat harness admits any auth; a subscription (OAuth) endpoint
        # found there is not servable here, so the background harness decides.
        auth = (model_def or {}).get("auth_mechanism")
        if auth is not None and auth not in ProtocolCompatibility.BACKGROUND_AUTH:
            last_error = IncompatibleModelError(
                f"{ref!r} resolves to a {auth} access path; background jobs need an API key or no auth")
            continue
        return model_def
    assert last_error is not None
    raise last_error


class UnsupportedBackgroundModel(ValueError):
    """The endpoint resolved, but no background client can serve it."""


#: Sent to endpoints that need no key. Never None: the generic clients treat a
#: missing key as "use the platform default key", which would leak an unrelated
#: provider's credential to this endpoint.
NO_KEY = "not-needed"


def endpoint_api_key(config: Any, model_def: dict) -> str:
    """API key for the endpoint's access path, from its owning provider connection.

    An explicit ``api_key`` on the definition is used only when no provider
    owns the path (custom endpoints). An
    ``api-key`` path with no connected key raises, so the job fails visibly
    instead of calling upstream unauthenticated. Other paths get :data:`NO_KEY`.
    """
    from .providers.registry import access_path_adapter

    # A provider-owned path takes its key only from that connection, so a
    # disconnected provider can't keep sending on an inline key (TASK-1764).
    adapter = access_path_adapter(config, model_def.get("access_path"))
    explicit = model_def.get("api_key")
    if explicit and adapter is None:
        return str(explicit)
    auth = model_def.get("auth_mechanism")
    if auth != "api-key":
        return NO_KEY
    reader = getattr(adapter, "api_key", None)
    key = reader() if callable(reader) else None
    if not key:
        raise UnsupportedBackgroundModel(
            f"no connected API key for access path {model_def.get('access_path')!r}")
    return key


def provider_label(config: Any, model_def: dict) -> str | None:
    """Stable provider id for a job model: the access path's credential owner,
    else the access path vendor (local engines), else the legacy type mapping."""
    from .providers.registry import access_path_adapter

    adapter = access_path_adapter(config, model_def.get("access_path"))
    if adapter is not None:
        return adapter.id
    vendor = model_def.get("access_vendor")
    if vendor:
        return str(vendor)
    model_type = model_def.get("type")
    return {"grok": "xai"}.get(model_type, model_type)


class BackgroundClientFactory(ABC):
    """Builds the client for one family of background endpoints."""

    #: Short label for logs.
    name: str = ""

    @abstractmethod
    def accepts(self, model_def: dict) -> bool:
        """Whether this factory serves ``model_def``."""

    @abstractmethod
    def build(self, config: Any, model_def: dict, alias: str) -> Any:
        """A ready client. Raise on missing credentials or config."""

    @staticmethod
    def model_id(model_def: dict, alias: str) -> str:
        return str(model_def.get("upstream_model_id") or model_def.get("model_id") or alias)


class LocalBridgeFactory(BackgroundClientFactory):
    """gguf/vllm run in the local model bridge, reached over Redis (TASK-276)."""

    name = "local bridge"

    def accepts(self, model_def: dict) -> bool:
        return model_def.get("type") in ("gguf", "vllm")

    def build(self, config: Any, model_def: dict, alias: str) -> Any:
        from ..clients.agent_backend_client import AgentBackendClient

        bridge_definition: dict[str, Any] = {
            "type": "agent_backend",
            "backend": "local",
            # The bridge resolves the same catalog entry by alias.
            "model_id": alias,
            "local_model_definition": dict(model_def),
        }
        for key in ("context_window", "max_tokens"):
            if model_def.get(key) is not None:
                bridge_definition[key] = model_def[key]
        return AgentBackendClient(
            backend_name="local", config=config,
            # Fixed, isolated identity: background work is stateless and shared.
            bot_config={"bot_id": "background", "user_id": "system"},
            model_definition=bridge_definition,
        )


class GrokFactory(BackgroundClientFactory):
    name = "xai"

    def accepts(self, model_def: dict) -> bool:
        return model_def.get("type") == "grok"

    def build(self, config: Any, model_def: dict, alias: str) -> Any:
        from ..clients.grok_client import GrokClient

        return GrokClient(model=model_def.get("model_id", alias), config=config,
                          model_definition=model_def)


class OpenAIChatFactory(BackgroundClientFactory):
    """OpenAI platform chat-completions (and legacy defs without a protocol)."""

    name = "openai chat"

    def accepts(self, model_def: dict) -> bool:
        return (model_def.get("type") == "openai"
                and model_def.get("protocol") in (None, "chat-completions"))

    def build(self, config: Any, model_def: dict, alias: str) -> Any:
        from ..clients.openai_client import OpenAIClient

        return OpenAIClient(model=model_def.get("model_id", alias), config=config,
                            model_definition=model_def)


class OllamaFactory(BackgroundClientFactory):
    """Ollama's OpenAI-compatible surface, same construction as the chat path."""

    name = "ollama"

    def accepts(self, model_def: dict) -> bool:
        return model_def.get("type") == "ollama"

    def build(self, config: Any, model_def: dict, alias: str) -> Any:
        from ..clients.openai_client import OpenAIClient

        base_url = model_def.get("base_url") or f"{config.OLLAMA_URL.rstrip('/')}/v1"
        return OpenAIClient(self.model_id(model_def, alias), config=config, base_url=base_url,
                            api_key=model_def.get("api_key") or "ollama",
                            model_definition=model_def)


class ChatCompletionsFactory(BackgroundClientFactory):
    """Any other OpenAI-compatible chat-completions endpoint with a base_url."""

    name = "chat-completions"

    def accepts(self, model_def: dict) -> bool:
        return (model_def.get("protocol") == "chat-completions"
                and model_def.get("type") == "openai-compatible")

    def build(self, config: Any, model_def: dict, alias: str) -> Any:
        from ..clients.openai_client import OpenAIClient

        base_url = model_def.get("base_url")
        if not base_url:
            raise UnsupportedBackgroundModel("chat-completions endpoint has no base_url")
        return OpenAIClient(self.model_id(model_def, alias), config=config, base_url=base_url,
                            api_key=endpoint_api_key(config, model_def),
                            model_definition=model_def)


class ResponsesFactory(BackgroundClientFactory):
    """Responses API endpoints with an API key (xAI Responses goes through Grok)."""

    name = "responses"

    def accepts(self, model_def: dict) -> bool:
        return model_def.get("protocol") == "responses" and model_def.get("type") != "grok"

    def build(self, config: Any, model_def: dict, alias: str) -> Any:
        from ..clients.responses_client import ResponsesClient

        base_url = model_def.get("base_url")
        if not base_url:
            raise UnsupportedBackgroundModel("responses endpoint has no base_url")
        return ResponsesClient(self.model_id(model_def, alias), config, base_url=base_url,
                               api_key=endpoint_api_key(config, model_def),
                               model_definition=model_def)


class AnthropicMessagesFactory(BackgroundClientFactory):
    name = "anthropic-messages"

    def accepts(self, model_def: dict) -> bool:
        return model_def.get("protocol") == "anthropic-messages"

    def build(self, config: Any, model_def: dict, alias: str) -> Any:
        from ..clients.anthropic_messages_client import AnthropicMessagesClient

        base_url = model_def.get("base_url")
        if not base_url:
            raise UnsupportedBackgroundModel("anthropic-messages endpoint has no base_url")
        return AnthropicMessagesClient(self.model_id(model_def, alias), config, base_url=base_url,
                                       api_key=endpoint_api_key(config, model_def),
                                       model_definition=model_def)


#: First match wins; order runs from the most to the least specific rule.
BACKGROUND_CLIENT_FACTORIES: tuple[BackgroundClientFactory, ...] = (
    LocalBridgeFactory(),
    GrokFactory(),
    OllamaFactory(),
    OpenAIChatFactory(),
    ChatCompletionsFactory(),
    ResponsesFactory(),
    AnthropicMessagesFactory(),
)


def factory_for(model_def: dict) -> BackgroundClientFactory | None:
    return next((f for f in BACKGROUND_CLIENT_FACTORIES if f.accepts(model_def)), None)


def build_background_client(config: Any, model_def: dict, alias: str) -> tuple[Any, str]:
    """``(client, factory name)``; raises :class:`UnsupportedBackgroundModel`."""
    factory = factory_for(model_def)
    if factory is None:
        raise UnsupportedBackgroundModel(
            f"no background client for type={model_def.get('type')!r} "
            f"protocol={model_def.get('protocol')!r}")
    from .usage.background_spend import install_spend_recorder

    # One hook for every factory: each billed call is priced into the balance
    # ledger as maintenance spend (TASK-1733).
    return install_spend_recorder(factory.build(config, model_def, alias), model_def), factory.name

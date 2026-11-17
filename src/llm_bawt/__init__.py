# Keep package import side-effect free: the agent-bridge image installs this
# project with --no-deps, so bridges that import a light subpackage (e.g.
# llm_bawt.api_auth.outbound) must not drag in the client stack (rich, ...).
# OpenAIClient stays available as `llm_bawt.OpenAIClient`, loaded on first use.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .clients.openai_client import OpenAIClient as OpenAIClient

__all__ = ["OpenAIClient"]


def __getattr__(name: str):
    if name == "OpenAIClient":
        from .clients.openai_client import OpenAIClient

        return OpenAIClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

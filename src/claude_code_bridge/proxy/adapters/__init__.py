"""Provider adapter registry.

Adapters declare a ``name`` class var; ``register`` indexes them by that name
and ``lookup`` resolves a name → adapter instance. Auto-registers the
OpenAI ChatGPT adapter so callers only need to import this package.
"""

from __future__ import annotations

import asyncio

from .base import ProviderAdapter
from .kimi_coding import KimiCodingAdapter
from .local import LocalAdapter
from .moonshot import MoonshotAdapter
from .openai import OpenAIAdapter
from .openai_chatgpt import OpenAIChatGPTAdapter
from .chatgpt_app import ChatGPTAppAdapter
from .openrouter import OpenRouterAdapter
from .xai import XaiAdapter
from .zai import ZaiAdapter

REGISTRY: dict[str, ProviderAdapter] = {}


def register(adapter: ProviderAdapter) -> None:
    REGISTRY[adapter.name] = adapter


def lookup(name: str) -> ProviderAdapter | None:
    return REGISTRY.get(name)


async def start_all() -> None:
    """Start every registered adapter's reusable connection pool."""
    await asyncio.gather(*(adapter.start() for adapter in REGISTRY.values()))


async def close_all() -> None:
    """Close every registered adapter exactly once during proxy shutdown."""
    await asyncio.gather(*(adapter.close() for adapter in REGISTRY.values()))


# Default registrations. Adding a new provider = create the adapter file,
# import it here, and register an instance.
register(OpenAIChatGPTAdapter())
register(ChatGPTAppAdapter())
register(OpenAIAdapter())
register(XaiAdapter())
register(ZaiAdapter())
register(MoonshotAdapter())
register(KimiCodingAdapter())
register(OpenRouterAdapter())
register(LocalAdapter())

__all__ = [
    "ProviderAdapter", "REGISTRY", "register", "lookup", "start_all", "close_all",
]

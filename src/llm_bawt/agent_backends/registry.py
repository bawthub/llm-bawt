"""Agent backend registry with entry point discovery.

Builtin backends are registered on import. Third-party backends are
discovered via the ``llm_bawt.agent_backends`` entry point group.
"""

import importlib.metadata
import logging
from typing import Type

from .base import AgentBackend

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, Type[AgentBackend]] = {}


def register_backend(name: str, cls: Type[AgentBackend]) -> None:
    """Register an agent backend class by name."""
    _BACKENDS[name] = cls


def get_backend(name: str) -> AgentBackend | None:
    """Get an agent backend instance by name.

    Returns a new instance each call, or None if not found.
    """
    cls = _BACKENDS.get(name)
    if cls is None:
        return None
    return cls()


def list_backends() -> dict[str, Type[AgentBackend]]:
    """Return a copy of the registered backends."""
    return _BACKENDS.copy()


def _load_entry_points() -> None:
    """Discover and register third-party agent backends via entry points."""
    try:
        eps = importlib.metadata.entry_points(group="llm_bawt.agent_backends")
    except Exception:
        return

    for ep in eps:
        if ep.name in _BACKENDS:
            continue
        try:
            cls = ep.load()
            register_backend(ep.name, cls)
            logger.debug(f"Loaded agent backend plugin: {ep.name}")
        except Exception as e:
            logger.warning(f"Failed to load agent backend '{ep.name}': {e}")


def _register_builtins() -> None:
    from .claude_code import ClaudeCodeBackend
    from .codex import CodexBackend
    from .local_model import LocalModelBackend

    register_backend("claude-code", ClaudeCodeBackend)
    register_backend("codex", CodexBackend)
    register_backend("local", LocalModelBackend)


_register_builtins()
_load_entry_points()

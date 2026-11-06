"""Curated Claude catalog for subscription discovery fallback (TASK-1296 D1).

A Claude subscription (OAuth) login may not be allowed to list
``/v1/models``. When the live call is refused, discovery offers this
maintained list instead so the wizard still has real Claude Code model ids to
choose from. Every row is labelled ``curated=True``: it is what the bundled
Claude Code CLI (2.1.317) knows how to run, not proof that the signed-in plan
is entitled to it.

Keep this list short and current: ids the CLI no longer accepts are worse than
a missing entry, because the catalog would advertise a model no session can
start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CURATED_KEY = "curated"


@dataclass(frozen=True)
class CuratedClaudeModel:
    """One Claude Code CLI model id with its uncapped CLI context window."""

    id: str
    display_name: str
    context_length: int


#: Per-model windows verified with CLI 2.1.317 ``-p /context --model <id>``,
#: with CLAUDE_CODE_AUTO_COMPACT_WINDOW unset. Bare Opus/Sonnet 5/5.5 and
#: Fable 5.1 use 1M; Haiku 4.5 still uses 200K. Compact policy is separate.
#: Discovery retains explicit ``[1m]`` aliases for existing catalog choices.
CURATED_CLAUDE_MODELS: tuple[CuratedClaudeModel, ...] = (
    CuratedClaudeModel("claude-opus-5-5", "Claude Opus 5.5", 1_000_000),
    CuratedClaudeModel("claude-opus-5", "Claude Opus 5", 1_000_000),
    CuratedClaudeModel("claude-sonnet-5-5", "Claude Sonnet 5.5", 1_000_000),
    CuratedClaudeModel("claude-sonnet-5", "Claude Sonnet 5", 1_000_000),
    CuratedClaudeModel("claude-fable-5-1", "Claude Fable 5.1", 1_000_000),
    CuratedClaudeModel("claude-haiku-4-5", "Claude Haiku 4.5", 200_000),
)


def curated_claude_rows() -> list[dict[str, Any]]:
    """Discovery-shaped rows for the curated list (before adding CLI aliases)."""
    return [
        {
            "id": model.id,
            "description": model.display_name,
            "context_length": model.context_length,
            CURATED_KEY: True,
        }
        for model in CURATED_CLAUDE_MODELS
    ]

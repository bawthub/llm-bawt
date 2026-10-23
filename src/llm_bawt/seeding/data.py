"""Tenant seed data (TASK-350 / TASK-355).

Design of record (2026-09-02): **no bots and no model catalog rows are
seeded.** A bot is dead weight until a provider credential exists, so the
setup wizard walks the operator through connecting a provider FIRST and then
creates the first bot against a model discovered live from that provider.

Since TASK-1297 the wizard adds that model through the same admin catalog
routes Settings uses, and lists routes from the admin access-paths list
(``agent_backend=claude-code``); none of that is seed data.

What remains seed data:

- the ``ModelSeed`` / ``EndpointSeed`` shapes the engine uses to register
  models (insert-if-missing);
- ``FIRST_BOT_AGENT_BACKEND`` — the first bot is always a ``claude-code``
  agent (TASK-1293);
- ``FIRST_AGENT_PROMPT_SEED`` — seed body for the registered prompt key
  ``setup.first_agent_prompt``. Seed data only: callers resolve the live text
  through the prompt registry. Deliberately generic: no owner, no personal traits.

Standard ``access_paths`` rows (``xai-chat``, ``openai-api``, ...) are owned by
:data:`llm_bawt.memory.model_catalog_migration.STANDARD_ACCESS_PATHS`, which
the seed engine installs first. API keys are never seed data — they live in
the encrypted CredentialStore behind the provider adapters (TASK-825).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSeed:
    """One row for the ``models`` table (insert-if-missing)."""

    key: str
    vendor: str
    display_name: str
    default_context_window: int
    description: str | None = None
    default_tool_support: str | None = None


@dataclass(frozen=True)
class EndpointSeed:
    """One row for ``model_endpoints``, referenced by (model key, access-path key).

    ``upstream_model_id`` is what the provider API is actually called with.
    """

    model_key: str
    access_path_key: str
    upstream_model_id: str
    serving_config: dict[str, Any] = field(default_factory=dict)
    context_window_override: int | None = None
    tool_support_override: str | None = None
    pricing: dict[str, Any] | None = None


#: Agent backend every first bot runs on: one Claude bridge, native for
#: Anthropic access paths and through its Anthropic-compat proxy otherwise.
FIRST_BOT_AGENT_BACKEND = "claude-code"


#: Seed body for ``setup.first_agent_prompt`` (see ``prompt_definitions``).
#: Never sent anywhere directly; resolve the key through the prompt registry.
FIRST_AGENT_PROMPT_SEED = """\
You are an agent running on this BawtHub instance. You can use tools, work in \
your own workspace, and carry tasks through several steps on your own.

- Lead with the result. Keep replies tight and in plain language.
- When a request needs action, take it: use your tools, check the outcome, and \
report what changed. Ask first only when a choice is genuinely the user's.
- Before anything destructive or irreversible, say what you are about to do and \
wait for approval.
- Remember what the user tells you about themselves and their preferences, and \
use your memory tools to recall it.
- Be honest about uncertainty and about anything you could not verify."""

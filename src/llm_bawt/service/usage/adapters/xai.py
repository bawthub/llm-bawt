"""xAI (Grok) usage adapter.

xAI is pure pay-per-token API-key billing — there is no subscription / plan
quota endpoint to poll (unlike Claude Max, ChatGPT Plus/codex, or z.ai GLM
Coding Plan). The adapter still registers under the ``xai`` provider id so:

  1. ``GET /v1/usage?bot_id=<grok-bot>`` resolves to this provider instead of
     incorrectly falling back to Claude (which is what made the chat context
     popup show the Anthropic sunburst + 5h/weekly Max limits for Grok).
  2. The all-providers view and ProviderIcon can identify the backend.

The snapshot returns ``status=ok`` with an empty ``limits`` list once a key is
connected. The UI treats that as "API usage only" and hides the subscription
section.
"""

from __future__ import annotations

from .api_key_balance import ApiKeyBalanceAdapter


class XaiUsageAdapter(ApiKeyBalanceAdapter):
    """API-key Grok — no plan limits; a user-set balance counted down per turn."""

    # No xAI balance API is wired yet (TASK-1473); until then a manual anchor.
    provider = "xai"
    display_name = "xAI · Grok"
    backend = "claude-code"  # typically reached via the claude-code bridge proxy
    access_path_keys = ("xai-chat", "xai-responses")
    connection_id = "xai"
    key_env_vars = ("XAI_API_KEY", "LLM_BAWT_XAI_API_KEY", "GROK_API_KEY")

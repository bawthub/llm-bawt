"""xAI (Grok) usage adapter.

xAI is pure pay-per-token API-key billing — there is no subscription / plan
quota endpoint to poll (unlike Claude Max, ChatGPT Plus/codex, or z.ai GLM
Coding Plan). The adapter still registers under the ``xai`` provider id so:

  1. ``GET /v1/usage?bot_id=<grok-bot>`` resolves to this provider instead of
     incorrectly falling back to Claude (which is what made the chat context
     popup show the Anthropic sunburst + 5h/weekly Max limits for Grok).
  2. The all-providers view and ProviderIcon can identify the backend.

The snapshot always returns ``status=ok`` with an empty ``limits`` list. The
UI treats that as "API usage only" and hides the subscription section.
"""

from __future__ import annotations

import os

from ..balance import BALANCE_ANCHOR
from ..base import UsageAdapter
from ..canonical import STATUS_OK, STATUS_UNAUTHORIZED


class XaiUsageAdapter(UsageAdapter):
    """API-key Grok — no plan limits; a user-set balance counted down per turn."""

    provider = "xai"
    display_name = "xAI · Grok"
    backend = "claude-code"  # typically reached via the claude-code bridge proxy
    access_path_keys = ("xai-chat", "xai-responses")
    kind = "balance"
    # No xAI balance API is wired yet (TASK-1473); until then a manual anchor.
    balance_mode = BALANCE_ANCHOR

    def _has_key(self) -> bool:
        for env in ("XAI_API_KEY", "LLM_BAWT_XAI_API_KEY", "GROK_API_KEY"):
            if os.getenv(env):
                return True
        return False

    async def fetch(self):
        if not self._has_key():
            return self._base(
                available=False,
                status=STATUS_UNAUTHORIZED,
                error=(
                    "No xAI API key configured "
                    "(set XAI_API_KEY / LLM_BAWT_XAI_API_KEY)."
                ),
                limits=[],
            )
        return self._base(
            available=True,
            status=STATUS_OK,
            limits=[],  # pay-per-token; no plan windows
        )

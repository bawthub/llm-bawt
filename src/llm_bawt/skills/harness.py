"""What each agent harness can do with assigned skills (TASK-1309).

Capability data, not prose: the UI maps these enums to its own copy, so a
caveat chip comes from the harness a bot runs on, never from a per-bot string.
Keyed by ``Bot.agent_backend`` (the bridge that executes the turn).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# How long an injected skill body stays in the model's context.
PERSISTENT = "persistent"  # re-sent with the pinned system prompt on every turn
UNTIL_FIRST_COMPACTION = "until_first_compaction"  # cold start until auto-compaction
NOT_APPLIED = "not_applied"

# How the harness is told which skills a bot may invoke.
ALLOW_LIST = "allow_list"  # the SDK's ``skills=[...]`` option
DENY_LIST_COMPLEMENT = "deny_list_complement"  # disable every other ambient skill
NONE = "none"

# Whether a resumed session re-renders injected skills.
YES = "yes"
NO = "no"
UNVERIFIED = "unverified"


@dataclass(frozen=True)
class HarnessCapabilities:
    harness: str
    supported: bool
    injection: str
    listing: str
    reinject_on_resume: str

    def to_dict(self) -> dict:
        return asdict(self)


HARNESS_CAPABILITIES: dict[str, HarnessCapabilities] = {
    "claude-code": HarnessCapabilities("claude-code", True, PERSISTENT, ALLOW_LIST, YES),
    # Codex keeps instructions from session start until its first automatic
    # compaction; whether ``exec resume`` re-renders them is not established.
    "codex": HarnessCapabilities("codex", True, UNTIL_FIRST_COMPACTION,
                                 DENY_LIST_COMPLEMENT, UNVERIFIED),
}


def capabilities_for(harness: str | None) -> HarnessCapabilities:
    """Known capabilities, or an unsupported record for any other harness."""
    known = HARNESS_CAPABILITIES.get(harness or "")
    if known is not None:
        return known
    return HarnessCapabilities(harness or "chat", False, NOT_APPLIED, NONE, NO)

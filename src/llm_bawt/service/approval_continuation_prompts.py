"""Seed-only approval continuation instructions; runtime uses the prompt registry."""

APPROVED_GUIDANCE = (
    "[The user APPROVED the {tool_name} action you requested. Re-issue "
    "exactly this call now and continue:\n\n{shown}]"
)
DENIED_GUIDANCE = (
    "[The user DENIED the {tool_name} action you requested:\n\n{shown}\n\n"
    "Do not attempt it again. Continue without it, or explain what you need.]"
)
RESPONDED_GUIDANCE = (
    "[The user did NOT run the {tool_name} action you requested:\n\n{shown}\n\n"
    "They responded with this guidance — follow it and do not re-issue the "
    "original call unless it tells you to:\n\n{message}]"
)
NO_RESPONSE_GUIDANCE = (
    "[The user reviewed the {tool_name} action you requested and chose not to "
    "run it, with no further instruction. Continue without it.]"
)

"""Invocation binding shared by the wake-up tool and schedule-create HTTP route."""

# Deliberately not an MCP tool name: a token for an unrelated MCP invocation
# must never double as HTTP authorship proof.
CREATE_PURPOSE = "http:prompt-schedules.create:v1"


def create_invocation(user: str, body: dict) -> dict:
    """Bind the normalized owner and complete parsed HTTP JSON, before coercion."""
    return {"user": user.strip().lower(), "body": body}

"""Agent bridge — generic transport shared by all bridge backends.

This package holds the wire-format envelope, event store, Redis
publisher/subscriber, and per-session queue that the Codex, Claude Code
and local-model bridges all use. Backend-specific code (SDK adapters,
process management) lives in its own package (``codex_bridge``,
``claude_code_bridge``, ``local_model_bridge``).
"""

from .approval import (
    ApprovalDecision,
    ApprovalPolicy,
    MatcherType,
    PolicyAction,
    PolicyBundle,
    Severity,
    compute_etag,
    evaluate,
    grant_key,
)
from .events import AgentEvent, AgentEventKind, synthesize_event_id
from .session_queue import SessionQueue

__all__ = [
    "AgentEvent",
    "AgentEventKind",
    "ApprovalDecision",
    "ApprovalPolicy",
    "MatcherType",
    "PolicyAction",
    "PolicyBundle",
    "SessionQueue",
    "Severity",
    "compute_etag",
    "evaluate",
    "grant_key",
    "synthesize_event_id",
]

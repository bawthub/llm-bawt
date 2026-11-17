"""Apply the existing approval evaluator before native handlers run.

REQUIRE_APPROVAL is a refusal, not a native card or a replay grant. Offline
executors without a configured policy database have no operator policy source.
Configured-source failures never authorize a handler.
"""
from __future__ import annotations

import logging

from agent_bridge.approval import PolicyAction, evaluate
from agent_bridge.mcp_call_context import canonical_invocation_hash

log = logging.getLogger(__name__)


class NativePolicyGate:
    def __init__(self, config=None, *, store=None):
        self.config = config
        self.store = store

    def _policy_store(self):
        if self.store is not None:
            return self.store
        password = getattr(self.config, "POSTGRES_PASSWORD", "")
        if not isinstance(password, str) or not password.strip():
            return None
        from ..service.dependencies import get_tool_approval_policy_store

        self.store = get_tool_approval_policy_store(self.config)
        return self.store

    def check(self, original, normalized, *, bot_id, user_id, turn_id=None) -> str | None:
        """Return a non-executing explanation for the strongest matched action."""
        try:
            store = self._policy_store()
            if store is None:
                return None  # Explicit no-policy-database/offline configuration.
            bundle = store.compile_bundle()
            identities = [(original.name, original.arguments)]
            if (normalized.name, normalized.arguments) != identities[0]:
                identities.append((normalized.name, normalized.arguments))
            actions = []
            for name, arguments in identities:
                decision = evaluate(bundle.policies, "native", name, arguments)
                store.record_decision(
                    decision=decision, bundle=bundle, backend="native",
                    tool_name=name,
                    invocation_hash=canonical_invocation_hash(name, arguments),
                    bot_id=bot_id, user_id=user_id, turn_id=turn_id,
                    source="native",
                )
                actions.append(decision.action)
            if PolicyAction.DENY in actions:
                return "Not executed: this native tool call is denied by approval policy."
            if PolicyAction.REQUIRE_APPROVAL in actions:
                return (
                    "Held: approval policy requires approval. Nothing was executed. "
                    "Native tools have no approval-card/resume transport; no card "
                    "or retry grant was created."
                )
            return None
        except Exception:
            # Do not log arbitrary tool arguments or exception messages.
            log.warning("Native approval policy source/audit unavailable; call held")
            return "Held: approval policies are unavailable. Nothing was executed."

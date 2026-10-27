"""Request-local proof for a Codex reset, optionally followed by one task.

The reset names the exact seed it applied (TASK-1579): an empty seed is a
clean start, a non-empty one is a fresh provider thread the bridge seeded and
bound before acknowledging.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent_bridge.continuity import seed_receipt


@dataclass
class CodexResetEvidence:
    thread_session_id: str
    previous_provider_id: str | None
    provider_session_id: str | None = None
    reset_only: bool = False
    seed: list | None = None

    @classmethod
    def from_fields(
        cls, fields: dict, *, thread_session_id: str | None,
        explicit_thread: bool, thread_resume_id: str | None,
    ) -> CodexResetEvidence | None:
        action = fields.get("bridge_reset_action")
        if not action:
            return None
        if action not in {"reset_only", "reset_follow_up"} or explicit_thread or not thread_session_id:
            raise ValueError("Invalid structured Codex reset command")
        seed = parse_seed(fields)
        reset_only = action == "reset_only"
        if reset_only and (seed is None or fields.get("thread_resume_id")):
            raise ValueError("Standalone Codex reset requires an explicit seed and no resume identity")
        return cls(thread_session_id, thread_resume_id, reset_only=reset_only, seed=seed)

    @property
    def clean_start(self) -> bool:
        """A standalone reset with nothing to seed opens no provider thread."""
        return self.reset_only and not self.seed

    async def bind(self, bridge: Any, *, bot_id: str, provider_id: Any, model: str) -> None:
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise RuntimeError("Codex reset has no fresh provider session identity")
        if provider_id == self.previous_provider_id:
            raise RuntimeError("Codex reset reused the previous provider session")
        if self.provider_session_id is not None:
            if self.provider_session_id != provider_id:
                raise RuntimeError("Codex reset provider session changed during the turn")
            return
        await bridge._set_thread_session(
            self.thread_session_id, bot_id, provider_id, model, required=True,
        )
        self.provider_session_id = provider_id

    def acknowledgment(self) -> dict:
        if not self.clean_start and not self.provider_session_id:
            raise RuntimeError("Codex reset provider binding is unverified")
        return {
            "thread_session_id": self.thread_session_id,
            "provider_session_id": self.provider_session_id,
            # A clean standalone reset deliberately creates no dummy provider
            # turn; the app must also prove its successor has no Codex key.
            "clean_start": self.clean_start,
            **seed_receipt(self.seed),
        }


def parse_seed(fields: dict) -> list | None:
    """The app's ``inject_messages`` field: absent, or an explicit list."""
    seed = fields.get("inject_messages")
    if seed is None or seed == "":
        return None
    if isinstance(seed, str):
        try:
            seed = json.loads(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid Codex continuity seed") from exc
    if not isinstance(seed, list) or not all(isinstance(m, dict) for m in seed):
        raise ValueError("Invalid Codex continuity seed")
    return seed

"""Request-local proof for Codex clean reset, optionally followed by one task."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class CodexResetEvidence:
    thread_session_id: str
    previous_provider_id: str | None
    provider_session_id: str | None = None
    reset_only: bool = False

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
        # Codex supports clean resets only. Never silently discard a retained
        # seed and then confirm that the requested continuity was applied.
        seed = fields.get("inject_messages")
        if isinstance(seed, str):
            try:
                seed = json.loads(seed)
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid Codex reset seed") from exc
        if seed is not None and seed != []:
            raise ValueError("Codex reset does not support retained history")
        reset_only = action == "reset_only"
        if reset_only and (seed != [] or fields.get("thread_resume_id")):
            raise ValueError("Standalone Codex reset requires explicit empty seed and no resume identity")
        return cls(thread_session_id, thread_resume_id, reset_only=reset_only)

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
        if not self.reset_only and not self.provider_session_id:
            raise RuntimeError("Codex reset provider binding is unverified")
        return {
            "thread_session_id": self.thread_session_id,
            "provider_session_id": self.provider_session_id,
            # Standalone clean reset deliberately creates no dummy provider turn.
            # The app must also prove its successor has no persisted Codex key.
            "clean_start": self.reset_only,
        }

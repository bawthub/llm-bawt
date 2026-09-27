"""Request-local execution handles, shared by stream workers and Stop.

A handle is registered before turn_start and retired only after finalization.
Cancellation is cooperative for native synchronous providers: acceptance does
not claim a blocked provider/tool has stopped. Bridge dispatch has a handshake
so Stop during setup cannot race ahead of chat.send and get lost.
"""
from __future__ import annotations

from threading import Event, Lock


class TurnExecutionCancelled(Exception):
    """The owning worker observed its request's cancellation."""


class TurnExecution:
    def __init__(self, turn_id: str, cancel_event: Event, *, is_agent: bool, store):
        self.turn_id = turn_id
        self.cancel_event = cancel_event
        self.is_agent = is_agent
        self.store = store
        self.backend: str | None = None
        self.session_key: str | None = None
        self.request_id: str | None = None
        self._dispatched = False
        self._lock = Lock()

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise TurnExecutionCancelled()

    def bind_bridge(self, *, backend: str, session_key: str, request_id: str) -> None:
        # Persist the actual request-local identity before command publication,
        # never from the bot's shared _active_request_id or first model output.
        self.store.update_turn(turn_id=self.turn_id, agent_request_id=request_id,
                               agent_session_key=session_key)
        with self._lock:
            self.backend, self.session_key, self.request_id = backend, session_key, request_id
        self.check_cancelled()

    def dispatched(self) -> bool:
        with self._lock:
            self._dispatched = True
            return self.cancel_event.is_set()

    def request_cancel(self) -> bool:
        """Signal the owner; return whether an already-sent bridge needs RPC."""
        with self._lock:
            self.cancel_event.set()
            return self.is_agent and self._dispatched


class TurnExecutions:
    def __init__(self):
        self._lock = Lock()
        self._turns: dict[str, TurnExecution] = {}

    def register(self, execution: TurnExecution) -> None:
        with self._lock:
            if execution.turn_id in self._turns:
                raise ValueError("Turn execution already registered")
            self._turns[execution.turn_id] = execution

    def get(self, turn_id: str) -> TurnExecution | None:
        with self._lock:
            return self._turns.get(turn_id)

    def remove(self, turn_id: str) -> None:
        with self._lock:
            self._turns.pop(turn_id, None)

    def active_ids(self) -> list[str]:
        with self._lock:
            return list(self._turns)


turn_executions = TurnExecutions()

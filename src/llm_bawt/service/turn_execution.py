"""Request-local execution handles, shared by stream workers and Stop.

A handle is registered before turn_start and retired only after finalization.
Cancellation is cooperative for native synchronous providers: acceptance does
not claim a blocked provider/tool has stopped. Bridge dispatch has a handshake
so Stop during setup cannot race ahead of chat.send and get lost.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from threading import Event, Lock
from typing import Any

from agent_bridge.run_contract import RunTerminal

from .turn_gates import execution_exit_record, stamp_turn_exit

log = logging.getLogger(__name__)


def exit_record_from_terminal(raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """B1 source (a): the ``execution_exit`` record a ``RUN_TERMINAL`` proves.

    ``raw`` is ``RunTerminal.to_dict()``, or ``{}`` for a pre-contract bridge
    (never evidence). The record keeps the terminal's ``teardown.exited`` as
    is: only ``True`` drains, and ``False``/``None`` are named blockers.
    """
    if not raw:
        return None
    terminal = RunTerminal.from_dict(raw)
    return execution_exit_record(source="a", request_id=terminal.request_id, backend=terminal.backend,
                                 boot_id=terminal.boot_id, exited=terminal.teardown.exited)


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
        self.store.bind_agent_bridge(
            turn_id=self.turn_id, backend=backend, request_id=request_id, session_key=session_key,
        )
        with self._lock:
            self.backend, self.session_key, self.request_id = backend, session_key, request_id
        self.check_cancelled()

    def record_exit(self, terminal: Mapping[str, Any] | None) -> bool:
        """Stamp this run's terminal on the turn row (B1 source (a), at settle).

        Never raises: a missing stamp only leaves a quiesce waiting, and the
        (b)/(c) sweep can still prove the exit. Returns whether it stamped.
        """
        engine = getattr(self.store, "engine", None)
        try:
            record = exit_record_from_terminal(terminal)
            if record is None or engine is None:
                return False
            with engine.begin() as conn:
                return stamp_turn_exit(conn, self.turn_id, record)
        except ValueError:  # RunContractError included: a malformed terminal or record
            log.error("Unusable run terminal for turn %s: %r", self.turn_id, terminal)
        except Exception:
            log.exception("Could not stamp the run terminal for turn %s", self.turn_id)
        return False

    def record_accept(self, accepted: Mapping[str, Any] | None) -> bool:
        """Durably record which bridge boot accepted this run (``RUN_ACCEPTED``, TASK-1233 B1 (b)).

        Never raises into the stream worker: a missing record only means the
        sweep's restart proof falls back to its own sighting, or the quiesce
        times out. Returns whether it wrote the record.
        """
        from .run_accepts import record_accept

        try:
            return record_accept(getattr(self.store, "engine", None), accepted, request_id=self.request_id)
        except ValueError:  # RunContractError included: a malformed or foreign payload
            log.error("Unusable RUN_ACCEPTED for turn %s: %r", self.turn_id, accepted)
        except Exception:
            log.exception("Could not record RUN_ACCEPTED for turn %s", self.turn_id)
        return False

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

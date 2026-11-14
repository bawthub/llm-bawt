"""Request-local lease heartbeat, including blocking session-setup windows."""
from __future__ import annotations

import logging
from threading import Event, Lock, Thread, current_thread

from .turn_execution import TurnExecution, TurnExecutionDetached, turn_executions
from .turn_terminal_protocol import FollowerFenceLost, FollowerLease, RENEW_SECONDS, TurnTerminalProtocol

log = logging.getLogger(__name__)


class TerminalFollower:
    """Carry the admission token; never claim a healthy turn a second time.

    A dedicated heartbeat is intentional: setup can block the asyncio loop on
    session locks, history or reset work. Losing observation detaches a bridge
    reader without cancelling/rerunning its SDK run. Native providers also get
    the existing cooperative cancellation signal.
    """

    def __init__(self, protocol, lease: FollowerLease, execution):
        if lease.turn_id != execution.turn_id:
            raise ValueError("follower lease does not belong to this execution")
        self.protocol = protocol
        self.execution = execution
        self._lease = lease
        self._lock = Lock()
        self._stop = Event()
        self._lost = Event()
        self._thread: Thread | None = None
        self._started = False
        execution.terminal_follower = self

    @property
    def lease(self) -> FollowerLease:
        with self._lock:
            return self._lease

    def start(self) -> TerminalFollower:
        """Confirm the INSERT token now, then renew through every setup wait."""
        if self._started or self._stop.is_set():
            raise RuntimeError("follower cannot be started twice or after close")
        self._started = True
        try:
            self.execution.check_detached()
            self._renew()
            self.execution.check_detached()
        except Exception:
            self._detach()
            raise
        self._thread = Thread(
            target=self._run, name=f"turn-follower-{self.lease.turn_id}", daemon=True,
        )
        self._thread.start()
        return self

    def _renew(self) -> None:
        renewed = self.protocol.renew(self.lease)
        with self._lock:
            self._lease = renewed

    def _detach(self) -> None:
        self._lost.set()
        self._stop.set()
        self.execution.detach()
        if not self.execution.is_agent:
            self.execution.request_cancel()

    def _run(self) -> None:
        while not self._stop.wait(RENEW_SECONDS):
            if self.execution.detach_requested():
                self._stop.set()
                return
            try:
                self._renew()
            except Exception:
                # A transport failure is not proof of a new owner, but is also
                # not authority to keep observing/writing with an unrenewed lease.
                # The protocol's fenced recovery path decides what happens next.
                log.warning(
                    "Follower renewal failed; detaching turn %s",
                    self.lease.turn_id, exc_info=True,
                )
                self._detach()
                return

    def check(self) -> None:
        """Setup/worker must stop before further side effects after fence loss."""
        if self._lost.is_set() or self._stop.is_set():
            raise TurnExecutionDetached()
        self.execution.check_detached()

    def write_partial(self, *, text_value: str, reasoning: str | None = None) -> None:
        """Checkpoint observed content under this exact admission lease."""
        self.check()
        try:
            self.protocol.write_partial(self.lease, text_value=text_value, reasoning=reasoning)
        except FollowerFenceLost as exc:
            # A synchronous write can discover lease loss before the heartbeat.
            # No more text, tools or events may leave this observation worker.
            self._detach()
            raise TurnExecutionDetached() from exc

    def close(self, *, timeout: float = 5.0) -> bool:
        """Stop heartbeats without pretending the provider itself has exited.

        No claim is released here: existing worker-exit safety stays independent.
        False truthfully reports an in-flight DB call still finishing. Once it
        returns, the stop event prevents another renewal.
        """
        self._stop.set()
        thread = self._thread
        if thread is None:
            return True
        if thread is not current_thread():
            thread.join(timeout=timeout)
        return not thread.is_alive()


class TerminalFollowerScope:
    """Transfer lease cleanup from request setup to the actual worker.

    Setup can fail or an SSE client can leave before worker launch. After
    launch, only the real worker's finally closes the heartbeat; cancellation
    of the HTTP task must not pretend a blocked worker has exited.
    Legacy admissions return None and leave their execution lifecycle alone.
    """

    def __init__(self, store):
        self.store = store
        self.follower: TerminalFollower | None = None
        self._launched = False
        self._inference_started = False
        self._closed = False
        self._on_close = None

    def admitted(self, lease, *, turn_id: str, cancel_event: Event, is_agent: bool, on_close=None):
        if lease is None:
            return
        if not isinstance(lease, FollowerLease):
            raise TypeError("terminal admission returned an invalid follower lease")
        execution = TurnExecution(turn_id, cancel_event, is_agent=is_agent, store=self.store)
        follower = TerminalFollower(TurnTerminalProtocol(self.store), lease, execution)
        turn_executions.register(execution)
        self.follower = follower
        self._on_close = on_close
        try:
            follower.start()
        except BaseException:
            self.close_if_unlaunched()
            raise

    def close_if_unlaunched(self):
        if not self._launched:
            self._close()

    def begin_inference(self):
        if self.follower is not None:
            self.follower.check()
        self._inference_started = True

    def fail_setup(self, error_text: str, *, from_worker: bool = False):
        """Fence an actual pre-launch failure; a prior Stop/owner still wins."""
        if self.follower is None or (
            self._launched and (not from_worker or self._inference_started)
        ):
            return
        try:
            self.follower.protocol.commit(
                self.follower.lease, status="error", end_reason="setup_error",
                error_text=error_text,
            )
        except FollowerFenceLost:
            # Neither claim a new token nor interrupt a newer owner's work.
            return
        except Exception:
            # Preserve the original setup error. Durable lease expiry allows
            # recovery to finish a failed write once the database is reachable.
            log.warning("Could not commit pre-launch terminal failure", exc_info=True)

    def _close(self):
        follower = self.follower
        if follower is None or self._closed:
            return
        self._closed = True
        follower.close()
        execution = follower.execution
        # Never remove a different execution that owns the same key.
        if turn_executions.get(execution.turn_id) is execution:
            turn_executions.remove(execution.turn_id)
        if self._on_close is not None:
            self._on_close()

    def launch_worker(self, loop, worker, *, execution=None):
        if self.follower is None:
            return (
                execution.launch_worker(loop, worker) if execution is not None
                else loop.run_in_executor(None, worker)
            )
        self.follower.check()

        def run():
            try:
                return worker()
            finally:
                self._close()

        self._launched = True
        try:
            return self.follower.execution.launch_worker(loop, run)
        except BaseException:
            self._launched = False
            self._close()
            raise

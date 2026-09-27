"""Per-session message serialization and active-task tracking.

Shared between OpenClaw and Claude Code bridges to ensure consistent
behavior: only one send runs at a time per session, and abort support
works uniformly.

Inspired by the Claude Code CLI's ``messageQueueManager`` which queues
messages arriving mid-turn and processes them sequentially after the
current turn completes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)


class SessionQueue:
    """Serialize sends per session and track active tasks for abort.

    Both the OpenClaw gateway and Claude Code SDK process one turn at a
    time per session.  Sending concurrent messages to the same session
    causes state corruption (run-queue overwrites, stale history fallback,
    tangled SDK session state).

    Usage in a bridge command listener::

        queue = SessionQueue()

        # In _command_listener, when dispatching a chat.send:
        task = asyncio.create_task(self._handle_send(fields, ...))
        queue.set_active_task(session_key, task)

        # Inside _handle_send, wrap the actual work:
        async with queue.lock(session_key):
            # ... do the send ...

        # For abort / chat.abort RPC:
        queue.cancel_active(session_key)
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._request_tasks: dict[tuple[str, str], asyncio.Task] = {}
        # Cooperative cancel signals — handlers check this between SDK
        # messages so an abort takes effect without waiting for the next
        # `await` point to fire CancelledError.
        self._cancel_events: dict[str, asyncio.Event] = {}
        # Opaque per-session abort handles.
        # - Claude Code bridge stores a ClaudeSDKClient and abort calls
        #   `disconnect()` to tear down the subprocess.
        # - Codex bridge stores an AbortController and abort calls `.abort()`.
        # Keep this registry duck-typed so shared agent_bridge code does not
        # import SDK-specific classes.
        self._active_clients: dict[str, Any] = {}

    # -- Lock management --------------------------------------------------

    def lock(self, session_key: str) -> asyncio.Lock:
        """Get or create the asyncio.Lock for a session.

        Returns the lock object for use with ``async with``.
        """
        existing = self._locks.get(session_key)
        if existing is None:
            existing = asyncio.Lock()
            self._locks[session_key] = existing
        return existing

    def is_busy(self, session_key: str) -> bool:
        """Return True if a session has an active send (lock is held)."""
        lk = self._locks.get(session_key)
        return lk is not None and lk.locked()

    # -- Active task tracking (for abort) ---------------------------------

    def set_active_task(self, session_key: str, task: asyncio.Task) -> None:
        """Register *task* as the active send for *session_key*.

        When the task completes (success or failure) it is automatically
        removed from the registry.
        """
        self._active_tasks[session_key] = task
        task.add_done_callback(
            lambda done, sk=session_key: self.clear_active_task(sk, done)
        )

    def clear_active_task(
        self, session_key: str, task: asyncio.Task | None = None
    ) -> bool:
        """Remove the active task only when it is still *task*.

        A completed task must never remove a newer task registered for the
        same session. That race made chat.abort report success while leaving
        the newer bridge subprocess running.
        """
        current = self._active_tasks.get(session_key)
        if current is None or (task is not None and current is not task):
            return False
        self._active_tasks.pop(session_key, None)
        return True

    @asynccontextmanager
    async def active(
        self,
        session_key: str,
        *,
        on_wait: Callable[[], None] | None = None,
        on_cancel: Callable[[], Awaitable[None]] | None = None,
        wait_interval: float = 30.0,
        request_id: str | None = None,
    ):
        """Serialize a send and expose only the lock holder to abort.

        Tasks waiting for the session lock are queued, not active. Registering
        them when they were created allowed a queued send to replace the
        running send in ``_active_tasks``, so abort cancelled the wrong task.
        ``on_wait`` provides a non-terminal heartbeat for callers subscribed to
        the queued request's run stream; without it, a legitimate long queue wait
        is indistinguishable from a bridge that accepted and then lost the send.
        """
        lock = self.lock(session_key)
        if wait_interval <= 0:
            raise ValueError("wait_interval must be positive")

        was_queued = lock.locked()
        if was_queued and on_wait is not None:
            on_wait()
        acquire_task = asyncio.create_task(lock.acquire())
        acquired = False
        task: asyncio.Task | None = None
        owner = asyncio.current_task()
        request_key = (session_key, request_id) if request_id else None
        if request_key is not None and owner is not None:
            self._request_tasks[request_key] = owner
        try:
            while True:
                done, _ = await asyncio.wait(
                    {acquire_task}, timeout=wait_interval
                )
                if acquire_task in done:
                    acquire_task.result()
                    acquired = True
                    break
                if on_wait is not None:
                    on_wait()

            task = asyncio.current_task()
            if task is not None:
                self.clear_cancel_event(session_key)
                self.set_active_task(session_key, task)
            yield
        except asyncio.CancelledError:
            # Also finalize callers cancelled while waiting for the session lock.
            if on_cancel is not None:
                await on_cancel()
            raise
        finally:
            if not acquire_task.done():
                acquire_task.cancel()
                await asyncio.gather(acquire_task, return_exceptions=True)
            elif not acquired:
                # Cancellation can race lock acquisition. If this waiter won the
                # lock before cancellation landed, release it for the next waiter.
                try:
                    acquire_task.result()
                except BaseException:
                    pass
                else:
                    acquired = True
            if task is not None:
                self.clear_active_task(session_key, task)
            if acquired:
                lock.release()
            if request_key is not None and self._request_tasks.get(request_key) is owner:
                self._request_tasks.pop(request_key, None)

    def cancel_request(self, session_key: str, request_id: str) -> bool:
        """Cancel exactly one queued/running send, never its session siblings."""
        task = self._request_tasks.get((session_key, request_id))
        if task is None or task.done():
            return False
        if self._active_tasks.get(session_key) is task:
            self.signal_cancel(session_key)
        # Repeated cancellation must not interrupt terminal publication/cleanup.
        if not task.cancelling():
            task.cancel()
        return True

    def cancel_active(self, session_key: str) -> bool:
        """Cancel the active task for *session_key*.

        Returns True if a running task was found and cancelled.
        """
        task = self._active_tasks.get(session_key)
        if task and not task.done():
            task.cancel()
            logger.info("Cancelled active task for session: %s", session_key)
            return True
        return False

    def has_active_task(self, session_key: str) -> bool:
        """Return True if *session_key* has a running (non-done) task."""
        task = self._active_tasks.get(session_key)
        return task is not None and not task.done()

    # -- Cooperative cancel events ---------------------------------------

    def cancel_event(self, session_key: str) -> asyncio.Event:
        """Get or create the cooperative cancel event for *session_key*.

        Handlers should check ``event.is_set()`` between SDK messages and
        bail out cleanly when set. The event survives across handler
        invocations until explicitly cleared with ``clear_cancel_event``.
        """
        existing = self._cancel_events.get(session_key)
        if existing is None:
            existing = asyncio.Event()
            self._cancel_events[session_key] = existing
        return existing

    def signal_cancel(self, session_key: str) -> bool:
        """Set the cooperative cancel event for *session_key*.

        Returns True if a fresh signal was raised (event went from clear
        to set), False if the event was already set or no event existed.
        Creates the event if it doesn't yet exist so a chat.abort that
        races a slow chat.send still arms the flag for the next iteration.
        """
        event = self.cancel_event(session_key)
        already_set = event.is_set()
        event.set()
        if not already_set:
            logger.info("Cooperative cancel signalled: %s", session_key)
        return not already_set

    def clear_cancel_event(self, session_key: str) -> None:
        """Drop the cancel event for *session_key*.

        Call this from the handler's `finally` block so the next send
        starts with a fresh, un-set event.
        """
        self._cancel_events.pop(session_key, None)

    # -- Active client tracking (for forced subprocess kill / interrupt) ---

    def set_active_client(self, session_key: str, client: Any) -> None:
        """Register the active per-session abort handle for *session_key*.

        The handle is bridge-specific: Claude Code stores a
        ``ClaudeSDKClient`` and abort calls ``disconnect()``; Codex stores
        an ``AbortController`` and abort calls ``abort()``.
        """
        self._active_clients[session_key] = client

    def pop_active_client(self, session_key: str) -> Any | None:
        """Remove and return the active abort handle for *session_key*."""
        return self._active_clients.pop(session_key, None)

    def get_active_client(self, session_key: str) -> Any | None:
        """Return the active abort handle for *session_key* without removing it."""
        return self._active_clients.get(session_key)

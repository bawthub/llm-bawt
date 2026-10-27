"""The Codex ``app-server`` JSON-RPC transport over stdio (TASK-1243 P2).

``codex exec`` runs one turn per process and can only be stopped by killing
it. ``codex app-server`` keeps the turn addressable while it runs, so the
bridge can add input to it (``turn/steer``) and stop it cleanly
(``turn/interrupt``). A bot opts in with ``agent_backend_config.codex_transport
= "app_server"``; everything else stays on exec.

The protocol is newline-delimited JSON-RPC on the child's stdin/stdout,
without the ``"jsonrpc"`` member: requests ``{"id", "method", "params"}``,
replies ``{"id", "result"}`` or ``{"id", "error"}``, notifications
``{"method", "params"}``. The server can also send requests of its own
(approvals, user input, tool calls, auth refresh). The bridge runs with
``approvalPolicy=never`` and has no client-side tools, so it answers every
one of them with a JSON-RPC error: fail closed, never an implied approval.

The pinned :data:`APP_SERVER_VERSION` is the version whose schema
(``tests/fixtures/codex_app_server/``) this client was written against. A
different binary is not trusted to speak it: the send falls back to exec for
that attempt, which cannot steer, so the bridge refuses steers to it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_bridge.steer_contract import STEER_RUN_FINISHING

from .exec_patch import read_event_line
from .process_ledger import register_process

logger = logging.getLogger(__name__)

APP_SERVER_VERSION = "0.157.1"
CLIENT_NAME = "llm_bawt_codex_bridge"
CLIENT_VERSION = "1"

# JSON-RPC error codes. app-server answers a steer or interrupt that has no
# live turn to act on (or names the wrong one) with INVALID_REQUEST.
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601

STREAM_LIMIT_BYTES = 1024 * 1024
STDERR_TAIL_LINES = 40
# Teardown escalation: close stdin (app-server exits on EOF), then SIGTERM,
# then SIGKILL, waiting this long after each step.
CLOSE_STEP_GRACE_S = 2.0
# One control RPC (steer/interrupt) must answer within this.
CONTROL_RPC_TIMEOUT_S = 10.0

INTERRUPT_SENT = "interrupt_sent"
TURN_INTERRUPTED = "turn_interrupted"
TURN_ENDED = "turn_ended"
INTERRUPT_TIMEOUT = "interrupt_timeout"
INTERRUPT_NO_TURN = "interrupt_no_turn"
# Stop reasons after which the turn is known to be over without a kill.
GRACEFUL_STOPS = frozenset({TURN_INTERRUPTED, TURN_ENDED})

_CLOSED = object()


class AppServerError(RuntimeError):
    """A JSON-RPC error reply. The message keeps app-server's own text, so the
    bridge's auth and session-error predicates still read it."""

    def __init__(self, method: str, code: Any, message: str) -> None:
        super().__init__(f"{method}: {message} (code {code})")
        self.method = method
        self.code = code


class AppServerClosed(RuntimeError):
    """app-server exited, or its pipe closed, before answering.

    ``sent`` is False when the request never reached app-server (it was
    already gone), so nothing it asked for can have happened; True when it was
    written and the reply was lost, so the outcome is unknown.
    """

    def __init__(self, message: str, *, sent: bool) -> None:
        super().__init__(message)
        self.sent = sent


class AppServerVersionMismatch(RuntimeError):
    """The binary is not the app-server version this client speaks."""


def server_version(initialize_result: Any) -> str | None:
    """The version in ``initialize``'s ``userAgent`` (``<client>/<version> (...)``)."""
    agent = initialize_result.get("userAgent") if isinstance(initialize_result, Mapping) else None
    if not isinstance(agent, str) or "/" not in agent:
        return None
    rest = agent.split("/", 1)[1].split()
    return rest[0] if rest else None


class JsonRpcStdioClient:
    """One app-server child process and its JSON-RPC conversation."""

    def __init__(self, proc: Any) -> None:
        self._proc = proc
        self._next_id = 0
        self._pending: dict[int, tuple[str, asyncio.Future]] = {}
        self._notes: asyncio.Queue = asyncio.Queue()
        self._observers: list[Callable[[dict], None]] = []
        self._write_lock = asyncio.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        self._refusals: set[asyncio.Task] = set()
        self.closed = asyncio.Event()
        self._close_task: asyncio.Task | None = None
        self._reader = asyncio.create_task(self._read_stdout())
        self._stderr_reader = asyncio.create_task(self._read_stderr())

    @property
    def pid(self) -> Any:
        return getattr(self._proc, "pid", None)

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def add_observer(self, observer: Callable[[dict], None]) -> None:
        """Call ``observer(note)`` for every notification, as it is read."""
        self._observers.append(observer)

    async def request(self, method: str, params: Any, *, timeout: float | None = None) -> Any:
        """Send one request and return its ``result``; raises :class:`AppServerError`."""
        if self.closed.is_set():
            raise self._closed_error(method, sent=False)
        self._next_id += 1
        request_id = self._next_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = (method, future)
        try:
            await self._write({"id": request_id, "method": method, "params": params})
            if timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: Any = None) -> None:
        message: dict[str, Any] = {"method": method}
        if params is not None:
            message["params"] = params
        await self._write(message)

    async def next_notification(self, timeout: float) -> dict | None:
        """The next notification, or ``None`` once app-server has exited.

        Raises ``TimeoutError`` if none arrives within ``timeout``.
        """
        item = await asyncio.wait_for(self._notes.get(), timeout)
        if item is _CLOSED:
            self._notes.put_nowait(_CLOSED)
            return None
        return item

    async def _write(self, message: dict) -> None:
        data = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        async with self._write_lock:
            stdin = self._proc.stdin
            name = str(message.get("method") or "reply")
            if stdin is None or stdin.is_closing():
                raise self._closed_error(name, sent=False)
            try:
                stdin.write(data)
                await stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                # Its stdin is gone: app-server can no longer read anything.
                raise self._closed_error(name, sent=False) from exc

    def _closed_error(self, method: str, *, sent: bool) -> AppServerClosed:
        tail = self.stderr_tail()
        return AppServerClosed(
            f"codex app-server exited before answering {method}" + (f": {tail}" if tail else ""),
            sent=sent,
        )

    async def _read_stdout(self) -> None:
        try:
            while True:
                line = await read_event_line(self._proc.stdout)
                if not line:
                    return
                try:
                    message = json.loads(line)
                except ValueError:
                    logger.warning("codex app-server wrote a non-JSON line: %.200r", line)
                    continue
                if isinstance(message, dict):
                    self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("codex app-server reader failed")
        finally:
            self.closed.set()
            for method, future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(self._closed_error(method, sent=True))
            self._notes.put_nowait(_CLOSED)

    async def _read_stderr(self) -> None:
        stream = getattr(self._proc, "stderr", None)
        if stream is None:
            return
        with contextlib.suppress(Exception):
            while True:
                line = await read_event_line(stream)
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._stderr_tail.append(text)
                    logger.debug("codex app-server: %s", text)

    def _dispatch(self, message: dict) -> None:
        method = message.get("method")
        if method is not None and "id" in message:
            task = asyncio.create_task(self._refuse(message["id"], str(method)))
            self._refusals.add(task)
            task.add_done_callback(self._refusals.discard)
            return
        if method is not None:
            for observer in self._observers:
                try:
                    observer(message)
                except Exception:
                    logger.exception("codex app-server observer failed on %s", method)
            self._notes.put_nowait(message)
            return
        pending = self._pending.get(message.get("id"))  # type: ignore[arg-type]
        if pending is None:
            logger.debug("codex app-server reply for no pending request: id=%r", message.get("id"))
            return
        name, future = pending
        if future.done():
            return
        error = message.get("error")
        if error is not None:
            code = error.get("code") if isinstance(error, Mapping) else None
            text = error.get("message") if isinstance(error, Mapping) else str(error)
            future.set_exception(AppServerError(name, code, str(text or "error")))
        else:
            future.set_result(message.get("result"))

    async def _refuse(self, request_id: Any, method: str) -> None:
        logger.warning("Refusing codex app-server request %s: the bridge answers none (fail closed)", method)
        with contextlib.suppress(AppServerClosed):
            await self._write({
                "id": request_id,
                "error": {"code": METHOD_NOT_FOUND, "message": f"{method} is not supported by this client"},
            })

    def close(self) -> asyncio.Task:
        """Start (once) the teardown and return its task; await it for the detail.

        The task outlives a cancelled awaiter, so ``await asyncio.shield(
        client.close())`` always finishes reaping the child.
        """
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._teardown())
        return self._close_task

    async def _teardown(self) -> str:
        proc = self._proc
        step = "already_exited"
        if proc.returncode is None:
            step = "stdin"
            stdin = proc.stdin
            if stdin is not None and not stdin.is_closing():
                with contextlib.suppress(Exception):
                    stdin.close()
            if not await _reaped(proc, CLOSE_STEP_GRACE_S):
                step = "sigterm"
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
                if not await _reaped(proc, CLOSE_STEP_GRACE_S):
                    step = "sigkill"
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    await _reaped(proc, CLOSE_STEP_GRACE_S)
        for task in (self._reader, self._stderr_reader, *self._refusals):
            if not task.done():
                task.cancel()
        await asyncio.gather(self._reader, self._stderr_reader, *self._refusals, return_exceptions=True)
        exited = proc.returncode is not None
        return f"teardown:close={step};exited={'true' if exited else 'false'}"


async def _reaped(proc: Any, timeout: float) -> bool:
    try:
        await asyncio.wait_for(proc.wait(), timeout)
    except TimeoutError:
        return False
    return True


async def spawn_app_server(
    executable: str,
    *,
    overrides: Iterable[str],
    env: Mapping[str, str],
    cwd: str | None,
    timeout: float,
) -> JsonRpcStdioClient:
    """Start ``codex [--config k=v ...] app-server`` and complete its handshake.

    ``--config`` pairs are global options, so they go before the subcommand.
    The process is ledgered before anything else can fail, so the send's
    teardown report covers it. Raises :class:`AppServerVersionMismatch` (after
    reaping the child) when the binary is not :data:`APP_SERVER_VERSION`.
    """
    args = [part for item in overrides for part in ("--config", item)]
    proc = await asyncio.create_subprocess_exec(
        executable, *args, "app-server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env),
        cwd=cwd,
        limit=STREAM_LIMIT_BYTES,
    )
    register_process(proc)
    client = JsonRpcStdioClient(proc)
    try:
        result = await client.request(
            "initialize",
            {"clientInfo": {"name": CLIENT_NAME, "title": None, "version": CLIENT_VERSION}},
            timeout=timeout,
        )
        version = server_version(result)
        if version != APP_SERVER_VERSION:
            raise AppServerVersionMismatch(
                f"codex app-server {version or 'unknown'} is not the supported {APP_SERVER_VERSION}"
            )
        await client.notify("initialized")
    except BaseException:
        await asyncio.shield(client.close())
        raise
    return client


@dataclass
class PendingSteer:
    """A steer sent into the live turn, until its outcome is known."""

    origin: str
    steer_rpc_id: str | None
    accepted: bool = False
    started: bool = False
    resolved: asyncio.Event = field(default_factory=asyncio.Event)


class CodexAppServerRun:
    """The send's live app-server turn: the steer and stop target.

    Registered as the session's active client once ``turn/start`` has
    returned a turn id, and removed when the send leaves the turn. The send
    owns the client and its teardown; this object only issues control RPCs
    and exposes the state an abort waits on.
    """

    def __init__(self, client: JsonRpcStdioClient, *, request_id: str, thread_id: str) -> None:
        self.client = client
        self.request_id = request_id
        self.thread_id = thread_id
        self.turn_id: str | None = None
        self.turn_status: str | None = None
        self.stopping = False
        self.turn_done = asyncio.Event()
        self.closed = asyncio.Event()
        self.teardown_detail: str | None = None
        self.pending_steers: dict[str, PendingSteer] = {}
        self._finished_turns: dict[str, str] = {}
        client.add_observer(self._observe)

    def set_turn(self, turn_id: str) -> None:
        self.turn_id = turn_id
        if turn_id in self._finished_turns:
            self.mark_finished(self._finished_turns[turn_id])

    def _observe(self, note: dict) -> None:
        if note.get("method") != "turn/completed":
            return
        turn = (note.get("params") or {}).get("turn") or {}
        turn_id = turn.get("id")
        if not isinstance(turn_id, str):
            return
        status = str(turn.get("status") or "")
        # The turn/start reply and this notification can be read back to
        # back, before the send has recorded the turn id.
        self._finished_turns[turn_id] = status
        if turn_id == self.turn_id:
            self.mark_finished(status)

    def mark_finished(self, status: str) -> None:
        """Record the turn's end once; the first status wins."""
        if self.turn_done.is_set():
            return
        self.turn_status = status
        self.turn_done.set()

    async def steer(self, text: str, *, message_id: str | None, origin: str,
                    steer_rpc_id: str | None) -> None:
        """Add ``text`` to the running turn (``turn/steer``).

        Raises ``RuntimeError(run_finishing)`` when nothing was delivered
        because the turn is stopping or over: a definite non-delivery.
        """
        if self.stopping or self.turn_done.is_set() or self.turn_id is None:
            raise RuntimeError(STEER_RUN_FINISHING)
        params: dict[str, Any] = {
            "threadId": self.thread_id,
            "expectedTurnId": self.turn_id,
            "input": [{"type": "text", "text": text, "text_elements": []}],
        }
        pending = None
        if message_id:
            params["clientUserMessageId"] = message_id
            pending = self.pending_steers[message_id] = PendingSteer(origin, steer_rpc_id)
        try:
            await self.client.request("turn/steer", params, timeout=CONTROL_RPC_TIMEOUT_S)
        except BaseException as exc:
            if message_id:
                self.pending_steers.pop(message_id, None)
            # Definite non-delivery: app-server refused it (no live turn, or
            # not this one), or never received it. A lost reply or a timeout
            # is not: the steer may be in the turn, so it surfaces as is.
            if (isinstance(exc, AppServerError) and exc.code == INVALID_REQUEST) or (
                isinstance(exc, AppServerClosed) and not exc.sent
            ):
                raise RuntimeError(STEER_RUN_FINISHING) from exc
            raise
        else:
            if pending is not None:
                pending.accepted = True
        finally:
            if pending is not None:
                pending.resolved.set()

    async def interrupt(self) -> str:
        """Ask app-server to interrupt the turn; returns a stop reason."""
        if self.turn_done.is_set():
            return TURN_ENDED
        if self.turn_id is None:
            return INTERRUPT_NO_TURN
        try:
            await self.client.request(
                "turn/interrupt", {"threadId": self.thread_id, "turnId": self.turn_id},
                timeout=CONTROL_RPC_TIMEOUT_S,
            )
        except AppServerError as exc:
            if exc.code == INVALID_REQUEST:
                # No live turn to interrupt: it is ending on its own.
                await self.turn_done.wait()
                return TURN_ENDED
            return f"interrupt_error:{exc.code}"
        except AppServerClosed:
            # app-server is gone, so the turn is too.
            self.mark_finished("interrupted")
            return TURN_ENDED
        except TimeoutError:
            return INTERRUPT_TIMEOUT
        return INTERRUPT_SENT

    async def stop(self, bound: float) -> str:
        """Interrupt and wait up to ``bound`` for the turn to end.

        ``turn_interrupted``/``turn_ended`` mean the turn is over without a
        kill (:data:`GRACEFUL_STOPS`); anything else means the caller must
        cancel the send.
        """
        self.stopping = True
        try:
            return await asyncio.wait_for(self._interrupt_and_wait(), bound)
        except TimeoutError:
            return INTERRUPT_TIMEOUT

    async def _interrupt_and_wait(self) -> str:
        reason = await self.interrupt()
        if reason != INTERRUPT_SENT:
            return reason
        await self.turn_done.wait()
        return TURN_INTERRUPTED if self.turn_status == "interrupted" else TURN_ENDED

    def track_close(self, close_task: asyncio.Task) -> None:
        """Record the client's teardown when ``close_task`` finishes.

        A done callback, not the awaiting send, so the record lands even if
        the send is cancelled again while it waits on the shielded close.
        """
        def _done(task: asyncio.Task) -> None:
            if task.cancelled():
                self.teardown_detail = "teardown:cancelled"
            elif task.exception() is not None:
                self.teardown_detail = f"teardown:error={type(task.exception()).__name__}"
            else:
                self.teardown_detail = task.result()
            self.mark_finished("closed")
            self.closed.set()

        close_task.add_done_callback(_done)

    async def wait_closed(self, bound: float) -> str:
        try:
            await asyncio.wait_for(self.closed.wait(), bound)
        except TimeoutError:
            return "teardown:pending"
        return self.teardown_detail or "teardown:unknown"


__all__ = [
    "APP_SERVER_VERSION", "AppServerClosed", "AppServerError", "AppServerVersionMismatch",
    "CodexAppServerRun", "GRACEFUL_STOPS", "INVALID_REQUEST", "JsonRpcStdioClient",
    "PendingSteer", "server_version", "spawn_app_server",
]

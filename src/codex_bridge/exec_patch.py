from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from .process_ledger import register_process

logger = logging.getLogger(__name__)

#: Hard ceiling for one JSONL event line from ``codex exec``. Lines are no
#: longer bounded by the StreamReader buffer limit (see ``read_event_line``);
#: this only stops a runaway child from growing bridge memory without bound.
#: A full-viewport Playwright screenshot event is a few MiB of base64.
MAX_EVENT_LINE_BYTES = 64 * 1024 * 1024


async def read_event_line(
    stream: asyncio.StreamReader, *, max_bytes: int = MAX_EVENT_LINE_BYTES,
) -> bytes:
    """Read one newline-terminated line of any length up to ``max_bytes``.

    ``StreamReader.readline()`` raises once a line outgrows the reader's
    buffer limit, which killed whole turns whenever a tool result carried a
    screenshot. This drains the over-limit line in buffer-sized pieces
    instead. Returns ``b""`` at EOF, or the trailing partial line if the
    stream ends without a newline (same contract as ``readline``).
    """
    parts: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = await stream.readuntil(b"\n")
        except asyncio.IncompleteReadError as exc:
            chunk = exc.partial
            done = True
        except asyncio.LimitOverrunError as exc:
            # The buffer holds ``exc.consumed`` bytes that contain no complete
            # line yet; take them and keep scanning for the separator.
            chunk = await stream.readexactly(exc.consumed)
            done = False
        else:
            done = True
        total += len(chunk)
        if total > max_bytes:
            from openai_codex_sdk.errors import CodexExecError

            raise CodexExecError(
                f"Codex event line exceeded {max_bytes} bytes; aborting turn"
            )
        parts.append(chunk)
        if done:
            if total > stream._limit:
                logger.info("Read oversized Codex event line (%d bytes)", total)
            return b"".join(parts)


def install(*, stream_limit: int = 1024 * 1024) -> None:
    """Make the SDK subprocess reader tolerate arbitrarily large JSON events.

    The upstream SDK uses StreamReader.readline(), which raises
    LimitOverrunError/ValueError once a line outgrows the reader limit
    (asyncio's default is 64 KiB). Codex then looks dead to the bridge and the
    turn is killed. Screenshot tool results routinely exceed even a raised
    limit, so we patch CodexExec.run once at startup to read lines with
    ``read_event_line`` (no buffer-limit ceiling) and keep ``stream_limit`` as
    the buffer size for the ordinary small-line case.
    """
    from openai_codex_sdk import exec as exec_mod
    from openai_codex_sdk.abort import AbortError, _format_abort_reason
    from openai_codex_sdk.errors import CodexExecError

    if getattr(exec_mod.CodexExec.run, "__llm_bawt_stream_limit_patch__", False):
        return

    async def patched_run(self, args) -> AsyncIterator[str]:
        if args.signal is not None and args.signal.aborted:
            raise AbortError(_format_abort_reason(args.signal.reason))

        command_args = self._build_command_args(args)
        env = self._build_env(args)

        proc = await asyncio.create_subprocess_exec(
            self.executable_path,
            *command_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        # TASK-1233: the send's teardown report checks this process was reaped.
        register_process(proc)

        if proc.stdin is None or proc.stdout is None:
            try:
                proc.kill()
            finally:
                raise CodexExecError("Child process missing stdin/stdout")

        for stream in (proc.stdout, proc.stderr):
            if stream is None:
                continue
            try:
                current = getattr(stream, "_limit", 0)
                if current < stream_limit:
                    stream._limit = stream_limit
            except Exception:
                pass

        stderr_task = asyncio.create_task(exec_mod._read_all(proc.stderr))
        abort_waiter = None
        if args.signal is not None:
            abort_waiter = asyncio.create_task(exec_mod._wait_abort(args.signal))

        try:
            proc.stdin.write(args.input.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            while True:
                line_task = asyncio.create_task(read_event_line(proc.stdout))

                if abort_waiter is None:
                    done, _pending = await asyncio.wait(
                        {line_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                else:
                    done, _pending = await asyncio.wait(
                        {line_task, abort_waiter},
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                if abort_waiter is not None and abort_waiter in done:
                    line_task.cancel()
                    await asyncio.gather(line_task, return_exceptions=True)
                    await exec_mod._terminate_process(proc)
                    raise AbortError(
                        _format_abort_reason(args.signal.reason if args.signal else None)
                    )

                line = line_task.result()
                if not line:
                    break

                yield line.decode("utf-8", errors="replace").rstrip("\n")

            returncode = await proc.wait()
            stderr = await stderr_task

            if returncode != 0:
                raise CodexExecError(
                    f"Codex Exec exited with code {returncode}: {stderr.decode('utf-8', errors='replace')}"
                )

        finally:
            if abort_waiter is not None:
                abort_waiter.cancel()
                await asyncio.gather(abort_waiter, return_exceptions=True)

            if proc.returncode is None:
                await exec_mod._terminate_process(proc)

            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)

    patched_run.__llm_bawt_stream_limit_patch__ = True
    exec_mod.CodexExec.run = patched_run
    logger.info("Installed Codex stream-limit patch (limit=%s bytes)", stream_limit)

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from .process_ledger import register_process

logger = logging.getLogger(__name__)


def install(*, stream_limit: int = 1024 * 1024) -> None:
    """Raise the SDK subprocess StreamReader limit to tolerate large JSON events.

    The upstream SDK uses StreamReader.readline() with asyncio's default
    64 KiB limit. Large Codex JSON event lines can exceed that and trigger
    LimitOverrunError/ValueError, which then looks like an idle timeout from
    the bridge. We patch CodexExec.run once at startup and bump stdout/stderr
    reader limits before reading.
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
                line_task = asyncio.create_task(proc.stdout.readline())

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

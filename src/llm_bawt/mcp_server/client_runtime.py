"""Neutral sync/async bridge shared by the memory client modules (TASK-1222).

Lives outside ``client.py`` so the client mixins can import it without a
cycle; ``client.py`` re-exports ``_run_async`` for existing callers.
"""

from __future__ import annotations

import asyncio


def _run_async(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're in an async context, create a new thread to run
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)

"""Model-independent supervision for ordinary Responses HTTP streams.

Reuse the same progress classifier/deadlines as WS/Lite, without changing the
HTTP payload or selecting a different protocol. The outer adapter owns retries.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from .chatgpt_transport import (
    ChatGPTStream,
    DEFAULT_PRODUCTIVE_IDLE_TIMEOUT,
)


class ResponsesSSEStream(ChatGPTStream):
    """HTTP headers and first-event consumption share one deadline.

    Header stalls enter the output-aware progress-stall classifier; HTTP/auth
    failures retain the initial-request path. Every retry creates a new stream;
    unfinished HTTP responses are closed.
    """

    def __init__(self, opener, *, context=None, first_event_timeout=60.0,
                 productive_idle_timeout=DEFAULT_PRODUCTIVE_IDLE_TIMEOUT):
        limits = SimpleNamespace(
            first_event_timeout=first_event_timeout, idle_timeout=240.0,
            productive_idle_timeout=productive_idle_timeout,
        )
        super().__init__(limits, None, SimpleNamespace(turn_state=None), context=context)
        self._opener = opener

    @property
    def transport(self):
        return "sse"

    @property
    def fallback_transport(self):
        return "sse"

    async def prepare(self):
        """Keep HTTP/auth errors at the adapter's existing initial boundary.

        Only liveness timeouts are deferred to iteration, where the shared
        progress-stall policy handles visibility and bounded recovery.
        """
        deadline, phase = self._deadline(time.monotonic())
        self._opening_timeout = None
        try:
            self.http = await asyncio.wait_for(
                self._opener(), max(0, deadline - time.monotonic()),
            )
        except asyncio.TimeoutError:
            self._opening_timeout = self._timeout(phase, time.monotonic())
            return
        self.response = getattr(self.http, "response", None)
        self.http_iterator = self.http.__aiter__()

    async def __anext__(self):
        if self.closed or self.complete:
            raise StopAsyncIteration
        if not hasattr(self, "_opening_timeout"):
            await self.prepare()
        if self._opening_timeout is not None:
            raise self._opening_timeout
        return await super().__anext__()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.http is not None:
            close = getattr(self.http, "close", None) or getattr(self.http, "aclose", None)
            if close is not None:
                await close()

    aclose = close

    async def discard(self):
        await self.close()

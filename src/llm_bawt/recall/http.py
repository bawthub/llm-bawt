"""Streaming byte limits applied before parsing external JSON."""
from __future__ import annotations

import json

import httpx

from .contracts import RecallError


class BoundedHTTP:
    def __init__(self, client: httpx.AsyncClient, *, max_calls: int, max_bytes: int):
        self.client = client
        self.max_calls = max_calls
        self.remaining_bytes = max_bytes
        self.calls = 0

    async def json(self, method: str, url: str, *, byte_limit: int = 65_536,
                   status_codes: dict[int, str] | None = None, **kwargs):
        """``status_codes`` maps an HTTP error status to the caller's own error code."""
        if self.calls >= self.max_calls or self.remaining_bytes <= 0:
            raise RecallError("retrieval_budget")
        self.calls += 1
        chunks = bytearray()
        try:
            async with self.client.stream(method, url, **kwargs) as response:
                if response.status_code == 504:
                    raise RecallError("timeout")
                if response.status_code == 429:
                    raise RecallError("busy")
                if response.status_code >= 400:
                    raise RecallError((status_codes or {}).get(response.status_code, "source_unavailable"))
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    self.remaining_bytes -= len(chunk)
                    if len(chunks) + len(chunk) > byte_limit or self.remaining_bytes < 0:
                        raise RecallError("retrieval_budget")
                    chunks.extend(chunk)
            return json.loads(chunks)
        except httpx.TimeoutException as error:
            raise RecallError("timeout") from error
        except (ValueError, httpx.HTTPError) as error:
            raise RecallError("source_unavailable") from error
